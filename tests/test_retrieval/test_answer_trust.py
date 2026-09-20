"""Behavioral regressions for scoped evidence and cache reuse (synthetic data only)."""
import re
import time
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.config import settings
from src.retrieval.models import ChunkMetadata, RetrievedChunk
from src.retrieval.query_scope import resolve_query_scope
from src.agent.synthesizer import build_evidence_pack, finalize_answer, synthesize_answer


def doc(did="d1", groups=None, dataset=1, **values):
    data = dict(doc_id=did, acl_groups=groups or ["team"], dataset_id=dataset,
                filename=did + ".pdf", doc_type="pdf", category="sdwan",
                content_hash="revision-one", chunk_count=2, source_url="https://docs.test/" + did)
    data.update(values)
    return SimpleNamespace(**data)


def chunk(text, did="d1", score=0.8, tier="medium"):
    return RetrievedChunk(text=text, score=score, metadata=ChunkMetadata(
        doc_id=did, filename=did + ".pdf", doc_type="pdf", chunk_index=0,
        start_char=0, acl_groups=["team"], chunk_size_tier=tier))


@pytest.fixture(autouse=True)
def isolated(monkeypatch, tmp_path):
    monkeypatch.setattr(settings, "lancedb_path", str(tmp_path / "lance"))
    monkeypatch.setattr(settings, "embedding_dimension", 3)
    monkeypatch.setattr("src.agent.synthesizer._source_urls", lambda ids: {d: "https://docs.test/" + d for d in ids})


@pytest.mark.asyncio
async def test_scope_revision_changes_on_delete_revoke_edit_add_and_settings(monkeypatch):
    docs = [doc(), doc("d2", groups=["other"])]
    store = AsyncMock()
    store.list_documents.side_effect = lambda *a: docs
    original = await resolve_query_scope(["team"], store)
    assert original.doc_ids == ("d1",)
    docs[0].content_hash = "revision-two"
    assert (await resolve_query_scope(["team"], store)).revision != original.revision
    docs[0].content_hash = "revision-one"
    docs[0].acl_groups = ["other"]
    assert (await resolve_query_scope(["team"], store)).doc_ids == ()
    docs[0].acl_groups = ["team"]
    docs.append(doc("d3"))
    assert (await resolve_query_scope(["team"], store)).revision != original.revision
    docs[:] = [docs[1]]
    assert (await resolve_query_scope(["team"], store)).doc_ids == ()
    docs[:] = [doc(), doc("d2", groups=["other"])]
    monkeypatch.setattr(settings, "answer_domain_instructions", "Require platform and version.")
    assert (await resolve_query_scope(["team"], store)).revision != original.revision


@pytest.mark.asyncio
async def test_empty_dataset_and_groups_never_expand_scope():
    store = AsyncMock()
    store.list_documents.return_value = [doc()]
    assert (await resolve_query_scope([], store)).doc_ids == ()
    assert (await resolve_query_scope(["ALL"], store, dataset_id=7)).doc_ids == ()
    assert (await resolve_query_scope(["ALL"], store, allowed_doc_ids=[])).doc_ids == ()
    full = await resolve_query_scope(["ALL"], store)
    vector = await resolve_query_scope(["ALL"], store, mode="vector_only")
    assert full.revision != vector.revision


def test_context_skips_oversized_passage_and_citations_match(monkeypatch):
    monkeypatch.setattr(settings, "llm_max_context", 170)
    pack = build_evidence_pack({"question": "deploy", "retrieved_chunks": [
        chunk("x" * 1000, "large", score=1), chunk("Use the documented command.", "small")]})
    assert "documented command" in pack.context
    assert [c.doc_id for c in pack.citations] == ["small"]
    assert pack.citations[0].snippet == "Use the documented command."
    assert pack.citations[0].source_url.endswith("small")
    assert len(pack.context) <= 170
    assert pack.warnings


def test_no_generation_when_nothing_fits(monkeypatch):
    monkeypatch.setattr(settings, "llm_max_context", 1)
    generator = MagicMock(side_effect=AssertionError("must not generate"))
    monkeypatch.setattr("src.agent.synthesizer.generate", generator)
    result = synthesize_answer({"question": "deploy", "retrieved_chunks": [chunk("details")]})
    assert result["citations"] == []
    generator.assert_not_called()


def test_final_citations_only_include_used_evidence():
    pack = build_evidence_pack({"question": "deploy", "retrieved_chunks": [chunk("one"), chunk("two", "d2")]})
    result = finalize_answer({}, f"Use one [{pack.citations[0].evidence_id}].", pack)
    assert len(result["citations"]) == 1
    assert result["citations"][0].doc_id == "d1"
    invalid = finalize_answer({}, "Invented instruction [Edeadbeefdead].", pack)
    assert "Invented instruction" not in invalid["answer"]
    assert invalid["citations"] == []
    uncited = finalize_answer({}, "Invented instruction.", pack)
    assert "Invented instruction" not in uncited["answer"]


def test_tier_identity_is_preserved():
    from src.agent.state import _merge_chunks
    a, b = chunk("same index", tier="small"), chunk("larger section", tier="medium")
    assert len(_merge_chunks([a], [a, b])) == 2


@pytest.mark.asyncio
async def test_graph_requires_every_contributing_source(monkeypatch):
    from src.knowledge import graph_rag as kg
    store = AsyncMock()
    store.list_documents.return_value = [doc(), doc("d2", groups=["other"])]
    monkeypatch.setattr("src.api.routes_ingest.get_metadata_store", lambda: store)
    rag = AsyncMock()
    rag.aquery_data.return_value = {"data": {
        "entities": [{"file_path": "d1.pdf<SEP>d2.pdf", "description": "restricted merged fact"}],
        "chunks": [{"file_path": "d1.pdf", "content": "allowed source passage"},
                   {"file_path": "d2.pdf", "content": "restricted source passage"}]}}
    get_rag = AsyncMock(return_value=rag)
    monkeypatch.setattr(kg, "get_lightrag", get_rag)
    empty = await kg.query_graph("q", user_groups=[])
    assert empty["context"] == ""
    get_rag.assert_not_awaited()
    result = await kg.query_graph("q", user_groups=["team"])
    assert "allowed source passage" in result["context"]
    assert "restricted" not in result["context"]
    rag.aquery.assert_not_awaited()
    store.list_documents.return_value.append(doc("duplicate", groups=["other"], filename="d1.pdf"))
    assert (await kg.query_graph("q", user_groups=["team"]))["context"] == ""


def test_cache_scopes_expiry_and_legacy_rows(monkeypatch):
    from src.retrieval import query_cache as qc
    monkeypatch.setattr(qc, "_cache_table", None)
    monkeypatch.setattr(settings, "query_cache_mode", "exact")
    citation = {"doc_id": "d1", "source_kind": "document", "evidence_id": "E123456789012"}
    qc.cache_store("deploy", [0.1, 0.2, 0.3], "answer", [citation], ["team"], ["d1"], scope_revision="r1")
    assert qc.cache_lookup([0.1, 0.2, 0.3], ["team"], scope_revision="r1", query_text="deploy")
    assert qc.cache_lookup([0.1, 0.2, 0.3], ["team"], scope_revision="r2", query_text="deploy") is None
    assert qc.cache_lookup([0.1, 0.2, 0.3], ["other"], scope_revision="r1", query_text="deploy") is None
    assert qc.cache_lookup([0.1, 0.2, 0.3], ["team"], scope_revision="r1", query_text="rollback") is None
    monkeypatch.setattr(qc.time, "time", lambda: 10**12)
    assert qc.cache_lookup([0.1, 0.2, 0.3], ["team"], scope_revision="r1", query_text="deploy") is None


@pytest.mark.asyncio
async def test_judge_outage_requires_fresh_retrieval(monkeypatch):
    from src.retrieval.query_cache import cache_judge
    monkeypatch.setattr("src.generation.llm_client.generate", MagicMock(side_effect=RuntimeError("down")))
    assert (await cache_judge("old", "new", "answer"))["applicable"] is False


def test_document_lookup_reads_beyond_global_search_limit(monkeypatch):
    from src.mcp.tools_low import lookup_document
    store = AsyncMock()
    store.list_documents.return_value = [doc()]
    vs = MagicMock()
    vs.read_document_page.return_value = ([chunk("a later passage")], True)
    result = lookup_document("d1", ["team"], vs, store, offset=200, limit=100)
    vs.search.assert_not_called()
    vs.read_document_page.assert_called_once_with("d1", ["team"], offset=200, limit=100)
    assert result["content"] == "a later passage"
    assert result["complete"] is False and result["next_offset"] == 201
    store.list_documents.return_value = [doc(groups=["other"])]
    assert lookup_document("d1", ["team"], vs, store)["content"] == ""


def test_date_filter_intersects_existing_document_scope(monkeypatch):
    from src.agent.strategies.lookup import retrieve_lookup
    monkeypatch.setattr("src.agent.strategies.lookup.embed_query", lambda q: [0.1, 0.2, 0.3])
    monkeypatch.setattr("src.agent.strategies.sweep._extract_date_filter", lambda *a: ["outside-dataset"])
    monkeypatch.setattr("src.agent.strategies.lookup.get_feedback_boosts_sync", lambda *a: {})
    vs = MagicMock()
    vs.hybrid_search_reranked.return_value = []
    vs.expand_window.return_value = []
    retrieve_lookup({"question": "on that date", "user_groups": ["team"], "allowed_doc_ids": ["d1"]}, vs)
    assert vs.hybrid_search_reranked.call_args.kwargs["doc_ids"] == []


def test_structured_tables_are_restricted_to_the_selected_dataset():
    from src.retrieval.query_scope import scoped_schemas
    from src.db.schema_registry import SchemaRegistry, TableSchema
    from src.ingestion.tabular_store import duckdb_table_name
    registry = SchemaRegistry()
    for did in ["d1", "d2"]:
        registry.register(TableSchema(database="spreadsheets", table=duckdb_table_name(did, "Sheet"), columns=[], acl_groups=["team"]))
    registry.register(TableSchema(database="external", table="unrelated", columns=[], acl_groups=["team"]))
    result = scoped_schemas(registry, {"user_groups": ["team"], "allowed_doc_ids": ["d1"], "dataset_id": 1})
    assert [s.table for s in result] == [duckdb_table_name("d1", "Sheet")]
    assert scoped_schemas(registry, {"user_groups": ["team"], "allowed_doc_ids": [], "dataset_id": 1}) == []


@pytest.mark.asyncio
async def test_empty_index_directory_has_a_valid_scope(tmp_path, monkeypatch):
    import lancedb
    lancedb.connect(settings.lancedb_path)
    store = AsyncMock()
    store.list_documents.return_value = []
    assert (await resolve_query_scope(["team"], store)).doc_ids == ()


def test_raw_search_does_not_reuse_stale_vector_acl(monkeypatch):
    from src.mcp.tools_low import search_documents
    store = AsyncMock()
    store.list_documents.return_value = [doc(groups=["other"])]
    vs = MagicMock()
    assert search_documents("deploy", ["team"], vs, metadata_store=store) == []
    vs.search.assert_not_called()


def test_model_aliases_only_cover_budgeted_authorized_passages(monkeypatch):
    from src.config import settings
    a = chunk("top passage", "allowed")
    b = chunk("x" * 2000, "too-large")
    hidden = chunk("secret", "denied")
    monkeypatch.setattr(settings, "llm_max_context", 250)
    pack = build_evidence_pack({"question": "test", "retrieved_chunks": [a, b, hidden],
        "allowed_doc_ids": ["allowed", "too-large"], "edition_decisions": {
            "allowed": {"source_revision": "a" * 64},
        }})
    assert len(pack.aliases) == len(pack.citations) == 1
    assert pack.model_context.startswith("[E1] Source:")
    assert "secret" not in pack.model_context and "x" * 2000 not in pack.model_context
    assert pack.citations[0].source_revision == "a" * 64
    assert len(pack.model_context) <= settings.llm_max_context
    assert any("excluded 1" in w for w in pack.warnings)
    result = finalize_answer({}, "Known [E1].", pack, aliases=pack.aliases)
    assert result["citations"][0].doc_id == "allowed"
    assert result["citations"][0].source_revision == "a" * 64
    assert finalize_answer({}, "Omitted [E2].", pack, aliases=pack.aliases)["response_kind"] == "insufficient_evidence"
