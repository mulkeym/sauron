"""Catalog-build helpers for file-metadata Q&A."""
from types import SimpleNamespace
from datetime import datetime, timezone

from src.agent.strategies.metadata_catalog import (
    _flatten_tags, build_catalog_connection, CATALOG_SCHEMA,
)
from src.ingestion.tabular_store import execute_duckdb_sql


def _doc(doc_id="d1", filename="pay.pdf", doc_type="pdf", dataset_id=2,
         category="payroll", tags=None):
    return SimpleNamespace(
        doc_id=doc_id, filename=filename, doc_type=doc_type, dataset_id=dataset_id,
        category=category, acl_groups=["executives"], chunk_count=5,
        source_url="", summary="active duty pay", uploaded_by="mike",
        created_at=datetime(2025, 5, 1, tzinfo=timezone.utc),
        metadata_tags=(tags if tags is not None else {"organizations": ["DoD"], "topics": ["Officer pay"]}),
    )


def test_flatten_tags_lowercases_and_joins():
    out = _flatten_tags({"organizations": ["DoD"], "topics": ["Officer Pay"], "amounts": ["$5,000"]})
    assert "dod" in out and "officer pay" in out and "$5,000" in out


def test_flatten_tags_empty():
    assert _flatten_tags(None) == ""
    assert _flatten_tags({}) == ""


def test_build_catalog_has_files_table_with_columns():
    con = build_catalog_connection([_doc()], dataset_names={2: "DoD OPM Policies"})
    rows = execute_duckdb_sql(con, "SELECT doc_id, filename, doc_type, dataset, category, "
                                   "uploaded_by, chunk_count, summary, tags FROM files",
                              allowed_tables={"files"})
    assert len(rows) == 1
    r = rows[0]
    assert r["filename"] == "pay.pdf"
    assert r["dataset"] == "DoD OPM Policies"
    assert "dod" in r["tags"]


def test_build_catalog_count_aggregate():
    docs = [_doc(doc_id="a", doc_type="pdf"), _doc(doc_id="b", doc_type="pdf"),
            _doc(doc_id="c", doc_type="xlsx")]
    con = build_catalog_connection(docs, dataset_names={2: "DoD"})
    rows = execute_duckdb_sql(con, "SELECT doc_type, COUNT(*) AS n FROM files GROUP BY doc_type ORDER BY doc_type",
                              allowed_tables={"files"})
    assert rows == [{"doc_type": "pdf", "n": 2}, {"doc_type": "xlsx", "n": 1}]


def test_build_catalog_tag_search_ilike():
    docs = [_doc(doc_id="a", tags={"topics": ["Officer pay"]}),
            _doc(doc_id="b", tags={"topics": ["Enlisted pay"]})]
    con = build_catalog_connection(docs, dataset_names={2: "DoD"})
    rows = execute_duckdb_sql(con, "SELECT doc_id FROM files WHERE tags ILIKE '%officer%'",
                              allowed_tables={"files"})
    assert rows == [{"doc_id": "a"}]


def test_build_catalog_optional_datasets_categories():
    con = build_catalog_connection(
        [_doc()], dataset_names={2: "DoD"},
        datasets=[SimpleNamespace(name="DoD", description="policies")],
        categories=[SimpleNamespace(name="payroll", description="pay records")])
    ds = execute_duckdb_sql(con, "SELECT name FROM datasets", allowed_tables={"datasets"})
    cat = execute_duckdb_sql(con, "SELECT name FROM categories", allowed_tables={"categories"})
    assert ds == [{"name": "DoD"}] and cat == [{"name": "payroll"}]


def test_catalog_schema_mentions_files_and_tags():
    assert "files" in CATALOG_SCHEMA and "tags" in CATALOG_SCHEMA


def test_build_catalog_created_at_date_filter():
    docs = [_doc(doc_id="may")]  # created_at = 2025-05-01 (UTC)
    apr = _doc(doc_id="apr")
    apr.created_at = datetime(2025, 4, 1, tzinfo=timezone.utc)
    con = build_catalog_connection([*docs, apr], dataset_names={2: "DoD"})
    rows = execute_duckdb_sql(
        con,
        "SELECT doc_id FROM files WHERE created_at >= '2025-05-01' AND created_at < '2025-06-01'",
        allowed_tables={"files"})
    assert rows == [{"doc_id": "may"}]


import pytest
from src.agent.strategies import metadata_catalog as mc


class _Store:
    def __init__(self, docs):
        self._docs = docs
    async def list_documents(self, user_groups=None):
        if user_groups is None:
            return self._docs
        return [d for d in self._docs if any(g in d.acl_groups for g in user_groups)]
    async def list_datasets(self, active_only=True):
        return [SimpleNamespace(id=2, name="DoD", description="")]
    async def list_categories(self):
        return [SimpleNamespace(name="payroll", description="")]


def _state(q="how many pdfs?", groups=("executives",)):
    return {"question": q, "user_groups": list(groups), "retrieval_attempts": 0}


@pytest.mark.asyncio
async def test_aggregate_returns_sql_rows(monkeypatch):
    store = _Store([_doc(doc_id="a"), _doc(doc_id="b")])
    def fake_sql(schema_prompt, question, generate_fn=None):
        return "SELECT COUNT(*) AS n FROM files"
    monkeypatch.setattr(mc, "generate_sql", fake_sql)
    out = await mc.retrieve_metadata_catalog(_state(), metadata_store=store)
    assert out["sql_results"] == [{"n": 2}]
    assert out["structured_trace"]["status"] == "ran"
    assert out["retrieved_chunks"] == []


@pytest.mark.asyncio
async def test_per_file_rows_yield_citation(monkeypatch):
    store = _Store([_doc(doc_id="a", filename="pay.pdf")])
    def fake_sql(schema_prompt, question, generate_fn=None):
        return "SELECT filename, doc_id FROM files"
    monkeypatch.setattr(mc, "generate_sql", fake_sql)
    out = await mc.retrieve_metadata_catalog(_state("list files"), metadata_store=store)
    assert len(out["retrieved_chunks"]) == 1
    assert out["retrieved_chunks"][0].metadata.doc_id == "a"
    assert out["retrieved_chunks"][0].metadata.filename == "pay.pdf"


@pytest.mark.asyncio
async def test_acl_scopes_documents(monkeypatch):
    visible = _doc(doc_id="a")
    hidden = _doc(doc_id="b")
    hidden.acl_groups = ["admins"]
    store = _Store([visible, hidden])
    def fake_sql(schema_prompt, question, generate_fn=None):
        return "SELECT COUNT(*) AS n FROM files"
    monkeypatch.setattr(mc, "generate_sql", fake_sql)
    out = await mc.retrieve_metadata_catalog(_state(groups=("executives",)), metadata_store=store)
    assert out["sql_results"] == [{"n": 1}]


@pytest.mark.asyncio
async def test_sql_error_falls_back_to_snapshot(monkeypatch):
    store = _Store([_doc(doc_id="a"), _doc(doc_id="b")])
    def boom(schema_prompt, question, generate_fn=None):
        raise RuntimeError("bad sql")
    monkeypatch.setattr(mc, "generate_sql", boom)
    out = await mc.retrieve_metadata_catalog(_state(), metadata_store=store)
    assert out["sql_results"] == []
    assert out["structured_trace"]["fell_back"] is True
    assert len(out["retrieved_chunks"]) == 1
    snap = out["retrieved_chunks"][0]
    assert snap.metadata.doc_id == "metadata-context"
    assert "Total documents" in snap.text
