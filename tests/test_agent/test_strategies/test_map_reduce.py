"""Tests for map-reduce retrieval helpers: term weighting, score fusion,
the pre-MAP relevance gate, and the failed-MAP retry queue."""
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.agent.state import AgentState, QueryType
from src.agent.strategies.map_reduce import (
    _term_idf,
    _meta_match_score,
    _rrf_fuse,
    _normalize_relevance,
    _map_documents,
    _prefilter_by_summary,
    retrieve_map_reduce,
    _classify_failure,
    _cap_content,
)
from src.generation.llm_client import LLMTimeoutError, LLMConnectionError
from src.retrieval.models import RetrievedChunk, ChunkMetadata


# --- Term-specificity weighting (IDF) -------------------------------------

def test_common_terms_get_near_zero_weight():
    # "march" appears in every candidate doc -> carries no signal.
    docs = [{"topics": ["march"]} for _ in range(10)]
    idf = _term_idf({"march"}, docs)
    assert idf["march"] == pytest.approx(0.0, abs=0.05)


def test_rare_terms_outweigh_common_terms():
    # "march" in all 10 docs, "navy" in just 1 -> navy should weigh more.
    docs = [{"topics": ["march"]} for _ in range(9)]
    docs.append({"organizations": ["navy"], "topics": ["march"]})
    idf = _term_idf({"march", "navy"}, docs)
    assert idf["navy"] > idf["march"]


def test_meta_score_rewards_specific_match_over_calendar_match():
    idf = {"navy": 1.7, "march": 0.0}
    navy_doc = _meta_match_score(
        {"navy", "march"}, {"organizations": ["Navy"], "topics": ["March"]}, idf
    )
    march_only = _meta_match_score({"navy", "march"}, {"topics": ["March"]}, idf)
    assert navy_doc > march_only


# --- Reciprocal rank fusion -----------------------------------------------

def test_rrf_rewards_docs_ranked_high_in_both_lists():
    vector_ranked = ["a", "b", "c"]
    meta_ranked = ["b", "a", "d"]
    scores = _rrf_fuse([vector_ranked, meta_ranked])
    # "b" tops one list and is near top of the other; beats list-tail-only docs.
    assert scores["b"] > scores["c"]
    assert scores["a"] > scores["d"]


# --- Relevance normalization ----------------------------------------------

def test_normalize_relevance_scales_top_to_one():
    norm = _normalize_relevance({"a": 0.03, "b": 0.015, "c": 0.0})
    assert norm["a"] == pytest.approx(1.0)
    assert norm["b"] == pytest.approx(0.5)
    assert norm["c"] == pytest.approx(0.0)


def test_normalize_relevance_handles_empty_and_zero():
    assert _normalize_relevance({}) == {}
    assert _normalize_relevance({"a": 0.0, "b": 0.0}) == {"a": 0.0, "b": 0.0}


# --- Pre-MAP relevance gate -----------------------------------------------

@pytest.mark.asyncio
async def test_prefilter_keeps_only_docs_judged_relevant():
    summaries = {"d1": "Navy contract award", "d2": "GS pay tables"}

    async def judge(question, summary):
        return "contract" in summary.lower()

    kept = await _prefilter_by_summary(summaries, "navy contracts", judge, concurrency=2)
    assert kept == ["d1"]


@pytest.mark.asyncio
async def test_prefilter_keeps_doc_when_judge_errors():
    # A failed gate call must not silently drop a possibly-relevant doc.
    summaries = {"d1": "anything"}

    async def judge(question, summary):
        raise RuntimeError("LLM timed out")

    kept = await _prefilter_by_summary(summaries, "q", judge, concurrency=2)
    assert kept == ["d1"]


# --- Failed-MAP retry queue -----------------------------------------------

@pytest.mark.asyncio
async def test_failed_map_is_retried_and_can_succeed():
    attempts = {}

    async def map_one(doc_id):
        attempts[doc_id] = attempts.get(doc_id, 0) + 1
        # d2 fails on first attempt, succeeds on retry
        if doc_id == "d2" and attempts[doc_id] == 1:
            return {"doc_id": doc_id, "filename": "f2", "extraction": "", "status": "failed"}
        return {"doc_id": doc_id, "filename": doc_id, "extraction": f"data-{doc_id}", "status": "ok"}

    results, failed = await _map_documents(
        ["d1", "d2", "d3"], map_one, concurrency=2, retry_concurrency=1
    )

    assert attempts["d2"] == 2  # retried
    assert failed == []
    by_id = {r["doc_id"]: r for r in results}
    assert by_id["d2"]["status"] == "ok"
    assert by_id["d2"]["extraction"] == "data-d2"


@pytest.mark.asyncio
async def test_persistent_failures_are_reported_not_dropped():
    async def map_one(doc_id):
        if doc_id == "bad":
            return {"doc_id": doc_id, "filename": "bad", "extraction": "", "status": "failed"}
        return {"doc_id": doc_id, "filename": doc_id, "extraction": "x", "status": "ok"}

    results, failed = await _map_documents(
        ["good", "bad"], map_one, concurrency=2, retry_concurrency=1, max_retries=2
    )

    assert failed == ["bad"]
    # The good doc's data is preserved.
    assert any(r["doc_id"] == "good" and r["status"] == "ok" for r in results)


@pytest.mark.asyncio
async def test_genuine_no_data_is_not_treated_as_failure():
    async def map_one(doc_id):
        # "empty" status = doc read fine, just no relevant content.
        return {"doc_id": doc_id, "filename": doc_id, "extraction": "", "status": "empty"}

    results, failed = await _map_documents(["d1"], map_one, concurrency=1, retry_concurrency=1)
    assert failed == []


# --- Integration: full retrieve_map_reduce flow ---------------------------

def _chunk(doc_id, text, score=0.5):
    return RetrievedChunk(
        text=text, score=score,
        metadata=ChunkMetadata(
            doc_id=doc_id, filename=f"{doc_id}.md", doc_type="md",
            chunk_index=0, start_char=0, acl_groups=["ALL"],
        ),
    )


@pytest.mark.asyncio
async def test_retrieve_map_reduce_gates_then_maps_relevant_docs():
    # Two docs surface from vector search; only the Navy one is relevant.
    summary_results = [_chunk("d1", "navy contract", 0.5), _chunk("d2", "pay tables", 0.4)]
    vector_store = MagicMock()
    vector_store.search.return_value = summary_results
    vector_store.get_chunks_by_doc.side_effect = lambda did, *a, **k: [
        _chunk(did, "Navy awarded $5M to Acme Corp" if did == "d1" else "GS pay rate tables")
    ]

    # Both docs mention "march" (common -> low IDF); only d1 mentions "navy"
    # (rare -> high IDF), so d1 should clearly outrank d2.
    docs = [
        SimpleNamespace(doc_id="d1", filename="d1.md",
                        metadata_tags={"organizations": ["Navy"], "topics": ["march"], "summary": "Navy contract award to Acme"}),
        SimpleNamespace(doc_id="d2", filename="d2.md",
                        metadata_tags={"topics": ["pay", "march"], "summary": "GS pay tables for March"}),
    ]
    store = MagicMock()
    store.list_documents = AsyncMock(return_value=docs)

    def fake_generate(system_prompt, user_prompt, **kw):
        if "Answer YES or NO" in user_prompt:          # pre-MAP gate
            return "YES" if "Navy" in user_prompt else "NO"
        if "Extract ONLY" in user_prompt:              # MAP
            return "Navy awarded $5M to Acme Corp" if "Acme" in user_prompt else "NO_RELEVANT_DATA"
        return "combined"                              # reduce (unused here)

    state = AgentState(question="what contracts did the navy award in march?",
                       user_groups=["ALL"], query_type=QueryType.SWEEP,
                       retrieved_chunks=[], retrieval_attempts=0)

    with patch("src.agent.strategies.map_reduce.embed_query", return_value=[0.1] * 1024), \
         patch("src.agent.strategies.map_reduce.generate", side_effect=fake_generate), \
         patch("src.agent.strategies.sweep._extract_date_filter", return_value=None), \
         patch("src.retrieval.prf.expand_query_with_prf", new=AsyncMock(return_value=("what contracts did the navy award in march?", [0.1] * 1024))), \
         patch("src.retrieval.feedback.get_feedback_boosts", new=AsyncMock(return_value={})), \
         patch("src.api.routes_ingest.get_metadata_store", return_value=store):
        result = await retrieve_map_reduce(state, vector_store=vector_store)

    chunks = result["retrieved_chunks"]
    assert len(chunks) == 1
    text = chunks[0].text
    assert "Acme" in text                  # relevant doc's data made it through
    assert "pay rate tables" not in text   # irrelevant doc was gated out
    # d2 was gated out, so its full-content MAP extraction was never fetched.
    fetched = {c.args[0] for c in vector_store.get_chunks_by_doc.call_args_list}
    assert "d2" not in fetched

    # A meaningful, normalized relevance is reported per candidate doc, so the
    # playground/citations no longer show 0.00 for fetched-by-id chunks.
    rel = result["doc_relevance"]
    assert rel["d1"] == pytest.approx(1.0)   # top-ranked
    assert rel["d1"] > rel.get("d2", 0.0)


# --- Failure classification & payload cap ---------------------------------

def test_timeout_failure_is_permanent():
    assert _classify_failure(LLMTimeoutError("x")) == "permanent"


def test_connection_failure_is_transient():
    assert _classify_failure(LLMConnectionError("x")) == "transient"


def test_unknown_error_is_permanent():
    # Unknown errors default to permanent — never retried, to avoid wasted timeouts.
    assert _classify_failure(ValueError("x")) == "permanent"


def test_cap_content_truncates_over_budget():
    capped = _cap_content("a" * 100, 10)
    assert capped.startswith("a" * 10)
    assert "[truncated]" in capped
    assert len(capped) < 100


def test_cap_content_leaves_small_content_untouched():
    assert _cap_content("short", 100) == "short"
