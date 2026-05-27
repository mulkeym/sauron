import pytest
from unittest.mock import patch
from src.agent.synthesizer import synthesize_answer
from src.agent.state import AgentState, QueryType
from src.retrieval.models import RetrievedChunk, ChunkMetadata


def _make_chunk(text, doc_id="d1", filename="policy.pdf", page=None, score=0.9):
    return RetrievedChunk(
        text=text,
        score=score,
        metadata=ChunkMetadata(
            doc_id=doc_id,
            filename=filename,
            doc_type="pdf",
            chunk_index=0,
            start_char=0,
            acl_groups=["finance"],
            page=page,
        ),
    )


def test_synthesize_with_chunks():
    with patch("src.agent.synthesizer.generate", return_value="Expenses over $500 need approval [1]."):
        state = AgentState(
            question="What is the expense policy?",
            user_groups=["finance"],
            query_type=QueryType.LOOKUP,
            retrieved_chunks=[_make_chunk("All expenses over $500 require manager approval.", page=12)],
            sql_results=[],
        )
        result = synthesize_answer(state)
    assert "approval" in result["answer"].lower() or "500" in result["answer"]
    assert len(result["citations"]) == 1
    assert result["citations"][0].filename == "policy.pdf"
    assert result["citations"][0].page == 12


def test_synthesize_with_sql_results():
    with patch("src.agent.synthesizer.generate", return_value="Q3 2026 revenue was $1,500,000."):
        state = AgentState(
            question="What was Q3 revenue?",
            user_groups=["finance"],
            query_type=QueryType.ANALYTICAL,
            retrieved_chunks=[],
            sql_results=[{"quarter": "Q3", "revenue": 1500000, "year": 2026}],
        )
        result = synthesize_answer(state)
    assert "1,500,000" in result["answer"] or "1500000" in result["answer"]


def test_synthesize_no_context():
    state = AgentState(
        question="Something with no results",
        user_groups=["finance"],
        query_type=QueryType.LOOKUP,
        retrieved_chunks=[],
        sql_results=[],
    )
    result = synthesize_answer(state)
    assert "could not find" in result["answer"].lower()
    assert result["citations"] == []


def test_synthesize_handles_context_cap_without_nameerror():
    """When the assembled context exceeds llm_max_context, the synthesizer logs a
    'Context cap reached' line and stops adding chunks. That branch referenced an
    undefined `priority_chunks` (NameError) — regression guard. With a tiny cap,
    the first chunk overflows and the branch must run cleanly."""
    with patch("src.agent.synthesizer.generate", return_value="ok"), \
         patch("src.config.settings.llm_max_context", 50):
        state = AgentState(
            question="What are the pay rates?",
            user_groups=["finance"],
            query_type=QueryType.LOOKUP,
            retrieved_chunks=[
                _make_chunk("x" * 200, doc_id="d1", score=0.95),
                _make_chunk("y" * 200, doc_id="d2", score=0.90),
            ],
            sql_results=[],
        )
        result = synthesize_answer(state)  # must not raise NameError
    assert result["answer"] == "ok"


def _chunk(text, doc_id="d1", tier="medium", score=0.9):
    return RetrievedChunk(
        text=text, score=score,
        metadata=ChunkMetadata(
            doc_id=doc_id, filename="pay.xlsx", doc_type="xlsx",
            chunk_index=0, start_char=0, acl_groups=["finance"],
            chunk_size_tier=tier,
        ),
    )


def test_sweep_keeps_structured_narratives_drops_raw_when_mapreduce():
    """With a map-reduce synthesis present, structured table_row narratives are
    kept in the synthesis context but bulky raw sweep chunks are dropped."""
    captured = {}

    def fake_generate(**kwargs):
        captured["user_prompt"] = kwargs.get("user_prompt", "")
        return "answer [1]"

    mr = _chunk("Map-reduce synthesis of pay docs.", doc_id="map-reduce", tier="medium")
    narrative = _chunk("locality=Tampa, grade=GS-12: salary is 86415", doc_id="dpay", tier="table_row")
    raw = _chunk("RAWSWEEPBLOB huge raw spreadsheet text", doc_id="dpay", tier="large")

    with patch("src.agent.synthesizer.generate", fake_generate):
        state = AgentState(
            question="GS rates in Tampa", user_groups=["finance"],
            query_type=QueryType.SWEEP,
            retrieved_chunks=[mr, narrative, raw], sql_results=[],
        )
        synthesize_answer(state)

    ctx = captured["user_prompt"]
    assert "Map-reduce synthesis" in ctx          # synthetic kept
    assert "locality=Tampa" in ctx                # structured narrative kept
    assert "RAWSWEEPBLOB" not in ctx              # raw sweep chunk dropped


def test_no_mapreduce_keeps_raw_chunks():
    """Regression: without a map-reduce chunk, raw chunks are still included."""
    captured = {}

    def fake_generate(**kwargs):
        captured["user_prompt"] = kwargs.get("user_prompt", "")
        return "answer [1]"

    raw = _chunk("RAWSWEEPBLOB huge raw spreadsheet text", doc_id="dpay", tier="large")
    with patch("src.agent.synthesizer.generate", fake_generate):
        state = AgentState(
            question="GS rates in Tampa", user_groups=["finance"],
            query_type=QueryType.LOOKUP,
            retrieved_chunks=[raw], sql_results=[],
        )
        synthesize_answer(state)
    assert "RAWSWEEPBLOB" in captured["user_prompt"]
