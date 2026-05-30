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


def test_build_synthesis_context_includes_sql_block():
    from src.agent.synthesizer import build_synthesis_context
    state = AgentState(
        question="GS rates in Tampa", user_groups=["finance"],
        query_type=QueryType.ANALYTICAL, retrieved_chunks=[],
        sql_results=[{"annual1": 23440.0}],
        structured_trace={
            "status": "ran", "row_count": 15,
            "sql": "SELECT * FROM all_gs WHERE locname = 'RUS'",
            "schema_context": "Table all_gs — GS pay\nValue meanings:\n  - locname: RUS = Rest of U.S.",
        },
    )
    ctx = build_synthesis_context(state)
    assert "WHERE locname = 'RUS'" in ctx
    assert "RUS = Rest of U.S." in ctx
    assert "23440" in ctx


def test_build_citations_dedupes_chunks_by_document():
    from src.agent.synthesizer import build_citations
    state = AgentState(
        question="policy?", user_groups=["finance"], query_type=QueryType.LOOKUP,
        retrieved_chunks=[_make_chunk("a", doc_id="d1", score=0.8),
                          _make_chunk("b", doc_id="d1", score=0.95)],
        sql_results=[],
    )
    cits = build_citations(state)
    assert len(cits) == 1                      # one per document
    assert cits[0].doc_id == "d1"
    assert cits[0].relevance == 0.95           # best score kept


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


def test_sql_context_includes_executed_sql_and_schema_reference():
    """The answer LLM must receive the executed SQL and the table/column
    reference (meanings + value glossary) alongside the raw rows — otherwise it
    sees only bare column keys/codes and cannot interpret them (the GS-rates
    failure: rows of step salaries with no grade/locality context)."""
    captured = {}

    def fake_generate(**kwargs):
        captured["user_prompt"] = kwargs.get("user_prompt", "")
        return "answer"

    with patch("src.agent.synthesizer.generate", fake_generate):
        state = AgentState(
            question="What are the GS salary rates in Tampa?",
            user_groups=["finance"], query_type=QueryType.ANALYTICAL,
            retrieved_chunks=[], sql_results=[{"annual1": 23440.0}],
            structured_trace={
                "status": "ran", "row_count": 15,
                "sql": "SELECT * FROM all_gs WHERE locname = 'RUS'",
                "schema_context": "Table all_gs — GS pay\nValue meanings:\n  - locname: RUS = Rest of U.S.",
            },
        )
        synthesize_answer(state)
    ctx = captured["user_prompt"]
    assert "WHERE locname = 'RUS'" in ctx        # #1: executed SQL is in context
    assert "RUS = Rest of U.S." in ctx           # #2: schema/value reference is in context
    assert "23440" in ctx                         # rows still present


def _sql_doc_ms(doc_id="docpay", filename="2026-pay.xlsx"):
    class _Doc:
        def __init__(self):
            self.doc_id = doc_id
            self.filename = filename
            self.doc_type = "xlsx"
            self.source_url = ""

    class _MS:
        async def list_documents(self, user_groups=None):
            return [_Doc()]

        async def get_document(self, did):
            return _Doc() if did == doc_id else None

    return _MS()


def test_sql_answer_cites_source_document(monkeypatch):
    from src.ingestion.tabular_store import duckdb_table_name
    tbl = duckdb_table_name("docpay", "pay")
    monkeypatch.setattr("src.api.routes_ingest.get_metadata_store", lambda: _sql_doc_ms())
    with patch("src.agent.synthesizer.generate", lambda **k: "answer"):
        state = AgentState(
            question="pay?", user_groups=["finance"], query_type=QueryType.ANALYTICAL,
            retrieved_chunks=[], sql_results=[{"salary": 91162}],
            structured_trace={"status": "ran", "row_count": 15, "sql": f'SELECT * FROM "{tbl}"'},
        )
        result = synthesize_answer(state)
    cits = [c for c in result["citations"] if c.doc_id == "docpay"]
    assert len(cits) == 1
    assert cits[0].filename == "2026-pay.xlsx"
    assert "15 rows" in cits[0].snippet
    assert cits[0].relevance == 1.0


def test_sql_block_labels_source_filename(monkeypatch):
    """The SQL-results context block must name its source document(s) by filename,
    so the answer LLM cites the original Excel filename rather than the raw DuckDB
    table name (e.g. doc_bb9025d1_..._allleo)."""
    from src.ingestion.tabular_store import duckdb_table_name
    tbl = duckdb_table_name("docpay", "AllLEO")
    monkeypatch.setattr("src.api.routes_ingest.get_metadata_store", lambda: _sql_doc_ms())
    captured = {}

    def fake_generate(**kwargs):
        captured["user_prompt"] = kwargs.get("user_prompt", "")
        return "answer"

    with patch("src.agent.synthesizer.generate", fake_generate):
        state = AgentState(
            question="pay range for an officer in Tampa?", user_groups=["finance"],
            query_type=QueryType.ANALYTICAL,
            retrieved_chunks=[], sql_results=[{"salary": 91162}],
            structured_trace={"status": "ran", "row_count": 15, "sql": f'SELECT * FROM "{tbl}"'},
        )
        synthesize_answer(state)
    ctx = captured["user_prompt"]
    assert "Source: 2026-pay.xlsx" in ctx          # filename labels the SQL block
    assert tbl not in ctx.split("Result rows:")[0].split("Executed SQL:")[0]  # no raw table name before the SQL itself


def test_sql_citation_deduped_with_chunk(monkeypatch):
    from src.ingestion.tabular_store import duckdb_table_name
    tbl = duckdb_table_name("docpay", "pay")
    monkeypatch.setattr("src.api.routes_ingest.get_metadata_store", lambda: _sql_doc_ms())
    chunk = _chunk("locality=Tampa: salary 91162", doc_id="docpay", tier="table_row")
    with patch("src.agent.synthesizer.generate", lambda **k: "answer"):
        state = AgentState(
            question="pay?", user_groups=["finance"], query_type=QueryType.SWEEP,
            retrieved_chunks=[chunk], sql_results=[{"salary": 91162}],
            structured_trace={"status": "ran", "row_count": 15, "sql": f'SELECT * FROM "{tbl}"'},
        )
        result = synthesize_answer(state)
    assert len([c for c in result["citations"] if c.doc_id == "docpay"]) == 1


def test_no_sql_citation_when_zero_rows(monkeypatch):
    from src.ingestion.tabular_store import duckdb_table_name
    tbl = duckdb_table_name("docpay", "pay")
    monkeypatch.setattr("src.api.routes_ingest.get_metadata_store", lambda: _sql_doc_ms())
    chunk = _chunk("some prose", doc_id="d1", tier="large")
    with patch("src.agent.synthesizer.generate", lambda **k: "answer"):
        state = AgentState(
            question="pay?", user_groups=["finance"], query_type=QueryType.ANALYTICAL,
            retrieved_chunks=[chunk], sql_results=[],
            structured_trace={"status": "ran", "row_count": 0, "sql": f'SELECT * FROM "{tbl}"'},
        )
        result = synthesize_answer(state)
    assert all(c.doc_id != "docpay" for c in result["citations"])


def test_synthesis_context_caps_wide_sql_result():
    from src.agent.synthesizer import build_synthesis_context
    from src.config import settings
    # Realistic OPM GS shape: 885 rows x 32 cols of real-looking values.
    big = [{f"col_{c}": f"value_{r}_{c}_xxxxxxxxxx" for c in range(32)} for r in range(885)]
    state = AgentState(
        question="what are the pay rates?",
        user_groups=["finance"],
        query_type=QueryType.ANALYTICAL,
        retrieved_chunks=[],
        sql_results=big,
        structured_trace={"sql": "SELECT * FROM gs_pay", "schema_context": "gs_pay(...)"},
    )
    ctx = build_synthesis_context(state)
    assert len(ctx) <= settings.llm_max_context          # never overflows
    assert "showing" in ctx and "of 885" in ctx          # truncation is disclosed
