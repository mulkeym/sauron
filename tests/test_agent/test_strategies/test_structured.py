"""Tests for shared structured retrieval (SQL core + gate + retrieve_structured)."""
from types import SimpleNamespace

import duckdb
import pytest

from src.agent.strategies.structured import StructuredLookupTrace

from src.ingestion.tabular import SheetGrid, SheetClassification
from src.ingestion.tabular_store import load_sheet_to_duckdb, schema_from_sheet
from src.agent.strategies import structured
from src.agent.strategies.structured import structured_sql_rows


def _pay_schema_and_db(tmp_path, monkeypatch):
    from src.config import settings
    monkeypatch.setattr(settings, "tabular_duckdb_path", str(tmp_path / "t.duckdb"))
    grid = SheetGrid("Pay", [["grade", "step", "salary"]] + [[f"GS-{g}", 5, 80000 + g] for g in range(10, 14)])
    cls = SheetClassification("Pay", "clean", 0, ["text", "number", "number"], "clean table")
    from src.ingestion.tabular_store import connect_tabular
    con = connect_tabular(read_only=False)
    load_sheet_to_duckdb(con, "doc1", "Pay", cls, grid)
    con.close()
    db = str(tmp_path / "t.duckdb")
    return schema_from_sheet("doc1", "Pay", cls, grid, acl_groups=["ALL"]), db


def test_structured_sql_rows_generates_and_runs(tmp_path, monkeypatch):
    schema, _ = _pay_schema_and_db(tmp_path, monkeypatch)
    monkeypatch.setattr(structured, "generate",
                        lambda **kw: f'SELECT grade, salary FROM "{schema.table}" WHERE step = 5 ORDER BY salary')
    rows = structured_sql_rows("engineer pay", [schema])
    assert rows[0] == {"grade": "GS-10", "salary": 80010.0}
    assert len(rows) == 4


def test_run_structured_lookup_captures_sql_and_schema_context(tmp_path, monkeypatch):
    """The trace must carry the executed SQL and a schema reference (column
    meanings + value glossary) for the table(s) the SQL touched, so the
    synthesizer can interpret the raw result rows."""
    from src.agent.strategies.structured import run_structured_lookup
    from src.agent.strategies.hint_resolver import ResolvedHints
    schema, _ = _pay_schema_and_db(tmp_path, monkeypatch)
    monkeypatch.setattr(structured, "generate",
                        lambda **kw: f'SELECT * FROM "{schema.table}" WHERE step = 5')
    hints = {schema.table: ResolvedHints(
        column_glossaries={"grade": {"GS-10": "Grade 10"}}, table_notes=["pay table"])}
    trace = run_structured_lookup("pay", [schema], "analytical", hints=hints)
    assert trace.status == "ran"
    assert "WHERE step = 5" in trace.sql
    assert "grade" in trace.schema_context              # column described
    assert "GS-10 = Grade 10" in trace.schema_context   # glossary rendered
    assert trace.to_dict()["schema_context"] == trace.schema_context
    schema, _ = _pay_schema_and_db(tmp_path, monkeypatch)
    monkeypatch.setattr(structured, "generate", lambda **kw: 'SELECT * FROM "doc_other_secret"')
    with pytest.raises(Exception):
        structured_sql_rows("x", [schema])  # allowlist rejects -> raises (caller falls back)


from src.agent.strategies.structured import tables_relevant_to
from src.db.schema_registry import TableSchema, ColumnSchema


def _schema(table, desc):
    return TableSchema(database="spreadsheets", table=table,
                       columns=[ColumnSchema("grade", "DOUBLE", "")],
                       description=desc, acl_groups=["ALL"])


def test_gate_keeps_only_matching_tables():
    pay = _schema("doc_pay", "GS pay rates by grade and locality")
    weather = _schema("doc_weather", "daily weather observations")
    # Deterministic fake embeddings: question vector == pay-text vector; weather orthogonal.
    eq = lambda q: [1.0, 0.0]
    et = lambda texts: [[1.0, 0.0] if "pay" in t.lower() else [0.0, 1.0] for t in texts]
    out = tables_relevant_to("what is the pay for grade 12", [pay, weather],
                             threshold=0.5, embed_query_fn=eq, embed_texts_fn=et)
    assert [s.table for s in out] == ["doc_pay"]


def test_gate_empty_when_no_schemas():
    assert tables_relevant_to("anything", []) == []


from unittest.mock import MagicMock
from src.agent.strategies.structured import retrieve_structured
from src.retrieval.models import RetrievedChunk, ChunkMetadata


def _narr_chunk():
    return RetrievedChunk(text="GS pay grade=GS-12: salary is 86415", score=0.9,
                          metadata=ChunkMetadata(doc_id="d", filename="f", doc_type="xlsx",
                          chunk_index=0, start_char=0, acl_groups=["ALL"], chunk_size_tier="table_row"))


def _reg(schemas):
    r = MagicMock()
    r.list_for_user.return_value = schemas
    return r


@pytest.mark.asyncio
async def test_retrieve_structured_returns_sql_and_narratives(monkeypatch):
    schemas = [SimpleNamespace(table="t_pay", description="pay", columns=[SimpleNamespace(name="salary")])]

    class _Reg:
        def list_for_user(self, g): return schemas

    monkeypatch.setattr(structured, "tables_relevant_scored",
                        lambda q, s: [(schemas[0], 0.71, True)])
    monkeypatch.setattr(structured, "run_structured_lookup",
                        lambda q, s, query_type, gate=None: StructuredLookupTrace(
                            query_type="sweep", gate=gate, status="ran", sql="SELECT 1",
                            row_count=1, sample_rows=[{"salary": 86415.0}], rows=[{"salary": 86415.0}]))
    monkeypatch.setattr(structured, "embed_query", lambda q: [0.0])

    class _VS:
        def search(self, **kw): return [SimpleNamespace(text="narr", metadata=SimpleNamespace(doc_id="d", chunk_index=0))]

    out = await structured.retrieve_structured(
        {"question": "gs pay", "user_groups": ["ALL"]}, vector_store=_VS(), schema_registry=_Reg())
    assert out["sql_results"] == [{"salary": 86415.0}]
    assert out["structured_trace"]["status"] == "ran"
    assert out["structured_trace"]["gate"] == [["t_pay", 0.71, True]]


@pytest.mark.asyncio
async def test_retrieve_structured_skipped_trace_when_gate_misses(monkeypatch):
    schemas = [SimpleNamespace(table="t_pay", description="pay", columns=[SimpleNamespace(name="salary")])]

    class _Reg:
        def list_for_user(self, g): return schemas

    monkeypatch.setattr(structured, "tables_relevant_scored",
                        lambda q, s: [(schemas[0], 0.10, False)])

    class _VS:
        def search(self, **kw): return []

    out = await structured.retrieve_structured(
        {"question": "weather", "user_groups": ["ALL"]}, vector_store=_VS(), schema_registry=_Reg())
    assert "sql_results" not in out
    assert out["structured_trace"]["status"] == "skipped"
    assert out["structured_trace"]["gate"] == [["t_pay", 0.1, False]]


@pytest.mark.asyncio
async def test_retrieve_structured_fail_open_on_gate_error(monkeypatch):
    class _Reg:
        def list_for_user(self, g): raise RuntimeError("embeddings down")

    out = await structured.retrieve_structured(
        {"question": "x", "user_groups": ["ALL"]}, vector_store=None, schema_registry=_Reg())
    assert out == {}


# --- _extract_sql: robust SQL extraction from LLM responses ---

def test_extract_sql_from_prose_with_fenced_block():
    """The real failure mode: model hedges in prose, then emits a ```sql fence."""
    resp = (
        'Wait, "TAMPA" is not in the locname values. However, if I must:\n\n'
        "```sql\nSELECT * FROM t WHERE locname = 'TU'\n```"
    )
    assert structured._extract_sql(resp) == "SELECT * FROM t WHERE locname = 'TU'"


def test_extract_sql_bare_statement():
    assert structured._extract_sql("SELECT * FROM t") == "SELECT * FROM t"


def test_extract_sql_fenced_without_language():
    assert structured._extract_sql("```\nSELECT 1\n```") == "SELECT 1"


def test_extract_sql_sql_fenced_block():
    assert structured._extract_sql("```sql\nSELECT 1\n```") == "SELECT 1"


def test_extract_sql_takes_last_fenced_block():
    resp = "```sql\nSELECT 1\n```\nthen reconsidering:\n```sql\nSELECT 2 FROM t\n```"
    assert structured._extract_sql(resp) == "SELECT 2 FROM t"


def test_extract_sql_from_prose_without_fence():
    resp = "Sure, here is the query: SELECT a FROM t WHERE x = 1"
    assert structured._extract_sql(resp) == "SELECT a FROM t WHERE x = 1"


from src.agent.strategies.structured import (
    StructuredLookupTrace, run_structured_lookup, tables_relevant_scored,
)


def test_run_structured_lookup_success(tmp_path, monkeypatch):
    schema, _ = _pay_schema_and_db(tmp_path, monkeypatch)
    monkeypatch.setattr(
        structured, "generate",
        lambda **kw: f'SELECT grade, salary FROM "{schema.table}" ORDER BY grade',
    )
    trace = run_structured_lookup("engineer pay", [schema], query_type="analytical")
    assert trace.status == "ran"
    assert trace.query_type == "analytical"
    assert trace.gate is None
    assert trace.row_count == 4
    assert trace.sample_rows == trace.rows[:5]
    assert "SELECT" in trace.sql
    d = trace.to_dict()
    assert d["status"] == "ran" and d["row_count"] == 4 and "rows" not in d


def test_run_structured_lookup_error_captures_sql(tmp_path, monkeypatch):
    schema, _ = _pay_schema_and_db(tmp_path, monkeypatch)
    monkeypatch.setattr(structured, "generate", lambda **kw: 'SELECT * FROM "doc_other_secret"')
    trace = run_structured_lookup("x", [schema], query_type="analytical")
    assert trace.status == "error"
    assert trace.error
    assert trace.fell_back is True
    assert 'doc_other_secret' in trace.sql


def test_run_structured_lookup_zero_rows(tmp_path, monkeypatch):
    schema, _ = _pay_schema_and_db(tmp_path, monkeypatch)
    monkeypatch.setattr(
        structured, "generate",
        lambda **kw: f'SELECT * FROM "{schema.table}" WHERE grade = \'NOPE\'',
    )
    trace = run_structured_lookup("x", [schema], query_type="sweep", gate=[["t", 0.5, True]])
    assert trace.status == "ran"
    assert trace.row_count == 0
    assert trace.sample_rows == []
    assert trace.gate == [["t", 0.5, True]]


@pytest.mark.asyncio
async def test_structured_applies_feedback_boosts_to_narratives(monkeypatch):
    from src.agent.strategies import structured as st
    from src.retrieval.models import RetrievedChunk, ChunkMetadata

    class S:
        table = "all_gs"
    monkeypatch.setattr(st, "tables_relevant_scored", lambda q, schemas: [(S(), 0.9, True)])

    class FakeTrace:
        rows = [{"x": 1}]
        def to_dict(self):
            return {"status": "ran"}
    monkeypatch.setattr(st, "run_structured_lookup", lambda *a, **k: FakeTrace())
    monkeypatch.setattr(st, "embed_query", lambda q: [0.0, 0.1])

    async def fake_boosts(qv, ug):
        return {"docB": 0.5}
    monkeypatch.setattr(st, "get_feedback_boosts", fake_boosts, raising=False)

    def _c(doc_id, idx, score):
        return RetrievedChunk(text="t", score=score,
            metadata=ChunkMetadata(doc_id=doc_id, filename="f", doc_type="text",
                                   chunk_index=idx, start_char=0, acl_groups=["ALL"]))

    class FakeVS:
        def search(self, **k):
            return [_c("docA", 0, 0.8), _c("docB", 1, 0.4)]

    class FakeRegistry:
        def list_for_user(self, ug):
            return [S()]

    state = {"question": "q", "user_groups": ["ALL"]}
    result = await st.retrieve_structured(state, vector_store=FakeVS(), schema_registry=FakeRegistry())
    chunks = result["retrieved_chunks"]
    assert chunks[0].metadata.doc_id == "docB"  # 0.4+0.5 > 0.8


def test_tables_relevant_scored_reports_all_scores(monkeypatch):
    from types import SimpleNamespace
    schemas = [
        SimpleNamespace(table="t_hi", description="pay", columns=[SimpleNamespace(name="salary")]),
        SimpleNamespace(table="t_lo", description="weather", columns=[SimpleNamespace(name="temp")]),
    ]
    monkeypatch.setattr(structured, "embed_query", lambda q: [1.0, 0.0])
    monkeypatch.setattr(structured, "embed_texts", lambda texts: [[1.0, 0.0], [0.0, 1.0]])
    scored = tables_relevant_scored("pay?", schemas)
    assert scored[0][0].table == "t_hi" and scored[0][2] is True
    assert scored[1][0].table == "t_lo" and scored[1][2] is False
    assert scored[0][1] > scored[1][1]
