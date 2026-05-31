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


def test_generate_run_fit_caps_schema_prompt_budget(tmp_path, monkeypatch):
    """The SQL-generation schema prompt must be rendered with the configured
    hard char budget, so it can never overflow the model context."""
    from src.config import settings
    monkeypatch.setattr(settings, "sql_schema_prompt_budget_chars", 12345)
    schema, _ = _pay_schema_and_db(tmp_path, monkeypatch)

    import src.ingestion.tabular_store as ts
    orig = ts.schema_prompt_with_values
    captured = {}

    def spy(schemas, con, **kw):
        captured["budget"] = kw.get("max_total_chars")
        return orig(schemas, con, **kw)
    monkeypatch.setattr(ts, "schema_prompt_with_values", spy)
    monkeypatch.setattr(structured, "generate", lambda **kw: f'SELECT * FROM "{schema.table}"')

    structured_sql_rows("pay", [schema])
    assert captured["budget"] == 12345


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
                        lambda q, s, query_type, gate=None, generate_fn=None, hints=None: StructuredLookupTrace(
                            query_type="sweep", gate=gate, status="ran", sql="SELECT 1",
                            row_count=1, sample_rows=[{"salary": 86415.0}], rows=[{"salary": 86415.0}]))

    async def _no_hints(schemas_arg, hint_store, metadata_store):
        return {}
    monkeypatch.setattr(structured, "resolve_hints_for_schemas", _no_hints)
    monkeypatch.setattr(structured, "embed_query", lambda q: [0.0])

    class _VS:
        def search(self, **kw): return [SimpleNamespace(text="narr", metadata=SimpleNamespace(doc_id="d", chunk_index=0))]

    out = await structured.retrieve_structured(
        {"question": "gs pay", "user_groups": ["ALL"]}, vector_store=_VS(), schema_registry=_Reg())
    assert out["sql_results"] == [{"salary": 86415.0}]
    assert out["structured_trace"]["status"] == "ran"
    assert out["structured_trace"]["gate"] == [["t_pay", 0.71, True]]


@pytest.mark.asyncio
async def test_retrieve_structured_routes_when_many_tables_pass(monkeypatch):
    """When the gate passes many tables, route them down before SQL so the
    schema prompt stays bounded — only the routed tables reach run_structured_lookup."""
    from src.config import settings
    monkeypatch.setattr(settings, "sql_table_routing_enabled", True)
    monkeypatch.setattr(settings, "sql_table_routing_max_selected", 8)
    monkeypatch.setattr(settings, "sql_table_routing_catalog_budget_chars", 10_000_000)

    schemas = [SimpleNamespace(table=f"t{i}", description=f"d{i}",
                               columns=[SimpleNamespace(name="a")]) for i in range(40)]

    class _Reg:
        def list_for_user(self, g):
            return schemas

    monkeypatch.setattr(structured, "tables_relevant_scored",
                        lambda q, s: [(x, 0.5, True) for x in s])   # gate passes ALL
    monkeypatch.setattr(structured, "generate", lambda **kw: '["t5"]')  # router picks t5

    captured = {}

    def fake_run(q, s, query_type, gate=None, generate_fn=None, hints=None):
        captured["tables"] = [x.table for x in s]
        return StructuredLookupTrace(query_type="sweep", gate=gate, status="ran",
                                     sql="SELECT 1", row_count=1,
                                     sample_rows=[{"a": 1}], rows=[{"a": 1}])
    monkeypatch.setattr(structured, "run_structured_lookup", fake_run)

    async def _no_hints(s, hs, ms):
        return {}
    monkeypatch.setattr(structured, "resolve_hints_for_schemas", _no_hints)
    monkeypatch.setattr(structured, "embed_query", lambda q: [0.0])

    class _VS:
        def search(self, **kw):
            return []

    await structured.retrieve_structured(
        {"question": "find t5", "user_groups": ["ALL"]}, vector_store=_VS(), schema_registry=_Reg())
    assert captured["tables"] == ["t5"]


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
async def test_retrieve_structured_passes_resolved_hints(monkeypatch):
    """The SWEEP structured path must resolve domain hints (value glossaries) and
    pass them to run_structured_lookup — parity with retrieve_analytical. Without
    this the SQL generator sees bare locality codes and can't map e.g. 'florida'."""
    schemas = [SimpleNamespace(table="t_pay", description="pay",
                               columns=[SimpleNamespace(name="locname")])]

    class _Reg:
        def list_for_user(self, g):
            return schemas

    monkeypatch.setattr(structured, "tables_relevant_scored",
                        lambda q, s: [(schemas[0], 0.71, True)])
    import src.api.routes_ingest as ri
    monkeypatch.setattr(ri, "get_hint_store", lambda: "HS")
    monkeypatch.setattr(ri, "get_metadata_store", lambda: "MS")

    SENTINEL = {"t_pay": "RESOLVED"}

    async def fake_resolve(schemas_arg, hint_store, metadata_store):
        return SENTINEL
    monkeypatch.setattr(structured, "resolve_hints_for_schemas", fake_resolve)

    captured = {}

    def fake_run(q, s, query_type, gate=None, generate_fn=None, hints=None):
        captured["hints"] = hints
        return StructuredLookupTrace(query_type="sweep", gate=gate, status="ran",
                                     sql="SELECT 1", row_count=1,
                                     sample_rows=[{"x": 1}], rows=[{"x": 1}])
    monkeypatch.setattr(structured, "run_structured_lookup", fake_run)
    monkeypatch.setattr(structured, "embed_query", lambda q: [0.0])

    class _VS:
        def search(self, **kw):
            return []

    await structured.retrieve_structured(
        {"question": "what is the pay rate for florida?", "user_groups": ["ALL"]},
        vector_store=_VS(), schema_registry=_Reg())
    assert captured["hints"] == SENTINEL


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


def test_extract_sql_returns_empty_when_no_sql_keyword():
    """Reasoning prose that never commits to a SELECT/WITH (e.g. thinking that ran
    out of tokens mid-ramble and leaked via reasoning_content) must NOT be passed
    to the executor as a query. Return '' so the caller fails cleanly into the
    repair loop instead of running reasoning text against DuckDB."""
    prose = "Actually, I'll just use locname='FL'. Wait, let me check the list again. Hmm."
    assert structured._extract_sql(prose) == ""


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


def test_run_structured_lookup_skips_when_model_declines(tmp_path, monkeypatch):
    """When text-to-SQL never produces a SELECT (the model judges that no
    available table can answer the question), that is a clean SKIP — not an
    error. The misleading 'Only SELECT queries are allowed' must not surface."""
    schema, _ = _pay_schema_and_db(tmp_path, monkeypatch)
    monkeypatch.setattr(structured, "generate",
                        lambda **kw: "These tables hold pay data, not contract awards, so I cannot answer.")
    trace = run_structured_lookup("list DHA contracts", [schema], query_type="analytical")
    assert trace.status == "skipped"
    assert trace.fell_back is True
    assert trace.error == ""                       # not surfaced as an error
    assert "Only SELECT" not in trace.skip_reason  # no confusing SQL-guard message
    assert trace.skip_reason                        # has a human reason


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
