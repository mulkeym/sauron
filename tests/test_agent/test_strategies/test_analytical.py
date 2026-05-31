"""Tests for analytical retrieval routed to DuckDB."""
from unittest.mock import MagicMock

import pytest

from src.agent.strategies import analytical
from src.agent.strategies.analytical import retrieve_analytical
import src.agent.strategies.structured as structured
from src.db.schema_registry import SchemaRegistry, TableSchema, ColumnSchema
from src.ingestion.tabular import SheetGrid, SheetClassification
from src.ingestion.tabular_store import connect_tabular, load_sheet_to_duckdb, schema_from_sheet


def _make_pay_duckdb(tmp_path, monkeypatch):
    """Create a tabular.duckdb with a doc_doc1_pay table; return (registry, table)."""
    from src.config import settings
    monkeypatch.setattr(settings, "tabular_duckdb_path", str(tmp_path / "t.duckdb"))
    grid = SheetGrid("Pay", [["grade", "step", "salary"]] + [[f"GS-{g}", 5, 80000 + g] for g in range(10, 14)])
    cls = SheetClassification("Pay", "clean", 0, ["text", "number", "number"], "clean table")
    con = connect_tabular(read_only=False)
    table, _ = load_sheet_to_duckdb(con, "doc1", "Pay", cls, grid)
    con.close()
    reg = SchemaRegistry()
    reg.register(schema_from_sheet("doc1", "Pay", cls, grid, acl_groups=["ALL"]))
    return reg, table


def _pay_schema_and_db(tmp_path, monkeypatch):
    """Build the pay tabular.duckdb and return (schema, db_path)."""
    from src.config import settings
    monkeypatch.setattr(settings, "tabular_duckdb_path", str(tmp_path / "t.duckdb"))
    grid = SheetGrid("Pay", [["grade", "step", "salary"]] + [[f"GS-{g}", 5, 80000 + g] for g in range(10, 14)])
    cls = SheetClassification("Pay", "clean", 0, ["text", "number", "number"], "clean table")
    con = connect_tabular(read_only=False)
    load_sheet_to_duckdb(con, "doc1", "Pay", cls, grid)
    con.close()
    return schema_from_sheet("doc1", "Pay", cls, grid, acl_groups=["ALL"]), str(tmp_path / "t.duckdb")


@pytest.mark.asyncio
async def test_analytical_runs_sql_against_duckdb(tmp_path, monkeypatch):
    reg, table = _make_pay_duckdb(tmp_path, monkeypatch)
    monkeypatch.setattr(structured, "generate",
                        lambda **kw: f'SELECT grade, salary FROM "{table}" WHERE step = 5 ORDER BY salary')

    state = {"question": "engineer pay", "user_groups": ["ALL"], "retrieval_attempts": 0}
    result = await retrieve_analytical(state, vector_store=MagicMock(), schema_registry=reg)

    assert result["sql_results"][0] == {"grade": "GS-10", "salary": 80010.0}
    assert len(result["sql_results"]) == 4


@pytest.mark.asyncio
async def test_analytical_falls_back_when_no_schemas(monkeypatch):
    import src.agent.strategies.map_reduce as mr
    async def fake_mr(state, vector_store):
        return {"retrieved_chunks": [], "fellback": True}
    monkeypatch.setattr(mr, "retrieve_map_reduce", fake_mr)

    state = {"question": "x", "user_groups": ["ALL"], "retrieval_attempts": 0}
    result = await retrieve_analytical(state, vector_store=MagicMock(), schema_registry=SchemaRegistry())
    assert result.get("fellback") is True


@pytest.mark.asyncio
async def test_analytical_falls_back_when_sql_references_disallowed_table(tmp_path, monkeypatch):
    reg, table = _make_pay_duckdb(tmp_path, monkeypatch)
    # LLM emits SQL referencing a table the user is NOT allowed -> allowlist rejects -> fallback
    monkeypatch.setattr(structured, "generate", lambda **kw: 'SELECT * FROM "doc_other_secret"')
    import src.agent.strategies.map_reduce as mr
    async def fake_mr(state, vector_store):
        return {"retrieved_chunks": [], "fellback": True}
    monkeypatch.setattr(mr, "retrieve_map_reduce", fake_mr)

    state = {"question": "x", "user_groups": ["ALL"], "retrieval_attempts": 0}
    result = await retrieve_analytical(state, vector_store=MagicMock(), schema_registry=reg)
    assert result.get("fellback") is True


@pytest.mark.asyncio
async def test_analytical_narrows_tables_before_sql(monkeypatch):
    """A large ACL-visible corpus must be routed down to the relevant table(s)
    before run_structured_lookup — otherwise the text-to-SQL prompt overflows."""
    from src.config import settings
    monkeypatch.setattr(settings, "sql_table_routing_enabled", True)
    monkeypatch.setattr(settings, "sql_table_routing_max_selected", 8)
    monkeypatch.setattr(settings, "sql_table_routing_catalog_budget_chars", 10_000_000)

    schemas = [TableSchema("db", f"t{i}", [ColumnSchema("a", "VARCHAR", "")], f"d{i}", ["ALL"])
               for i in range(40)]

    class _Reg:
        def list_for_user(self, g):
            return schemas

    # router LLM picks t5; run_structured_lookup captures what it actually received
    monkeypatch.setattr(structured, "generate", lambda **kw: '["t5"]')
    captured = {}

    def fake_lookup(q, s, query_type, gate=None, generate_fn=None, hints=None):
        captured["tables"] = [x.table for x in s]
        return structured.StructuredLookupTrace(
            query_type=query_type, status="ran", sql="SELECT 1",
            row_count=1, rows=[{"a": 1}], sample_rows=[{"a": 1}])

    monkeypatch.setattr(structured, "run_structured_lookup", fake_lookup)

    async def _no_hints(s, hs, ms):
        return {}
    monkeypatch.setattr(structured, "resolve_hints_for_schemas", _no_hints)

    result = await retrieve_analytical(
        {"question": "find t5", "user_groups": ["ALL"], "retrieval_attempts": 0},
        vector_store=MagicMock(), schema_registry=_Reg())
    assert captured["tables"] == ["t5"]
    assert result["structured_trace"]["status"] == "ran"


@pytest.mark.asyncio
async def test_analytical_emits_structured_trace(tmp_path, monkeypatch):
    from src.agent.strategies import structured
    schema, _ = _pay_schema_and_db(tmp_path, monkeypatch)
    monkeypatch.setattr(structured, "generate",
                        lambda **kw: f'SELECT grade, salary FROM "{schema.table}"')

    class _Reg:
        def list_for_user(self, g): return [schema]

    result = await retrieve_analytical(
        {"question": "pay", "user_groups": ["ALL"], "retrieval_attempts": 0},
        vector_store=None, schema_registry=_Reg())
    assert result["structured_trace"]["query_type"] == "analytical"
    assert result["structured_trace"]["gate"] is None
    assert result["structured_trace"]["status"] == "ran"
    assert result["structured_trace"]["row_count"] == 4
