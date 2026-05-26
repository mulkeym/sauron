"""Tests for shared structured retrieval (SQL core + gate + retrieve_structured)."""
import duckdb
import pytest

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
    return schema_from_sheet("doc1", "Pay", cls, grid, acl_groups=["ALL"])


def test_structured_sql_rows_generates_and_runs(tmp_path, monkeypatch):
    schema = _pay_schema_and_db(tmp_path, monkeypatch)
    monkeypatch.setattr(structured, "generate",
                        lambda **kw: f'SELECT grade, salary FROM "{schema.table}" WHERE step = 5 ORDER BY salary')
    rows = structured_sql_rows("engineer pay", [schema])
    assert rows[0] == {"grade": "GS-10", "salary": 80010.0}
    assert len(rows) == 4


def test_structured_sql_rows_raises_on_disallowed_table(tmp_path, monkeypatch):
    schema = _pay_schema_and_db(tmp_path, monkeypatch)
    monkeypatch.setattr(structured, "generate", lambda **kw: 'SELECT * FROM "doc_other_secret"')
    with pytest.raises(Exception):
        structured_sql_rows("x", [schema])  # allowlist rejects -> raises (caller falls back)
