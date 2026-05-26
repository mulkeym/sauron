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
