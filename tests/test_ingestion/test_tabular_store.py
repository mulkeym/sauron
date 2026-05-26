"""Tests for src/ingestion/tabular_store.py — DuckDB storage + schema for clean sheets."""
import pytest
import duckdb

from src.ingestion.tabular import SheetGrid, SheetClassification
from src.ingestion.tabular_store import duckdb_table_name, _safe_column_names
from src.ingestion.tabular_store import _to_number, load_sheet_to_duckdb
from src.ingestion.tabular_store import execute_duckdb_sql
from src.ingestion.tabular_store import schema_from_sheet, DUCKDB_DATABASE
from src.db.schema_registry import TableSchema


def test_table_name_is_sql_safe_and_deterministic():
    n1 = duckdb_table_name("a1b2-c3d4", "Pay Rates 2024")
    n2 = duckdb_table_name("a1b2-c3d4", "Pay Rates 2024")
    assert n1 == n2                      # deterministic
    assert n1 == "doc_a1b2_c3d4_pay_rates_2024"
    assert not n1[0].isdigit()           # never starts with a digit


def test_safe_column_names_sanitizes_and_dedupes():
    cols = _safe_column_names(["GS Grade", "Step (1)", "", "Step (1)", "2024"])
    # spaces/punct -> underscores, blanks -> col_N, collisions -> suffixed, digit-leading -> prefixed
    assert cols == ["gs_grade", "step_1", "col_2", "step_1_1", "c_2024"]


@pytest.mark.parametrize("value,expected", [
    (5, 5.0), (3.14, 3.14), ("86415", 86415.0), ("$86,415", 86415.0),
    ("12%", 12.0), ("", None), (None, None), ("N/A", None),
    ("nan", None), ("inf", None), ("-inf", None),
])
def test_to_number(value, expected):
    assert _to_number(value) == expected


def _pay_grid():
    rows = [["grade", "step", "salary"]] + [[f"GS-{g}", 5, 80000 + g] for g in range(10, 14)]
    return SheetGrid("Pay", rows)


def test_load_sheet_creates_typed_table_with_rows():
    grid = _pay_grid()
    cls = SheetClassification("Pay", "clean", 0, ["text", "number", "number"], "clean table")
    con = duckdb.connect()  # in-memory
    table, cols = load_sheet_to_duckdb(con, "doc1", "Pay", cls, grid)
    assert table == "doc_doc1_pay"
    assert cols == ["grade", "step", "salary"]
    # 4 data rows loaded; salary stored as a number so SQL math works
    n, total = con.execute(f'SELECT COUNT(*), SUM(salary) FROM "{table}"').fetchone()
    assert n == 4
    assert total == sum(80000 + g for g in range(10, 14))


def test_load_sheet_coerces_bad_numbers_to_null():
    rows = [["k", "v"], ["a", 1], ["b", "oops"], ["c", 3]]
    cls = SheetClassification("S", "clean", 0, ["text", "number"], "clean table")
    con = duckdb.connect()
    table, _ = load_sheet_to_duckdb(con, "d", "S", cls, grid := SheetGrid("S", rows))
    vals = [r[0] for r in con.execute(f'SELECT v FROM "{table}" ORDER BY k').fetchall()]
    assert vals == [1.0, None, 3.0]  # "oops" -> NULL, not a crash


def test_execute_duckdb_sql_returns_dicts():
    con = duckdb.connect()
    cls = SheetClassification("Pay", "clean", 0, ["text", "number", "number"], "clean table")
    table, _ = load_sheet_to_duckdb(con, "doc1", "Pay", cls, _pay_grid())
    rows = execute_duckdb_sql(con, f'SELECT grade, salary FROM "{table}" WHERE step = 5 ORDER BY salary')
    assert rows[0] == {"grade": "GS-10", "salary": 80010.0}
    assert len(rows) == 4


@pytest.mark.parametrize("bad_sql", [
    "DROP TABLE foo",
    "INSERT INTO foo VALUES (1)",
    "UPDATE foo SET x = 1",
    "SELECT 1; DROP TABLE foo",
])
def test_execute_duckdb_sql_rejects_non_select(bad_sql):
    con = duckdb.connect()
    with pytest.raises(ValueError):
        execute_duckdb_sql(con, bad_sql)


def test_schema_from_sheet_builds_registrable_tableschema():
    cls = SheetClassification("Pay", "clean", 0, ["text", "number", "number"], "clean table")
    schema = schema_from_sheet("doc1", "Pay", cls, _pay_grid(), acl_groups=["ALL"])
    assert isinstance(schema, TableSchema)
    assert schema.database == DUCKDB_DATABASE
    assert schema.table == "doc_doc1_pay"
    assert [c.name for c in schema.columns] == ["grade", "step", "salary"]
    assert [c.dtype for c in schema.columns] == ["VARCHAR", "DOUBLE", "DOUBLE"]
    # original header text is preserved as the column description (Plan 2b enriches it)
    assert [c.description for c in schema.columns] == ["grade", "step", "salary"]
    assert schema.acl_groups == ["ALL"]
