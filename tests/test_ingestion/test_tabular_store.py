"""Tests for src/ingestion/tabular_store.py — DuckDB storage + schema for clean sheets."""
import pytest
import duckdb

from src.ingestion.tabular import SheetGrid, SheetClassification
from src.ingestion.tabular_store import duckdb_table_name, _safe_column_names
from src.ingestion.tabular_store import _to_number, load_sheet_to_duckdb
from src.ingestion.tabular_store import execute_duckdb_sql
from src.ingestion.tabular_store import schema_from_sheet, DUCKDB_DATABASE
from src.ingestion.tabular_store import connect_tabular
from src.ingestion.tabular_store import _referenced_tables, _cte_names
from src.ingestion.tabular_store import distinct_values, schema_prompt_with_values
from src.db.schema_registry import TableSchema


def test_connect_tabular_creates_writes_and_reads(tmp_path, monkeypatch):
    from src.config import settings
    monkeypatch.setattr(settings, "tabular_duckdb_path", str(tmp_path / "t.duckdb"))
    con = connect_tabular()
    con.execute("CREATE TABLE t (x INTEGER)")
    con.execute("INSERT INTO t VALUES (1), (2)")
    assert con.execute("SELECT SUM(x) FROM t").fetchone()[0] == 3
    con.close()


def test_connect_tabular_read_only_sees_committed_data(tmp_path, monkeypatch):
    from src.config import settings
    monkeypatch.setattr(settings, "tabular_duckdb_path", str(tmp_path / "t.duckdb"))
    w = connect_tabular()
    w.execute("CREATE TABLE t (x INTEGER)")
    w.execute("INSERT INTO t VALUES (7)")
    w.close()
    r = connect_tabular(read_only=True)
    assert r.execute("SELECT x FROM t").fetchone()[0] == 7
    r.close()


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


def test_load_sheet_raises_on_dtype_mismatch():
    rows = [["a", "b", "c"], [1, 2, 3]]
    cls = SheetClassification("S", "clean", 0, ["number", "number"], "clean table")  # 2 dtypes, 3 cols
    with pytest.raises(ValueError, match="header has 3 columns but classification has 2"):
        load_sheet_to_duckdb(duckdb.connect(), "d", "S", cls, SheetGrid("S", rows))


def test_schema_from_sheet_raises_on_dtype_mismatch():
    rows = [["a", "b", "c"], [1, 2, 3]]
    cls = SheetClassification("S", "clean", 0, ["number", "number"], "clean table")
    with pytest.raises(ValueError, match="header has 3 columns but classification has 2"):
        schema_from_sheet("d", "S", cls, SheetGrid("S", rows))


def test_connect_tabular_blocks_external_file_access(tmp_path, monkeypatch):
    from src.config import settings
    monkeypatch.setattr(settings, "tabular_duckdb_path", str(tmp_path / "t.duckdb"))
    con = connect_tabular()
    con.execute("CREATE TABLE t (x INTEGER)")
    con.execute("INSERT INTO t VALUES (1)")
    assert con.execute("SELECT SUM(x) FROM t").fetchone()[0] == 1  # normal queries still work
    # external file access is disabled -> read_csv on a real file is refused
    with pytest.raises(Exception):
        con.execute("SELECT * FROM read_csv('/etc/hostname')").fetchall()
    con.close()


def test_referenced_tables_from_and_join():
    assert _referenced_tables('SELECT * FROM "doc_x_pay" WHERE step = 5') == {"doc_x_pay"}
    refs = _referenced_tables("SELECT * FROM a JOIN b ON a.id = b.id")
    assert refs == {"a", "b"}


def test_referenced_tables_is_case_insensitive_and_unquoted():
    assert _referenced_tables('select * from "Doc_X"') == {"doc_x"}


def test_cte_names_collects_with_aliases():
    sql = "WITH totals AS (SELECT 1), avgs AS (SELECT 2) SELECT * FROM totals"
    assert _cte_names(sql) == {"totals", "avgs"}


def test_cte_names_empty_when_no_with():
    assert _cte_names('SELECT * FROM "doc_x_pay"') == set()


def _con_with_pay():
    con = duckdb.connect()
    cls = SheetClassification("Pay", "clean", 0, ["text", "number", "number"], "clean table")
    load_sheet_to_duckdb(con, "doc1", "Pay", cls, _pay_grid())
    return con, "doc_doc1_pay"


def test_execute_allows_query_on_allowed_table():
    con, table = _con_with_pay()
    rows = execute_duckdb_sql(con, f'SELECT grade FROM "{table}" ORDER BY grade',
                              allowed_tables={table})
    assert rows[0]["grade"] == "GS-10"


def test_execute_rejects_table_outside_allowlist():
    con, table = _con_with_pay()
    with pytest.raises(ValueError, match="outside the allowed set"):
        execute_duckdb_sql(con, 'SELECT * FROM "doc_other_secret"', allowed_tables={table})


def test_execute_allows_cte_referencing_allowed_table():
    con, table = _con_with_pay()
    sql = f'WITH hi AS (SELECT * FROM "{table}" WHERE salary > 80011) SELECT grade FROM hi'
    rows = execute_duckdb_sql(con, sql, allowed_tables={table})
    assert len(rows) >= 1  # CTE alias "hi" is allowed; the real table is in the allowlist


@pytest.mark.parametrize("bad_sql", [
    "SELECT * FROM read_csv('/etc/hostname')",
    "SELECT * FROM read_parquet('/tmp/x.parquet')",
    "ATTACH 'other.db' AS o",
    "INSTALL httpfs",
])
def test_execute_rejects_forbidden_constructs(bad_sql):
    con = duckdb.connect()
    with pytest.raises(ValueError):
        execute_duckdb_sql(con, bad_sql, allowed_tables={"whatever"})


def test_execute_without_allowlist_skips_table_check():
    # allowed_tables=None preserves the internal/back-compat behavior (no ACL check).
    con, table = _con_with_pay()
    rows = execute_duckdb_sql(con, f'SELECT COUNT(*) AS n FROM "{table}"')
    assert rows[0]["n"] == 4


def test_distinct_values_returns_low_cardinality():
    con, table = _con_with_pay()
    assert sorted(distinct_values(con, table, "grade", max_distinct=100)) == [
        "GS-10", "GS-11", "GS-12", "GS-13"]


def test_distinct_values_none_when_high_cardinality():
    con, table = _con_with_pay()
    assert distinct_values(con, table, "grade", max_distinct=2) is None


def test_schema_prompt_lists_categorical_values():
    con, table = _con_with_pay()
    cls = SheetClassification("Pay", "clean", 0, ["text", "number", "number"], "clean table")
    schema = schema_from_sheet("doc1", "Pay", cls, _pay_grid(), acl_groups=["ALL"])
    prompt = schema_prompt_with_values([schema], con)
    # VARCHAR 'grade' gets its real values listed in the prompt.
    grade_line = next(l for l in prompt.splitlines() if l.strip().startswith("- grade"))
    salary_line = next(l for l in prompt.splitlines() if l.strip().startswith("- salary"))
    assert "values:" in grade_line and "GS-12" in grade_line
    assert "values:" not in salary_line  # numeric column: no value list


def test_referenced_tables_captures_comma_joins():
    assert _referenced_tables("SELECT * FROM a, b, c WHERE a.id = b.id") == {"a", "b", "c"}
    assert _referenced_tables('SELECT * FROM "t1", "t2"') == {"t1", "t2"}


def test_execute_rejects_comma_joined_table_outside_allowlist():
    con, table = _con_with_pay()
    with pytest.raises(ValueError, match="outside the allowed set"):
        execute_duckdb_sql(con, f'SELECT * FROM "{table}", "doc_other_secret"', allowed_tables={table})


from src.agent.strategies.hint_resolver import ResolvedHints


def test_schema_prompt_hints_none_is_unchanged():
    con, table = _con_with_pay()
    cls = SheetClassification("Pay", "clean", 0, ["text", "number", "number"], "clean table")
    schema = schema_from_sheet("doc1", "Pay", cls, _pay_grid(), acl_groups=["ALL"])
    assert schema_prompt_with_values([schema], con, hints=None) == schema_prompt_with_values([schema], con)


def test_schema_prompt_annotates_glossary_values():
    con, table = _con_with_pay()
    cls = SheetClassification("Pay", "clean", 0, ["text", "number", "number"], "clean table")
    schema = schema_from_sheet("doc1", "Pay", cls, _pay_grid(), acl_groups=["ALL"])
    hints = {schema.table: ResolvedHints(column_glossaries={"grade": {"GS-12": "Senior"}})}
    prompt = schema_prompt_with_values([schema], con, hints=hints)
    assert "GS-12 (Senior)" in prompt


def test_schema_prompt_adds_column_and_table_notes():
    con, table = _con_with_pay()
    cls = SheetClassification("Pay", "clean", 0, ["text", "number", "number"], "clean table")
    schema = schema_from_sheet("doc1", "Pay", cls, _pay_grid(), acl_groups=["ALL"])
    hints = {schema.table: ResolvedHints(
        column_notes={"grade": "GS pay grade"}, table_notes=["OPM 2022 GS pay"])}
    prompt = schema_prompt_with_values([schema], con, hints=hints)
    assert "GS pay grade" in prompt
    assert "Notes: OPM 2022 GS pay" in prompt


from src.ingestion.tabular_store import referenced_source_docs


def test_referenced_source_docs_single():
    sql = f'SELECT * FROM "{duckdb_table_name("live1", "pay")}" WHERE x = 1'
    assert referenced_source_docs(sql, ["live1", "other"]) == ["live1"]


def test_referenced_source_docs_join_two_order_stable():
    t1 = duckdb_table_name("a", "s")
    t2 = duckdb_table_name("b", "s")
    sql = f'SELECT * FROM "{t1}" JOIN "{t2}" ON 1=1'
    assert referenced_source_docs(sql, ["a", "b"]) == ["a", "b"]


def test_referenced_source_docs_skips_unmatched():
    sql = f'SELECT * FROM "{duckdb_table_name("ghost", "pay")}"'
    assert referenced_source_docs(sql, ["live1"]) == []


def test_referenced_source_docs_ignores_cte():
    t = duckdb_table_name("live1", "pay")
    sql = f'WITH tmp AS (SELECT 1) SELECT * FROM "{t}", tmp'
    assert referenced_source_docs(sql, ["live1"]) == ["live1"]


def test_distinct_values_sorted_independent_of_insertion_order():
    # distinct_values feeds the categorical value list in the text-to-SQL schema
    # prompt; a stable (sorted) order keeps that prompt byte-identical run-to-run
    # so seed+temperature=0 yields deterministic SQL. DISTINCT without ORDER BY
    # has no order guarantee in DuckDB.
    con = duckdb.connect()
    con.execute('CREATE TABLE t (loc VARCHAR)')
    for v in ["TU", "AK", "SF", "ATL", "RUS"]:
        con.execute('INSERT INTO t VALUES (?)', [v])
    assert distinct_values(con, "t", "loc", max_distinct=100) == ["AK", "ATL", "RUS", "SF", "TU"]
