# Tabular Ingestion — Plan 3a: Harden `execute_duckdb_sql` for LLM/User SQL

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `execute_duckdb_sql` safe to run LLM- or user-generated SQL: block DuckDB's file-reading/extension functions, and enforce a per-request table-name allowlist (the ACL boundary, since the DuckDB file holds every user's tables).

**Architecture:** Defense in depth: (1) the tabular DuckDB connection is opened with `enable_external_access=False` so `read_csv`/`read_parquet`/`ATTACH`/`COPY`/httpfs cannot reach the filesystem or network; (2) `execute_duckdb_sql` gains a forbidden-token reject list and an optional `allowed_tables` set — it parses the table identifiers a query references (FROM/JOIN, minus CTE aliases) and rejects any outside the allowed set. SELECT/WITH-only + single-statement guards stay. Pure functions, fully testable; no pipeline or routing changes (that's Plan 3b).

**Tech Stack:** Python 3.11, DuckDB, pytest. Tests run inside the app image: `docker run --rm -v "$(pwd)":/app -w /app sauron-api python -m pytest <path> -q`.

**Spec:** `docs/superpowers/specs/2026-05-25-tabular-spreadsheet-ingestion-design.md` (Component 3 — the "DuckDB allowlist / CTE hardening before exposing to LLM SQL" follow-up flagged during Plan 2a review). **Depends on:** Plan 2a (`execute_duckdb_sql`, `connect_tabular` in `tabular_store.py`).

**Why now:** Plan 3b routes `ANALYTICAL` questions to `execute_duckdb_sql` with LLM-generated SQL. That SQL must not be able to read arbitrary files or query another user's table. This plan adds those guards first, in isolation.

---

## File Structure

- `src/ingestion/tabular_store.py` — **modify**: harden `connect_tabular` (open with `enable_external_access=False`); add `_referenced_tables` and `_cte_names` helpers; extend `execute_duckdb_sql` with forbidden-token rejection + `allowed_tables` enforcement, and accept `WITH` as a statement opener.
- `tests/test_ingestion/test_tabular_store.py` — **modify**: add hardening tests.

No new files; this is a focused hardening of one existing module.

---

### Task 1: Lock down `connect_tabular` (no external file/network access)

**Files:**
- Modify: `src/ingestion/tabular_store.py`
- Test: `tests/test_ingestion/test_tabular_store.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_ingestion/test_tabular_store.py` (the file already imports `connect_tabular`):

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `docker run --rm -v "$(pwd)":/app -w /app sauron-api python -m pytest tests/test_ingestion/test_tabular_store.py::test_connect_tabular_blocks_external_file_access -q`
Expected: FAIL — `read_csv('/etc/hostname')` currently succeeds (external access is on), so `pytest.raises(Exception)` is not triggered.

- [ ] **Step 3: Open the connection with external access disabled**

In `src/ingestion/tabular_store.py`, change the `return` line of `connect_tabular` from:

```python
    return duckdb.connect(path, read_only=read_only)
```

to:

```python
    # enable_external_access=False blocks read_csv/read_parquet/ATTACH/COPY/httpfs,
    # so untrusted (LLM-generated) SELECTs cannot reach the filesystem or network.
    return duckdb.connect(path, read_only=read_only, config={"enable_external_access": False})
```

- [ ] **Step 4: Run test to verify it passes**

Run: `docker run --rm -v "$(pwd)":/app -w /app sauron-api python -m pytest tests/test_ingestion/test_tabular_store.py::test_connect_tabular_blocks_external_file_access -q`
Expected: PASS (1 passed).

- [ ] **Step 5: Run the existing tabular_store + tabular_ingest suites (regression — CREATE/INSERT must still work under the locked config)**

Run: `docker run --rm -v "$(pwd)":/app -w /app sauron-api python -m pytest tests/test_ingestion/test_tabular_store.py tests/test_ingestion/test_tabular_ingest.py -q`
Expected: PASS (all — the lockdown only blocks external access; internal CREATE/INSERT/SELECT used by ingestion are unaffected).

- [ ] **Step 6: Commit**

```bash
git add src/ingestion/tabular_store.py tests/test_ingestion/test_tabular_store.py
git commit -m "feat: lock down tabular DuckDB connection (no external access)

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task 2: `_referenced_tables` + `_cte_names` (parse table identifiers)

**Files:**
- Modify: `src/ingestion/tabular_store.py`
- Test: `tests/test_ingestion/test_tabular_store.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_ingestion/test_tabular_store.py` (extend the import to add `_referenced_tables`, `_cte_names`):

```python
from src.ingestion.tabular_store import _referenced_tables, _cte_names


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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `docker run --rm -v "$(pwd)":/app -w /app sauron-api python -m pytest tests/test_ingestion/test_tabular_store.py -k "referenced_tables or cte_names" -q`
Expected: FAIL — `ImportError: cannot import name '_referenced_tables'`.

- [ ] **Step 3: Implement the parsers**

Add to `src/ingestion/tabular_store.py` (near the top, after the `import re` / `DUCKDB_DATABASE` area):

```python
def _referenced_tables(sql: str) -> set[str]:
    """Lowercased identifiers used as tables (the token after FROM or JOIN).

    Best-effort lexical parse — combined with the connection's
    enable_external_access=False and the allowed_tables check in
    execute_duckdb_sql, it bounds which tables a query can read.
    """
    return {
        m.lower()
        for m in re.findall(r'\b(?:FROM|JOIN)\s+"?([A-Za-z_][A-Za-z0-9_]*)"?', sql, re.IGNORECASE)
    }


def _cte_names(sql: str) -> set[str]:
    """Lowercased CTE aliases defined via ``WITH <name> AS (`` so they count as
    allowed 'tables' when validating references."""
    return {
        m.lower()
        for m in re.findall(r'([A-Za-z_][A-Za-z0-9_]*)\s+AS\s*\(', sql, re.IGNORECASE)
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `docker run --rm -v "$(pwd)":/app -w /app sauron-api python -m pytest tests/test_ingestion/test_tabular_store.py -k "referenced_tables or cte_names" -q`
Expected: PASS (4 passed).

- [ ] **Step 5: Commit**

```bash
git add src/ingestion/tabular_store.py tests/test_ingestion/test_tabular_store.py
git commit -m "feat: parse table + CTE identifiers from SQL for allowlisting

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task 3: Forbidden-token reject + `allowed_tables` enforcement in `execute_duckdb_sql`

**Files:**
- Modify: `src/ingestion/tabular_store.py`
- Test: `tests/test_ingestion/test_tabular_store.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_ingestion/test_tabular_store.py` (the file already imports `execute_duckdb_sql`, `duckdb`, `SheetGrid`, `SheetClassification`, `load_sheet_to_duckdb`; reuse the `_pay_grid` helper defined earlier in the file):

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `docker run --rm -v "$(pwd)":/app -w /app sauron-api python -m pytest tests/test_ingestion/test_tabular_store.py -k "allowlist or forbidden or allowed_table or cte_referencing" -q`
Expected: FAIL — `execute_duckdb_sql` does not yet accept `allowed_tables` (TypeError) / does not reject the forbidden constructs.

- [ ] **Step 3: Extend `execute_duckdb_sql`**

In `src/ingestion/tabular_store.py`, replace the entire `execute_duckdb_sql` function with:

```python
_FORBIDDEN_SQL_TOKENS = (
    "read_csv", "read_parquet", "read_json", "read_text",
    "parquet_scan", "csv_scan", "glob", "attach", "install",
)


def execute_duckdb_sql(con, sql: str, allowed_tables=None) -> list[dict]:
    """Run a single read-only SELECT/WITH against a DuckDB connection.

    Guards (in addition to the connection's enable_external_access=False):
      - single statement only, must start with SELECT or WITH;
      - reject DuckDB file/extension functions (read_csv, attach, install, ...);
      - if ``allowed_tables`` is given, every FROM/JOIN identifier must be in it
        (plus any CTE alias defined in the query). This is the ACL boundary —
        callers pass the set of tables the user may read. ``None`` skips the
        check (trusted/internal callers only).

    Returns ``list[dict]``.
    """
    sql = sql.strip().rstrip(";")
    if ";" in sql:
        raise ValueError("Only a single SELECT statement is allowed")
    if not re.match(r"^\s*(SELECT|WITH)\b", sql, re.IGNORECASE):
        raise ValueError("Only SELECT queries are allowed")

    lowered = sql.lower()
    for tok in _FORBIDDEN_SQL_TOKENS:
        if re.search(r"\b" + re.escape(tok) + r"\b", lowered):
            raise ValueError(f"Disallowed SQL construct: {tok}")

    if allowed_tables is not None:
        allowed = {t.lower() for t in allowed_tables} | _cte_names(sql)
        extra = _referenced_tables(sql) - allowed
        if extra:
            raise ValueError(f"Query references tables outside the allowed set: {sorted(extra)}")

    cur = con.execute(sql)
    columns = [d[0] for d in cur.description]
    return [dict(zip(columns, row)) for row in cur.fetchall()]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `docker run --rm -v "$(pwd)":/app -w /app sauron-api python -m pytest tests/test_ingestion/test_tabular_store.py -k "allowlist or forbidden or allowed_table or cte_referencing or without_allowlist" -q`
Expected: PASS (8 passed: 1 allowed + 1 reject-outside + 1 cte + 4 forbidden params + 1 no-allowlist).

- [ ] **Step 5: Run the full tabular_store suite (regression — Plan 2a's existing SELECT-only/reject tests must still pass)**

Run: `docker run --rm -v "$(pwd)":/app -w /app sauron-api python -m pytest tests/test_ingestion/test_tabular_store.py -q`
Expected: PASS (all). The existing `test_execute_duckdb_sql_returns_dicts` (no `allowed_tables`) and the `test_execute_duckdb_sql_rejects_non_select` params still hold; allowing `WITH` as an opener does not affect them.

- [ ] **Step 6: Commit**

```bash
git add src/ingestion/tabular_store.py tests/test_ingestion/test_tabular_store.py
git commit -m "feat: forbidden-token + table allowlist guards in execute_duckdb_sql

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Final verification

- [ ] Run the affected suites:

```bash
docker run --rm -v "$(pwd)":/app -w /app sauron-api python -m pytest \
  tests/test_ingestion/test_tabular_store.py tests/test_ingestion/test_tabular_ingest.py -q
```
Expected: all PASS.

- [ ] Confirm only the two intended files changed:

```bash
git diff --stat <first-commit-of-this-plan>^..HEAD
```
Expected: only `src/ingestion/tabular_store.py` and `tests/test_ingestion/test_tabular_store.py`.

## Notes for the implementer

- **Two layers, on purpose.** `enable_external_access=False` (Task 1) is the hard boundary against filesystem/network reads — it cannot be regex-evaded. The forbidden-token list (Task 3) is a cheap early reject for clearer error messages and defense in depth. The `allowed_tables` check is the **ACL** boundary (the DuckDB file holds every user's tables; a user must only read their own).
- **The allowlist is best-effort lexical**, not a full SQL parser. It is acceptable because it is backed by the locked-down connection and the read-only scope, and because the LLM only ever sees the user's own table names in its schema prompt (Plan 3b). A crafted query that evades the regex could at worst read another *spreadsheet* table in the same file — never the filesystem. Tightening to a real parser is a future option, not required here.
- **`WITH` is now an allowed opener** so legitimate analytic CTEs work; CTE aliases are added to the allowed set so `FROM <cte_alias>` passes. This resolves the CTE-rejection limitation noted in the Plan 2a review.
- **`allowed_tables=None` preserves back-compat** for any internal/trusted caller; Plan 3b will always pass the user's allowed set.
- **This plan does not wire routing.** `retrieve_analytical` still uses the sqlite `execute_sql` and the classifier is unchanged — Plan 3b makes `ANALYTICAL` questions generate SQL, pass the allowed table set, and execute against the tabular DuckDB read-only connection.
```
