# Tabular Ingestion — Plan 2a: DuckDB Store + Schema Persistence

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give the system a queryable structured home for clean spreadsheet tables — load a classified sheet into a DuckDB table, build a registrable `TableSchema` for it, run SELECT-only SQL against it, and persist registered schemas across restarts.

**Architecture:** A new `src/ingestion/tabular_store.py` turns a Plan‑1 `SheetGrid` + `SheetClassification` into a typed DuckDB table and a `TableSchema`, and runs validated SELECT SQL. A new `RegisteredSchema` SQLAlchemy model + `MetadataStore` methods persist schemas (the registry is in‑memory today and always empty). No LLM, no pipeline changes, no query‑routing changes — this is the storage plumbing that Plan 2b (profiler + row narratives + pipeline wiring) and Plan 3 (query routing) build on.

**Tech Stack:** Python 3.11, DuckDB, SQLAlchemy + aiosqlite, openpyxl/xlrd (via Plan 1), pytest/pytest-asyncio. Tests run inside the app image: `docker run --rm -v "$(pwd)":/app -w /app sauron-api python -m pytest <path> -q`.

**Spec:** `docs/superpowers/specs/2026-05-25-tabular-spreadsheet-ingestion-design.md` (Component 2 — storage half). **Depends on:** Plan 1 (`src/ingestion/tabular.py`: `SheetGrid`, `SheetClassification`, already merged).

---

## File Structure

- `requirements.txt` — **modify**: add `duckdb`.
- `src/ingestion/tabular_store.py` — **create**. Owns: table naming (`duckdb_table_name`), column-name sanitization (`_safe_column_names`), number coercion (`_to_number`), `load_sheet_to_duckdb`, `execute_duckdb_sql`, and `schema_from_sheet` (grid → `TableSchema`). Single responsibility: the structured (DuckDB) side of a clean sheet.
- `src/db/models.py` — **modify**: add `RegisteredSchema` model (auto-created by `Base.metadata.create_all`).
- `src/db/metadata.py` — **modify**: add `save_schema`, `load_all_schemas`, `delete_schema` to `MetadataStore`.
- `tests/test_ingestion/test_tabular_store.py` — **create**.
- `tests/test_db/test_schema_persistence.py` — **create**.

## Constant

`tabular_store.py` defines `DUCKDB_DATABASE = "spreadsheets"` — the logical `database` name all spreadsheet-derived `TableSchema`s use (the `database` field in `TableSchema`/`schema_registry`).

---

### Task 1: Add the DuckDB dependency and rebuild the test image

**Files:**
- Modify: `requirements.txt`

- [ ] **Step 1: Add the dependency**

Append this line to `requirements.txt` (keep existing lines unchanged):

```
duckdb>=1.0.0
```

- [ ] **Step 2: Rebuild the app image so test runs have duckdb**

The test command mounts the working tree into the `sauron-api` image, which is built from `requirements.txt`. Adding the line does nothing until the image is rebuilt. Rebuild only the `api` image (this does NOT restart the live `sauron-api-1` container — it keeps running the old image until a future `docker compose up`):

Run: `docker compose build api`
Expected: `Image sauron-api Built` (a few minutes).

- [ ] **Step 3: Verify duckdb imports in the rebuilt image**

Run: `docker run --rm sauron-api python -c "import duckdb; print(duckdb.__version__)"`
Expected: prints a version (e.g. `1.x.y`), exit 0.

- [ ] **Step 4: Commit**

```bash
git add requirements.txt
git commit -m "build: add duckdb dependency for tabular store

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task 2: Table naming + column-name sanitization

**Files:**
- Create: `src/ingestion/tabular_store.py`
- Test: `tests/test_ingestion/test_tabular_store.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_ingestion/test_tabular_store.py`:

```python
"""Tests for src/ingestion/tabular_store.py — DuckDB storage + schema for clean sheets."""
import pytest

from src.ingestion.tabular_store import duckdb_table_name, _safe_column_names


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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `docker run --rm -v "$(pwd)":/app -w /app sauron-api python -m pytest tests/test_ingestion/test_tabular_store.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.ingestion.tabular_store'`.

- [ ] **Step 3: Create the module with the two helpers**

Create `src/ingestion/tabular_store.py`:

```python
"""Structured (DuckDB) storage for clean spreadsheet tables.

Turns a Plan-1 ``SheetGrid`` + ``SheetClassification`` into a typed DuckDB
table and a registrable ``TableSchema``, and runs SELECT-only SQL against a
DuckDB connection. No LLM, no embeddings — that is Plan 2b.
"""
from __future__ import annotations

import re

from src.db.schema_registry import ColumnSchema, TableSchema
from src.ingestion.tabular import SheetClassification, SheetGrid

DUCKDB_DATABASE = "spreadsheets"


def duckdb_table_name(doc_id: str, sheet_name: str) -> str:
    """Deterministic, SQL-safe table name for one doc's sheet."""
    safe_doc = re.sub(r"[^0-9a-zA-Z]+", "_", str(doc_id)).strip("_")
    safe_sheet = re.sub(r"[^0-9a-zA-Z]+", "_", str(sheet_name)).strip("_")
    return f"doc_{safe_doc}_{safe_sheet}".lower()


def _safe_column_names(header: list) -> list[str]:
    """SQL-safe, unique column identifiers from a header row.

    Non-alphanumerics collapse to underscores; blanks become ``col_<i>``;
    digit-leading names get a ``c_`` prefix; collisions get a numeric suffix.
    """
    names: list[str] = []
    for i, h in enumerate(header):
        base = re.sub(r"[^0-9a-zA-Z]+", "_", str(h).strip()).strip("_").lower()
        if not base:
            base = f"col_{i}"
        if base[0].isdigit():
            base = f"c_{base}"
        name = base
        suffix = 0
        while name in names:
            suffix += 1
            name = f"{base}_{suffix}"
        names.append(name)
    return names
```

- [ ] **Step 4: Run test to verify it passes**

Run: `docker run --rm -v "$(pwd)":/app -w /app sauron-api python -m pytest tests/test_ingestion/test_tabular_store.py -q`
Expected: PASS (2 passed).

- [ ] **Step 5: Commit**

```bash
git add src/ingestion/tabular_store.py tests/test_ingestion/test_tabular_store.py
git commit -m "feat: SQL-safe table and column naming for tabular store

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task 3: `_to_number` + `load_sheet_to_duckdb`

**Files:**
- Modify: `src/ingestion/tabular_store.py`
- Test: `tests/test_ingestion/test_tabular_store.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_ingestion/test_tabular_store.py` (extend the import to add `_to_number`, `load_sheet_to_duckdb`; add `import duckdb` and the Plan-1 imports):

```python
import duckdb

from src.ingestion.tabular import SheetGrid, SheetClassification
from src.ingestion.tabular_store import _to_number, load_sheet_to_duckdb


@pytest.mark.parametrize("value,expected", [
    (5, 5.0), (3.14, 3.14), ("86415", 86415.0), ("$86,415", 86415.0),
    ("12%", 12.0), ("", None), (None, None), ("N/A", None),
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `docker run --rm -v "$(pwd)":/app -w /app sauron-api python -m pytest tests/test_ingestion/test_tabular_store.py -k "to_number or load_sheet" -q`
Expected: FAIL — `ImportError: cannot import name '_to_number'`.

- [ ] **Step 3: Implement `_to_number` and `load_sheet_to_duckdb`**

Add to `src/ingestion/tabular_store.py`:

```python
def _to_number(value) -> float | None:
    """Coerce a cell to float, or None if it isn't numeric.

    Handles native numbers and numeric strings with $, comma, % formatting.
    Bools are treated as non-numeric (consistent with tabular._cell_kind).
    """
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    cleaned = str(value).replace("$", "").replace(",", "").replace("%", "").strip()
    if not cleaned:
        return None
    try:
        return float(cleaned)
    except ValueError:
        return None


def load_sheet_to_duckdb(con, doc_id: str, sheet_name: str,
                         classification: SheetClassification, grid: SheetGrid) -> tuple[str, list[str]]:
    """Create a typed DuckDB table for a clean sheet and insert its data rows.

    ``number`` columns become DOUBLE (cells coerced via ``_to_number``); all
    other columns become VARCHAR. Returns ``(table_name, column_names)``.
    Replaces any existing table with the same name (re-ingest is idempotent).
    """
    header_idx = classification.header_row_index
    header = grid.rows[header_idx]
    col_names = _safe_column_names(header)
    dtypes = classification.column_dtypes
    table = duckdb_table_name(doc_id, sheet_name)

    col_defs = ", ".join(
        f'"{name}" {"DOUBLE" if dt == "number" else "VARCHAR"}'
        for name, dt in zip(col_names, dtypes)
    )
    con.execute(f'DROP TABLE IF EXISTS "{table}"')
    con.execute(f'CREATE TABLE "{table}" ({col_defs})')

    ncols = len(col_names)
    placeholders = ", ".join(["?"] * ncols)
    for row in grid.rows[header_idx + 1:]:
        values = []
        for c in range(ncols):
            raw = row[c] if c < len(row) else None
            if dtypes[c] == "number":
                values.append(_to_number(raw))
            else:
                values.append(None if raw is None else str(raw))
        con.execute(f'INSERT INTO "{table}" VALUES ({placeholders})', values)
    return table, col_names
```

- [ ] **Step 4: Run test to verify it passes**

Run: `docker run --rm -v "$(pwd)":/app -w /app sauron-api python -m pytest tests/test_ingestion/test_tabular_store.py -k "to_number or load_sheet" -q`
Expected: PASS (10 passed).

- [ ] **Step 5: Commit**

```bash
git add src/ingestion/tabular_store.py tests/test_ingestion/test_tabular_store.py
git commit -m "feat: load a clean sheet into a typed DuckDB table

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task 4: `execute_duckdb_sql` (SELECT-only)

**Files:**
- Modify: `src/ingestion/tabular_store.py`
- Test: `tests/test_ingestion/test_tabular_store.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_ingestion/test_tabular_store.py` (extend import to add `execute_duckdb_sql`):

```python
from src.ingestion.tabular_store import execute_duckdb_sql


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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `docker run --rm -v "$(pwd)":/app -w /app sauron-api python -m pytest tests/test_ingestion/test_tabular_store.py -k execute -q`
Expected: FAIL — `ImportError: cannot import name 'execute_duckdb_sql'`.

- [ ] **Step 3: Implement `execute_duckdb_sql`**

Add to `src/ingestion/tabular_store.py`:

```python
def execute_duckdb_sql(con, sql: str) -> list[dict]:
    """Run a single SELECT against a DuckDB connection; return list[dict].

    Mirrors the guardrails in src/db/sql_executor.py: a single statement only,
    and it must be a SELECT.
    """
    sql = sql.strip().rstrip(";")
    if ";" in sql:
        raise ValueError("Only a single SELECT statement is allowed")
    if not re.match(r"^\s*SELECT\b", sql, re.IGNORECASE):
        raise ValueError("Only SELECT queries are allowed")
    cur = con.execute(sql)
    columns = [d[0] for d in cur.description]
    return [dict(zip(columns, row)) for row in cur.fetchall()]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `docker run --rm -v "$(pwd)":/app -w /app sauron-api python -m pytest tests/test_ingestion/test_tabular_store.py -k execute -q`
Expected: PASS (5 passed).

- [ ] **Step 5: Commit**

```bash
git add src/ingestion/tabular_store.py tests/test_ingestion/test_tabular_store.py
git commit -m "feat: SELECT-only SQL execution against DuckDB

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task 5: `schema_from_sheet` (grid → registrable `TableSchema`)

**Files:**
- Modify: `src/ingestion/tabular_store.py`
- Test: `tests/test_ingestion/test_tabular_store.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_ingestion/test_tabular_store.py` (extend import to add `schema_from_sheet`, `DUCKDB_DATABASE`):

```python
from src.ingestion.tabular_store import schema_from_sheet, DUCKDB_DATABASE
from src.db.schema_registry import TableSchema


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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `docker run --rm -v "$(pwd)":/app -w /app sauron-api python -m pytest tests/test_ingestion/test_tabular_store.py -k schema_from_sheet -q`
Expected: FAIL — `ImportError: cannot import name 'schema_from_sheet'`.

- [ ] **Step 3: Implement `schema_from_sheet`**

Add to `src/ingestion/tabular_store.py`:

```python
def schema_from_sheet(doc_id: str, sheet_name: str, classification: SheetClassification,
                      grid: SheetGrid, acl_groups: list[str] | None = None) -> TableSchema:
    """Build a registrable ``TableSchema`` from a clean sheet.

    Column names match ``load_sheet_to_duckdb``; dtypes map number->DOUBLE and
    everything else->VARCHAR; descriptions default to the original header text
    (Plan 2b's profiler replaces these with richer descriptions).
    """
    header = grid.rows[classification.header_row_index]
    col_names = _safe_column_names(header)
    columns = [
        ColumnSchema(
            name=name,
            dtype="DOUBLE" if dt == "number" else "VARCHAR",
            description=str(orig).strip(),
        )
        for name, dt, orig in zip(col_names, classification.column_dtypes, header)
    ]
    return TableSchema(
        database=DUCKDB_DATABASE,
        table=duckdb_table_name(doc_id, sheet_name),
        columns=columns,
        description=f"Sheet '{sheet_name}' from document {doc_id}",
        acl_groups=acl_groups or [],
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `docker run --rm -v "$(pwd)":/app -w /app sauron-api python -m pytest tests/test_ingestion/test_tabular_store.py -k schema_from_sheet -q`
Expected: PASS (1 passed).

- [ ] **Step 5: Commit**

```bash
git add src/ingestion/tabular_store.py tests/test_ingestion/test_tabular_store.py
git commit -m "feat: build a registrable TableSchema from a clean sheet

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task 6: Persist registered schemas (`RegisteredSchema` model + `MetadataStore` methods)

**Files:**
- Modify: `src/db/models.py`
- Modify: `src/db/metadata.py`
- Test: `tests/test_db/test_schema_persistence.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_db/test_schema_persistence.py`:

```python
"""Round-trip persistence for registered table schemas."""
import pytest

from src.db.metadata import MetadataStore
from src.db.schema_registry import TableSchema, ColumnSchema


def _schema(table="doc_x_pay"):
    return TableSchema(
        database="spreadsheets",
        table=table,
        columns=[
            ColumnSchema(name="grade", dtype="VARCHAR", description="Pay grade"),
            ColumnSchema(name="salary", dtype="DOUBLE", description="Annual salary"),
        ],
        description="GS pay table",
        acl_groups=["ALL"],
    )


@pytest.mark.asyncio
async def test_save_and_load_schema_round_trip(tmp_path):
    store = MetadataStore(database_url=f"sqlite+aiosqlite:///{tmp_path}/m.db")
    await store.init()
    await store.save_schema(_schema())

    loaded = await store.load_all_schemas()
    assert len(loaded) == 1
    s = loaded[0]
    assert (s.database, s.table) == ("spreadsheets", "doc_x_pay")
    assert [(c.name, c.dtype, c.description) for c in s.columns] == [
        ("grade", "VARCHAR", "Pay grade"), ("salary", "DOUBLE", "Annual salary")]
    assert s.acl_groups == ["ALL"]


@pytest.mark.asyncio
async def test_save_is_idempotent_on_same_table(tmp_path):
    store = MetadataStore(database_url=f"sqlite+aiosqlite:///{tmp_path}/m.db")
    await store.init()
    await store.save_schema(_schema())
    await store.save_schema(_schema())  # re-save same db.table => no duplicate
    assert len(await store.load_all_schemas()) == 1


@pytest.mark.asyncio
async def test_delete_schema(tmp_path):
    store = MetadataStore(database_url=f"sqlite+aiosqlite:///{tmp_path}/m.db")
    await store.init()
    await store.save_schema(_schema("doc_a"))
    await store.save_schema(_schema("doc_b"))
    await store.delete_schema("spreadsheets", "doc_a")
    remaining = [s.table for s in await store.load_all_schemas()]
    assert remaining == ["doc_b"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `docker run --rm -v "$(pwd)":/app -w /app sauron-api python -m pytest tests/test_db/test_schema_persistence.py -q`
Expected: FAIL — `AttributeError: 'MetadataStore' object has no attribute 'save_schema'`.

- [ ] **Step 3: Add the `RegisteredSchema` model**

In `src/db/models.py`, the existing imports already include `JSON, DateTime, String, UniqueConstraint` and `datetime, timezone`. Add this class at the end of the file:

```python
class RegisteredSchema(Base):
    __tablename__ = "registered_schemas"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    database: Mapped[str] = mapped_column(String, nullable=False)
    table_name: Mapped[str] = mapped_column(String, nullable=False)
    columns: Mapped[list] = mapped_column(JSON, default=list)   # [{"name","dtype","description"}]
    description: Mapped[str] = mapped_column(String, default="")
    acl_groups: Mapped[list] = mapped_column(JSON, default=list)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
    )
    __table_args__ = (UniqueConstraint("database", "table_name", name="uq_registered_schema"),)
```

Note: the column is `table_name` (not `table`) because `table` collides with SQLAlchemy's declarative API. It maps to `TableSchema.table`.

- [ ] **Step 4: Add the `MetadataStore` methods**

In `src/db/metadata.py`, add `RegisteredSchema` to the existing model import line (the `from src.db.models import Base, DocumentRecord, ...` line), then add these three methods to the `MetadataStore` class:

```python
    async def save_schema(self, schema) -> None:
        """Persist a TableSchema, replacing any existing one with the same database.table."""
        from sqlalchemy import delete as sa_delete
        from src.db.models import RegisteredSchema
        cols = [{"name": c.name, "dtype": c.dtype, "description": c.description} for c in schema.columns]
        async with self.session_factory() as session:
            await session.execute(sa_delete(RegisteredSchema).where(
                RegisteredSchema.database == schema.database,
                RegisteredSchema.table_name == schema.table,
            ))
            session.add(RegisteredSchema(
                database=schema.database, table_name=schema.table,
                columns=cols, description=schema.description, acl_groups=schema.acl_groups,
            ))
            await session.commit()

    async def load_all_schemas(self) -> list:
        """Load every persisted schema as a list of TableSchema."""
        from src.db.models import RegisteredSchema
        from src.db.schema_registry import TableSchema, ColumnSchema
        async with self.session_factory() as session:
            result = await session.execute(select(RegisteredSchema))
            records = result.scalars().all()
        schemas = []
        for r in records:
            columns = [ColumnSchema(name=c["name"], dtype=c["dtype"], description=c.get("description", ""))
                       for c in (r.columns or [])]
            schemas.append(TableSchema(database=r.database, table=r.table_name,
                                       columns=columns, description=r.description,
                                       acl_groups=r.acl_groups or []))
        return schemas

    async def delete_schema(self, database: str, table: str) -> None:
        """Remove a persisted schema by database.table."""
        from sqlalchemy import delete as sa_delete
        from src.db.models import RegisteredSchema
        async with self.session_factory() as session:
            await session.execute(sa_delete(RegisteredSchema).where(
                RegisteredSchema.database == database,
                RegisteredSchema.table_name == table,
            ))
            await session.commit()
```

(`select` is already imported at the top of `metadata.py`.)

- [ ] **Step 5: Run test to verify it passes**

Run: `docker run --rm -v "$(pwd)":/app -w /app sauron-api python -m pytest tests/test_db/test_schema_persistence.py -q`
Expected: PASS (3 passed). `Base.metadata.create_all` in `store.init()` auto-creates `registered_schemas`; no migration entry is needed (the `_migrate` list is only for adding columns to pre-existing tables).

- [ ] **Step 6: Commit**

```bash
git add src/db/models.py src/db/metadata.py tests/test_db/test_schema_persistence.py
git commit -m "feat: persist registered table schemas in the metadata DB

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Final verification

- [ ] Run both new test files:

```bash
docker run --rm -v "$(pwd)":/app -w /app sauron-api python -m pytest \
  tests/test_ingestion/test_tabular_store.py tests/test_db/test_schema_persistence.py -q
```
Expected: all PASS — **21 total**. `test_tabular_store.py` = 18 (Task 2: 2; Task 3: 8 `_to_number` params + 2 load; Task 4: 1 + 4 reject params; Task 5: 1), `test_schema_persistence.py` = 3. (`_pay_grid` and `_schema` are helpers, not tests.)

- [ ] Confirm no unintended files changed:

```bash
git diff --stat <first-commit-of-this-plan>^..HEAD
```
Expected: only `requirements.txt`, `src/ingestion/tabular_store.py`, `src/db/models.py`, `src/db/metadata.py`, and the two new test files.

## Notes for the implementer

- **DuckDB connections are passed in** (`con`) rather than opened inside the store functions, so tests use fast `duckdb.connect()` in-memory connections and a later plan can choose where the persistent `.duckdb` file lives. Do NOT hardcode a file path here.
- **Idempotent re-ingest:** `load_sheet_to_duckdb` drops-and-recreates its table, and `save_schema` deletes-then-inserts — re-ingesting a document replaces its rows/schema cleanly.
- **This plan does NOT wire anything into ingestion or query routing.** `RegisteredSchema` persists schemas, but nothing populates the in-memory `SchemaRegistry` from it yet, and `retrieve_analytical` still uses sqlite `execute_sql`. Those connections are Plan 2b (pipeline wiring + registry-load-on-startup) and Plan 3 (route analytical queries to `execute_duckdb_sql`). Keeping them separate means this plan merges without changing live behavior.
- **`table` vs `table_name`:** the DB column is `table_name` (SQLAlchemy reserves `table`); always map it back to `TableSchema.table` in `load_all_schemas`.
- **SELECT-only guard** intentionally mirrors `src/db/sql_executor.py` rather than importing it, because that module is sqlite/SQLAlchemy-specific; DuckDB execution is synchronous and separate.
```
