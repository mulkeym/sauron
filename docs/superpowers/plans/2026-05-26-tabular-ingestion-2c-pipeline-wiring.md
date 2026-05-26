# Tabular Ingestion — Plan 2c: Pipeline Wiring (clean sheets go live)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Wire Plans 1/2a/2b together so that ingesting a spreadsheet loads each clean sheet into the persistent DuckDB store, registers+persists its schema, and embeds deterministic row narratives — and so persisted schemas reload into the in-memory registry at startup.

**Architecture:** A new orchestrator `src/ingestion/tabular_ingest.py` (`ingest_spreadsheet_tables`) does the clean-sheet work and is the testable integration unit. `ingest_document` gains a small, fail-open spreadsheet branch that calls it **in addition to** the existing text-chunk path (so messy sheets and a full-text fallback are unchanged — nothing is lost). `main.py` loads persisted schemas into the registry at startup. The existing full-text chunking is intentionally left in place; de-duplicating clean-sheet text is a Plan 4 optimization.

**Tech Stack:** Python 3.11, DuckDB, openpyxl/xlrd, LanceDB (`vector_store`), SQLAlchemy/aiosqlite (`MetadataStore`), pytest/pytest-asyncio. Tests run inside the app image: `docker run --rm -v "$(pwd)":/app -w /app sauron-api python -m pytest <path> -q`.

**Spec:** `docs/superpowers/specs/2026-05-25-tabular-spreadsheet-ingestion-design.md` (Component 2 wiring + Component 4 startup-load). **Depends on (all merged):** Plan 1 (`tabular.py`), Plan 2a (`tabular_store.py`, `RegisteredSchema`, `MetadataStore.save_schema/load_all_schemas`), Plan 2b (`table_profiler.py`).

> ⚠️ **This plan changes live behavior** and requires an image rebuild + redeploy after merge (`docker compose build && docker compose up -d`). It is the first tabular plan that is NOT dormant.

---

## File Structure

- `src/config.py` — **modify**: add `tabular_duckdb_path` setting.
- `src/ingestion/tabular_store.py` — **modify**: add `connect_tabular` (open the persistent DuckDB).
- `src/ingestion/tabular_ingest.py` — **create**. Owns: `ingest_spreadsheet_tables` (orchestrate clean-sheet ingest) and `populate_schema_registry` (startup load). The only module that ties parse→classify→store→profile→narrative→embed→upsert together.
- `src/ingestion/pipeline.py` — **modify**: add the fail-open spreadsheet branch to `ingest_document`.
- `src/main.py` — **modify**: call `populate_schema_registry` in `lifespan`.
- `tests/test_ingestion/test_tabular_store.py` — **modify**: add `connect_tabular` tests.
- `tests/test_ingestion/test_tabular_ingest.py` — **create**.

## Existing signatures this plan consumes (already in the repo)

- `tabular.read_sheets(path) -> list[SheetGrid]`, `tabular.classify_sheet(grid) -> SheetClassification` (fields: `sheet_name`, `route`, `header_row_index`, `column_dtypes`).
- `tabular_store.load_sheet_to_duckdb(con, doc_id, sheet_name, classification, grid) -> (table, col_names)`, `tabular_store.schema_from_sheet(doc_id, sheet_name, classification, grid, acl_groups=None) -> TableSchema`.
- `table_profiler.profile_table(sheet_name, col_names, column_dtypes, sample_rows, generate_fn=None) -> TableProfile` (fields: `column_descriptions`, `key_columns`, `measure_columns`, `table_description`), `table_profiler.build_row_narratives(col_names, profile, data_rows, context="") -> list[str]`.
- `embedder.embed_texts(texts) -> list[list[float]]`.
- `vector_store.upsert(texts, vectors, metadatas: list[ChunkMetadata])`.
- `ChunkMetadata(doc_id, filename, doc_type, chunk_index, start_char, acl_groups, category=..., chunk_size_tier=...)`.
- `MetadataStore.save_schema(schema)`, `MetadataStore.load_all_schemas() -> list[TableSchema]`.
- `routes_ingest.get_schema_registry() -> SchemaRegistry` (singleton); `SchemaRegistry.register(schema)`, `.list_for_user(groups)`.

---

### Task 1: `tabular_duckdb_path` setting + `connect_tabular`

**Files:**
- Modify: `src/config.py`
- Modify: `src/ingestion/tabular_store.py`
- Test: `tests/test_ingestion/test_tabular_store.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_ingestion/test_tabular_store.py` (extend the import to add `connect_tabular`):

```python
from src.ingestion.tabular_store import connect_tabular


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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `docker run --rm -v "$(pwd)":/app -w /app sauron-api python -m pytest tests/test_ingestion/test_tabular_store.py -k connect_tabular -q`
Expected: FAIL — `ImportError: cannot import name 'connect_tabular'`.

- [ ] **Step 3: Add the setting and the helper**

In `src/config.py`, after the LanceDB settings (the `lancedb_table_name` line), add:

```python
    # Tabular store (structured spreadsheet querying)
    tabular_duckdb_path: str = "data/tabular.duckdb"
```

In `src/ingestion/tabular_store.py`, add at the end of the file:

```python
def connect_tabular(read_only: bool = False):
    """Open a connection to the persistent tabular DuckDB database.

    Path comes from ``settings.tabular_duckdb_path`` (on the mounted data
    volume). Ingestion opens read-write (short-lived); query-time opens
    read_only. The parent directory is created if missing.
    """
    import os
    import duckdb
    from src.config import settings

    path = settings.tabular_duckdb_path
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    return duckdb.connect(path, read_only=read_only)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `docker run --rm -v "$(pwd)":/app -w /app sauron-api python -m pytest tests/test_ingestion/test_tabular_store.py -k connect_tabular -q`
Expected: PASS (2 passed).

- [ ] **Step 5: Commit**

```bash
git add src/config.py src/ingestion/tabular_store.py tests/test_ingestion/test_tabular_store.py
git commit -m "feat: persistent DuckDB connection helper (connect_tabular)

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task 2: `ingest_spreadsheet_tables` orchestrator (happy path)

**Files:**
- Create: `src/ingestion/tabular_ingest.py`
- Test: `tests/test_ingestion/test_tabular_ingest.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_ingestion/test_tabular_ingest.py`:

```python
"""Integration tests for the spreadsheet structured-ingest orchestrator."""
import json
from unittest.mock import MagicMock

import openpyxl
import pytest

from src.db.metadata import MetadataStore
from src.db.schema_registry import SchemaRegistry
from src.ingestion.tabular_store import connect_tabular, duckdb_table_name


def _write_xlsx(path, sheets):
    wb = openpyxl.Workbook()
    wb.remove(wb.active)
    for name, rows in sheets.items():
        ws = wb.create_sheet(title=name)
        for row in rows:
            ws.append(row)
    wb.save(path)


def _fake_profile_generate(system_prompt, user_prompt, **kwargs):
    return json.dumps({
        "column_descriptions": {"grade": "Pay grade", "step": "Step", "salary": "Annual salary"},
        "key_columns": ["grade", "step"],
        "measure_columns": ["salary"],
        "table_description": "GS pay by grade and step",
    })


@pytest.mark.asyncio
async def test_ingest_clean_sheet_stores_rows_schema_and_narratives(tmp_path, monkeypatch):
    from src.config import settings
    monkeypatch.setattr(settings, "tabular_duckdb_path", str(tmp_path / "t.duckdb"))

    # embed_texts is patched where the orchestrator looks it up
    import src.ingestion.tabular_ingest as ti
    monkeypatch.setattr(ti, "embed_texts", lambda texts: [[0.1, 0.2, 0.3] for _ in texts])

    xlsx = tmp_path / "pay.xlsx"
    _write_xlsx(xlsx, {
        "Pay": [["grade", "step", "salary"]] + [[f"GS-{g}", 5, 80000 + g] for g in range(10, 14)],
    })

    vector_store = MagicMock()
    store = MetadataStore(database_url=f"sqlite+aiosqlite:///{tmp_path}/m.db")
    await store.init()
    registry = SchemaRegistry()

    from src.ingestion.tabular_ingest import ingest_spreadsheet_tables
    n = await ingest_spreadsheet_tables(
        str(xlsx), "doc1", "pay.xlsx", "xlsx", ["ALL"], "",
        vector_store, store, schema_registry=registry, generate_fn=_fake_profile_generate,
    )

    assert n == 1
    # narratives embedded + upserted, one per data row, marked as table_row
    vector_store.upsert.assert_called_once()
    kwargs = vector_store.upsert.call_args.kwargs
    assert len(kwargs["texts"]) == 4
    assert all("Pay grade=" in t for t in kwargs["texts"])
    assert all(m.chunk_size_tier == "table_row" for m in kwargs["metadatas"])
    # schema registered in-memory AND persisted
    assert registry.list_for_user(["ALL"])
    persisted = await store.load_all_schemas()
    assert persisted[0].table == duckdb_table_name("doc1", "Pay")
    # rows landed in DuckDB
    con = connect_tabular(read_only=True)
    cnt = con.execute(f'SELECT COUNT(*) FROM "{duckdb_table_name("doc1", "Pay")}"').fetchone()[0]
    con.close()
    assert cnt == 4
```

- [ ] **Step 2: Run test to verify it fails**

Run: `docker run --rm -v "$(pwd)":/app -w /app sauron-api python -m pytest tests/test_ingestion/test_tabular_ingest.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.ingestion.tabular_ingest'`.

- [ ] **Step 3: Create the orchestrator**

Create `src/ingestion/tabular_ingest.py`:

```python
"""Wire clean spreadsheet sheets into the structured store + row narratives.

Called from the ingestion pipeline for spreadsheet documents. For each CLEAN
sheet: load rows into DuckDB, profile the columns (one LLM call), register +
persist the schema, and embed deterministic per-row narratives (the raw rows
stay authoritative in DuckDB). Fail-open per sheet — one bad sheet never aborts
the others or the surrounding document ingestion.
"""
import logging

from src.ingestion.tabular import read_sheets, classify_sheet
from src.ingestion.tabular_store import (
    connect_tabular, load_sheet_to_duckdb, schema_from_sheet,
)
from src.ingestion.table_profiler import profile_table, build_row_narratives
from src.ingestion.embedder import embed_texts
from src.retrieval.models import ChunkMetadata

logger = logging.getLogger(__name__)


async def ingest_spreadsheet_tables(file_path, doc_id, filename, doc_type, acl_groups,
                                    category, vector_store, metadata_store,
                                    schema_registry=None, generate_fn=None) -> int:
    """Process every CLEAN sheet of a spreadsheet. Returns the count processed."""
    if schema_registry is None:
        from src.api.routes_ingest import get_schema_registry
        schema_registry = get_schema_registry()

    try:
        grids = read_sheets(file_path)
    except Exception as e:
        logger.warning(f"Tabular ingest: could not read sheets from {filename}: {e}")
        return 0

    clean_count = 0
    chunk_index = 0
    con = None
    try:
        con = connect_tabular(read_only=False)
        for grid in grids:
            cls = classify_sheet(grid)
            if cls.route != "clean":
                continue
            try:
                _, col_names = load_sheet_to_duckdb(con, doc_id, grid.sheet_name, cls, grid)
                data_rows = grid.rows[cls.header_row_index + 1:]
                profile = profile_table(
                    grid.sheet_name, col_names, cls.column_dtypes, data_rows[:5],
                    generate_fn=generate_fn,
                )
                # Build + enrich the schema with the profile's labels/description.
                schema = schema_from_sheet(doc_id, grid.sheet_name, cls, grid, acl_groups=acl_groups)
                for col in schema.columns:
                    if col.name in profile.column_descriptions:
                        col.description = profile.column_descriptions[col.name]
                if profile.table_description:
                    schema.description = profile.table_description
                schema_registry.register(schema)
                await metadata_store.save_schema(schema)

                # Deterministic row narratives -> embeddings (raw rows live in DuckDB).
                narratives = [
                    n for n in build_row_narratives(
                        col_names, profile, data_rows, context=profile.table_description
                    ) if n.strip()
                ]
                if narratives:
                    vectors = embed_texts(narratives)
                    metadatas = []
                    for _ in narratives:
                        metadatas.append(ChunkMetadata(
                            doc_id=doc_id, filename=filename, doc_type=doc_type,
                            chunk_index=chunk_index, start_char=0, acl_groups=acl_groups,
                            category=category, chunk_size_tier="table_row",
                        ))
                        chunk_index += 1
                    vector_store.upsert(texts=narratives, vectors=vectors, metadatas=metadatas)
                clean_count += 1
            except Exception as e:
                logger.warning(f"Tabular ingest: failed on sheet '{grid.sheet_name}' of {filename}: {e}")
                continue
    finally:
        if con is not None:
            con.close()

    logger.info(f"Tabular ingest: processed {clean_count} clean sheet(s) from {filename}")
    return clean_count
```

- [ ] **Step 4: Run test to verify it passes**

Run: `docker run --rm -v "$(pwd)":/app -w /app sauron-api python -m pytest tests/test_ingestion/test_tabular_ingest.py -q`
Expected: PASS (1 passed).

- [ ] **Step 5: Commit**

```bash
git add src/ingestion/tabular_ingest.py tests/test_ingestion/test_tabular_ingest.py
git commit -m "feat: orchestrate clean-sheet ingest (DuckDB + schema + narratives)

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task 3: orchestrator — messy-only yields nothing; per-sheet failure is contained

**Files:**
- Modify: `tests/test_ingestion/test_tabular_ingest.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_ingestion/test_tabular_ingest.py`:

```python
@pytest.mark.asyncio
async def test_messy_only_workbook_stores_nothing(tmp_path, monkeypatch):
    from src.config import settings
    monkeypatch.setattr(settings, "tabular_duckdb_path", str(tmp_path / "t.duckdb"))
    import src.ingestion.tabular_ingest as ti
    monkeypatch.setattr(ti, "embed_texts", lambda texts: [[0.1] for _ in texts])

    xlsx = tmp_path / "notes.xlsx"
    _write_xlsx(xlsx, {"Readme": [["This is a narrative note."], ["Updated 2024."]]})

    vector_store = MagicMock()
    store = MetadataStore(database_url=f"sqlite+aiosqlite:///{tmp_path}/m.db")
    await store.init()
    registry = SchemaRegistry()

    from src.ingestion.tabular_ingest import ingest_spreadsheet_tables
    n = await ingest_spreadsheet_tables(
        str(xlsx), "doc2", "notes.xlsx", "xlsx", ["ALL"], "",
        vector_store, store, schema_registry=registry, generate_fn=_fake_profile_generate,
    )
    assert n == 0
    vector_store.upsert.assert_not_called()
    assert await store.load_all_schemas() == []


@pytest.mark.asyncio
async def test_per_sheet_failure_is_contained(tmp_path, monkeypatch):
    from src.config import settings
    monkeypatch.setattr(settings, "tabular_duckdb_path", str(tmp_path / "t.duckdb"))
    import src.ingestion.tabular_ingest as ti
    # embedding blows up -> the clean sheet fails, but the call must not raise
    monkeypatch.setattr(ti, "embed_texts", lambda texts: (_ for _ in ()).throw(RuntimeError("embed down")))

    xlsx = tmp_path / "pay.xlsx"
    _write_xlsx(xlsx, {
        "Pay": [["grade", "step", "salary"]] + [[f"GS-{g}", 5, 80000 + g] for g in range(10, 14)],
    })

    vector_store = MagicMock()
    store = MetadataStore(database_url=f"sqlite+aiosqlite:///{tmp_path}/m.db")
    await store.init()
    registry = SchemaRegistry()

    from src.ingestion.tabular_ingest import ingest_spreadsheet_tables
    n = await ingest_spreadsheet_tables(  # must NOT raise
        str(xlsx), "doc3", "pay.xlsx", "xlsx", ["ALL"], "",
        vector_store, store, schema_registry=registry, generate_fn=_fake_profile_generate,
    )
    assert n == 0  # the sheet failed at the embed step, so it was not counted
    vector_store.upsert.assert_not_called()
```

- [ ] **Step 2: Run test to verify it fails (or already passes)**

Run: `docker run --rm -v "$(pwd)":/app -w /app sauron-api python -m pytest tests/test_ingestion/test_tabular_ingest.py -k "messy_only or per_sheet_failure" -q`
Expected: These exercise existing orchestrator behavior; they should PASS once the orchestrator from Task 2 exists. (If a test fails, it has found a real gap — fix the orchestrator, not the test.) Run them to confirm the fail-open and routing behavior.

- [ ] **Step 3: No implementation needed if green**

The orchestrator already routes only `cls.route == "clean"` sheets and wraps each sheet body in `try/except`, so both behaviors are covered. If either test fails, make the minimal orchestrator fix so the per-sheet `try/except` (Task 2) contains the failure and messy sheets are skipped.

- [ ] **Step 4: Run test to verify it passes**

Run: `docker run --rm -v "$(pwd)":/app -w /app sauron-api python -m pytest tests/test_ingestion/test_tabular_ingest.py -q`
Expected: PASS (3 passed).

- [ ] **Step 5: Commit**

```bash
git add tests/test_ingestion/test_tabular_ingest.py
git commit -m "test: messy-only and per-sheet-failure behavior of tabular ingest

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task 4: wire the spreadsheet branch into `ingest_document`

**Files:**
- Modify: `src/ingestion/pipeline.py`
- Test: `tests/test_ingestion/test_tabular_ingest.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_ingestion/test_tabular_ingest.py`. This drives `ingest_document` with all heavy collaborators patched, asserting the spreadsheet branch invokes the orchestrator for a spreadsheet and skips it for a `.txt` file:

```python
@pytest.mark.asyncio
async def test_ingest_document_invokes_tabular_branch_for_spreadsheet(tmp_path, monkeypatch):
    import src.ingestion.pipeline as pipe
    import src.ingestion.tabular_ingest as ti
    import src.generation.llm_client as llm
    import src.knowledge.graph_rag as kg

    # Patch heavy collaborators so we exercise only the branch logic.
    monkeypatch.setattr(pipe, "embed_texts", lambda texts: [[0.1] for _ in texts])
    monkeypatch.setattr(llm, "generate", lambda **kw: "summary")

    async def _noop_insert(*a, **k):
        return None
    monkeypatch.setattr(kg, "insert_document", _noop_insert)

    calls = []

    async def _fake_orch(*args, **kwargs):
        calls.append(args)
        return 1
    monkeypatch.setattr(ti, "ingest_spreadsheet_tables", _fake_orch)

    store = MetadataStore(database_url=f"sqlite+aiosqlite:///{tmp_path}/m.db")
    await store.init()
    vector_store = MagicMock()

    xlsx = tmp_path / "pay.xlsx"
    _write_xlsx(xlsx, {"Pay": [["grade", "salary"], ["GS-12", 86415], ["GS-13", 102000], ["GS-14", 120000]]})
    await pipe.ingest_document(str(xlsx), ["ALL"], "tester", vector_store, store)
    assert len(calls) == 1  # orchestrator invoked for the spreadsheet

    txt = tmp_path / "note.txt"
    txt.write_text("just some text\n", encoding="utf-8")
    await pipe.ingest_document(str(txt), ["ALL"], "tester", vector_store, store)
    assert len(calls) == 1  # NOT invoked again for a non-spreadsheet
```

- [ ] **Step 2: Run test to verify it fails**

Run: `docker run --rm -v "$(pwd)":/app -w /app sauron-api python -m pytest tests/test_ingestion/test_tabular_ingest.py -k invokes_tabular_branch -q`
Expected: FAIL — `assert len(calls) == 1` is `0` (no branch wired yet).

- [ ] **Step 3: Add the spreadsheet branch to `ingest_document`**

In `src/ingestion/pipeline.py`, in `ingest_document`, insert this block immediately AFTER the `await metadata_store.add_document(...)` call (currently lines 111-119) and BEFORE the `# Ensure category exists` block:

```python
    # Structured handling for spreadsheets: clean sheets -> DuckDB + row narratives.
    # Fail-open: a structured-ingest error must never break document ingestion.
    if parsed.doc_type in ("xlsx", "xls", "csv", "tsv"):
        try:
            from src.ingestion.tabular_ingest import ingest_spreadsheet_tables
            await ingest_spreadsheet_tables(
                file_path, doc_id, parsed.filename, parsed.doc_type,
                acl_groups, category, vector_store, metadata_store,
            )
        except Exception as e:
            logging.getLogger(__name__).warning(f"Spreadsheet structured ingest failed: {e}")
```

(`logging` is already imported inside `ingest_document` at line 65.)

- [ ] **Step 4: Run test to verify it passes**

Run: `docker run --rm -v "$(pwd)":/app -w /app sauron-api python -m pytest tests/test_ingestion/test_tabular_ingest.py -k invokes_tabular_branch -q`
Expected: PASS (1 passed).

Note: the test patches `src.ingestion.tabular_ingest.ingest_spreadsheet_tables`; because `ingest_document` imports it locally (`from src.ingestion.tabular_ingest import ingest_spreadsheet_tables`) at call time, the patch on the source module takes effect.

- [ ] **Step 5: Commit**

```bash
git add src/ingestion/pipeline.py tests/test_ingestion/test_tabular_ingest.py
git commit -m "feat: route spreadsheet ingestion through the structured orchestrator

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task 5: load persisted schemas into the registry at startup

**Files:**
- Modify: `src/ingestion/tabular_ingest.py`
- Modify: `src/main.py`
- Test: `tests/test_ingestion/test_tabular_ingest.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_ingestion/test_tabular_ingest.py`:

```python
@pytest.mark.asyncio
async def test_populate_schema_registry_loads_persisted(tmp_path):
    from src.db.schema_registry import TableSchema, ColumnSchema
    from src.ingestion.tabular_ingest import populate_schema_registry

    store = MetadataStore(database_url=f"sqlite+aiosqlite:///{tmp_path}/m.db")
    await store.init()
    await store.save_schema(TableSchema(
        database="spreadsheets", table="doc_x_pay",
        columns=[ColumnSchema(name="grade", dtype="VARCHAR", description="Pay grade")],
        description="GS pay", acl_groups=["ALL"],
    ))

    registry = SchemaRegistry()
    await populate_schema_registry(store, registry)

    loaded = registry.list_for_user(["ALL"])
    assert len(loaded) == 1
    assert loaded[0].table == "doc_x_pay"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `docker run --rm -v "$(pwd)":/app -w /app sauron-api python -m pytest tests/test_ingestion/test_tabular_ingest.py -k populate_schema_registry -q`
Expected: FAIL — `ImportError: cannot import name 'populate_schema_registry'`.

- [ ] **Step 3: Add `populate_schema_registry` and wire it into startup**

In `src/ingestion/tabular_ingest.py`, add at the end:

```python
async def populate_schema_registry(metadata_store, schema_registry) -> int:
    """Load every persisted TableSchema into the in-memory registry. Returns count."""
    schemas = await metadata_store.load_all_schemas()
    for schema in schemas:
        schema_registry.register(schema)
    return len(schemas)
```

In `src/main.py`, in `lifespan`, after the LightRAG init `try/except` block and before `yield`, add:

```python
    # Load persisted table schemas into the in-memory registry
    try:
        from src.api.routes_ingest import get_schema_registry
        from src.ingestion.tabular_ingest import populate_schema_registry
        n = await populate_schema_registry(store, get_schema_registry())
        import logging
        logging.getLogger(__name__).info(f"Loaded {n} persisted table schema(s) into the registry")
    except Exception as e:
        import logging
        logging.getLogger(__name__).warning(f"Schema registry load deferred: {e}")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `docker run --rm -v "$(pwd)":/app -w /app sauron-api python -m pytest tests/test_ingestion/test_tabular_ingest.py -k populate_schema_registry -q`
Expected: PASS (1 passed).

- [ ] **Step 5: Commit**

```bash
git add src/ingestion/tabular_ingest.py src/main.py tests/test_ingestion/test_tabular_ingest.py
git commit -m "feat: load persisted table schemas into the registry at startup

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Final verification

- [ ] Run the affected suites:

```bash
docker run --rm -v "$(pwd)":/app -w /app sauron-api python -m pytest \
  tests/test_ingestion/test_tabular_ingest.py \
  tests/test_ingestion/test_tabular_store.py \
  tests/test_ingestion/test_tabular.py \
  tests/test_ingestion/test_table_profiler.py -q
```
Expected: all PASS (tabular_ingest: 5; tabular_store: 25; tabular: 26; table_profiler: 14). If the tabular_store/tabular/table_profiler counts differ slightly from earlier plans' growth, that's fine — what matters is zero failures.

- [ ] Confirm the changed files are only those intended:

```bash
git diff --stat <first-commit-of-this-plan>^..HEAD
```
Expected: `src/config.py`, `src/ingestion/tabular_store.py`, `src/ingestion/tabular_ingest.py`, `src/ingestion/pipeline.py`, `src/main.py`, and the two test files.

- [ ] **Deploy (this plan changes live behavior):** after merge, rebuild and restart so the branch + startup load take effect:

```bash
docker compose build && docker compose up -d
```
Then confirm health and that startup logged the schema-registry load:
```bash
docker logs sauron-api-1 2>&1 | grep -i "persisted table schema"
```

## Notes for the implementer

- **Additive, not destructive:** `ingest_document` still runs its full-text chunking for ALL documents (including spreadsheets). The spreadsheet branch ADDS structured storage + row narratives for clean sheets. This means clean-sheet content is briefly duplicated (text chunks + narratives + DuckDB); de-duplicating is a Plan 4 optimization. Do NOT remove or gate the existing text path in this plan.
- **Raw rows live in DuckDB**, not in chunk metadata — `ChunkMetadata` has no free-form field and changing the LanceDB schema is out of scope. The narrative text carries the human-readable values; the authoritative row is queryable in DuckDB via `duckdb_table_name(doc_id, sheet)`. Plan 3 uses the DuckDB table for exact SQL.
- **Row narratives are marked `chunk_size_tier="table_row"`** so later code can distinguish them from the small/medium/large/xlarge text chunks.
- **Fail-open is load-bearing:** the branch in `ingest_document` is wrapped in `try/except`, and the orchestrator wraps each sheet. A profiling/embedding/DuckDB error degrades to "no structured data for this sheet," never a failed upload.
- **DuckDB single-writer:** prod runs `max_parallel_ingestion=1`, and `connect_tabular(read_only=False)` is opened per-document and closed promptly, so write-lock contention is avoided. Cross-process concurrent writes (e.g., if the MCP container also ingested) are a known DuckDB limitation to revisit if that path is ever enabled.
- **`embed_texts` is patched in tests** at `src.ingestion.tabular_ingest.embed_texts` (the orchestrator's lookup) and `src.ingestion.pipeline.embed_texts` (the text path) — both are module-level imports, so `monkeypatch.setattr` on those names works.
```
