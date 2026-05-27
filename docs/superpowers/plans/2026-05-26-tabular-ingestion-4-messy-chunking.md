# Tabular Ingestion Plan 4 — Messy-Sheet Structure-Aware Chunking + Conditional Clean-Sheet De-dup Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make messy/narrative spreadsheet sheets retrievable with row-atomic, header-repeated chunks (no cell data lost) plus deterministic restate-only narratives over any table-like region inside them, and stop emitting redundant full-text chunks for clean sheets that were already loaded into the structured DuckDB store — but only when that structured ingest actually succeeded (fail-open fallback preserved).

**Architecture:** A new pure module `src/ingestion/tabular_chunker.py` provides sheet-aware chunking, table-region detection, and region narratives — all operating on in-memory grids, no I/O, trivially unit-testable. `tabular_ingest.py` gains `ingest_structured_sheets`, which reads each spreadsheet's sheets exactly once, sends CLEAN sheets to the existing DuckDB+schema+narrative path (collecting the set that fully succeeded), and embeds deterministic region narratives for MESSY sheets. Both ingestion entry points (`pipeline.py` sync + `queue.py` async worker) then build their per-tier text chunks from `tabular_chunker` instead of the generic `chunk_text`, feeding it only the sheets that still need text representation — every messy sheet plus any clean sheet whose structured ingest failed. The existing 4 chunk tiers (small/medium/large/xlarge) are preserved so the tier-specific retrieval strategies (sweep/map-reduce query `xlarge`, lookup/cross_reference query `medium`) keep finding spreadsheet content unchanged; region narratives go to the existing `table_row` tier so structured retrieval finds them alongside clean-row narratives.

**Tech Stack:** Python 3, dataclasses, pytest, openpyxl (test fixtures), DuckDB + LanceDB (existing stores). No new dependencies.

---

## Background — read before starting

This is Plan 4 of the tabular-spreadsheet-ingestion roadmap. Plans 1–3b are already live: clean sheets are routed to DuckDB with a registered schema and answered by text-to-SQL, and deterministic per-row narratives (`build_row_narratives`) are embedded at `chunk_size_tier="table_row"`. The design spec is `docs/superpowers/specs/2026-05-25-tabular-spreadsheet-ingestion-design.md` (Component 3 = this plan).

**Today's redundancy (the thing we fix).** `parser._parse_spreadsheet` flattens the *whole* workbook to `header | val | val` text into `parsed.text`. Both ingestion paths then chunk that blob at 4 tiers via `chunk_text` and embed every tier — for *every* sheet, clean or messy. So a clean GS pay table is stored ~3 ways: full-text chunks (4 tiers) + DuckDB rows + row narratives. The full-text chunks are the redundant part for clean sheets, and they are the ones that arbitrarily split mid-row and produce the giant number-grid chunks.

**Two ingestion paths.** `src/ingestion/pipeline.py::ingest_document` (sync) AND `src/ingestion/queue.py::IngestionQueue._process_job` (async worker, what the admin UI uses) BOTH chunk + embed + call the spreadsheet helper. Any change to ingestion MUST touch both, or the queue path silently diverges. Shared logic goes in the tabular modules; only thin per-tier branching lives in the two files.

**Decisions locked for this plan (do not re-litigate):**
- **Conditional de-dup**: suppress a clean sheet's full-text chunks ONLY if its structured ingest fully succeeded; a failed clean sheet falls back to structure-aware full-text chunks.
- **Messy narratives = deterministic region narratives**: detect a table-like rectangular region inside a messy sheet and run the EXISTING deterministic `build_row_narratives` on it via the LLM-free `_heuristic_profile`. No LLM on the messy path. No fabrication.
- **Tiers unchanged**: structure-aware chunks populate the same small/medium/large/xlarge tiers; region narratives use `table_row`.

**Out of scope (note as deferred, do not implement):** LLM restate narratives for messy sheets; multi-region detection within one messy sheet (we take the first/largest contiguous region); feeding messy sheets to LightRAG differently (`lightrag_insert(parsed.text, ...)` stays as-is); the `table_summary` discovery chunk (a Plan 2 gap).

---

## File Structure

- **Create** `src/ingestion/tabular_chunker.py` — pure functions: `structure_aware_chunks`, `build_tier_chunks`, `find_table_region`, `messy_region_narratives`, `sheets_needing_text`. No I/O.
- **Create** `tests/test_ingestion/test_tabular_chunker.py` — unit tests for the above.
- **Modify** `src/ingestion/tabular_ingest.py` — add `ingest_structured_sheets` (reads sheets once, structured-ingests clean sheets, embeds messy region narratives, returns `(grids, classifications, ingested_names)`); refactor the per-sheet clean logic out of `ingest_spreadsheet_tables` into a reusable `_ingest_one_clean_sheet`; promote `_SPREADSHEET_DOC_TYPES` → `SPREADSHEET_DOC_TYPES`; remove `maybe_ingest_spreadsheet`.
- **Modify** `tests/test_ingestion/test_tabular_ingest.py` — update for the new return shape / entry point.
- **Modify** `src/ingestion/pipeline.py` — spreadsheet branch: read+classify once via `ingest_structured_sheets`, build per-tier chunks from `tabular_chunker`.
- **Modify** `src/ingestion/queue.py` — same spreadsheet branch; keep `chunks` defined (line ~399 uses `len(chunks)`).
- **Modify** `tests/test_ingestion/test_pipeline.py` — regression for the spreadsheet branch.

---

## Task 1: `structure_aware_chunks` — row-atomic, header-repeated chunks for one sheet

**Files:**
- Create: `src/ingestion/tabular_chunker.py`
- Test: `tests/test_ingestion/test_tabular_chunker.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_ingestion/test_tabular_chunker.py
"""Tests for src/ingestion/tabular_chunker.py — sheet-aware chunking + region narratives."""
from src.ingestion.chunker import Chunk
from src.ingestion.tabular import SheetGrid, SheetClassification
from src.ingestion.tabular_chunker import (
    structure_aware_chunks,
    build_tier_chunks,
    find_table_region,
    messy_region_narratives,
    sheets_needing_text,
)


def test_structure_aware_chunks_repeats_header_and_marks_sheet():
    rows = [["grade", "step", "salary"], ["GS-12", "5", "86415"], ["GS-13", "1", "90000"]]
    chunks = structure_aware_chunks("Pay", rows, header_row_index=0, chunk_size=1000)
    assert len(chunks) == 1
    assert chunks[0].text.startswith("Sheet: Pay\ngrade | step | salary")
    assert "GS-12 | 5 | 86415" in chunks[0].text
    assert "GS-13 | 1 | 90000" in chunks[0].text


def test_structure_aware_chunks_never_splits_mid_row():
    # 6 data rows; a tiny chunk_size forces multiple chunks, each carrying the header.
    rows = [["a", "b"]] + [[f"r{i}", f"v{i}"] for i in range(6)]
    chunks = structure_aware_chunks("S", rows, header_row_index=0, chunk_size=30)
    assert len(chunks) > 1
    for c in chunks:
        assert c.text.startswith("Sheet: S\na | b")
        # every non-header line is a whole "rN | vN" row, never a fragment
        body_lines = c.text.split("\n")[2:]
        for line in body_lines:
            assert line.count("|") == 1
            assert line.split(" | ")[0].startswith("r")


def test_structure_aware_chunks_no_header_emits_all_rows_with_marker():
    rows = [["intro note"], ["x", "y"], ["1", "2"]]
    chunks = structure_aware_chunks("Mess", rows, header_row_index=-1, chunk_size=1000)
    assert len(chunks) == 1
    assert chunks[0].text.startswith("Sheet: Mess\n")
    assert "intro note" in chunks[0].text
    assert "x | y" in chunks[0].text


def test_structure_aware_chunks_indices_and_start_offsets():
    rows = [["a", "b"]] + [[f"r{i}", f"v{i}"] for i in range(6)]
    chunks = structure_aware_chunks("S", rows, header_row_index=0, chunk_size=30,
                                    start_index=10, start_char=500)
    assert chunks[0].index == 10
    assert chunks[1].index == 11
    assert chunks[0].start_char == 500
    assert chunks[1].start_char == 500 + len(chunks[0].text) + 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_ingestion/test_tabular_chunker.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.ingestion.tabular_chunker'`

- [ ] **Step 3: Create the module with `structure_aware_chunks`**

```python
# src/ingestion/tabular_chunker.py
"""Structure-aware text chunking + table-region narratives for spreadsheets.

Clean sheets go to the structured DuckDB store; this module owns the TEXT side.
Messy sheets (and any clean sheet whose structured ingest failed) get row-atomic,
header-repeated chunks so no cell data is lost, plus deterministic restate-only
narratives over any table-like region detected inside them. All functions are
pure and operate on in-memory grids — no file or store I/O.
"""
from __future__ import annotations

from src.ingestion.chunker import Chunk
from src.ingestion.tabular import (
    SheetGrid, SheetClassification, detect_header_row, infer_column_dtypes,
    _cell_kind, COLUMN_CONSISTENCY, MIN_DATA_ROWS,
)
from src.ingestion.table_profiler import _heuristic_profile, build_row_narratives
from src.ingestion.tabular_store import _safe_column_names


def _fmt_row(row: list) -> str:
    """Render one grid row as pipe-joined text (None -> empty cell)."""
    return " | ".join("" if c is None else str(c) for c in row)


def structure_aware_chunks(sheet_name: str, rows: list, header_row_index: int,
                           chunk_size: int = 2048, start_index: int = 0,
                           start_char: int = 0) -> list[Chunk]:
    """Row-atomic chunks for one sheet.

    Every chunk leads with a ``Sheet: <name>`` marker and (if detected) the
    header row, then whole data rows joined by newlines. A new chunk starts when
    appending the next row would push the chunk past ``chunk_size`` — rows are
    NEVER split mid-row, so a single row wider than ``chunk_size`` simply forms
    its own oversized chunk. ``header_row_index`` < 0 means no header: all rows
    are emitted as data under the marker. Chunks are numbered from
    ``start_index``; ``start_char`` seeds the first chunk's offset and each
    subsequent chunk's offset follows the previous chunk's text length + 1.
    """
    marker = f"Sheet: {sheet_name}"
    if 0 <= header_row_index < len(rows):
        header_line = _fmt_row(rows[header_row_index])
        data_rows = rows[header_row_index + 1:]
    else:
        header_line = ""
        data_rows = rows
    lead = marker + (f"\n{header_line}" if header_line else "")

    chunks: list[Chunk] = []
    cur: list[str] = []
    char_pos = start_char

    def flush():
        nonlocal char_pos
        if not cur:
            return
        text = lead + "\n" + "\n".join(cur)
        chunks.append(Chunk(text=text, index=start_index + len(chunks), start_char=char_pos))
        char_pos += len(text) + 1

    for row in data_rows:
        line = _fmt_row(row)
        projected = len(lead) + 1 + sum(len(r) + 1 for r in cur) + len(line)
        if cur and projected > chunk_size:
            flush()
            cur = []
        cur.append(line)
    flush()
    return chunks
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_ingestion/test_tabular_chunker.py -v`
Expected: PASS (4 tests)

- [ ] **Step 5: Commit**

```bash
git add src/ingestion/tabular_chunker.py tests/test_ingestion/test_tabular_chunker.py
git commit -m "feat: structure-aware row-atomic chunking for spreadsheet sheets"
```

---

## Task 2: `find_table_region` — locate a table-like block inside a messy sheet

**Files:**
- Modify: `src/ingestion/tabular_chunker.py`
- Test: `tests/test_ingestion/test_tabular_chunker.py`

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_ingestion/test_tabular_chunker.py
def test_find_table_region_in_messy_sheet():
    # title banner + blank-ish preamble, then a clean rectangular block, then trailing note
    rows = [
        ["2026 GS Pay Table"],            # 0 single-cell banner
        ["grade", "step", "salary"],      # 1 header
        ["GS-12", "5", "86415"],          # 2 data
        ["GS-13", "1", "90000"],          # 3 data
        ["GS-14", "2", "99000"],          # 4 data
        ["note: rates effective Jan 1"],  # 5 trailing (width mismatch ends region)
    ]
    region = find_table_region(rows)
    assert region == (1, 5)  # header at row 1, data rows [2,5)


def test_find_table_region_returns_none_when_no_block():
    rows = [["just"], ["some"], ["prose"], ["lines"]]
    assert find_table_region(rows) is None


def test_find_table_region_requires_min_data_rows():
    rows = [["a", "b"], ["1", "2"]]  # only one data row beneath header
    assert find_table_region(rows) is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_ingestion/test_tabular_chunker.py::test_find_table_region_in_messy_sheet -v`
Expected: FAIL — `AttributeError`/`ImportError` (function not yet defined) — note the import at the top of the test file already lists `find_table_region`, so collection will error until the function exists.

- [ ] **Step 3: Add `find_table_region`**

```python
# add to src/ingestion/tabular_chunker.py
def find_table_region(rows: list) -> tuple[int, int] | None:
    """Locate the largest contiguous table-like block inside a sheet.

    Returns ``(header_row_index, data_end_exclusive)`` for the run of rows
    directly beneath a detected header that share the header's column count and
    are column-type-consistent (same thresholds as clean-sheet classification),
    or None if no qualifying block exists (too few data rows, no header, or
    inconsistent columns). Detection is modest by design: it finds the first
    header in the leading rows and the contiguous width-matching run beneath it
    — stacked secondary tables are deferred.
    """
    header_idx = detect_header_row(rows)
    if header_idx < 0:
        return None
    ncols = len(rows[header_idx])
    end = header_idx + 1
    while end < len(rows) and len(rows[end]) == ncols:
        end += 1
    data = rows[header_idx + 1:end]
    if len(data) < MIN_DATA_ROWS:
        return None
    for col in range(ncols):
        counts = {"number": 0, "text": 0}
        for r in data:
            kind = _cell_kind(r[col]) if col < len(r) else "empty"
            if kind != "empty":
                counts[kind] += 1
        total = counts["number"] + counts["text"]
        if total and max(counts["number"], counts["text"]) / total < COLUMN_CONSISTENCY:
            return None
    return (header_idx, end)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_ingestion/test_tabular_chunker.py -v`
Expected: PASS (7 tests)

- [ ] **Step 5: Commit**

```bash
git add src/ingestion/tabular_chunker.py tests/test_ingestion/test_tabular_chunker.py
git commit -m "feat: detect table-like region inside a messy spreadsheet sheet"
```

---

## Task 3: `messy_region_narratives` — deterministic restate-only narratives for a region

**Files:**
- Modify: `src/ingestion/tabular_chunker.py`
- Test: `tests/test_ingestion/test_tabular_chunker.py`

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_ingestion/test_tabular_chunker.py
def test_messy_region_narratives_restates_rows_no_llm():
    rows = [
        ["2026 GS Pay Table"],
        ["locality", "grade", "salary"],
        ["Tampa", "GS-12", "86415"],
        ["Boston", "GS-12", "92000"],
        ["Denver", "GS-13", "99000"],
    ]
    grid = SheetGrid(sheet_name="Pay", rows=rows)
    region = find_table_region(rows)
    assert region is not None
    narratives = messy_region_narratives(grid, region)
    assert len(narratives) == 3  # one per data row in the region
    # keys (text cols) as context, measures (number cols) restated; raw values, no math
    joined = " ".join(narratives)
    assert "Tampa" in joined and "86415" in joined
    assert "Pay" in narratives[0]  # context defaults to sheet name


def test_messy_region_narratives_missing_cell_not_fabricated():
    rows = [
        ["locality", "grade", "salary"],
        ["Tampa", "GS-12", "86415"],
        ["Boston", "GS-12"],            # missing salary cell
        ["Denver", "GS-13", "99000"],
    ]
    grid = SheetGrid(sheet_name="Pay", rows=rows)
    region = find_table_region(rows)
    narratives = messy_region_narratives(grid, region)
    assert any("(not specified)" in n for n in narratives)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_ingestion/test_tabular_chunker.py::test_messy_region_narratives_restates_rows_no_llm -v`
Expected: FAIL — function not defined.

- [ ] **Step 3: Add `messy_region_narratives`**

```python
# add to src/ingestion/tabular_chunker.py
def messy_region_narratives(grid: SheetGrid, region: tuple[int, int],
                            context: str = "") -> list[str]:
    """Deterministic restate-only narratives for a detected table-like region.

    No LLM: builds the dtype heuristic profile (number columns are measures, the
    rest are keys) and runs the SAME per-row narrative builder used for clean
    tables. Missing cells render as "(not specified)" — nothing is fabricated.
    ``context`` defaults to the sheet name so a retrieved narrative is
    self-describing. Empty narratives are dropped.
    """
    header_idx, end = region
    header = grid.rows[header_idx]
    col_names = _safe_column_names(header)
    dtypes = infer_column_dtypes(grid.rows[:end], header_idx)
    profile = _heuristic_profile(col_names, dtypes)
    data_rows = grid.rows[header_idx + 1:end]
    ctx = context or grid.sheet_name
    return [n for n in build_row_narratives(col_names, profile, data_rows, context=ctx)
            if n.strip()]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_ingestion/test_tabular_chunker.py -v`
Expected: PASS (9 tests)

- [ ] **Step 5: Commit**

```bash
git add src/ingestion/tabular_chunker.py tests/test_ingestion/test_tabular_chunker.py
git commit -m "feat: deterministic restate-only narratives for messy-sheet table regions"
```

---

## Task 4: `sheets_needing_text` + `build_tier_chunks` — conditional de-dup + multi-sheet tier builder

**Files:**
- Modify: `src/ingestion/tabular_chunker.py`
- Test: `tests/test_ingestion/test_tabular_chunker.py`

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_ingestion/test_tabular_chunker.py
def _grid(name, rows):
    return SheetGrid(sheet_name=name, rows=rows)


def _cls(name, route, hdr=0):
    return SheetClassification(sheet_name=name, route=route, header_row_index=hdr)


def test_sheets_needing_text_skips_ingested_clean_keeps_messy_and_failed_clean():
    grids = [_grid("Clean", [["a", "b"]]), _grid("Messy", [["x"]]), _grid("Failed", [["c", "d"]])]
    clss = [_cls("Clean", "clean"), _cls("Messy", "messy", hdr=-1), _cls("Failed", "clean")]
    ingested = {"Clean"}  # only "Clean" structured-ingest succeeded
    out = sheets_needing_text(grids, clss, ingested)
    names = [g.sheet_name for g, _ in out]
    assert names == ["Messy", "Failed"]  # ingested clean dropped; messy + failed-clean kept


def test_build_tier_chunks_concatenates_across_sheets_with_continuous_indices():
    grids = [
        _grid("A", [["h1", "h2"], ["a", "1"], ["b", "2"]]),
        _grid("B", [["h3", "h4"], ["c", "3"], ["d", "4"]]),
    ]
    clss = [_cls("A", "messy"), _cls("B", "messy")]
    text_sheets = list(zip(grids, clss))
    chunks = build_tier_chunks(text_sheets, chunk_size=1000)
    assert [c.index for c in chunks] == list(range(len(chunks)))
    assert any("Sheet: A" in c.text for c in chunks)
    assert any("Sheet: B" in c.text for c in chunks)


def test_build_tier_chunks_empty_when_no_sheets():
    assert build_tier_chunks([], chunk_size=1000) == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_ingestion/test_tabular_chunker.py::test_sheets_needing_text_skips_ingested_clean_keeps_messy_and_failed_clean -v`
Expected: FAIL — functions not defined.

- [ ] **Step 3: Add `sheets_needing_text` and `build_tier_chunks`**

```python
# add to src/ingestion/tabular_chunker.py
def sheets_needing_text(grids: list, classifications: list,
                        ingested_sheets: set) -> list:
    """Return ``[(grid, classification), ...]`` for every sheet that still needs a
    TEXT representation: all messy sheets, plus any clean sheet NOT in
    ``ingested_sheets`` (its structured ingest failed, so full text is its
    fallback). Clean sheets that were structured-ingested are dropped — this is
    the conditional de-dup."""
    out = []
    for grid, cls in zip(grids, classifications):
        if cls.route == "clean" and cls.sheet_name in ingested_sheets:
            continue
        out.append((grid, cls))
    return out


def build_tier_chunks(text_sheets: list, chunk_size: int) -> list[Chunk]:
    """Structure-aware chunks for all ``text_sheets`` concatenated into one list
    with continuous chunk indices and running char offsets. ``text_sheets`` is
    ``[(grid, classification), ...]`` (typically from ``sheets_needing_text``)."""
    chunks: list[Chunk] = []
    char_pos = 0
    for grid, cls in text_sheets:
        sheet_chunks = structure_aware_chunks(
            grid.sheet_name, grid.rows, cls.header_row_index,
            chunk_size=chunk_size, start_index=len(chunks), start_char=char_pos,
        )
        chunks.extend(sheet_chunks)
        if sheet_chunks:
            last = sheet_chunks[-1]
            char_pos = last.start_char + len(last.text) + 1
    return chunks
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_ingestion/test_tabular_chunker.py -v`
Expected: PASS (12 tests)

- [ ] **Step 5: Commit**

```bash
git add src/ingestion/tabular_chunker.py tests/test_ingestion/test_tabular_chunker.py
git commit -m "feat: conditional clean-sheet de-dup + multi-sheet tier chunk builder"
```

---

## Task 5: Refactor `tabular_ingest` — extract per-sheet clean ingest, add `ingest_structured_sheets`

**Files:**
- Modify: `src/ingestion/tabular_ingest.py`
- Test: `tests/test_ingestion/test_tabular_ingest.py`

This task introduces the orchestrator that reads sheets once, structured-ingests clean sheets (collecting the set that fully succeed), and embeds deterministic region narratives for messy sheets. It reuses the existing per-sheet clean logic (extracted into a helper) so behavior for clean sheets is unchanged.

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_ingestion/test_tabular_ingest.py
import pytest
from src.ingestion.tabular_ingest import ingest_structured_sheets, SPREADSHEET_DOC_TYPES


class _FakeVectorStore:
    def __init__(self):
        self.upserts = []  # list of (texts, metadatas)

    def upsert(self, texts, vectors, metadatas):
        self.upserts.append((list(texts), list(metadatas)))


class _FakeMetadataStore:
    def __init__(self):
        self.saved = []

    async def save_schema(self, schema):
        self.saved.append(schema)


class _FakeRegistry:
    def __init__(self):
        self.registered = []

    def register(self, schema):
        self.registered.append(schema)


@pytest.mark.asyncio
async def test_ingest_structured_sheets_clean_ingested_messy_gets_region_narratives(tmp_path, monkeypatch):
    import openpyxl
    p = tmp_path / "book.xlsx"
    wb = openpyxl.Workbook()
    wb.remove(wb.active)
    clean = wb.create_sheet("Pay")
    for row in [["locality", "grade", "salary"], ["Tampa", "GS-12", 86415],
                ["Boston", "GS-12", 92000], ["Denver", "GS-13", 99000]]:
        clean.append(row)
    messy = wb.create_sheet("Notes")
    for row in [["2026 Pay Notes"], ["locality", "grade", "salary"],
                ["Reno", "GS-9", 55000], ["Mesa", "GS-9", 56000], ["Ames", "GS-9", 57000],
                ["effective Jan 1 — see appendix"]]:
        messy.append(row)
    wb.save(p)

    # avoid real embeddings: return a fixed-width vector per text
    monkeypatch.setattr("src.ingestion.tabular_ingest.embed_texts",
                        lambda texts, *a, **k: [[0.0, 0.0, 0.0] for _ in texts])

    vs, ms, reg = _FakeVectorStore(), _FakeMetadataStore(), _FakeRegistry()
    grids, clss, ingested = await ingest_structured_sheets(
        str(p), "doc1", "book.xlsx", "xlsx", ["g1"], "cat",
        vs, ms, schema_registry=reg, generate_fn=lambda **k: "{}",
    )
    assert [g.sheet_name for g in grids] == ["Pay", "Notes"]
    assert "Pay" in ingested              # clean sheet fully ingested
    assert "Notes" not in ingested        # messy sheet not "ingested" to DuckDB
    # messy region narratives were embedded at the table_row tier
    tiers = [m.chunk_size_tier for _, metas in vs.upserts for m in metas]
    assert "table_row" in tiers
    # narrative text restates a messy-region row
    all_texts = [t for texts, _ in vs.upserts for t in texts]
    assert any("Reno" in t for t in all_texts)


@pytest.mark.asyncio
async def test_ingest_structured_sheets_read_failure_returns_empty(monkeypatch):
    vs, ms, reg = _FakeVectorStore(), _FakeMetadataStore(), _FakeRegistry()
    grids, clss, ingested = await ingest_structured_sheets(
        "/no/such/file.xlsx", "doc1", "x.xlsx", "xlsx", ["g1"], "cat",
        vs, ms, schema_registry=reg, generate_fn=lambda **k: "{}",
    )
    assert grids == [] and clss == [] and ingested == set()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_ingestion/test_tabular_ingest.py::test_ingest_structured_sheets_clean_ingested_messy_gets_region_narratives -v`
Expected: FAIL — `ImportError: cannot import name 'ingest_structured_sheets'`

- [ ] **Step 3: Refactor `ingest_spreadsheet_tables` and add the orchestrator**

In `src/ingestion/tabular_ingest.py`, add the new imports near the top (after the existing imports):

```python
from src.ingestion.tabular import read_sheets, classify_sheet, SheetGrid, SheetClassification
from src.ingestion.tabular_chunker import find_table_region, messy_region_narratives
```

(Keep the existing `from src.ingestion.tabular import read_sheets, classify_sheet` — merge into the line above rather than duplicating.)

Extract the per-sheet clean body of `ingest_spreadsheet_tables` into a reusable helper. Replace the body of the `for grid in grids:` loop (the `cls = classify_sheet(grid)` block through `clean_count += 1`) so the loop delegates:

```python
def _ingest_one_clean_sheet(con, grid, cls, doc_id, filename, doc_type, acl_groups,
                            category, vector_store, metadata_store, schema_registry,
                            generate_fn, chunk_index):
    """Structured-ingest ONE already-classified clean sheet: DuckDB rows +
    registered/persisted schema + embedded per-row narratives. Returns the next
    chunk_index. Raises on any failure (caller decides fail-open). This is the
    exact logic previously inlined in ``ingest_spreadsheet_tables``."""
    from src.ingestion.tabular_store import load_sheet_to_duckdb, schema_from_sheet
    _, col_names = load_sheet_to_duckdb(con, doc_id, grid.sheet_name, cls, grid)
    data_rows = grid.rows[cls.header_row_index + 1:]
    profile = profile_table(grid.sheet_name, col_names, cls.column_dtypes,
                            data_rows[:5], generate_fn=generate_fn)
    schema = schema_from_sheet(doc_id, grid.sheet_name, cls, grid, acl_groups=acl_groups)
    for col in schema.columns:
        if col.name in profile.column_descriptions:
            col.description = profile.column_descriptions[col.name]
    if profile.table_description:
        schema.description = profile.table_description
    schema_registry.register(schema)
    # save_schema is async; this helper is sync, so the async caller awaits it.
    return col_names, profile, data_rows, schema, chunk_index


async def _save_and_embed_clean(metadata_store, vector_store, schema, col_names, profile,
                                data_rows, doc_id, filename, doc_type, acl_groups, category,
                                chunk_index):
    """Persist a clean sheet's schema and embed its per-row narratives. Returns
    the next chunk_index."""
    await metadata_store.save_schema(schema)
    narratives = [n for n in build_row_narratives(
        col_names, profile, data_rows, context=profile.table_description) if n.strip()]
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
    return chunk_index
```

> NOTE: `profile_table`, `build_row_narratives`, `embed_texts`, `ChunkMetadata` are already imported at the top of `tabular_ingest.py`. Keep `ingest_spreadsheet_tables` working by having its loop call these two helpers (clean count = number of sheets that complete both without raising), OR — simpler — have `ingest_spreadsheet_tables` delegate entirely to `ingest_structured_sheets` and return `len(ingested)`. Choose delegation:

Replace `ingest_spreadsheet_tables`'s body with:

```python
async def ingest_spreadsheet_tables(file_path, doc_id, filename, doc_type, acl_groups,
                                    category, vector_store, metadata_store,
                                    schema_registry=None, generate_fn=None) -> int:
    """Backward-compatible wrapper: structured-ingest clean sheets only and
    return the count fully processed. New callers should use
    ``ingest_structured_sheets`` (which also returns grids/classifications and
    handles messy region narratives)."""
    _, _, ingested = await ingest_structured_sheets(
        file_path, doc_id, filename, doc_type, acl_groups, category,
        vector_store, metadata_store, schema_registry=schema_registry,
        generate_fn=generate_fn,
    )
    return len(ingested)
```

Now add the orchestrator:

```python
async def ingest_structured_sheets(file_path, doc_id, filename, doc_type, acl_groups,
                                   category, vector_store, metadata_store,
                                   schema_registry=None, generate_fn=None):
    """Read a spreadsheet's sheets once, structured-ingest clean sheets, and embed
    deterministic region narratives for messy sheets.

    Returns ``(grids, classifications, ingested_names)`` where ``ingested_names``
    is the set of CLEAN sheet names whose DuckDB rows + schema + per-row
    narratives ALL succeeded. The caller uses that set (via
    ``tabular_chunker.sheets_needing_text``) to decide which sheets still need
    full-text chunks. Fully fail-open: a read failure returns ([], [], set());
    a per-sheet failure is logged and the sheet is simply absent from
    ``ingested_names`` (so it falls back to text chunks).
    """
    if schema_registry is None:
        from src.api.routes_ingest import get_schema_registry
        schema_registry = get_schema_registry()

    try:
        grids = read_sheets(Path(file_path))
    except Exception as e:
        logger.warning(f"Tabular ingest: could not read sheets from {filename}: {e}")
        return [], [], set()

    classifications = [classify_sheet(g) for g in grids]
    ingested: set[str] = set()
    chunk_index = 0
    con = None
    try:
        con = connect_tabular(read_only=False)
        for grid, cls in zip(grids, classifications):
            if cls.route == "clean":
                try:
                    col_names, profile, data_rows, schema, chunk_index = _ingest_one_clean_sheet(
                        con, grid, cls, doc_id, filename, doc_type, acl_groups,
                        category, vector_store, metadata_store, schema_registry,
                        generate_fn, chunk_index)
                    chunk_index = await _save_and_embed_clean(
                        metadata_store, vector_store, schema, col_names, profile,
                        data_rows, doc_id, filename, doc_type, acl_groups, category,
                        chunk_index)
                    ingested.add(grid.sheet_name)
                except Exception as e:
                    logger.warning(
                        f"Tabular ingest: failed on clean sheet '{grid.sheet_name}' "
                        f"of {filename}: {e}")
                    continue
            else:  # messy: deterministic region narratives (no LLM, no DuckDB)
                try:
                    region = find_table_region(grid.rows)
                    if region is None:
                        continue
                    narratives = messy_region_narratives(grid, region)
                    if not narratives:
                        continue
                    vectors = embed_texts(narratives)
                    metadatas = [ChunkMetadata(
                        doc_id=doc_id, filename=filename, doc_type=doc_type,
                        chunk_index=chunk_index + i, start_char=0, acl_groups=acl_groups,
                        category=category, chunk_size_tier="table_row",
                    ) for i in range(len(narratives))]
                    chunk_index += len(narratives)
                    vector_store.upsert(texts=narratives, vectors=vectors, metadatas=metadatas)
                except Exception as e:
                    logger.warning(
                        f"Tabular ingest: region narratives failed on messy sheet "
                        f"'{grid.sheet_name}' of {filename}: {e}")
                    continue
    finally:
        if con is not None:
            con.close()

    logger.info(f"Tabular ingest: structured {len(ingested)} clean sheet(s) from {filename}")
    return grids, classifications, ingested
```

Add `Path` import if not present (`from pathlib import Path` is already at the top).

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_ingestion/test_tabular_ingest.py -v`
Expected: PASS (existing tests + 2 new). If an existing test asserted `ingest_spreadsheet_tables(...) == <int>`, it still holds (wrapper returns `len(ingested)`).

- [ ] **Step 5: Commit**

```bash
git add src/ingestion/tabular_ingest.py tests/test_ingestion/test_tabular_ingest.py
git commit -m "feat: ingest_structured_sheets — clean DuckDB ingest + messy region narratives, one read"
```

---

## Task 6: Promote `SPREADSHEET_DOC_TYPES` and remove `maybe_ingest_spreadsheet`

**Files:**
- Modify: `src/ingestion/tabular_ingest.py`

- [ ] **Step 1: Rename the constant**

Replace `_SPREADSHEET_DOC_TYPES = ("xlsx", "xls", "csv", "tsv")` with:

```python
SPREADSHEET_DOC_TYPES = ("xlsx", "xls", "csv", "tsv")
```

- [ ] **Step 2: Delete `maybe_ingest_spreadsheet`**

Remove the entire `async def maybe_ingest_spreadsheet(...)` function (it is being replaced by the spreadsheet branch in Tasks 7–8). Leave `cleanup_spreadsheet_tables` and `populate_schema_registry` untouched.

- [ ] **Step 3: Verify nothing else imports the removed name**

Run: `grep -rn "maybe_ingest_spreadsheet\|_SPREADSHEET_DOC_TYPES" src/ tests/`
Expected: only matches will be in `pipeline.py` and `queue.py` (fixed in Tasks 7–8). If any test references `maybe_ingest_spreadsheet`, update or delete that test now.

- [ ] **Step 4: Commit**

```bash
git add src/ingestion/tabular_ingest.py
git commit -m "refactor: promote SPREADSHEET_DOC_TYPES, drop maybe_ingest_spreadsheet"
```

---

## Task 7: Wire the spreadsheet branch into the sync pipeline

**Files:**
- Modify: `src/ingestion/pipeline.py:88-127`
- Test: `tests/test_ingestion/test_pipeline.py`

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_ingestion/test_pipeline.py
import pytest


@pytest.mark.asyncio
async def test_ingest_document_spreadsheet_dedups_clean_sheet(tmp_path, monkeypatch):
    """A clean spreadsheet that is structured-ingested emits NO full-text tier
    chunks (only its DuckDB rows + table_row narratives); a messy sheet still
    produces structure-aware text chunks."""
    import openpyxl
    from src.ingestion import pipeline

    p = tmp_path / "pay.xlsx"
    wb = openpyxl.Workbook(); wb.remove(wb.active)
    clean = wb.create_sheet("Pay")
    for r in [["locality", "grade", "salary"], ["Tampa", "GS-12", 86415],
              ["Boston", "GS-12", 92000], ["Denver", "GS-13", 99000]]:
        clean.append(r)
    wb.save(p)

    captured = {"tiers": []}

    class FakeVS:
        def upsert(self, texts, vectors, metadatas):
            captured["tiers"] += [m.chunk_size_tier for m in metadatas]

    class FakeMS:
        async def add_document(self, **k): pass
        async def get_category(self, name): return None
        async def add_category(self, **k): pass
        async def save_schema(self, s): pass

    # stub out everything external
    monkeypatch.setattr(pipeline, "embed_texts", lambda texts, *a, **k: [[0.0] for _ in texts])
    monkeypatch.setattr("src.ingestion.tabular_ingest.embed_texts", lambda texts, *a, **k: [[0.0] for _ in texts])
    monkeypatch.setattr("src.generation.llm_client.generate", lambda **k: "")
    monkeypatch.setattr("src.knowledge.graph_rag.insert_document",
                        _async_noop := (lambda *a, **k: _coro()))

    async def _coro(): return None
    monkeypatch.setattr("src.knowledge.graph_rag.insert_document", lambda *a, **k: _coro())
    # schema registry: in-memory fake
    monkeypatch.setattr("src.api.routes_ingest.get_schema_registry",
                        lambda: type("R", (), {"register": lambda self, s: None,
                                               "remove": lambda self, *a: None})())

    res = await pipeline.ingest_document(
        str(p), acl_groups=["g1"], uploaded_by="t", vector_store=FakeVS(),
        metadata_store=FakeMS(), category="cat",
    )
    # the clean "Pay" sheet was de-duped: no small/medium/large/xlarge text tiers,
    # only table_row narratives.
    assert "table_row" in captured["tiers"]
    assert not ({"small", "medium", "large", "xlarge"} & set(captured["tiers"]))
```

> NOTE: `_parse_spreadsheet` will still set `parsed.text` to the flattened blob; the test asserts the *chunk tiers emitted*, which is what the branch controls. If `ingest_document`'s LLM summary call or LightRAG insert need additional stubs in this repo's test harness, mirror the stubbing already used in the other tests in this file.

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_ingestion/test_pipeline.py::test_ingest_document_spreadsheet_dedups_clean_sheet -v`
Expected: FAIL — clean-sheet text tiers (small/medium/large/xlarge) are still emitted because the branch does not exist yet.

- [ ] **Step 3: Add the spreadsheet branch**

In `src/ingestion/pipeline.py`, add imports near the top:

```python
from src.ingestion.tabular_ingest import ingest_structured_sheets, SPREADSHEET_DOC_TYPES
from src.ingestion.tabular_chunker import sheets_needing_text, build_tier_chunks
```

Replace the chunking section (currently lines ~87-110, the `total_chunks = 0` line through the tier loop) with:

```python
    total_chunks = 0
    chunks = []  # medium-tier chunks, retained for the return/entity count

    is_spreadsheet = parsed.doc_type in SPREADSHEET_DOC_TYPES
    text_sheets = None
    if is_spreadsheet:
        # Structured: clean sheets -> DuckDB + schema + row narratives; messy
        # sheets -> deterministic region narratives. Returns which clean sheets
        # fully succeeded so we can de-dup their full text below.
        grids, classifications, ingested = await ingest_structured_sheets(
            file_path, doc_id, parsed.filename, parsed.doc_type,
            acl_groups, category, vector_store, metadata_store,
        )
        text_sheets = sheets_needing_text(grids, classifications, ingested)

    for tier_name, tier_size, tier_overlap in CHUNK_TIERS:
        if is_spreadsheet:
            # Structure-aware, row-atomic chunks for messy + failed-clean sheets
            # only. Clean sheets already in the structured store contribute none.
            tier_chunks = build_tier_chunks(text_sheets, chunk_size=tier_size)
        else:
            tier_chunks = chunk_text(parsed.text, chunk_size=tier_size, chunk_overlap=tier_overlap)
        texts = [f"{doc_context}\n\n{c.text}" for c in tier_chunks]
        metadatas = [
            ChunkMetadata(
                doc_id=doc_id,
                filename=parsed.filename,
                doc_type=parsed.doc_type,
                chunk_index=c.index,
                start_char=c.start_char,
                acl_groups=acl_groups,
                category=category,
                chunk_size_tier=tier_name,
            )
            for c in tier_chunks
        ]
        vectors = embed_texts(texts) if texts else []
        if vectors:
            vector_store.upsert(texts=texts, vectors=vectors, metadatas=metadatas)
        if tier_name == "medium":
            total_chunks = len(tier_chunks)
            chunks = tier_chunks
```

Then DELETE the now-obsolete block that called `maybe_ingest_spreadsheet` (the `from src.ingestion.tabular_ingest import maybe_ingest_spreadsheet` + the `await maybe_ingest_spreadsheet(...)` lines, ~120-127). The structured ingest now happens inside the `is_spreadsheet` branch above.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_ingestion/test_pipeline.py -v`
Expected: PASS (new test + existing pipeline tests unchanged — non-spreadsheet docs take the `else` branch, identical to before).

- [ ] **Step 5: Commit**

```bash
git add src/ingestion/pipeline.py tests/test_ingestion/test_pipeline.py
git commit -m "feat: sheet-aware spreadsheet chunking + conditional de-dup in sync pipeline"
```

---

## Task 8: Wire the same branch into the async queue worker

**Files:**
- Modify: `src/ingestion/queue.py:282-351`

The queue path must match the pipeline exactly (the two-paths divergence risk). It additionally emits `update_step` progress and embeds via `asyncio.to_thread` with per-tier batch sizes, and uses `chunks` for `complete_job(chunk_count=len(chunks))` at the end — so `chunks` MUST stay defined.

- [ ] **Step 1: Add imports**

Near the top of `src/ingestion/queue.py` (with the other ingestion imports):

```python
from src.ingestion.tabular_ingest import ingest_structured_sheets, SPREADSHEET_DOC_TYPES
from src.ingestion.tabular_chunker import sheets_needing_text, build_tier_chunks
```

- [ ] **Step 2: Insert the structured-ingest call before the tier loop**

Immediately before `total_chunks = 0` (line ~282), add:

```python
        chunks = []  # ensure defined even if all sheets de-dup to zero text chunks
        is_spreadsheet = parsed.doc_type in SPREADSHEET_DOC_TYPES
        text_sheets = None
        if is_spreadsheet:
            self.update_step(job.job_id, IngestStep.STORING, "Structured spreadsheet ingest (DuckDB + narratives)")
            grids, classifications, ingested = await ingest_structured_sheets(
                file_path, doc_id, parsed.filename, parsed.doc_type,
                job.acl_groups, category, vector_store, metadata_store,
            )
            text_sheets = sheets_needing_text(grids, classifications, ingested)
```

- [ ] **Step 3: Swap the chunk source inside the tier loop**

Replace the line `tier_chunks = chunk_text(parsed.text, chunk_size=tier_size, chunk_overlap=tier_overlap)` (line ~289) with:

```python
            if is_spreadsheet:
                tier_chunks = build_tier_chunks(text_sheets, chunk_size=tier_size)
            else:
                tier_chunks = chunk_text(parsed.text, chunk_size=tier_size, chunk_overlap=tier_overlap)
```

- [ ] **Step 4: Delete the obsolete `maybe_ingest_spreadsheet` block**

Remove the block at lines ~345-351 (the comment + `from src.ingestion.tabular_ingest import maybe_ingest_spreadsheet` + `await maybe_ingest_spreadsheet(...)`). Structured ingest now happens in Step 2's branch.

- [ ] **Step 5: Verify the worker still imports and `chunks` is always bound**

Run: `python -c "import src.ingestion.queue"`
Expected: no ImportError.

Run: `grep -n "chunks" src/ingestion/queue.py | sed -n '1,20p'`
Expected: confirm `chunks = []` is set before the loop and `chunks = tier_chunks` is still inside `if tier_name == "medium":`, and `complete_job(... chunk_count=len(chunks) ...)` is unchanged.

- [ ] **Step 6: Run the queue tests**

Run: `pytest tests/test_ingestion/test_queue_active.py tests/test_ingestion/test_queue_cleanup.py -v`
Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add src/ingestion/queue.py
git commit -m "feat: sheet-aware spreadsheet chunking + conditional de-dup in async queue worker"
```

---

## Task 9: Full regression + manual smoke

**Files:** none (verification only)

- [ ] **Step 1: Run the full ingestion test suite**

Run: `pytest tests/test_ingestion/ -v`
Expected: all PASS. (`tests/test_agent/` has ~8 PRE-EXISTING unrelated failures — do not chase them here; confirm they are the same set as on `master` before this branch with `git stash` + rerun if unsure.)

- [ ] **Step 2: Manual smoke — messy sheet captures all rows, clean sheet de-dups**

Run:

```bash
python - <<'PY'
import asyncio
from pathlib import Path
import openpyxl
from src.ingestion.tabular import read_sheets, classify_sheet
from src.ingestion.tabular_chunker import find_table_region, messy_region_narratives, structure_aware_chunks

p = Path("/tmp/smoke.xlsx")
wb = openpyxl.Workbook(); wb.remove(wb.active)
m = wb.create_sheet("Messy")
for r in [["2026 Notes"], ["locality","grade","salary"],
          ["Reno","GS-9",55000],["Mesa","GS-9",56000],["Ames","GS-9",57000],
          ["effective Jan 1"]]:
    m.append(r)
wb.save(p)

grids = read_sheets(p)
g = grids[0]
cls = classify_sheet(g)
print("route:", cls.route)
region = find_table_region(g.rows)
print("region:", region)
print("narratives:", messy_region_narratives(g, region))
chunks = structure_aware_chunks(g.sheet_name, g.rows, cls.header_row_index if cls.route=="clean" else -1, chunk_size=2048)
print("chunk count:", len(chunks))
# assert every data cell value is present somewhere in the chunk text (no data loss)
blob = "\n".join(c.text for c in chunks)
for val in ["Reno","Mesa","Ames","55000","56000","57000","effective Jan 1"]:
    assert val in blob, f"MISSING: {val}"
print("ALL CELL VALUES PRESENT — no data loss")
PY
```

Expected output: `route: messy`, a non-None region, three narratives mentioning Reno/Mesa/Ames, and `ALL CELL VALUES PRESENT — no data loss`.

- [ ] **Step 3: Update the roadmap memory**

Edit `/home/mike/.claude/projects/-home-mike-sauron/memory/tabular-spreadsheet-ingestion-roadmap.md`: mark Plan 4 DONE (structure-aware chunking + conditional clean-sheet de-dup + deterministic messy region narratives), and note the deferred items still open (LLM restate narratives, multi-region detection, LightRAG still gets the full pipe blob).

- [ ] **Step 4: Commit any remaining changes**

```bash
git add -A && git commit -m "docs: mark tabular Plan 4 done; note deferred follow-ups"
```

---

## Self-Review (completed during authoring)

**Spec coverage (Component 3 of the design spec):**
- "Structure-aware chunking: repeat the detected header atop each chunk, never split mid-row, mark sheet boundaries" → Task 1 (`structure_aware_chunks`), Task 4 (`build_tier_chunks` across sheets).
- "Optional restate-only row-GROUP narratives for table-like regions" → realized as deterministic region narratives, Tasks 2–3 + wired in Task 5. (LLM variant explicitly deferred per locked decision.)
- "Answered via the existing sweep + map-reduce path" → chunks land in the same small/medium/large/xlarge tiers those strategies query (Tasks 7–8); verified against the tier-filter audit in retrieval.
- De-dup of redundant clean-sheet full text (roadmap item) → conditional suppression, Task 4 (`sheets_needing_text`) + Tasks 7–8, gated on `ingested` set from Task 5.

**Placeholder scan:** no TBD/TODO/"handle edge cases"; every code step shows complete code; commands have expected output.

**Type consistency:** `structure_aware_chunks` returns `list[Chunk]`; `build_tier_chunks` consumes `[(SheetGrid, SheetClassification)]` and returns `list[Chunk]`; `sheets_needing_text` returns the same pair-list it feeds; `find_table_region` returns `tuple[int,int] | None` consumed by `messy_region_narratives`; `ingest_structured_sheets` returns `(grids, classifications, set[str])` consumed by `sheets_needing_text` in both Tasks 7 and 8. `chunk_size_tier` values used: existing `small/medium/large/xlarge` (text) + `table_row` (narratives) — no new tier invented (would be invisible to retrieval).

**Fail-open audit:** read failure → `([], [], set())` → `sheets_needing_text([], [], set())` → `[]` → zero chunks, document still ingested. Per-sheet clean failure → sheet absent from `ingested` → falls into `sheets_needing_text` → full-text fallback. Per-sheet messy-narrative failure → logged, skipped; structure-aware text chunks for that sheet still emitted by the tier loop.
