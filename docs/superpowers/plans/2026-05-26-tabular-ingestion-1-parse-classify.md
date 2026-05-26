# Tabular Ingestion — Plan 1: Structured Parse + Clean/Messy Classification

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Parse spreadsheets into structured per-sheet grids and classify each sheet as a clean data table or messy/narrative — the routing foundation the later tabular-ingestion plans build on.

**Architecture:** A new self-contained module `src/ingestion/tabular.py` with pure functions over in-memory grids (header detection, column dtype inference, clean/messy classification) plus a thin file reader that reuses the existing magic-byte sniff. Nothing in the existing parser/chunker/pipeline is modified — this plan only adds new, independently testable code.

**Tech Stack:** Python 3.11, openpyxl, xlrd, csv, pytest. Tests run inside the app image (host lacks some deps): `docker run --rm -v "$(pwd)":/app -w /app sauron-api python -m pytest <path> -q`.

**Spec:** `docs/superpowers/specs/2026-05-25-tabular-spreadsheet-ingestion-design.md` (Component 1).

**Why isolated:** This is Plan 1 of 4. Later plans consume `classify_sheet`/`read_sheets`: Plan 2 (clean → DuckDB + schema + row narratives), Plan 3 (query routing), Plan 4 (messy → structure-aware chunking). Keeping Plan 1 free of pipeline changes means it can merge without altering live ingestion behavior.

---

## File Structure

- `src/ingestion/tabular.py` — **create**. Owns: `SheetGrid` (one sheet's raw rows), `read_sheets(path)` (file → grids, reusing `parser._sniff_workbook_format`), `_cell_kind`, `detect_header_row`, `infer_column_dtypes`, `SheetClassification`, `classify_sheet`, `analyze_spreadsheet`. Single responsibility: turn a spreadsheet file into per-sheet structure + a routing decision. No DB, no embeddings, no LLM.
- `tests/test_ingestion/test_tabular.py` — **create**. Unit tests for every function; pure-function tests use in-memory grids, the reader test builds tiny files in `tmp_path`.

## Module constants (defined in Task 1, used throughout)

```python
MAX_HEADER_SCAN = 10        # rows to scan when looking for the header
HEADER_MIN_FILLED = 0.6     # >= this fraction of header cells must be non-empty
HEADER_MAX_NUMERIC = 0.3    # <= this fraction of header cells may be numeric
MIN_HEADER_CELLS = 2        # a header needs >= this many non-empty cells (skips single-cell title banners)
MIN_DATA_ROWS = 3           # a clean table needs at least this many data rows
RECTANGULAR_RATIO = 0.8     # >= this fraction of data rows must match header width
COLUMN_CONSISTENCY = 0.8    # >= this fraction of a column's non-empty cells share one kind
```

---

### Task 1: `SheetGrid` + `read_sheets` (file → per-sheet raw grids)

**Files:**
- Create: `src/ingestion/tabular.py`
- Test: `tests/test_ingestion/test_tabular.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_ingestion/test_tabular.py`:

```python
"""Tests for src/ingestion/tabular.py — structured parse + clean/messy classification."""
from pathlib import Path

import openpyxl
import pytest

from src.ingestion.tabular import SheetGrid, read_sheets


def _write_xlsx(path: Path, sheets: dict[str, list[list]]):
    wb = openpyxl.Workbook()
    wb.remove(wb.active)
    for name, rows in sheets.items():
        ws = wb.create_sheet(title=name)
        for row in rows:
            ws.append(row)
    wb.save(path)


def test_read_sheets_xlsx_returns_one_grid_per_sheet(tmp_path):
    p = tmp_path / "book.xlsx"
    _write_xlsx(p, {
        "Pay": [["grade", "step", "salary"], ["GS-12", 5, 86415]],
        "Notes": [["just a note"]],
    })
    grids = read_sheets(p)
    assert [g.sheet_name for g in grids] == ["Pay", "Notes"]
    assert grids[0].rows[0] == ["grade", "step", "salary"]
    assert grids[0].rows[1] == ["GS-12", 5, 86415]


def test_read_sheets_csv_returns_single_grid(tmp_path):
    p = tmp_path / "data.csv"
    p.write_text("grade,step,salary\nGS-12,5,86415\n", encoding="utf-8")
    grids = read_sheets(p)
    assert len(grids) == 1
    assert grids[0].sheet_name == "data"
    assert grids[0].rows[0] == ["grade", "step", "salary"]
    assert grids[0].rows[1] == ["GS-12", "5", "86415"]  # csv cells are strings
```

- [ ] **Step 2: Run test to verify it fails**

Run: `docker run --rm -v "$(pwd)":/app -w /app sauron-api python -m pytest tests/test_ingestion/test_tabular.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.ingestion.tabular'`.

- [ ] **Step 3: Create the module with constants, `SheetGrid`, and `read_sheets`**

Create `src/ingestion/tabular.py`:

```python
"""Structured parse + clean/messy classification for spreadsheets.

Turns a spreadsheet file into per-sheet grids (lists of cell rows) and decides
whether each sheet is a clean data table (route to structured/SQL handling) or
messy/narrative (route to text RAG). Pure functions operate on in-memory grids
so they are trivially testable; ``read_sheets`` is the only file-touching entry.
"""
from __future__ import annotations

import csv
import io
from dataclasses import dataclass, field
from pathlib import Path

from src.ingestion.parser import _sniff_workbook_format

MAX_HEADER_SCAN = 10
HEADER_MIN_FILLED = 0.6
HEADER_MAX_NUMERIC = 0.3
MIN_HEADER_CELLS = 2
MIN_DATA_ROWS = 3
RECTANGULAR_RATIO = 0.8
COLUMN_CONSISTENCY = 0.8


@dataclass
class SheetGrid:
    """One sheet's raw rows, exactly as read (no header interpretation yet)."""
    sheet_name: str
    rows: list[list] = field(default_factory=list)


def read_sheets(path: Path) -> list[SheetGrid]:
    """Read a spreadsheet/CSV into one ``SheetGrid`` per sheet.

    Dispatches on magic bytes (not extension) so a mislabeled workbook still
    parses with the right reader. .xlsx/.xlsm via openpyxl, legacy .xls via
    xlrd, everything else as delimited text.
    """
    fmt = _sniff_workbook_format(path)
    if fmt == "ooxml":
        return _read_ooxml(path)
    if fmt == "ole":
        return _read_ole(path)
    return _read_delimited(path)


def _read_ooxml(path: Path) -> list[SheetGrid]:
    from openpyxl import load_workbook

    with open(path, "rb") as fh:
        wb = load_workbook(io.BytesIO(fh.read()), read_only=True, data_only=True)
    grids = []
    for name in wb.sheetnames:
        ws = wb[name]
        rows = [list(r) for r in ws.iter_rows(values_only=True)]
        grids.append(SheetGrid(sheet_name=name, rows=rows))
    wb.close()
    return grids


def _read_ole(path: Path) -> list[SheetGrid]:
    import xlrd

    book = xlrd.open_workbook(str(path))
    grids = []
    for sheet in book.sheets():
        rows = [list(sheet.row_values(r)) for r in range(sheet.nrows)]
        grids.append(SheetGrid(sheet_name=sheet.name, rows=rows))
    return grids


def _read_delimited(path: Path) -> list[SheetGrid]:
    if b"\x00" in path.read_bytes()[:8192]:
        raise ValueError(f"{path.name}: appears to be binary, not delimited text; refusing to read")
    delimiter = "\t" if path.suffix.lower() == ".tsv" else ","
    with open(path, newline="", encoding="utf-8", errors="replace") as f:
        rows = [list(r) for r in csv.reader(f, delimiter=delimiter)]
    return [SheetGrid(sheet_name=path.stem, rows=rows)]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `docker run --rm -v "$(pwd)":/app -w /app sauron-api python -m pytest tests/test_ingestion/test_tabular.py -q`
Expected: PASS (2 passed).

- [ ] **Step 5: Commit**

```bash
git add src/ingestion/tabular.py tests/test_ingestion/test_tabular.py
git commit -m "feat: read spreadsheets into per-sheet structured grids

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task 2: `_cell_kind` (classify a single cell)

**Files:**
- Modify: `src/ingestion/tabular.py`
- Test: `tests/test_ingestion/test_tabular.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_ingestion/test_tabular.py` (add `_cell_kind` to the import from `src.ingestion.tabular`):

```python
from src.ingestion.tabular import _cell_kind


@pytest.mark.parametrize("value,expected", [
    (None, "empty"),
    ("", "empty"),
    ("   ", "empty"),
    (5, "number"),
    (3.14, "number"),
    ("86415", "number"),
    ("$86,415", "number"),
    ("12%", "number"),
    ("GS-12", "text"),
    ("grade", "text"),
])
def test_cell_kind(value, expected):
    assert _cell_kind(value) == expected
```

- [ ] **Step 2: Run test to verify it fails**

Run: `docker run --rm -v "$(pwd)":/app -w /app sauron-api python -m pytest tests/test_ingestion/test_tabular.py::test_cell_kind -q`
Expected: FAIL — `ImportError: cannot import name '_cell_kind'`.

- [ ] **Step 3: Implement `_cell_kind`**

Add to `src/ingestion/tabular.py` (after the constants, before `SheetGrid`):

```python
def _cell_kind(value) -> str:
    """Classify a cell as 'empty', 'number', or 'text'.

    Numbers include native int/float and numeric-looking strings with common
    formatting ($, commas, %, surrounding whitespace).
    """
    if value is None:
        return "empty"
    if isinstance(value, bool):
        return "text"  # bools are not measures here
    if isinstance(value, (int, float)):
        return "number"
    s = str(value).strip()
    if not s:
        return "empty"
    cleaned = s.replace("$", "").replace(",", "").replace("%", "").strip()
    try:
        float(cleaned)
        return "number"
    except ValueError:
        return "text"
```

- [ ] **Step 4: Run test to verify it passes**

Run: `docker run --rm -v "$(pwd)":/app -w /app sauron-api python -m pytest tests/test_ingestion/test_tabular.py::test_cell_kind -q`
Expected: PASS (10 passed).

- [ ] **Step 5: Commit**

```bash
git add src/ingestion/tabular.py tests/test_ingestion/test_tabular.py
git commit -m "feat: classify spreadsheet cells as empty/number/text

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task 3: `detect_header_row`

**Files:**
- Modify: `src/ingestion/tabular.py`
- Test: `tests/test_ingestion/test_tabular.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_ingestion/test_tabular.py` (add `detect_header_row` to the import):

```python
from src.ingestion.tabular import detect_header_row


def test_header_is_first_row_when_clean():
    rows = [["grade", "step", "salary"], ["GS-12", 5, 86415], ["GS-13", 5, 102000]]
    assert detect_header_row(rows) == 0


def test_header_after_title_and_blank_rows():
    rows = [
        ["2024 General Schedule"],     # title banner
        [None, None, None],            # blank
        ["grade", "step", "salary"],   # real header at index 2
        ["GS-12", 5, 86415],
    ]
    assert detect_header_row(rows) == 2


def test_no_header_returns_negative_one():
    # All-numeric grid with no labels => no detectable header.
    rows = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    assert detect_header_row(rows) == -1


def test_empty_grid_returns_negative_one():
    assert detect_header_row([]) == -1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `docker run --rm -v "$(pwd)":/app -w /app sauron-api python -m pytest tests/test_ingestion/test_tabular.py -k header -q`
Expected: FAIL — `ImportError: cannot import name 'detect_header_row'`.

- [ ] **Step 3: Implement `detect_header_row`**

Add to `src/ingestion/tabular.py`:

```python
def detect_header_row(rows: list[list]) -> int:
    """Index of the row that looks like column headers, or -1 if none.

    A header row has >= MIN_HEADER_CELLS non-empty cells (so a single-cell title
    banner is skipped), is mostly-filled, mostly-non-numeric, and is followed by
    at least one more row (the data). Scans only the first ``MAX_HEADER_SCAN``
    rows so a giant sheet stays cheap.
    """
    for i, row in enumerate(rows[:MAX_HEADER_SCAN]):
        if i + 1 >= len(rows):
            break  # a header needs at least one data row beneath it
        kinds = [_cell_kind(c) for c in row]
        if not kinds:
            continue
        filled = [k for k in kinds if k != "empty"]
        if len(filled) < MIN_HEADER_CELLS:
            continue  # single-cell title banner or near-empty row, not a header
        filled_ratio = len(filled) / len(kinds)
        numeric_ratio = sum(1 for k in filled if k == "number") / len(filled)
        if filled_ratio >= HEADER_MIN_FILLED and numeric_ratio <= HEADER_MAX_NUMERIC:
            return i
    return -1
```

- [ ] **Step 4: Run test to verify it passes**

Run: `docker run --rm -v "$(pwd)":/app -w /app sauron-api python -m pytest tests/test_ingestion/test_tabular.py -k header -q`
Expected: PASS (4 passed).

- [ ] **Step 5: Commit**

```bash
git add src/ingestion/tabular.py tests/test_ingestion/test_tabular.py
git commit -m "feat: detect the header row in a spreadsheet grid

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task 4: `infer_column_dtypes`

**Files:**
- Modify: `src/ingestion/tabular.py`
- Test: `tests/test_ingestion/test_tabular.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_ingestion/test_tabular.py` (add `infer_column_dtypes` to the import):

```python
from src.ingestion.tabular import infer_column_dtypes


def test_infer_dtypes_per_column():
    rows = [
        ["grade", "step", "salary"],
        ["GS-12", 5, 86415],
        ["GS-13", 5, 102000],
        ["GS-14", 6, 120000],
    ]
    assert infer_column_dtypes(rows, header_row_index=0) == ["text", "number", "number"]


def test_infer_dtypes_dominant_kind_wins_with_one_outlier():
    rows = [
        ["amount"],
        [100],
        [200],
        ["N/A"],   # one text outlier in a numeric column
        [300],
    ]
    assert infer_column_dtypes(rows, header_row_index=0) == ["number"]


def test_infer_dtypes_all_empty_column_is_empty():
    rows = [["a", "b"], [1, None], [2, None]]
    assert infer_column_dtypes(rows, header_row_index=0) == ["number", "empty"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `docker run --rm -v "$(pwd)":/app -w /app sauron-api python -m pytest tests/test_ingestion/test_tabular.py -k dtypes -q`
Expected: FAIL — `ImportError: cannot import name 'infer_column_dtypes'`.

- [ ] **Step 3: Implement `infer_column_dtypes`**

Add to `src/ingestion/tabular.py`:

```python
def infer_column_dtypes(rows: list[list], header_row_index: int) -> list[str]:
    """Dominant kind ('number'/'text'/'empty') of each column's DATA cells.

    Looks only at rows below the header. A column with no non-empty data cells
    is 'empty'; otherwise the most common non-empty kind wins (ties -> 'text').
    Column count is taken from the header row.
    """
    header = rows[header_row_index] if 0 <= header_row_index < len(rows) else []
    ncols = len(header)
    data = rows[header_row_index + 1:]
    dtypes = []
    for col in range(ncols):
        counts = {"number": 0, "text": 0}
        for row in data:
            if col >= len(row):
                continue
            kind = _cell_kind(row[col])
            if kind == "empty":
                continue
            counts[kind] += 1
        if counts["number"] == 0 and counts["text"] == 0:
            dtypes.append("empty")
        elif counts["number"] > counts["text"]:
            dtypes.append("number")
        else:
            dtypes.append("text")
    return dtypes
```

- [ ] **Step 4: Run test to verify it passes**

Run: `docker run --rm -v "$(pwd)":/app -w /app sauron-api python -m pytest tests/test_ingestion/test_tabular.py -k dtypes -q`
Expected: PASS (3 passed).

- [ ] **Step 5: Commit**

```bash
git add src/ingestion/tabular.py tests/test_ingestion/test_tabular.py
git commit -m "feat: infer per-column data types from a grid

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task 5: `SheetClassification` + `classify_sheet`

**Files:**
- Modify: `src/ingestion/tabular.py`
- Test: `tests/test_ingestion/test_tabular.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_ingestion/test_tabular.py` (add `SheetClassification` and `classify_sheet` to the import):

```python
from src.ingestion.tabular import SheetClassification, classify_sheet


def test_clean_table_is_classified_clean():
    rows = [["grade", "step", "salary"]] + [[f"GS-{g}", 5, 80000 + g] for g in range(10, 20)]
    result = classify_sheet(SheetGrid("Pay", rows))
    assert isinstance(result, SheetClassification)
    assert result.route == "clean"
    assert result.header_row_index == 0
    assert result.column_dtypes == ["text", "number", "number"]


def test_too_few_rows_is_messy():
    rows = [["grade", "salary"], ["GS-12", 86415]]  # only 1 data row (< MIN_DATA_ROWS)
    result = classify_sheet(SheetGrid("Tiny", rows))
    assert result.route == "messy"


def test_no_header_is_messy():
    rows = [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]]
    result = classify_sheet(SheetGrid("Raw", rows))
    assert result.route == "messy"
    assert result.header_row_index == -1


def test_ragged_non_rectangular_is_messy():
    # Wildly varying row widths => not a rectangular table.
    rows = [
        ["a", "b", "c"],
        ["note spanning"],
        ["x", "y"],
        ["only one"],
        ["p", "q", "r", "s", "t"],
    ]
    result = classify_sheet(SheetGrid("Messy", rows))
    assert result.route == "messy"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `docker run --rm -v "$(pwd)":/app -w /app sauron-api python -m pytest tests/test_ingestion/test_tabular.py -k classify -q`
Expected: FAIL — `ImportError: cannot import name 'SheetClassification'`.

- [ ] **Step 3: Implement `SheetClassification` and `classify_sheet`**

Add to `src/ingestion/tabular.py` (the dataclass near `SheetGrid`, the function after `infer_column_dtypes`):

```python
@dataclass
class SheetClassification:
    """Routing decision for one sheet."""
    sheet_name: str
    route: str               # "clean" or "messy"
    header_row_index: int    # -1 if no header detected
    column_dtypes: list[str] = field(default_factory=list)
    reason: str = ""


def classify_sheet(grid: SheetGrid) -> SheetClassification:
    """Decide whether a sheet is a clean data table or messy/narrative.

    Clean requires ALL of: a detected header, >= MIN_DATA_ROWS data rows, a
    rectangular shape (>= RECTANGULAR_RATIO of data rows match header width),
    and every non-empty column being type-consistent (>= COLUMN_CONSISTENCY of
    its non-empty cells share one kind). Anything else is messy.
    """
    rows = grid.rows
    header_idx = detect_header_row(rows)
    if header_idx < 0:
        return SheetClassification(grid.sheet_name, "messy", -1, [], "no header detected")

    header = rows[header_idx]
    ncols = len(header)
    data = rows[header_idx + 1:]
    if len(data) < MIN_DATA_ROWS:
        return SheetClassification(grid.sheet_name, "messy", header_idx, [], "too few data rows")

    matching = sum(1 for r in data if len(r) == ncols)
    if matching / len(data) < RECTANGULAR_RATIO:
        return SheetClassification(grid.sheet_name, "messy", header_idx, [], "not rectangular")

    dtypes = infer_column_dtypes(rows, header_idx)
    for col in range(ncols):
        counts = {"number": 0, "text": 0}
        for r in data:
            if col >= len(r):
                continue
            kind = _cell_kind(r[col])
            if kind != "empty":
                counts[kind] += 1
        total = counts["number"] + counts["text"]
        if total == 0:
            continue  # empty column doesn't disqualify
        dominant = max(counts["number"], counts["text"])
        if dominant / total < COLUMN_CONSISTENCY:
            return SheetClassification(grid.sheet_name, "messy", header_idx, dtypes,
                                       f"column {col} not type-consistent")

    return SheetClassification(grid.sheet_name, "clean", header_idx, dtypes, "clean table")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `docker run --rm -v "$(pwd)":/app -w /app sauron-api python -m pytest tests/test_ingestion/test_tabular.py -k classify -q`
Expected: PASS (4 passed).

- [ ] **Step 5: Commit**

```bash
git add src/ingestion/tabular.py tests/test_ingestion/test_tabular.py
git commit -m "feat: classify a sheet as clean table or messy

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task 6: `analyze_spreadsheet` (file → per-sheet classifications)

**Files:**
- Modify: `src/ingestion/tabular.py`
- Test: `tests/test_ingestion/test_tabular.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_ingestion/test_tabular.py` (add `analyze_spreadsheet` to the import; reuses the `_write_xlsx` helper from Task 1):

```python
from src.ingestion.tabular import analyze_spreadsheet


def test_analyze_spreadsheet_routes_each_sheet(tmp_path):
    p = tmp_path / "mixed.xlsx"
    _write_xlsx(p, {
        "Pay": [["grade", "step", "salary"]] + [[f"GS-{g}", 5, 80000 + g] for g in range(10, 20)],
        "Readme": [["This workbook contains GS pay tables."], ["Updated 2024."]],
    })
    results = analyze_spreadsheet(p)
    by_name = {r.sheet_name: r for r in results}
    assert by_name["Pay"].route == "clean"
    assert by_name["Readme"].route == "messy"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `docker run --rm -v "$(pwd)":/app -w /app sauron-api python -m pytest tests/test_ingestion/test_tabular.py -k analyze -q`
Expected: FAIL — `ImportError: cannot import name 'analyze_spreadsheet'`.

- [ ] **Step 3: Implement `analyze_spreadsheet`**

Add to `src/ingestion/tabular.py` (at the end):

```python
def analyze_spreadsheet(path: Path) -> list[SheetClassification]:
    """Read a spreadsheet file and classify every sheet."""
    return [classify_sheet(grid) for grid in read_sheets(path)]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `docker run --rm -v "$(pwd)":/app -w /app sauron-api python -m pytest tests/test_ingestion/test_tabular.py -k analyze -q`
Expected: PASS (1 passed).

- [ ] **Step 5: Commit**

```bash
git add src/ingestion/tabular.py tests/test_ingestion/test_tabular.py
git commit -m "feat: analyze a spreadsheet file into per-sheet classifications

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Final verification

- [ ] Run the whole new test file:

```bash
docker run --rm -v "$(pwd)":/app -w /app sauron-api python -m pytest tests/test_ingestion/test_tabular.py -q
```
Expected: all PASS (24 tests: 2 + 10 + 4 + 3 + 4 + 1).

- [ ] Confirm nothing else changed (this plan adds only new files):

```bash
git diff --stat <first-commit-of-this-plan>^..HEAD
```
Expected: only `src/ingestion/tabular.py` and `tests/test_ingestion/test_tabular.py`.

## Notes for the implementer

- **`read_sheets` reuses `parser._sniff_workbook_format`** so detection logic stays in one place. Importing it pulls in `parser`'s module-level `openpyxl`/`docx` imports, which exist in the app image — that's why tests run in the container.
- **`data_only=True`** in `_read_ooxml` makes openpyxl return computed cell values rather than formula strings — important for pay tables with formulas.
- **This plan deliberately does NOT touch** `parser.py`, `chunker.py`, `pipeline.py`, or any retrieval code. The classification output is consumed by Plan 2 (clean-table → DuckDB + schema + narratives) and Plan 4 (messy → structure-aware chunking). Do not wire it into the pipeline here.
- **Header detection is intentionally simple** (first plausible row in the first 10). Multi-header / merged-cell handling is out of scope; such sheets fall to "messy", which is the safe route.
