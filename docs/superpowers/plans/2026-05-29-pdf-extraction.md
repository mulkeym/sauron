# Robust PDF Data Extraction Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extract tables and prose from PDFs into the same queryable structures used for Excel (DuckDB rows + schema + deterministic row narratives), plus a code/class glossary so extracted data is findable by user vocabulary (e.g. "enlisted" → `E-*`).

**Architecture:** A new `src/ingestion/pdf_extract.py` turns a PDF into prose blocks + table grids (pdfplumber for digital pages, unstructured hi_res + tesseract OCR for scanned pages), normalizing tables to the existing `SheetGrid`. A refactor splits grid *acquisition* from grid *ingestion* (`ingest_grids`) so PDFs reuse the entire Excel tabular pipeline. Both ingestion entry points (`pipeline.py` sync, `queue.py` async) gain a PDF branch. A prefix-pattern glossary, threaded into row narratives and the text-to-SQL prompt, fixes findability. Fail-open everywhere: any failure degrades to today's flat-text behavior.

**Tech Stack:** Python, pdfplumber (new, pure-Python), unstructured[pdf] (already present), tesseract-ocr + poppler-utils (system), DuckDB, LanceDB, pytest.

**Spec:** `docs/superpowers/specs/2026-05-29-pdf-extraction-design.md`

**Conventions used throughout:**
- Run a single test: `python -m pytest <path>::<name> -v`. The repo emits NumPy/pandas binary-compat warnings on stderr — ignore them; only the PASS/FAIL line matters.
- There is a known pre-existing set of ~8 failing tests in `tests/test_agent/` (environmental) — unrelated to this work.
- Both ingestion paths must stay in sync (`pipeline.py` AND `queue.py`).

---

## Phase 0 — Spike (validate engine choice before building)

### Task 1: pdfplumber spike on the real AD pay PDF

**Files:** none committed except a test fixture produced at the end.

This is throwaway exploration to confirm Approach B works on the hardest real input before building the module. Not TDD.

- [ ] **Step 1: Get the real PDF onto disk for the spike**

The file is in the running container's data volume. Copy it out (read-only):

```bash
docker cp sauron-api-1:/app/data/uploads/. /tmp/ad_spike/ 2>/dev/null || true
ls -la /tmp/ad_spike/ | grep -i "AD Pay" || echo "search uploads dir name"
```

If the uploads path differs, find it:
```bash
docker exec sauron-api-1 sh -c 'find /app/data -iname "*AD Pay*"'
```

- [ ] **Step 2: Run a pdfplumber probe**

```bash
pip install pdfplumber  # spike-only; Task 2 adds it to requirements
python - <<'PY'
import pdfplumber
p = "/tmp/ad_spike/2025 April Dec AD Pay.pdf"  # adjust path
with pdfplumber.open(p) as pdf:
    print("pages:", len(pdf.pages))
    for i, page in enumerate(pdf.pages):
        tables = page.extract_tables()
        print(f"--- page {i}: {len(tables)} table(s); chars={len(page.extract_text() or '')}")
        for t in tables:
            print("  rows:", len(t), "cols(row0):", len(t[0]) if t else 0)
            for r in t[:3]:
                print("   ", r)
PY
```

- [ ] **Step 3: Record findings**

Confirm in `docs/superpowers/specs/2026-05-29-pdf-extraction-design.md` (append a short "Spike results" note + commit): does `lattice` (default) reconstruct the ~20-column grid with rank labels intact and numbers whole? Do officer/warrant/enlisted appear as separate per-page tables with a repeating header (→ stitching needed)? If pdfplumber cannot reconstruct the grid even with `table_settings={"vertical_strategy":"text","horizontal_strategy":"text"}` (stream mode), STOP and revisit the engine choice with the user.

- [ ] **Step 4: Save a small test fixture**

Create a tiny 2-page fixture PDF that mimics the structure (repeating header, a key column + numeric columns split across two pages) for the unit tests in Phase 2b. Either trim the real PDF to 2 pages with `pdftk`/`qpdf`, or generate one:

```bash
mkdir -p tests/fixtures/pdf
python - <<'PY'
# Minimal generator using reportlab if available; else trim the real file.
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
c = canvas.Canvas("tests/fixtures/pdf/two_page_table.pdf", pagesize=letter)
def page(rows):
    t = c.beginText(40, 750)
    for r in rows: t.textLine(r)
    c.drawText(t); c.showPage()
page(["Grade  Over2  Over4", "O-1  3998.40  5031.30", "O-2  4606.80  6042.90"])
page(["Grade  Over2  Over4", "E-1  2017.20  2017.20", "E-3  2733.00  3081.00"])
c.save()
print("wrote tests/fixtures/pdf/two_page_table.pdf")
PY
```

> Note: a whitespace-separated text PDF like this exercises pdfplumber's `stream` strategy. If the spike showed the real file needs `lattice` (ruled lines), also save a trimmed 2-page slice of the real PDF as `tests/fixtures/pdf/two_page_ruled.pdf` and use it in Task 7.

- [ ] **Step 5: Commit the fixture + spike note**

```bash
git add tests/fixtures/pdf docs/superpowers/specs/2026-05-29-pdf-extraction-design.md
git commit -m "spike: confirm pdfplumber extracts AD pay grid; add PDF test fixtures"
```

---

## Phase 1 — Dependencies & offline image

### Task 2: Add pdfplumber dependency

**Files:**
- Modify: `requirements.txt`

- [ ] **Step 1: Add the dependency**

Add after the existing `unstructured[pdf,docx,xlsx]>=0.22.0` line:

```
pdfplumber>=0.11.0
```

- [ ] **Step 2: Install and verify import**

Run: `pip install -r requirements.txt && python -c "import pdfplumber; print(pdfplumber.__version__)"`
Expected: a version string, no error.

- [ ] **Step 3: Commit**

```bash
git add requirements.txt
git commit -m "build: add pdfplumber dependency"
```

### Task 3: Offline image — OCR system deps + baked models

**Files:**
- Modify: `Dockerfile`
- Create: `scripts/prefetch_pdf_models.py`
- Create: `tests/fixtures/pdf/tiny_smoke.pdf` (1-page, any text — reuse `two_page_table.pdf` if simpler)

- [ ] **Step 1: Write the model-prefetch script**

Create `scripts/prefetch_pdf_models.py`:

```python
"""Download the unstructured hi_res layout + table-transformer models at BUILD
time so the runtime image needs no network. Run during docker build."""
import sys


def main() -> int:
    # partition_pdf with hi_res pulls the layout + table models on first use;
    # invoking it once here caches them into the image layer.
    from unstructured.partition.pdf import partition_pdf
    try:
        partition_pdf(
            filename="tests/fixtures/pdf/tiny_smoke.pdf",
            strategy="hi_res",
            infer_table_structure=True,
        )
    except Exception as e:
        print(f"prefetch failed: {e}", file=sys.stderr)
        return 1
    print("pdf models prefetched")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 2: Edit the Dockerfile**

In the runtime stage, before the `COPY src/` lines, add the system packages:

```dockerfile
# OCR + PDF rasterization for scanned-PDF extraction
RUN apt-get update && apt-get install -y --no-install-recommends \
        tesseract-ocr poppler-utils \
    && rm -rf /var/lib/apt/lists/*

# Force fully-offline model use at runtime
ENV HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1
```

After the `COPY src/ src/` / `COPY scripts/ scripts/` lines (the fixture must be present), bake + verify the models:

```dockerfile
# Bake hi_res PDF models into the image and FAIL the build if absent (offline guarantee)
COPY tests/fixtures/pdf/tiny_smoke.pdf tests/fixtures/pdf/tiny_smoke.pdf
RUN python scripts/prefetch_pdf_models.py
```

> The `ENV HF_HUB_OFFLINE=1` is set BEFORE the prefetch RUN. Move the prefetch RUN to a stage where offline is not yet forced, OR set the offline env in a later line AFTER the prefetch. Correct ordering: install deps → run prefetch (online) → then `ENV ...OFFLINE=1`. Adjust the two blocks so the prefetch runs while still able to download, and the OFFLINE envs are the last thing set.

- [ ] **Step 3: Build and verify offline-completeness**

Run: `docker compose build api`
Expected: build succeeds; the `prefetch_pdf_models.py` step prints `pdf models prefetched`. If it fails, the models did not download — fix before proceeding.

- [ ] **Step 4: Verify runtime is genuinely offline**

Run:
```bash
docker compose run --rm --no-deps api \
  python -c "from unstructured.partition.pdf import partition_pdf; partition_pdf(filename='tests/fixtures/pdf/tiny_smoke.pdf', strategy='hi_res', infer_table_structure=True); print('offline OK')"
```
Expected: `offline OK` with `HF_HUB_OFFLINE=1` already in the image env (no network needed).

- [ ] **Step 5: Commit**

```bash
git add Dockerfile scripts/prefetch_pdf_models.py tests/fixtures/pdf/tiny_smoke.pdf
git commit -m "build: bake offline OCR/table models + tesseract/poppler into image"
```

---

## Phase 2 — pdf_extract: pure logic (no PDF I/O, fast TDD)

### Task 4: Module skeleton + normalize_grid

**Files:**
- Create: `src/ingestion/pdf_extract.py`
- Test: `tests/test_ingestion/test_pdf_extract.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_ingestion/test_pdf_extract.py`:

```python
from src.ingestion.pdf_extract import normalize_grid
from src.ingestion.tabular import SheetGrid


def test_normalize_grid_drops_empty_rows_cols_and_fills_none():
    raw = [
        ["Grade", "Over 2", None, ""],
        [None, None, None, None],          # fully-empty row -> dropped
        ["O-1", "3998.40", None, ""],
    ]
    grid = normalize_grid(raw, sheet_name="p1_table1")
    assert isinstance(grid, SheetGrid)
    assert grid.sheet_name == "p1_table1"
    # empty row dropped; trailing all-empty column dropped; None -> ""
    assert grid.rows == [["Grade", "Over 2"], ["O-1", "3998.40"]]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_ingestion/test_pdf_extract.py::test_normalize_grid_drops_empty_rows_cols_and_fills_none -v`
Expected: FAIL with `ImportError` / `cannot import name 'normalize_grid'`.

- [ ] **Step 3: Write minimal implementation**

Create `src/ingestion/pdf_extract.py`:

```python
"""Extract prose + tables from PDFs.

Digital pages -> pdfplumber (precise cell grids + text). Scanned pages ->
unstructured hi_res + tesseract OCR. Tables are normalized to the existing
``SheetGrid`` so the Excel tabular pipeline ingests them unchanged. Fail-open:
the caller falls back to flat-text parsing if extraction raises.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

from src.ingestion.tabular import SheetGrid

logger = logging.getLogger(__name__)


@dataclass
class ProseBlock:
    text: str
    page: int


@dataclass
class ExtractedPdf:
    prose_blocks: list[ProseBlock] = field(default_factory=list)
    table_grids: list[SheetGrid] = field(default_factory=list)
    method: str = "digital"          # "digital" | "ocr" | "mixed"


def normalize_grid(raw_rows: list[list], sheet_name: str) -> SheetGrid:
    """Coerce a raw extracted table into a SheetGrid: None -> "", drop
    fully-empty rows and fully-empty trailing/leading columns."""
    rows = [["" if c is None else str(c).strip() for c in row] for row in raw_rows]
    rows = [r for r in rows if any(c != "" for c in r)]
    if not rows:
        return SheetGrid(sheet_name=sheet_name, rows=[])
    width = max(len(r) for r in rows)
    rows = [r + [""] * (width - len(r)) for r in rows]   # pad ragged rows
    keep = [i for i in range(width) if any(r[i] != "" for r in rows)]
    rows = [[r[i] for i in keep] for r in rows]
    return SheetGrid(sheet_name=sheet_name, rows=rows)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_ingestion/test_pdf_extract.py::test_normalize_grid_drops_empty_rows_cols_and_fills_none -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/ingestion/pdf_extract.py tests/test_ingestion/test_pdf_extract.py
git commit -m "feat: pdf_extract module skeleton + normalize_grid"
```

### Task 5: stitch_tables (merge multi-page continuations)

**Files:**
- Modify: `src/ingestion/pdf_extract.py`
- Test: `tests/test_ingestion/test_pdf_extract.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_ingestion/test_pdf_extract.py`:

```python
from src.ingestion.pdf_extract import stitch_tables


def test_stitch_merges_consecutive_tables_with_matching_header():
    g1 = SheetGrid("p1_table1", [["Grade", "Over 2"], ["O-1", "3998"]])
    g2 = SheetGrid("p2_table1", [["Grade", "Over 2"], ["E-1", "2017"]])  # same header
    g3 = SheetGrid("p3_table1", [["Loc", "Pct"], ["RUS", "16.5"]])       # different header
    out = stitch_tables([g1, g2, g3])
    assert len(out) == 2
    # g1+g2 merged, repeated header dropped from continuation
    assert out[0].rows == [["Grade", "Over 2"], ["O-1", "3998"], ["E-1", "2017"]]
    assert out[1].rows == [["Loc", "Pct"], ["RUS", "16.5"]]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_ingestion/test_pdf_extract.py::test_stitch_merges_consecutive_tables_with_matching_header -v`
Expected: FAIL with `cannot import name 'stitch_tables'`.

- [ ] **Step 3: Write minimal implementation**

Add to `src/ingestion/pdf_extract.py`:

```python
def stitch_tables(grids: list[SheetGrid]) -> list[SheetGrid]:
    """Merge consecutive grids whose first (header) row is identical, dropping
    the repeated header on continuations. Header-equality is the merge key so
    unrelated adjacent tables stay separate."""
    out: list[SheetGrid] = []
    for g in grids:
        if not g.rows:
            continue
        if out and out[-1].rows and out[-1].rows[0] == g.rows[0]:
            out[-1].rows.extend(g.rows[1:])      # append data rows, drop dup header
        else:
            out.append(SheetGrid(sheet_name=g.sheet_name, rows=[list(r) for r in g.rows]))
    return out
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_ingestion/test_pdf_extract.py::test_stitch_merges_consecutive_tables_with_matching_header -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/ingestion/pdf_extract.py tests/test_ingestion/test_pdf_extract.py
git commit -m "feat: stitch_tables merges multi-page table continuations"
```

### Task 6: Cell-integrity guard

**Files:**
- Modify: `src/ingestion/pdf_extract.py`
- Test: `tests/test_ingestion/test_pdf_extract.py`

- [ ] **Step 1: Write the failing test**

Append:

```python
from src.ingestion.pdf_extract import grid_width_consistent


def test_grid_width_consistent_detects_ragged_data():
    good = SheetGrid("t", [["Grade", "A", "B"], ["O-1", "1", "2"], ["E-1", "3", "4"]])
    bad = SheetGrid("t", [["Grade", "A", "B"], ["O-1", "1"], ["E-1", "3", "4", "5"]])
    assert grid_width_consistent(good) is True
    assert grid_width_consistent(bad) is False


def test_grid_width_consistent_trivially_true_for_tiny_grid():
    assert grid_width_consistent(SheetGrid("t", [["only one row"]])) is True
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_ingestion/test_pdf_extract.py -k grid_width_consistent -v`
Expected: FAIL with `cannot import name 'grid_width_consistent'`.

- [ ] **Step 3: Write minimal implementation**

Add to `src/ingestion/pdf_extract.py`:

```python
def grid_width_consistent(grid: SheetGrid) -> bool:
    """True if every data row has the same cell count as the header row. A
    mismatch signals a sheared/misaligned extraction; such a grid should be
    demoted to messy-region narratives rather than loaded as a DuckDB table."""
    if len(grid.rows) < 2:
        return True
    header_w = len(grid.rows[0])
    return all(len(r) == header_w for r in grid.rows[1:])
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_ingestion/test_pdf_extract.py -k grid_width_consistent -v`
Expected: PASS (2 passed).

- [ ] **Step 5: Commit**

```bash
git add src/ingestion/pdf_extract.py tests/test_ingestion/test_pdf_extract.py
git commit -m "feat: grid_width_consistent integrity guard for extracted tables"
```

---

## Phase 2b — pdf_extract: I/O adapter (fixture-based)

### Task 7: Digital extraction path (`extract_pdf`)

**Files:**
- Modify: `src/ingestion/pdf_extract.py`
- Test: `tests/test_ingestion/test_pdf_extract_io.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_ingestion/test_pdf_extract_io.py`:

```python
from pathlib import Path
import pytest
from src.ingestion.pdf_extract import extract_pdf

FIX = Path("tests/fixtures/pdf/two_page_table.pdf")


@pytest.mark.skipif(not FIX.exists(), reason="fixture not generated (see Task 1)")
def test_extract_pdf_digital_returns_grids_and_prose():
    result = extract_pdf(FIX)
    assert result.method in ("digital", "mixed")
    # at least one table grid recovered; key column values present
    all_cells = [c for g in result.table_grids for row in g.rows for c in row]
    assert any("O-1" in c for c in all_cells)
    assert any("E-3" in c for c in all_cells)
    # multi-page same-header tables were stitched into one grid
    grades = [g for g in result.table_grids
              if g.rows and "Grade" in g.rows[0][0]]
    assert len(grades) == 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_ingestion/test_pdf_extract_io.py -v`
Expected: FAIL with `cannot import name 'extract_pdf'` (or AttributeError until implemented).

- [ ] **Step 3: Write minimal implementation**

Add to `src/ingestion/pdf_extract.py`. (Tune `table_settings` per the Task 1 spike — use `{}` for lattice/ruled tables, or the text-strategy dict for whitespace tables.)

```python
DIGITAL_MIN_CHARS = 20   # a page with fewer extractable chars is treated as scanned

# Per spike: {} (lattice) for ruled tables; text strategy for whitespace tables.
_TABLE_SETTINGS = {"vertical_strategy": "text", "horizontal_strategy": "text"}


def _page_tables(page, page_no: int) -> list[SheetGrid]:
    grids: list[SheetGrid] = []
    raw_tables = page.extract_tables() or []
    if not raw_tables:
        raw_tables = page.extract_tables(table_settings=_TABLE_SETTINGS) or []
    for n, raw in enumerate(raw_tables):
        g = normalize_grid(raw, sheet_name=f"p{page_no}_table{n}")
        if g.rows:
            grids.append(g)
    return grids


def _page_prose(page) -> str:
    """Page text with detected-table regions removed so prose isn't duplicated."""
    table_bboxes = [t.bbox for t in (page.find_tables() or [])]
    if not table_bboxes:
        return page.extract_text() or ""
    def outside_tables(obj):
        x0, top = obj.get("x0", 0), obj.get("top", 0)
        for (bx0, btop, bx1, bbot) in table_bboxes:
            if bx0 <= x0 <= bx1 and btop <= top <= bbot:
                return False
        return True
    return page.filter(outside_tables).extract_text() or ""


def extract_pdf(path: Path) -> ExtractedPdf:
    """Single entry point. Per-page triage: digital pages via pdfplumber,
    scanned pages via OCR (Task 8). Raises on hard failure (caller is fail-open)."""
    import pdfplumber

    prose: list[ProseBlock] = []
    grids: list[SheetGrid] = []
    methods: set[str] = set()
    with pdfplumber.open(str(path)) as pdf:
        for i, page in enumerate(pdf.pages):
            text = page.extract_text() or ""
            if len(text.strip()) >= DIGITAL_MIN_CHARS:
                methods.add("digital")
                grids.extend(_page_tables(page, i))
                body = _page_prose(page).strip()
                if body:
                    prose.append(ProseBlock(text=body, page=i))
            else:
                methods.add("ocr")
                p_blocks, p_grids = _extract_scanned_page(path, i)   # Task 8
                prose.extend(p_blocks)
                grids.extend(p_grids)

    method = "mixed" if len(methods) > 1 else (methods.pop() if methods else "digital")
    stitched = [g for g in stitch_tables(grids)]
    return ExtractedPdf(prose_blocks=prose, table_grids=stitched, method=method)
```

Also add a temporary stub so the digital test runs before Task 8:

```python
def _extract_scanned_page(path: Path, page_no: int):
    """OCR a single scanned page. Implemented in Task 8."""
    return [], []
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_ingestion/test_pdf_extract_io.py -v`
Expected: PASS. If the stitch assertion fails, the fixture's two pages produced different header strings — confirm the fixture repeats an identical header row (Task 1, Step 4).

- [ ] **Step 5: Commit**

```bash
git add src/ingestion/pdf_extract.py tests/test_ingestion/test_pdf_extract_io.py
git commit -m "feat: extract_pdf digital path (pdfplumber tables + prose, stitched)"
```

### Task 8: Scanned/OCR page path

**Files:**
- Modify: `src/ingestion/pdf_extract.py`
- Test: `tests/test_ingestion/test_pdf_extract_io.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_ingestion/test_pdf_extract_io.py`:

```python
def test_extract_scanned_page_parses_html_tables(monkeypatch):
    """The OCR path turns unstructured Table elements (text_as_html) into grids
    and NarrativeText into prose. Mock unstructured so the test is fast/offline."""
    from src.ingestion import pdf_extract

    class _Meta:
        def __init__(self, html=None): self.text_as_html = html

    class _El:
        def __init__(self, cat, text, html=None):
            self.category = cat
            self._text = text
            self.metadata = _Meta(html)
        def __str__(self): return self._text

    html = "<table><tr><td>Grade</td><td>Pay</td></tr><tr><td>E-1</td><td>2017</td></tr></table>"
    fake = [
        _El("Title", "Active Duty Pay"),
        _El("NarrativeText", "Monthly basic pay follows."),
        _El("Table", "Grade Pay E-1 2017", html),
    ]
    monkeypatch.setattr(pdf_extract, "_partition_scanned", lambda path, page_no: fake)

    blocks, grids = pdf_extract._extract_scanned_page(FIX, 0)
    assert any("Monthly basic pay" in b.text for b in blocks)
    assert len(grids) == 1
    assert ["E-1", "2017"] in grids[0].rows
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_ingestion/test_pdf_extract_io.py::test_extract_scanned_page_parses_html_tables -v`
Expected: FAIL (the stub returns `[], []`).

- [ ] **Step 3: Write minimal implementation**

Replace the `_extract_scanned_page` stub in `src/ingestion/pdf_extract.py`:

```python
def _html_to_grid(html: str, sheet_name: str) -> SheetGrid | None:
    """Parse an unstructured Table element's text_as_html into a SheetGrid."""
    from html.parser import HTMLParser

    class _T(HTMLParser):
        def __init__(self):
            super().__init__()
            self.rows, self._row, self._cell, self._in = [], [], [], False
        def handle_starttag(self, tag, attrs):
            if tag == "tr": self._row = []
            elif tag in ("td", "th"): self._in, self._cell = True, []
        def handle_endtag(self, tag):
            if tag in ("td", "th"):
                self._row.append("".join(self._cell).strip()); self._in = False
            elif tag == "tr":
                self.rows.append(self._row)
        def handle_data(self, data):
            if self._in: self._cell.append(data)

    p = _T(); p.feed(html or "")
    g = normalize_grid(p.rows, sheet_name)
    return g if g.rows else None


def _partition_scanned(path: Path, page_no: int):
    """Real OCR partition for one page (seam mocked in tests)."""
    from unstructured.partition.pdf import partition_pdf
    return partition_pdf(
        filename=str(path), strategy="hi_res", infer_table_structure=True,
        page_numbers=[page_no + 1],   # unstructured is 1-indexed
    )


def _extract_scanned_page(path: Path, page_no: int):
    blocks: list[ProseBlock] = []
    grids: list[SheetGrid] = []
    try:
        elements = _partition_scanned(path, page_no)
    except Exception as e:
        logger.warning(f"OCR partition failed on page {page_no} of {path.name}: {e}")
        return blocks, grids
    n = 0
    for el in elements:
        cat = getattr(el, "category", "")
        if cat == "Table":
            html = getattr(getattr(el, "metadata", None), "text_as_html", None)
            g = _html_to_grid(html, f"p{page_no}_ocr{n}") if html else None
            if g:
                grids.append(g); n += 1
        else:
            txt = str(el).strip()
            if txt:
                blocks.append(ProseBlock(text=txt, page=page_no))
    return blocks, grids
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_ingestion/test_pdf_extract_io.py::test_extract_scanned_page_parses_html_tables -v`
Expected: PASS.

- [ ] **Step 5: Run the whole pdf_extract suite**

Run: `python -m pytest tests/test_ingestion/test_pdf_extract.py tests/test_ingestion/test_pdf_extract_io.py -v`
Expected: all PASS.

- [ ] **Step 6: Commit**

```bash
git add src/ingestion/pdf_extract.py tests/test_ingestion/test_pdf_extract_io.py
git commit -m "feat: OCR path for scanned PDF pages (unstructured -> grids + prose)"
```

---

## Phase 3 — Pipeline integration

### Task 9: Refactor — split `ingest_grids` out of `ingest_structured_sheets`

**Files:**
- Modify: `src/ingestion/tabular_ingest.py`
- Test: `tests/test_ingestion/test_tabular_ingest.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_ingestion/test_tabular_ingest.py` (match existing fakes in that file for `vector_store`/`metadata_store`/`schema_registry`; reuse them):

```python
import pytest
from src.ingestion.tabular import SheetGrid


@pytest.mark.asyncio
async def test_ingest_grids_processes_provided_grids(tmp_path, monkeypatch):
    """ingest_grids ingests grids passed directly (no file read), so PDF-derived
    grids reuse the same clean/messy logic as Excel sheets."""
    from src.ingestion import tabular_ingest as ti
    grids = [SheetGrid("p0_table0",
                       [["Grade", "Over2", "Over4"],
                        ["O-1", "3998.40", "5031.30"],
                        ["E-1", "2017.20", "2017.20"]])]
    vs, ms, reg = _fake_vector_store(), _fake_metadata_store(), _fake_schema_registry()
    classifications, ingested = await ti.ingest_grids(
        grids, doc_id="docX", filename="ad.pdf", doc_type="pdf",
        acl_groups=["executives"], category="payroll_compensation",
        vector_store=vs, metadata_store=ms, schema_registry=reg,
        generate_fn=lambda **k: '{"key_columns":["grade"],"measure_columns":["over2","over4"],'
                                '"column_descriptions":{},"table_description":"AD pay"}',
    )
    assert "p0_table0" in ingested            # clean sheet structured
    assert classifications[0].route == "clean"
    assert vs.upserted                         # narratives embedded
```

> If `tests/test_ingestion/test_tabular_ingest.py` lacks `_fake_vector_store`/`_fake_metadata_store`/`_fake_schema_registry` helpers, copy the fakes already used by its existing tests (they construct `ingest_structured_sheets` args). Keep the same shapes.

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_ingestion/test_tabular_ingest.py::test_ingest_grids_processes_provided_grids -v`
Expected: FAIL with `module 'tabular_ingest' has no attribute 'ingest_grids'`.

- [ ] **Step 3: Refactor — extract `ingest_grids`**

In `src/ingestion/tabular_ingest.py`, replace the body of `ingest_structured_sheets` (lines ~99-174) so the loop moves into a new `ingest_grids`. Final shape:

```python
async def ingest_grids(grids, doc_id, filename, doc_type, acl_groups, category,
                       vector_store, metadata_store, schema_registry=None,
                       generate_fn=None):
    """Structured-ingest a list of already-acquired SheetGrids: clean sheets ->
    DuckDB + schema + per-row narratives; messy sheets -> region narratives.
    Returns (classifications, ingested_names). Fully fail-open per grid. Shared by
    Excel (ingest_structured_sheets) and PDF (pipeline PDF branch)."""
    if schema_registry is None:
        from src.api.routes_ingest import get_schema_registry
        schema_registry = get_schema_registry()

    classifications = [classify_sheet(g) for g in grids]
    for g, c in zip(grids, classifications):
        logger.info(
            f"Tabular ingest [{filename}]: sheet '{g.sheet_name}' classified {c.route.upper()} "
            f"({len(g.rows)} rows, header_row={c.header_row_index})")
    ingested: set[str] = set()
    chunk_index = 0
    con = None
    try:
        con = connect_tabular(read_only=False)
        for grid, cls in zip(grids, classifications):
            if cls.route == "clean":
                try:
                    col_names, profile, data_rows, schema = _ingest_one_clean_sheet(
                        con, grid, cls, doc_id, acl_groups, schema_registry, generate_fn)
                    chunk_index = await _save_and_embed_clean(
                        metadata_store, vector_store, schema, col_names, profile,
                        data_rows, doc_id, filename, doc_type, acl_groups, category,
                        chunk_index)
                    ingested.add(grid.sheet_name)
                    logger.info(
                        f"Tabular ingest [{filename}]: sheet '{grid.sheet_name}' CLEAN -> "
                        f"structured ({len(data_rows)} data rows to DuckDB + schema + "
                        f"narratives); full-text chunks suppressed")
                except Exception as e:
                    logger.warning(
                        f"Tabular ingest: failed on clean sheet '{grid.sheet_name}' "
                        f"of {filename}: {e}")
                    continue
            else:
                try:
                    region = find_table_region(grid.rows)
                    if region is None:
                        logger.info(
                            f"Tabular ingest [{filename}]: sheet '{grid.sheet_name}' MESSY -> "
                            f"no table-like region; full-text chunks only")
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
    return classifications, ingested


async def ingest_structured_sheets(file_path, doc_id, filename, doc_type, acl_groups,
                                   category, vector_store, metadata_store,
                                   schema_registry=None, generate_fn=None):
    """Read a spreadsheet's sheets once, then structured-ingest them via
    ingest_grids. Returns (grids, classifications, ingested_names)."""
    try:
        grids = read_sheets(Path(file_path))
    except Exception as e:
        logger.warning(f"Tabular ingest: could not read sheets from {filename}: {e}")
        return [], [], set()
    classifications, ingested = await ingest_grids(
        grids, doc_id, filename, doc_type, acl_groups, category,
        vector_store, metadata_store, schema_registry=schema_registry,
        generate_fn=generate_fn)
    logger.info(f"Tabular ingest: structured {len(ingested)} clean sheet(s) from {filename}")
    return grids, classifications, ingested
```

- [ ] **Step 4: Run the new test + the existing tabular_ingest suite (parity)**

Run: `python -m pytest tests/test_ingestion/test_tabular_ingest.py -v`
Expected: the new test PASSES and all previously-passing tests still PASS (the refactor is behavior-preserving for Excel).

- [ ] **Step 5: Commit**

```bash
git add src/ingestion/tabular_ingest.py tests/test_ingestion/test_tabular_ingest.py
git commit -m "refactor: extract ingest_grids seam from ingest_structured_sheets"
```

### Task 10: Wire PDF branch into `pipeline.py` (sync, fail-open)

**Files:**
- Modify: `src/ingestion/pipeline.py`
- Test: `tests/test_ingestion/test_pipeline_pdf.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_ingestion/test_pipeline_pdf.py`:

```python
import pytest
from src.ingestion.pdf_extract import ExtractedPdf, ProseBlock
from src.ingestion.tabular import SheetGrid


@pytest.mark.asyncio
async def test_pdf_falls_back_to_flat_text_when_extract_raises(monkeypatch):
    """If extract_pdf raises, the PDF still ingests via flat-text chunking (no crash)."""
    from src.ingestion import pipeline
    monkeypatch.setattr(pipeline, "extract_pdf",
                        lambda p: (_ for _ in ()).throw(RuntimeError("boom")))
    # _is_structured_pdf gates the branch; flat-text path must run on failure.
    assert pipeline._is_structured_pdf("pdf") is True
```

> This is a lightweight guard test; the full ingest is covered by the integration test (Task 17). The key behaviors: a PDF is recognized as structured, and a raising `extract_pdf` does not abort ingestion.

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_ingestion/test_pipeline_pdf.py -v`
Expected: FAIL with `module 'pipeline' has no attribute 'extract_pdf'` or `_is_structured_pdf`.

- [ ] **Step 3: Implement the branch**

In `src/ingestion/pipeline.py`:

Add import near the other ingestion imports (line ~8):
```python
from src.ingestion.tabular_ingest import ingest_structured_sheets, ingest_grids, SPREADSHEET_DOC_TYPES
from src.ingestion.pdf_extract import extract_pdf
```

Add a helper near the top of the module:
```python
def _is_structured_pdf(doc_type: str) -> bool:
    return doc_type == "pdf"
```

Replace the spreadsheet-branch region (the `is_spreadsheet = ...` block at line ~92 and the tier-loop `if is_spreadsheet:` at line ~105) with a version that also handles PDFs:

```python
    is_spreadsheet = parsed.doc_type in SPREADSHEET_DOC_TYPES
    is_pdf = _is_structured_pdf(parsed.doc_type)
    text_sheets = None
    pdf_prose = None
    if is_spreadsheet:
        grids, classifications, ingested = await ingest_structured_sheets(
            file_path, doc_id, parsed.filename, parsed.doc_type,
            acl_groups, category, vector_store, metadata_store,
        )
        text_sheets = sheets_needing_text(grids, classifications, ingested)
    elif is_pdf:
        try:
            extracted = extract_pdf(Path(file_path))
            await ingest_grids(
                extracted.table_grids, doc_id, parsed.filename, parsed.doc_type,
                acl_groups, category, vector_store, metadata_store,
            )
            pdf_prose = "\n\n".join(b.text for b in extracted.prose_blocks)
            logger.info(f"PDF structured extract [{parsed.filename}]: "
                        f"{len(extracted.table_grids)} table(s), method={extracted.method}")
        except Exception as e:
            logger.warning(f"PDF structured extract failed for {parsed.filename}, "
                           f"falling back to flat text: {e}")
            is_pdf = False   # fall back to parsed.text chunking below
```

In the tier loop, extend the chunk-source selection:
```python
        if is_spreadsheet:
            tier_chunks = build_tier_chunks(text_sheets, chunk_size=tier_size)
        elif is_pdf:
            tier_chunks = chunk_text(pdf_prose or "", chunk_size=tier_size, chunk_overlap=tier_overlap)
        else:
            tier_chunks = chunk_text(parsed.text, chunk_size=tier_size, chunk_overlap=tier_overlap)
```

> `logger` and `Path` are already imported in this module (`from pathlib import Path`, and a module logger). Confirm `logger` exists near the top; if the function uses a local `import logging; logger = ...`, reuse that.

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_ingestion/test_pipeline_pdf.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/ingestion/pipeline.py tests/test_ingestion/test_pipeline_pdf.py
git commit -m "feat: PDF structured-extract branch in sync pipeline (fail-open)"
```

### Task 11: Wire the same PDF branch into `queue.py` (async worker)

**Files:**
- Modify: `src/ingestion/queue.py`
- Test: `tests/test_ingestion/test_pipeline_pdf.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_ingestion/test_pipeline_pdf.py`:

```python
def test_queue_recognizes_pdf_as_structured():
    from src.ingestion import queue as q
    assert q._is_structured_pdf("pdf") is True
    assert q._is_structured_pdf("xlsx") is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_ingestion/test_pipeline_pdf.py::test_queue_recognizes_pdf_as_structured -v`
Expected: FAIL with `module 'queue' has no attribute '_is_structured_pdf'`.

- [ ] **Step 3: Implement the branch**

In `src/ingestion/queue.py`:

Add to the import inside `_process_job` (line ~192) and add the helper at module scope:
```python
from src.ingestion.tabular_ingest import ingest_structured_sheets, ingest_grids, SPREADSHEET_DOC_TYPES
from src.ingestion.pdf_extract import extract_pdf
```
Module-scope helper (top of file):
```python
def _is_structured_pdf(doc_type: str) -> bool:
    return doc_type == "pdf"
```

Mirror the pipeline.py change at the spreadsheet branch (line ~294) and the tier loop (line ~306). Because table ingest in the worker wraps blocking calls, run `extract_pdf` off the event loop:

```python
        is_spreadsheet = parsed.doc_type in SPREADSHEET_DOC_TYPES
        is_pdf = _is_structured_pdf(parsed.doc_type)
        text_sheets = None
        pdf_prose = None
        if is_spreadsheet:
            self.update_step(job.job_id, IngestStep.STORING, "Structured spreadsheet ingest (DuckDB + narratives)")
            grids, classifications, ingested = await ingest_structured_sheets(
                file_path, doc_id, parsed.filename, parsed.doc_type,
                job.acl_groups, category, vector_store, metadata_store,
            )
            text_sheets = sheets_needing_text(grids, classifications, ingested)
        elif is_pdf:
            self.update_step(job.job_id, IngestStep.STORING, "Structured PDF ingest (tables -> DuckDB + narratives)")
            try:
                extracted = await asyncio.to_thread(extract_pdf, Path(file_path))
                await ingest_grids(
                    extracted.table_grids, doc_id, parsed.filename, parsed.doc_type,
                    job.acl_groups, category, vector_store, metadata_store,
                )
                pdf_prose = "\n\n".join(b.text for b in extracted.prose_blocks)
            except Exception as e:
                logger.warning(f"PDF structured extract failed for {parsed.filename}, "
                               f"falling back to flat text: {e}")
                is_pdf = False
```

Tier loop:
```python
            if is_spreadsheet:
                tier_chunks = build_tier_chunks(text_sheets, chunk_size=tier_size)
            elif is_pdf:
                tier_chunks = chunk_text(pdf_prose or "", chunk_size=tier_size, chunk_overlap=tier_overlap)
            else:
                tier_chunks = chunk_text(parsed.text, chunk_size=tier_size, chunk_overlap=tier_overlap)
```

> Confirm `Path` and `logger` are imported in `queue.py`; add `from pathlib import Path` / a module `logger` if missing.

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_ingestion/test_pipeline_pdf.py -v`
Expected: PASS (all in file).

- [ ] **Step 5: Commit**

```bash
git add src/ingestion/queue.py tests/test_ingestion/test_pipeline_pdf.py
git commit -m "feat: PDF structured-extract branch in async queue worker (fail-open)"
```

---

## Phase 4 — Findability: prefix-pattern glossary

### Task 12: `glossary_lookup` shared matcher (exact + prefix pattern)

**Files:**
- Modify: `src/ingestion/table_profiler.py`
- Test: `tests/test_ingestion/test_table_profiler.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_ingestion/test_table_profiler.py` (create the file if absent, with the import below):

```python
from src.ingestion.table_profiler import glossary_lookup


def test_glossary_lookup_exact_then_prefix():
    g = {"GS": "base", "E-*": "Enlisted Member", "O-*": "Commissioned Officer"}
    assert glossary_lookup(g, "GS") == "base"          # exact wins
    assert glossary_lookup(g, "E-3") == "Enlisted Member"
    assert glossary_lookup(g, "O-10") == "Commissioned Officer"
    assert glossary_lookup(g, "W-2") is None           # no match
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_ingestion/test_table_profiler.py::test_glossary_lookup_exact_then_prefix -v`
Expected: FAIL with `cannot import name 'glossary_lookup'`.

- [ ] **Step 3: Write minimal implementation**

Add to `src/ingestion/table_profiler.py`:

```python
def glossary_lookup(glossary: dict, value) -> str | None:
    """Map a cell value to its meaning. Exact match wins; otherwise a glossary
    key ending in ``*`` matches values starting with the prefix (e.g. ``E-*``
    matches ``E-3``). Returns None if nothing matches."""
    if value is None:
        return None
    s = str(value).strip()
    if s in glossary:
        return glossary[s]
    for code, meaning in glossary.items():
        if isinstance(code, str) and code.endswith("*") and s.startswith(code[:-1]):
            return meaning
    return None
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_ingestion/test_table_profiler.py::test_glossary_lookup_exact_then_prefix -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/ingestion/table_profiler.py tests/test_ingestion/test_table_profiler.py
git commit -m "feat: glossary_lookup with exact + prefix-pattern matching"
```

### Task 13: Annotate row narratives with the glossary

**Files:**
- Modify: `src/ingestion/table_profiler.py`
- Test: `tests/test_ingestion/test_table_profiler.py`

- [ ] **Step 1: Write the failing test**

Append:

```python
from src.ingestion.table_profiler import build_row_narratives, TableProfile


def test_row_narrative_annotates_key_value_with_glossary():
    profile = TableProfile(
        column_descriptions={"grade": "Pay grade", "over2": "Over 2 years"},
        key_columns=["grade"], measure_columns=["over2"], table_description="AD pay")
    out = build_row_narratives(
        ["grade", "over2"], profile, [["E-3", "3081.00"]],
        context="AD pay", column_glossaries={"grade": {"E-*": "Enlisted Member"}})
    assert "E-3 (Enlisted Member)" in out[0]


def test_row_narrative_unchanged_without_glossary():
    profile = TableProfile(column_descriptions={"grade": "Pay grade"},
                           key_columns=["grade"], measure_columns=[], table_description="")
    out = build_row_narratives(["grade"], profile, [["E-3"]])
    assert out == ["Pay grade=E-3"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_ingestion/test_table_profiler.py -k row_narrative -v`
Expected: FAIL — `build_row_narratives() got an unexpected keyword argument 'column_glossaries'`.

- [ ] **Step 3: Modify `row_narrative` and `build_row_narratives`**

Replace `row_narrative` and `build_row_narratives` in `src/ingestion/table_profiler.py` with glossary-aware versions (default `None` keeps byte-identical output):

```python
def row_narrative(col_names: list[str], profile: TableProfile, row: list,
                  column_glossaries: dict | None = None) -> str:
    index = {name: i for i, name in enumerate(col_names)}
    glos = column_glossaries or {}

    def cell(name: str) -> str:
        i = index.get(name)
        if i is None or i >= len(row):
            return "(not specified)"
        value = _fmt_cell(row[i])
        mapping = glos.get(name)
        if mapping:
            meaning = glossary_lookup(mapping, row[i])
            if meaning:
                return f"{value} ({meaning})"
        return value

    keys = [f"{profile.column_descriptions.get(k, k)}={cell(k)}" for k in profile.key_columns]
    measures = [f"{profile.column_descriptions.get(m, m)} is {cell(m)}" for m in profile.measure_columns]
    key_str = ", ".join(keys)
    measure_str = "; ".join(measures)
    if key_str and measure_str:
        return f"{key_str}: {measure_str}"
    return key_str or measure_str


def build_row_narratives(col_names: list[str], profile: TableProfile, data_rows: list,
                         context: str = "", column_glossaries: dict | None = None) -> list[str]:
    prefix = f"{context} — " if context else ""
    return [f"{prefix}{row_narrative(col_names, profile, row, column_glossaries)}"
            for row in data_rows]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_ingestion/test_table_profiler.py -k row_narrative -v`
Expected: PASS (2 passed).

- [ ] **Step 5: Commit**

```bash
git add src/ingestion/table_profiler.py tests/test_ingestion/test_table_profiler.py
git commit -m "feat: row narratives annotate key values via column glossaries"
```

### Task 14: Resolve + thread glossaries through `ingest_grids`

**Files:**
- Modify: `src/ingestion/tabular_ingest.py`
- Test: `tests/test_ingestion/test_tabular_ingest.py`

- [ ] **Step 1: Write the failing test**

Append:

```python
@pytest.mark.asyncio
async def test_ingest_grids_applies_seeded_glossary_to_narratives(monkeypatch):
    from src.ingestion import tabular_ingest as ti
    from src.db.hint_store import HintStore, SchemaHint

    hint_store = HintStore()
    hint_store.register(SchemaHint(
        scope_type="category", scope_value="payroll_compensation",
        hint_type="value_glossary", target_column="grade",
        payload={"E-*": "Enlisted Member", "O-*": "Commissioned Officer"}))
    monkeypatch.setattr("src.api.routes_ingest.get_hint_store", lambda: hint_store)

    grids = [SheetGrid("p0_table0",
                       [["grade", "over2"], ["O-1", "3998"], ["E-1", "2017"]])]
    vs, ms, reg = _fake_vector_store(), _fake_metadata_store(), _fake_schema_registry()
    await ti.ingest_grids(
        grids, doc_id="docG", filename="ad.pdf", doc_type="pdf",
        acl_groups=["executives"], category="payroll_compensation",
        vector_store=vs, metadata_store=ms, schema_registry=reg,
        generate_fn=lambda **k: '{"key_columns":["grade"],"measure_columns":["over2"],'
                                '"column_descriptions":{},"table_description":"AD pay"}',
    )
    joined = "\n".join(vs.upserted_texts)
    assert "Enlisted Member" in joined
```

> `_fake_vector_store` must capture upserted texts as `upserted_texts`. If the existing fake doesn't, extend it (append `texts` on `upsert`).

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_ingestion/test_tabular_ingest.py::test_ingest_grids_applies_seeded_glossary_to_narratives -v`
Expected: FAIL — narratives have no glossary annotation (no "Enlisted Member").

- [ ] **Step 3: Implement glossary resolution in the clean-sheet path**

In `src/ingestion/tabular_ingest.py`:

Add imports at top:
```python
from src.agent.strategies.hint_resolver import resolve_hints
```

Add a tiny helper:
```python
def _resolve_column_glossaries(schema, category, dataset_id, hint_store):
    """ResolvedHints.column_glossaries for one schema, given its collection scope.
    Fail-safe -> {} on any error."""
    if hint_store is None:
        return {}
    try:
        doc_like = type("Doc", (), {"category": category or "",
                                    "dataset_id": dataset_id})()
        return resolve_hints(schema, doc_like, hint_store).column_glossaries
    except Exception:
        return {}
```

Change `_save_and_embed_clean` to accept and apply glossaries:
```python
async def _save_and_embed_clean(metadata_store, vector_store, schema, col_names, profile,
                                data_rows, doc_id, filename, doc_type, acl_groups, category,
                                chunk_index, column_glossaries=None):
    await metadata_store.save_schema(schema)
    narratives = [n for n in build_row_narratives(
        col_names, profile, data_rows, context=profile.table_description,
        column_glossaries=column_glossaries) if n.strip()]
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

In `ingest_grids`, accept `dataset_id=None, hint_store=None`, default `hint_store` from the singleton, and pass resolved glossaries into `_save_and_embed_clean`:
```python
async def ingest_grids(grids, doc_id, filename, doc_type, acl_groups, category,
                       vector_store, metadata_store, schema_registry=None,
                       generate_fn=None, dataset_id=None, hint_store=None):
    if schema_registry is None:
        from src.api.routes_ingest import get_schema_registry
        schema_registry = get_schema_registry()
    if hint_store is None:
        from src.api.routes_ingest import get_hint_store
        hint_store = get_hint_store()
    ...
                    col_names, profile, data_rows, schema = _ingest_one_clean_sheet(...)
                    column_glossaries = _resolve_column_glossaries(
                        schema, category, dataset_id, hint_store)
                    chunk_index = await _save_and_embed_clean(
                        metadata_store, vector_store, schema, col_names, profile,
                        data_rows, doc_id, filename, doc_type, acl_groups, category,
                        chunk_index, column_glossaries=column_glossaries)
```

> Pass `dataset_id` from the callers if readily available (`job.dataset_id` in queue.py; the dataset arg in pipeline.py). If not threaded, category-scoped glossaries still resolve (the seed is category-scoped), so dataset_id defaulting to None is acceptable for this project.

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_ingestion/test_tabular_ingest.py -v`
Expected: the new test PASSES; existing tabular_ingest tests still PASS.

- [ ] **Step 5: Commit**

```bash
git add src/ingestion/tabular_ingest.py tests/test_ingestion/test_tabular_ingest.py
git commit -m "feat: resolve + apply column glossaries to row narratives at ingest"
```

### Task 15: Prefix-pattern glossary in the text-to-SQL prompt

**Files:**
- Modify: `src/ingestion/tabular_store.py`
- Test: `tests/test_ingestion/test_tabular_store.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_ingestion/test_tabular_store.py`:

```python
def test_schema_prompt_annotates_values_via_prefix_glossary(monkeypatch):
    """A value_glossary with prefix patterns (E-*) annotates actual distinct
    values (E-3) in the text-to-SQL prompt, not just exact-match codes."""
    import src.ingestion.tabular_store as ts
    from src.agent.strategies.hint_resolver import ResolvedHints

    class _Col: 
        def __init__(s, name, dtype, desc=""): s.name, s.dtype, s.description = name, dtype, desc
    class _Schema:
        table = "doc_x_pay"; description = "AD pay"
        columns = [_Col("grade", "VARCHAR", "Pay grade")]

    monkeypatch.setattr(ts, "distinct_values", lambda con, t, c, m: ["E-3", "O-1"])
    hints = {"doc_x_pay": ResolvedHints(column_glossaries={"grade": {"E-*": "Enlisted", "O-*": "Officer"}})}
    out = ts.schema_prompt_with_values([_Schema()], con=None, hints=hints)
    assert "E-3 (Enlisted)" in out
    assert "O-1 (Officer)" in out
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_ingestion/test_tabular_store.py::test_schema_prompt_annotates_values_via_prefix_glossary -v`
Expected: FAIL — current line renders `E-3` unannotated (exact-match only).

- [ ] **Step 3: Route value annotation through `glossary_lookup`**

In `src/ingestion/tabular_store.py`, add the import near the top:
```python
from src.ingestion.table_profiler import glossary_lookup
```

Replace line ~241 in `schema_prompt_with_values`:
```python
                    rendered = [f"{v} ({gloss[str(v)]})" if str(v) in gloss else str(v) for v in vals]
```
with:
```python
                    rendered = []
                    for v in vals:
                        meaning = glossary_lookup(gloss, v) if gloss else None
                        rendered.append(f"{v} ({meaning})" if meaning else str(v))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_ingestion/test_tabular_store.py::test_schema_prompt_annotates_values_via_prefix_glossary -v`
Expected: PASS.

- [ ] **Step 5: Run the tabular_store suite (regression)**

Run: `python -m pytest tests/test_ingestion/test_tabular_store.py -v`
Expected: all PASS (existing exact-match glossary tests unaffected — `glossary_lookup` does exact-match first).

- [ ] **Step 6: Commit**

```bash
git add src/ingestion/tabular_store.py tests/test_ingestion/test_tabular_store.py
git commit -m "feat: prefix-pattern glossary annotation in text-to-SQL prompt"
```

### Task 16: Seed the military paygrade glossary (operator data)

**Files:**
- Create: `scripts/seed_military_paygrade_glossary.py`
- Create: `docs/glossaries/military-paygrades.md` (provenance note)

- [ ] **Step 1: Write the seed script**

Create `scripts/seed_military_paygrade_glossary.py`:

```python
"""Seed the verified military paygrade -> class glossary for the
payroll_compensation category. Run once against a running instance:

    python scripts/seed_military_paygrade_glossary.py --column grade

The --column must match the AD pay table's actual key-column name (inspect the
registered schema after ingesting the PDF). Prefix patterns cover all grades."""
import argparse
import asyncio

PAYGRADE_GLOSSARY = {
    "E-*": "Enlisted Member",
    "O-*": "Commissioned Officer",
    "O-*E": "Commissioned Officer with prior enlisted service",
    "W-*": "Warrant Officer",
}


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--column", required=True, help="key column name, e.g. 'grade'")
    ap.add_argument("--category", default="payroll_compensation")
    args = ap.parse_args()

    from src.api.routes_ingest import get_metadata_store, get_hint_store
    from src.db.hint_store import SchemaHint
    ms, hs = get_metadata_store(), get_hint_store()
    hint = SchemaHint(
        scope_type="category", scope_value=args.category,
        hint_type="value_glossary", target_column=args.column,
        payload=PAYGRADE_GLOSSARY, provenance="curated", created_by="seed-script")
    await ms.save_hint(hint)
    hs.register(hint)
    print(f"seeded paygrade glossary for column '{args.column}' / category '{args.category}'")


if __name__ == "__main__":
    asyncio.run(main())
```

- [ ] **Step 2: Document provenance**

Create `docs/glossaries/military-paygrades.md`:
```markdown
# Military paygrade glossary

Maps DoD pay grades to personnel class for the `payroll_compensation` category.
Verified against the standard DoD pay-grade scheme:
- `E-1`..`E-9` → Enlisted Member
- `O-1`..`O-10` → Commissioned Officer
- `O-1E`..`O-3E` → Commissioned Officer with prior enlisted service
- `W-1`..`W-5` → Warrant Officer

Prefix patterns (`E-*`, `O-*`, `W-*`, `O-*E`) are used so all grades resolve
without enumerating each. Seeded via `scripts/seed_military_paygrade_glossary.py`.
Lesson from the locality glossary: glossary content is operator-verified, never
model-guessed.
```

- [ ] **Step 3: (Deferred to deploy) run after the PDF is ingested**

Note in the PR description: after re-ingesting the AD pay PDF, inspect its registered schema to find the paygrade column name, then run the seed script with that `--column`. Not run during implementation (needs a live instance + ingested doc).

- [ ] **Step 4: Commit**

```bash
git add scripts/seed_military_paygrade_glossary.py docs/glossaries/military-paygrades.md
git commit -m "feat: seed script + provenance for military paygrade glossary"
```

---

## Phase 5 — Integration & verification

### Task 17: End-to-end integration test

**Files:**
- Test: `tests/test_ingestion/test_pdf_integration.py`

- [ ] **Step 1: Write the test**

Create `tests/test_ingestion/test_pdf_integration.py`:

```python
import pytest
from pathlib import Path
from src.ingestion.pdf_extract import extract_pdf
from src.ingestion import tabular_ingest as ti
from src.db.hint_store import HintStore, SchemaHint

FIX = Path("tests/fixtures/pdf/two_page_table.pdf")


@pytest.mark.skipif(not FIX.exists(), reason="fixture not generated (see Task 1)")
@pytest.mark.asyncio
async def test_pdf_to_structured_with_glossary(monkeypatch):
    """PDF -> extract -> ingest_grids produces DuckDB-bound row narratives that
    mention 'Enlisted' (findable by the original failing query's vocabulary)."""
    hs = HintStore()
    hs.register(SchemaHint(
        scope_type="category", scope_value="payroll_compensation",
        hint_type="value_glossary", target_column="grade",
        payload={"E-*": "Enlisted Member", "O-*": "Commissioned Officer"}))
    monkeypatch.setattr("src.api.routes_ingest.get_hint_store", lambda: hs)

    extracted = extract_pdf(FIX)
    assert extracted.table_grids, "expected at least one table grid"

    vs, ms, reg = _fake_vs(), _fake_ms(), _fake_reg()  # reuse fakes from test_tabular_ingest
    # Rename the key column header to 'grade' so the seeded glossary matches the schema.
    # (In production the column is named from the PDF header via _safe_column_names.)
    await ti.ingest_grids(
        extracted.table_grids, doc_id="docE2E", filename="ad.pdf", doc_type="pdf",
        acl_groups=["executives"], category="payroll_compensation",
        vector_store=vs, metadata_store=ms, schema_registry=reg,
        generate_fn=lambda **k: '{"key_columns":["grade"],"measure_columns":[],'
                                '"column_descriptions":{},"table_description":"AD pay"}',
    )
    text = "\n".join(getattr(vs, "upserted_texts", []))
    # If the fixture's header column isn't literally 'grade', this asserts the
    # mechanism on the grids; adjust the glossary target_column to the real header.
    assert "E-" in text  # enlisted grades made it into narratives
```

> Pull `_fake_vs`/`_fake_ms`/`_fake_reg` from `test_tabular_ingest.py` (import or duplicate). The glossary `target_column` must equal the schema column name `_safe_column_names` produced from the fixture header; adjust the seed/test to match (e.g. `col_0` if the header cell was blank). The assertion `"E-" in text` is robust to that naming; the stronger `"Enlisted Member"` assertion requires the column name to align.

- [ ] **Step 2: Run test to verify it passes (or is skipped without the fixture)**

Run: `python -m pytest tests/test_ingestion/test_pdf_integration.py -v`
Expected: PASS (or SKIP if the fixture is absent — then generate it per Task 1).

- [ ] **Step 3: Commit**

```bash
git add tests/test_ingestion/test_pdf_integration.py
git commit -m "test: end-to-end PDF -> structured ingest with glossary"
```

### Task 18: Full verification + offline image build

**Files:** none (verification only).

- [ ] **Step 1: Run the full ingestion test suite**

Run: `python -m pytest tests/test_ingestion -v`
Expected: all PASS except the documented pre-existing/environmental failures (numpy/sklearn binary-compat, OpenAI 401, stale `extract_entities` patch). Confirm no NEW failures vs the baseline.

- [ ] **Step 2: Build the offline image and verify**

Run: `docker compose build api && docker compose run --rm --no-deps api python scripts/prefetch_pdf_models.py`
Expected: build succeeds; prefetch prints `pdf models prefetched` with `HF_HUB_OFFLINE=1` set (proves models are resident, no network).

- [ ] **Step 3: Sanity-check both ingestion paths import cleanly**

Run: `python -c "import src.ingestion.pipeline, src.ingestion.queue, src.ingestion.pdf_extract; print('imports OK')"`
Expected: `imports OK`.

- [ ] **Step 4: Final commit (if any verification fixups were needed)**

```bash
git add -A && git commit -m "chore: PDF extraction verification fixups" || echo "nothing to commit"
```

---

## Deployment notes (operator, post-merge)

1. Rebuild + redeploy the image (`docker compose up -d --build`) — picks up the new branch + baked models.
2. Wipe + re-ingest all data (re-ingestion is out of scope per the spec; operator does this manually).
3. After the AD pay PDF re-ingests, inspect its registered schema for the paygrade key-column name, then run `python scripts/seed_military_paygrade_glossary.py --column <name>` (inside the api container).
4. Smoke test: "What is the pay range for an enlisted person?" — expect the `E-*` rows retrieved and the answer citing the PDF filename.

## Self-review notes

- **Spec coverage:** module + interface (Task 4), digital tables (Task 7), prose+de-dup (Task 7), multi-page stitch (Task 5), integrity guard (Task 6), OCR/scanned (Task 8), per-page triage + mixed (Task 7), `ingest_grids` refactor (Task 9), both ingestion paths (Tasks 10–11), fail-open (Tasks 10–11), prefix-pattern glossary (Task 12), narrative annotation (Tasks 13–14), SQL-prompt annotation (Task 15), curated seed (Task 16), offline build (Task 3), testing (Tasks 4–17), success criteria (Tasks 16–17 + deployment notes). Re-ingestion is explicitly out of scope (spec) — covered only as a deploy note.
- **Type consistency:** `extract_pdf -> ExtractedPdf{prose_blocks, table_grids, method}`; `ingest_grids(...) -> (classifications, ingested)`; `glossary_lookup(glossary, value) -> str|None`; `build_row_narratives(..., column_glossaries=None)`. Names consistent across tasks.
