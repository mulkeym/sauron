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
    fully-empty rows and fully-empty columns."""
    rows = [["" if c is None else str(c).strip() for c in row] for row in raw_rows]
    rows = [r for r in rows if any(c != "" for c in r)]
    if not rows:
        return SheetGrid(sheet_name=sheet_name, rows=[])
    width = max(len(r) for r in rows)
    rows = [r + [""] * (width - len(r)) for r in rows]   # pad ragged rows
    keep = [i for i in range(width) if any(r[i] != "" for r in rows)]
    rows = [[r[i] for i in keep] for r in rows]
    return SheetGrid(sheet_name=sheet_name, rows=rows)


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


def grid_width_consistent(grid: SheetGrid) -> bool:
    """True if every data row has the same cell count as the header row. A
    mismatch signals a sheared/misaligned extraction; such a grid should be
    demoted to messy-region narratives rather than loaded as a DuckDB table."""
    if len(grid.rows) < 2:
        return True
    header_w = len(grid.rows[0])
    return all(len(r) == header_w for r in grid.rows[1:])


DIGITAL_MIN_CHARS = 20   # a page with fewer extractable chars is treated as scanned

# Spike confirmed default lattice works for ruled grids; text strategy is the
# fallback for borderless tables.
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


def _extract_scanned_page(path: Path, page_no: int):
    """OCR a single scanned page. Implemented in Cycle 2."""
    return [], []


def extract_pdf(path: Path) -> ExtractedPdf:
    """Single entry point. Per-page triage: digital pages via pdfplumber, scanned
    pages via OCR. Raises on hard failure (caller is fail-open)."""
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
                p_blocks, p_grids = _extract_scanned_page(path, i)
                prose.extend(p_blocks)
                grids.extend(p_grids)

    method = "mixed" if len(methods) > 1 else (methods.pop() if methods else "digital")
    stitched = stitch_tables(grids)
    return ExtractedPdf(prose_blocks=prose, table_grids=stitched, method=method)
