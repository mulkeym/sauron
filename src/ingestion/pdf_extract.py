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
