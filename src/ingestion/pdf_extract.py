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
from tempfile import SpooledTemporaryFile

from src.ingestion.tabular import SheetGrid

logger = logging.getLogger(__name__)


@dataclass
class ProseBlock:
    text: str
    page: int
    bbox: tuple[float, float, float, float] | None = None
    font_size: float | None = None
    content_type: str = "text"
    figure_id: str = ""


@dataclass
class ExtractedPdf:
    prose_blocks: list[ProseBlock] = field(default_factory=list)
    table_grids: list[SheetGrid] = field(default_factory=list)
    method: str = "digital"          # "digital" | "ocr" | "mixed"
    # Positioned digital text lines used to place figures between surrounding
    # prose. The public prose stream remains page-sized until enrichment.
    layout_blocks: list[ProseBlock] = field(default_factory=list)
    figure_records: list = field(default_factory=list)


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


def _page_tables(page, page_no: int) -> tuple[list[SheetGrid], list[tuple]]:
    """Extract tables once and keep their bounds for prose/figure placement."""
    grids: list[SheetGrid] = []
    bboxes: list[tuple] = []
    tables = page.find_tables() or []
    if not tables:
        tables = page.find_tables(table_settings=_TABLE_SETTINGS) or []
    for n, table in enumerate(tables):
        raw = table.extract()
        g = normalize_grid(raw, sheet_name=f"p{page_no}_table{n}")
        if g.rows:
            # Check integrity on the RAW table (before normalize pads it), so a
            # sheared/mis-aligned extraction is visible rather than silently
            # rectangularized. Fail-open: still ingest; classify_sheet routes it.
            if not grid_width_consistent(SheetGrid(sheet_name=g.sheet_name, rows=raw)):
                logger.warning(
                    "PDF table %s has inconsistent row widths (possible "
                    "mis-extraction); ingesting anyway", g.sheet_name)
            grids.append(g)
            bboxes.append(table.bbox)
    return grids, bboxes


def _page_layout_lines(page, page_no: int, table_bboxes: list[tuple]) -> list[ProseBlock]:
    """Positioned text lines outside table regions, in page reading order."""
    lines: list[ProseBlock] = []
    try:
        raw_lines = page.extract_text_lines(return_chars=True) or []
    except MemoryError:
        raise
    except Exception:
        raw_lines = []
    for line in raw_lines:
        text = (line.get("text") or "").strip()
        if not text:
            continue
        x0 = float(line.get("x0", 0) or 0)
        top = float(line.get("top", 0) or 0)
        x1 = float(line.get("x1", x0) or x0)
        bottom = float(line.get("bottom", top) or top)
        inside_table = False
        for (bx0, btop, bx1, bbot) in table_bboxes:
            cx, cy = (x0 + x1) / 2, (top + bottom) / 2
            if bx0 <= cx <= bx1 and btop <= cy <= bbot:
                inside_table = True
                break
        if inside_table:
            continue
        chars = line.get("chars") or []
        sizes = [float(c.get("size", 0) or 0) for c in chars if c.get("size")]
        font_size = max(sizes) if sizes else None
        lines.append(ProseBlock(
            text=text, page=page_no, bbox=(x0, top, x1, bottom),
            font_size=font_size,
        ))
    lines.sort(key=lambda b: (
        b.bbox[1] if b.bbox else 0,
        b.bbox[0] if b.bbox else 0,
    ))
    return lines


def _page_prose(page, table_bboxes: list[tuple]) -> str:
    """Fallback text using the table bounds already detected on this page."""
    if not table_bboxes:
        return page.extract_text() or ""

    def outside_tables(obj):
        x0, top = obj.get("x0", 0), obj.get("top", 0)
        return not any(
            bx0 <= x0 <= bx1 and btop <= top <= bbot
            for (bx0, btop, bx1, bbot) in table_bboxes
        )

    filtered = page.filter(outside_tables)
    try:
        return filtered.extract_text() or ""
    finally:
        filtered.close()


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
    """OCR a one-page PDF; Unstructured does not support page_numbers."""
    from pypdf import PdfReader, PdfWriter
    from unstructured.partition.pdf import partition_pdf

    # Spill large pages to an anonymous temporary file. Closing it (or a
    # killed worker exiting) removes the file without leaving a PDF behind.
    with SpooledTemporaryFile(max_size=1024 * 1024) as single_page:
        # A reader caches resolved objects, including image streams. Close it
        # per page before OCR so those resources do not accumulate across pages.
        with path.open("rb") as source, PdfReader(source) as reader, PdfWriter() as writer:
            # Article threads can reference other pages; OCR needs this page only.
            writer.add_page(reader.pages[page_no], excluded_keys=["/B"])
            writer.write(single_page)
        del writer
        single_page.seek(0)
        return partition_pdf(
            file=single_page, strategy="hi_res", infer_table_structure=True,
            starting_page_number=page_no + 1,
        )


def _extract_scanned_page(path: Path, page_no: int):
    blocks: list[ProseBlock] = []
    grids: list[SheetGrid] = []
    try:
        elements = _partition_scanned(path, page_no)
    except MemoryError:
        raise
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


def extract_pdf(path: Path) -> ExtractedPdf:
    """Single entry point. Per-page triage: digital pages via pdfplumber, scanned
    pages via OCR. Raises on hard failure (caller is fail-open)."""
    import pdfplumber

    prose: list[ProseBlock] = []
    layout: list[ProseBlock] = []
    grids: list[SheetGrid] = []
    methods: set[str] = set()
    with pdfplumber.open(str(path)) as pdf:
        for i, page in enumerate(pdf.pages):
            try:
                text = page.extract_text() or ""
                if len(text.strip()) >= DIGITAL_MIN_CHARS:
                    methods.add("digital")
                    page_grids, table_bboxes = _page_tables(page, i)
                    grids.extend(page_grids)
                    page_layout = _page_layout_lines(page, i, table_bboxes)
                    layout.extend(page_layout)
                    body = "\n".join(block.text for block in page_layout).strip()
                    if not body:
                        body = _page_prose(page, table_bboxes).strip()
                    if body:
                        prose.append(ProseBlock(text=body, page=i))
                else:
                    methods.add("ocr")
                    # Triage is done; release pdfplumber caches before loading OCR.
                    page.close()
                    p_blocks, p_grids = _extract_scanned_page(path, i)
                    prose.extend(p_blocks)
                    grids.extend(p_grids)
            finally:
                page.close()

    method = "mixed" if len(methods) > 1 else (methods.pop() if methods else "digital")
    stitched = stitch_tables(grids)
    return ExtractedPdf(
        prose_blocks=prose, table_grids=stitched, method=method,
        layout_blocks=layout,
    )
