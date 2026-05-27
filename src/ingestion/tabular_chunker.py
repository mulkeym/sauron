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


def find_table_region(rows: list) -> tuple[int, int] | None:
    """Locate the largest contiguous table-like block inside a sheet.

    Returns ``(header_row_index, data_end_exclusive)`` for the run of rows
    directly beneath a detected header that look like data for it and are
    column-type-consistent (same thresholds as clean-sheet classification), or
    None if no qualifying block exists (too few data rows, no header, or
    inconsistent columns). Detection is modest by design: it finds the first
    header in the leading rows and the contiguous data run beneath it — stacked
    secondary tables are deferred. A row counts as data while it has more than
    one cell and no more than the header's column count; spreadsheet readers
    drop trailing empty cells, so a data row with a missing last value arrives
    shorter than the header — it stays in the region (its gap renders as "(not
    specified)" later), whereas a lone single-cell banner/note bounds it.
    """
    header_idx = detect_header_row(rows)
    if header_idx < 0:
        return None
    ncols = len(rows[header_idx])
    end = header_idx + 1
    while end < len(rows) and 1 < len(rows[end]) <= ncols:
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
