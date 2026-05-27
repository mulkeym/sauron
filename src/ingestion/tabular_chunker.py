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
    SheetGrid, SheetClassification, infer_column_dtypes, _cell_kind,
    COLUMN_CONSISTENCY, MIN_DATA_ROWS, MAX_HEADER_SCAN, MIN_HEADER_CELLS,
    HEADER_MIN_FILLED, HEADER_MAX_NUMERIC,
)
from src.ingestion.table_profiler import _heuristic_profile, build_row_narratives
from src.ingestion.tabular_store import _safe_column_names


def _fmt_row(row: list) -> str:
    """Render one grid row as pipe-joined text (None -> empty cell)."""
    return " | ".join("" if c is None else str(c) for c in row)


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


def _looks_like_header(row: list) -> bool:
    """Per-row header test, identical to the criteria ``detect_header_row`` uses:
    at least ``MIN_HEADER_CELLS`` non-empty cells (so a lone title banner is
    skipped), mostly filled, and mostly non-numeric."""
    kinds = [_cell_kind(c) for c in row]
    filled = [k for k in kinds if k != "empty"]
    if not kinds or len(filled) < MIN_HEADER_CELLS:
        return False
    numeric_ratio = sum(1 for k in filled if k == "number") / len(filled)
    return len(filled) / len(kinds) >= HEADER_MIN_FILLED and numeric_ratio <= HEADER_MAX_NUMERIC


def _consistent_run(rows: list, header_idx: int) -> int:
    """Greedily extend a data run beneath ``rows[header_idx]`` and return its
    exclusive end. A row joins the run only while it keeps EVERY column
    type-consistent (cumulative dominant-kind ratio >= ``COLUMN_CONSISTENCY``)
    and is structurally data-like (more than one cell, no wider than the header).
    The first conflicting-type or structurally-broken row bounds the run — this
    is what lets a clean block be carved out of a messy sheet whose later rows
    (totals, prose footnotes) would poison a whole-sheet consistency check."""
    ncols = len(rows[header_idx])
    counts = [{"number": 0, "text": 0} for _ in range(ncols)]
    end = header_idx + 1
    while end < len(rows):
        row = rows[end]
        if not (1 < len(row) <= ncols):
            break  # ragged-narrow note/banner or wider-than-header row bounds it
        trial = [dict(c) for c in counts]
        for col in range(ncols):
            kind = _cell_kind(row[col]) if col < len(row) else "empty"
            if kind != "empty":
                trial[col][kind] += 1
        consistent = all(
            (c["number"] + c["text"]) == 0
            or max(c["number"], c["text"]) / (c["number"] + c["text"]) >= COLUMN_CONSISTENCY
            for c in trial
        )
        if not consistent:
            break
        counts = trial
        end += 1
    return end


def find_table_region(rows: list) -> tuple[int, int] | None:
    """Locate the largest contiguous table-like block inside a sheet.

    Returns ``(header_row_index, data_end_exclusive)`` for the biggest run of
    type-consistent data rows beneath any header-like row, or None if no block
    has >= ``MIN_DATA_ROWS`` rows. Unlike clean-sheet classification (which
    judges a sheet whole, so one stray total/footnote row marks the entire sheet
    messy — and spreadsheet readers pad every row to the sheet width, hiding
    structural breaks), this scans candidate headers in the leading
    ``MAX_HEADER_SCAN`` rows and, for each, greedily grows the consistent run
    beneath it (see ``_consistent_run``), then keeps the largest. That carves a
    clean block out of an otherwise-messy sheet. A data row whose trailing empty
    cells were dropped by the reader stays in the region (its gap renders as
    "(not specified)" later); a lone single-cell banner/note bounds it. Stacked
    secondary tables beyond the largest block are deferred.
    """
    best: tuple[int, int] | None = None
    best_len = 0
    for header_idx in range(min(len(rows), MAX_HEADER_SCAN)):
        if header_idx + 1 >= len(rows) or not _looks_like_header(rows[header_idx]):
            continue
        end = _consistent_run(rows, header_idx)
        data_len = end - (header_idx + 1)
        if data_len >= MIN_DATA_ROWS and data_len > best_len:
            best, best_len = (header_idx, end), data_len
    return best


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
