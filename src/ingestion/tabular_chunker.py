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
