"""Structured (DuckDB) storage for clean spreadsheet tables.

Turns a Plan-1 ``SheetGrid`` + ``SheetClassification`` into a typed DuckDB
table and a registrable ``TableSchema``, and runs SELECT-only SQL against a
DuckDB connection. No LLM, no embeddings — that is Plan 2b.
"""
from __future__ import annotations

import math
import re

from src.db.schema_registry import ColumnSchema, TableSchema
from src.ingestion.tabular import SheetClassification, SheetGrid

DUCKDB_DATABASE = "spreadsheets"


def duckdb_table_name(doc_id: str, sheet_name: str) -> str:
    """Deterministic, SQL-safe table name for one doc's sheet."""
    safe_doc = re.sub(r"[^0-9a-zA-Z]+", "_", str(doc_id)).strip("_")
    safe_sheet = re.sub(r"[^0-9a-zA-Z]+", "_", str(sheet_name)).strip("_")
    return f"doc_{safe_doc}_{safe_sheet}".lower()


def _safe_column_names(header: list) -> list[str]:
    """SQL-safe, unique column identifiers from a header row.

    Non-alphanumerics collapse to underscores; blanks become ``col_<i>``;
    digit-leading names get a ``c_`` prefix; collisions get a numeric suffix.
    """
    names: list[str] = []
    for i, h in enumerate(header):
        base = re.sub(r"[^0-9a-zA-Z]+", "_", str(h).strip()).strip("_").lower()
        if not base:
            base = f"col_{i}"
        if base[0].isdigit():
            base = f"c_{base}"
        name = base
        suffix = 0
        while name in names:
            suffix += 1
            name = f"{base}_{suffix}"
        names.append(name)
    return names


def _to_number(value) -> float | None:
    """Coerce a cell to float, or None if it isn't numeric.

    Handles native numbers and numeric strings with $, comma, % formatting.
    Bools are treated as non-numeric (consistent with tabular._cell_kind).
    """
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    cleaned = str(value).replace("$", "").replace(",", "").replace("%", "").strip()
    if not cleaned:
        return None
    try:
        result = float(cleaned)
    except ValueError:
        return None
    if math.isnan(result) or math.isinf(result):
        return None  # "nan"/"inf" would silently corrupt DuckDB aggregates
    return result


def load_sheet_to_duckdb(con, doc_id: str, sheet_name: str,
                         classification: SheetClassification, grid: SheetGrid) -> tuple[str, list[str]]:
    """Create a typed DuckDB table for a clean sheet and insert its data rows.

    ``number`` columns become DOUBLE (cells coerced via ``_to_number``); all
    other columns become VARCHAR. Returns ``(table_name, column_names)``.
    Replaces any existing table with the same name (re-ingest is idempotent).
    """
    header_idx = classification.header_row_index
    header = grid.rows[header_idx]
    col_names = _safe_column_names(header)
    dtypes = classification.column_dtypes
    if len(col_names) != len(dtypes):
        raise ValueError(
            f"Sheet '{sheet_name}': header has {len(col_names)} columns but "
            f"classification has {len(dtypes)} dtypes"
        )
    table = duckdb_table_name(doc_id, sheet_name)

    col_defs = ", ".join(
        f'"{name}" {"DOUBLE" if dt == "number" else "VARCHAR"}'
        for name, dt in zip(col_names, dtypes)
    )
    con.execute(f'DROP TABLE IF EXISTS "{table}"')
    con.execute(f'CREATE TABLE "{table}" ({col_defs})')

    ncols = len(col_names)
    placeholders = ", ".join(["?"] * ncols)
    for row in grid.rows[header_idx + 1:]:
        values = []
        for c in range(ncols):
            raw = row[c] if c < len(row) else None
            if dtypes[c] == "number":
                values.append(_to_number(raw))
            else:
                values.append(None if raw is None else str(raw))
        con.execute(f'INSERT INTO "{table}" VALUES ({placeholders})', values)
    return table, col_names


def execute_duckdb_sql(con, sql: str) -> list[dict]:
    """Run a single SELECT against a DuckDB connection; return list[dict].

    Mirrors the guardrails in src/db/sql_executor.py: a single statement only,
    and it must be a SELECT.

    NOTE: this guard does NOT sandbox DuckDB's file-reading table functions
    (read_csv/read_parquet/read_json/etc.). It is safe for trusted/internal SQL
    only; before exposing this to LLM- or user-generated SQL (Plan 3), add a
    table-name allowlist.
    """
    sql = sql.strip().rstrip(";")
    if ";" in sql:
        raise ValueError("Only a single SELECT statement is allowed")
    if not re.match(r"^\s*SELECT\b", sql, re.IGNORECASE):
        raise ValueError("Only SELECT queries are allowed")
    cur = con.execute(sql)
    columns = [d[0] for d in cur.description]
    return [dict(zip(columns, row)) for row in cur.fetchall()]


def connect_tabular(read_only: bool = False):
    """Open a connection to the persistent tabular DuckDB database.

    Path comes from ``settings.tabular_duckdb_path`` (on the mounted data
    volume). Ingestion opens read-write (short-lived); query-time opens
    read_only. The parent directory is created if missing.

    NOTE: DuckDB permits only one read-write connection to a file per process at
    a time. Concurrent ingestion (settings.max_parallel_ingestion > 1) can
    therefore contend on the write lock; keep it at 1 while tabular ingest is
    enabled, or serialize writers. (A failed connect is fail-open: the caller
    logs a warning and skips structured ingest for that document.)
    """
    import os
    import duckdb
    from src.config import settings

    path = settings.tabular_duckdb_path
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    return duckdb.connect(path, read_only=read_only)


def schema_from_sheet(doc_id: str, sheet_name: str, classification: SheetClassification,
                      grid: SheetGrid, acl_groups: list[str] | None = None) -> TableSchema:
    """Build a registrable ``TableSchema`` from a clean sheet.

    Column names match ``load_sheet_to_duckdb``; dtypes map number->DOUBLE and
    everything else->VARCHAR; descriptions default to the original header text
    (Plan 2b's profiler replaces these with richer descriptions).
    """
    header = grid.rows[classification.header_row_index]
    col_names = _safe_column_names(header)
    if len(col_names) != len(classification.column_dtypes):
        raise ValueError(
            f"Sheet '{sheet_name}': header has {len(col_names)} columns but "
            f"classification has {len(classification.column_dtypes)} dtypes"
        )
    columns = [
        ColumnSchema(
            name=name,
            dtype="DOUBLE" if dt == "number" else "VARCHAR",
            description=str(orig).strip(),
        )
        for name, dt, orig in zip(col_names, classification.column_dtypes, header)
    ]
    return TableSchema(
        database=DUCKDB_DATABASE,
        table=duckdb_table_name(doc_id, sheet_name),
        columns=columns,
        description=f"Sheet '{sheet_name}' from document {doc_id}",
        acl_groups=acl_groups or [],
    )
