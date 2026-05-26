"""Structured (DuckDB) storage for clean spreadsheet tables.

Turns a Plan-1 ``SheetGrid`` + ``SheetClassification`` into a typed DuckDB
table and a registrable ``TableSchema``, and runs SELECT-only SQL against a
DuckDB connection. No LLM, no embeddings — that is Plan 2b.
"""
from __future__ import annotations

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
