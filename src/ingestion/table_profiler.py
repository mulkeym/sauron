"""Per-table profiling (one LLM call) and deterministic per-row narratives.

``profile_table`` labels a clean table's columns and tags keys vs. measures
with a single LLM call (falling back to a dtype heuristic on any failure).
``build_row_narratives`` then restates each row as a natural-language string
deterministically — no per-row LLM calls — so the embeddings carry semantic
signal while the raw row stays the source of truth.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class TableProfile:
    column_descriptions: dict = field(default_factory=dict)  # safe_col_name -> label
    key_columns: list = field(default_factory=list)
    measure_columns: list = field(default_factory=list)
    table_description: str = ""


def _heuristic_profile(col_names: list[str], column_dtypes: list[str]) -> TableProfile:
    """LLM-free profile: number columns are measures, the rest are keys.

    Used as the fallback whenever the LLM profiling call fails or returns
    unusable output, so ingestion never breaks on a profiling error.
    """
    key_columns = [c for c, dt in zip(col_names, column_dtypes) if dt != "number"]
    measure_columns = [c for c, dt in zip(col_names, column_dtypes) if dt == "number"]
    return TableProfile(
        column_descriptions={c: c for c in col_names},
        key_columns=key_columns,
        measure_columns=measure_columns,
        table_description="Table with columns: " + ", ".join(col_names),
    )
