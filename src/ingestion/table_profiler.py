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


_PROFILE_SYSTEM = (
    "You label spreadsheet columns. Given a table's column names and a few "
    "sample rows, return ONLY a JSON object with these keys:\n"
    '  "column_descriptions": object mapping each column name to a short human label,\n'
    '  "key_columns": array of the column names that identify a row (categories/dimensions),\n'
    '  "measure_columns": array of the numeric value columns,\n'
    '  "table_description": one sentence describing the table.\n'
    "Use ONLY the provided column names. Output JSON only, no prose."
)


def profile_table(sheet_name: str, col_names: list[str], column_dtypes: list[str],
                  sample_rows: list, generate_fn=None) -> TableProfile:
    """Profile a clean table with a single LLM call; fall back to a heuristic.

    ``generate_fn`` defaults to the real LLM client; tests inject a fake. Any
    failure (LLM error, unparseable output, missing keys) yields the dtype
    heuristic so ingestion never breaks. The returned profile's key/measure
    columns are filtered to names actually present in ``col_names``, and every
    column gets a description (defaulting to its own name).
    """
    try:
        if generate_fn is None:
            from src.generation.llm_client import generate as generate_fn
        from src.generation.llm_client import parse_json_response
        sample_text = "\n".join(" | ".join("" if c is None else str(c) for c in row)
                                for row in sample_rows[:5])
        raw = generate_fn(
            system_prompt=_PROFILE_SYSTEM,
            user_prompt=f"Sheet: {sheet_name}\nColumns: {col_names}\nSample rows:\n{sample_text}",
            temperature=0.0,
            max_tokens=1024,
        )
        data = parse_json_response(raw)
        valid = set(col_names)
        key_columns = [c for c in data.get("key_columns", []) if c in valid]
        measure_columns = [c for c in data.get("measure_columns", []) if c in valid]
        raw_desc = data.get("column_descriptions", {}) or {}
        descriptions = {c: str(raw_desc.get(c, c)) for c in col_names}
        table_description = str(data.get("table_description", "")) or (
            "Table with columns: " + ", ".join(col_names)
        )
        if not key_columns and not measure_columns:
            # LLM gave us nothing usable about structure -> heuristic
            return _heuristic_profile(col_names, column_dtypes)
        return TableProfile(
            column_descriptions=descriptions,
            key_columns=key_columns,
            measure_columns=measure_columns,
            table_description=table_description,
        )
    except Exception as e:
        logger.warning(f"Table profiling failed for '{sheet_name}', using heuristic: {e}")
        return _heuristic_profile(col_names, column_dtypes)


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


def _fmt_cell(value) -> str:
    """Render a cell for a narrative; missing values are explicit, never faked."""
    if value is None:
        return "(not specified)"
    s = str(value).strip()
    return s if s else "(not specified)"


def row_narrative(col_names: list[str], profile: TableProfile, row: list,
                  column_glossaries: dict | None = None) -> str:
    """One deterministic sentence for a row: keys as context, then measures.

    Uses the profile's column descriptions as human labels. Cells absent from
    the row (shorter row) render as "(not specified)" — nothing is fabricated.
    When ``column_glossaries`` is provided, key-column values are annotated with
    their glossary meaning (e.g. ``E-3 (Enlisted Member)``).
    """
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
    """One narrative string per data row, optionally prefixed with ``context``
    (e.g. the table description) so a retrieved narrative is self-describing.
    Pass ``column_glossaries`` to annotate key-column values with their meaning."""
    prefix = f"{context} — " if context else ""
    return [f"{prefix}{row_narrative(col_names, profile, row, column_glossaries)}"
            for row in data_rows]
