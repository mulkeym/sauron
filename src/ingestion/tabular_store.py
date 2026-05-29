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
from src.ingestion.table_profiler import glossary_lookup

DUCKDB_DATABASE = "spreadsheets"


def _referenced_tables(sql: str) -> set[str]:
    """Lowercased identifiers used as tables (after FROM/JOIN, including
    comma-joined tables like ``FROM a, b, c``).

    Best-effort lexical parse — combined with the connection's
    enable_external_access=False and the allowed_tables check in
    execute_duckdb_sql, it bounds which tables a query can read. Note: it does
    not resolve schema-qualified names (``schema.table``); spreadsheet tables
    all live in DuckDB's default ``main`` schema so this is not a concern here.
    """
    found = set(re.findall(r'\b(?:FROM|JOIN)\s+"?([A-Za-z_][A-Za-z0-9_]*)"?', sql, re.IGNORECASE))
    # Comma-joined tables: pull each FROM clause (up to the next major keyword)
    # and take the leading identifier of every comma-separated part.
    for clause in re.findall(
        r'\bFROM\s+(.*?)(?=\bWHERE\b|\bGROUP\b|\bORDER\b|\bHAVING\b|\bLIMIT\b|\bJOIN\b|\)|$)',
        sql, re.IGNORECASE | re.DOTALL,
    ):
        for part in clause.split(','):
            m = re.match(r'\s*"?([A-Za-z_][A-Za-z0-9_]*)"?', part)
            if m:
                found.add(m.group(1))
    return {m.lower() for m in found}


def _cte_names(sql: str) -> set[str]:
    """Lowercased CTE aliases defined via ``WITH <name> AS (`` so they count as
    allowed 'tables' when validating references."""
    return {
        m.lower()
        for m in re.findall(r'([A-Za-z_][A-Za-z0-9_]*)\s+AS\s*\(', sql, re.IGNORECASE)
    }


def duckdb_table_name(doc_id: str, sheet_name: str) -> str:
    """Deterministic, SQL-safe table name for one doc's sheet."""
    safe_doc = re.sub(r"[^0-9a-zA-Z]+", "_", str(doc_id)).strip("_")
    safe_sheet = re.sub(r"[^0-9a-zA-Z]+", "_", str(sheet_name)).strip("_")
    return f"doc_{safe_doc}_{safe_sheet}".lower()


def referenced_source_docs(sql: str, live_doc_ids: list[str]) -> list[str]:
    """Source document ids whose DuckDB tables the SQL references.

    Parses table names via ``_referenced_tables`` (minus CTE aliases) and matches
    each against the per-doc table-name prefix ``duckdb_table_name(doc_id, "")``.
    Order-stable by ``live_doc_ids``; deduped; referenced tables with no live doc
    (e.g. since-deleted docs, or non-doc names) are skipped. Pure — no I/O."""
    tables = _referenced_tables(sql) - _cte_names(sql)   # both already lowercased
    out: list[str] = []
    for doc_id in live_doc_ids:
        prefix = duckdb_table_name(doc_id, "")
        if doc_id not in out and any(t.startswith(prefix) for t in tables):
            out.append(doc_id)
    return out


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


_FORBIDDEN_SQL_TOKENS = (
    "read_csv", "read_parquet", "read_json", "read_text",
    "parquet_scan", "csv_scan", "attach", "install",
)


def execute_duckdb_sql(con, sql: str, allowed_tables=None) -> list[dict]:
    """Run a single read-only SELECT/WITH against a DuckDB connection.

    Guards (in addition to the connection's enable_external_access=False):
      - single statement only, must start with SELECT or WITH;
      - reject DuckDB file/extension functions (read_csv, attach, install, ...);
      - if ``allowed_tables`` is given, every FROM/JOIN identifier must be in it
        (plus any CTE alias defined in the query). This is the ACL boundary —
        callers pass the set of tables the user may read. ``None`` skips the
        check (trusted/internal callers only).

    Returns ``list[dict]``.
    """
    sql = sql.strip().rstrip(";")
    if ";" in sql:
        raise ValueError("Only a single SELECT statement is allowed")
    if not re.match(r"^\s*(SELECT|WITH)\b", sql, re.IGNORECASE):
        raise ValueError("Only SELECT queries are allowed")

    lowered = sql.lower()
    for tok in _FORBIDDEN_SQL_TOKENS:
        if re.search(r"\b" + re.escape(tok) + r"\b", lowered):
            raise ValueError(f"Disallowed SQL construct: {tok}")

    if allowed_tables is not None:
        allowed = {t.lower() for t in allowed_tables} | _cte_names(sql)
        extra = _referenced_tables(sql) - allowed
        if extra:
            raise ValueError(f"Query references tables outside the allowed set: {sorted(extra)}")

    cur = con.execute(sql)
    columns = [d[0] for d in cur.description]
    return [dict(zip(columns, row)) for row in cur.fetchall()]


def distinct_values(con, table: str, column: str, max_distinct: int = 100) -> list | None:
    """Distinct non-null values of a column, or None if it has more than
    ``max_distinct`` (i.e. it's high-cardinality, not a useful categorical hint).

    ``table``/``column`` are our own sanitized identifiers, so the f-string is
    safe. Used to show the LLM the actual category codes (e.g. locality codes)
    so it filters on real values instead of guessing human phrasings.
    """
    rows = con.execute(
        f'SELECT DISTINCT "{column}" FROM "{table}" '
        f'WHERE "{column}" IS NOT NULL ORDER BY "{column}" LIMIT {int(max_distinct) + 1}'
    ).fetchall()
    if len(rows) > max_distinct:
        return None
    return [r[0] for r in rows]


def schema_prompt_with_values(schemas, con, max_distinct: int = 100, hints=None) -> str:
    """Render registered schemas for the text-to-SQL prompt, appending the
    distinct values of low-cardinality VARCHAR columns so the LLM filters on
    real category codes (the gap that made 'Rest of U.S.' match 0 rows).

    ``hints`` (optional) maps table name -> ResolvedHints; when present its
    glossaries annotate the values list (``CODE (meaning)``), column notes append
    to the column description, and table notes render as a ``Notes:`` line. With
    ``hints=None`` (or no entry for a table) output is byte-identical to before.
    """
    parts = []
    for s in schemas:
        th = (hints or {}).get(s.table)
        glossaries = th.column_glossaries if th else {}
        col_notes = th.column_notes if th else {}
        col_lines = []
        for c in s.columns:
            desc = c.description
            if c.name in col_notes:
                desc = f"{desc} — {col_notes[c.name]}" if desc else col_notes[c.name]
            line = f"  - {c.name} ({c.dtype}): {desc}"
            if c.dtype == "VARCHAR":
                vals = distinct_values(con, s.table, c.name, max_distinct)
                if vals:
                    gloss = glossaries.get(c.name, {})
                    rendered = []
                    for v in vals:
                        meaning = glossary_lookup(gloss, v) if gloss else None
                        rendered.append(f"{v} ({meaning})" if meaning else str(v))
                    line += " | values: " + ", ".join(rendered)
            col_lines.append(line)
        header = f"Table: {s.table}\nDescription: {s.description}"
        if th and th.table_notes:
            header += "\nNotes: " + "; ".join(th.table_notes)
        parts.append(header + "\nColumns:\n" + "\n".join(col_lines))
    return "\n\n".join(parts)


def schema_context_for_synthesis(schemas, hints=None) -> str:
    """Compact description of the queried table(s) for the *answer* LLM: column
    meanings plus the value-code glossary, so it can interpret SQL result rows
    that carry only raw column keys/codes (e.g. read a ``locname='RUS'`` row as
    "Rest of U.S.", or know ``annual1`` is a step-1 salary).

    Unlike ``schema_prompt_with_values`` (the text-to-SQL prompt) this needs no
    DB connection and omits the exhaustive distinct-values dump — the result
    rows already contain the actual values; what the synthesizer lacks is their
    *meaning*. ``hints`` maps table name -> ResolvedHints (optional).
    """
    parts = []
    for s in schemas:
        th = (hints or {}).get(s.table)
        glossaries = th.column_glossaries if th else {}
        col_notes = th.column_notes if th else {}
        col_lines = []
        for c in s.columns:
            desc = c.description
            if c.name in col_notes:
                desc = f"{desc} — {col_notes[c.name]}" if desc else col_notes[c.name]
            col_lines.append(f"  - {c.name} ({c.dtype}): {desc}" if desc
                             else f"  - {c.name} ({c.dtype})")
        header = f"Table {s.table} — {s.description}"
        if th and th.table_notes:
            header += "\nNotes: " + "; ".join(th.table_notes)
        block = header + "\nColumns:\n" + "\n".join(col_lines)
        gloss_lines = [
            f"  - {col}: " + ", ".join(f"{code} = {meaning}" for code, meaning in mapping.items())
            for col, mapping in glossaries.items() if mapping
        ]
        if gloss_lines:
            block += "\nValue meanings:\n" + "\n".join(gloss_lines)
        parts.append(block)
    return "\n\n".join(parts)


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
    # enable_external_access=False blocks read_csv/read_parquet/ATTACH/COPY/httpfs,
    # so untrusted (LLM-generated) SELECTs cannot reach the filesystem or network.
    return duckdb.connect(path, read_only=read_only, config={"enable_external_access": False})


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
