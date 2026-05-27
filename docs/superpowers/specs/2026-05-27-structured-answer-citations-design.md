# Citations for Structured/SQL Answers — Design

**Date:** 2026-05-27
**Status:** Approved (brainstorming) — pending spec review before implementation planning

## Goal

When an answer draws on SQL results, emit a `Citation` to the **source document of each
table the executed SQL actually referenced**. This closes the gap where ANALYTICAL answers
(which return `retrieved_chunks: []` and only `sql_results`) carry **zero citations** — the
user gets numbers with no provenance, and can't tell which document/year they came from.

## Background

- Citations are built **only from `retrieved_chunks`** in `synthesize_answer`
  (`src/agent/synthesizer.py:162-205`): one deduped `Citation` per real (non-synthetic)
  doc, plus a `source_url` lookup via the metadata store. `sql_results` are added to the LLM
  context (line 148) but contribute no citations.
- The **ANALYTICAL** path (`retrieve_analytical`) returns `{retrieved_chunks: [], sql_results,
  structured_trace, ...}` → the citation loop sees no chunks → **no citations at all**.
  CROSS_REFERENCE's SQL portion has the same gap when no chunks back it.
- **SWEEP** usually still cites the source doc, because `retrieve_structured` also returns
  `table_row` narrative chunks whose `metadata.doc_id` is the spreadsheet's doc — those
  produce citations through the normal chunk path. So the primary gap is the pure-SQL paths.
- **Table → document link.** Neither the in-memory `TableSchema` nor the persisted
  `RegisteredSchema` carries a `doc_id`. The only link is the table **name**:
  `duckdb_table_name(doc_id, sheet)` → `doc_<safe_doc_id>_<sheet>`. The codebase already
  reverse-matches by prefix (`cleanup_spreadsheet_tables`, `purge_orphan_schemas`): for a
  table name, the owning doc is the live document whose `duckdb_table_name(doc_id, "")`
  prefix the table name starts with.
- The executed SQL and its row count are available on `state["structured_trace"]`
  (`{sql, status, row_count, ...}`, added by the Structured Lookup feature). `_referenced_tables(sql)`
  in `tabular_store.py` already parses table names out of SQL (used for the allowlist).

## Trigger

In `synthesize_answer`, add SQL-source citations only when `state.get("structured_trace")`
has `status == "ran"` **and** `row_count > 0` — i.e., SQL genuinely contributed to the
answer. For `skipped` / `error` / 0-row traces, add nothing (the answer there comes from
fallback chunks, which already cite themselves).

## Component 1 — `referenced_source_docs` (pure resolver)

New pure function in `src/ingestion/tabular_store.py` (beside the existing table utilities):

```python
def referenced_source_docs(sql: str, live_doc_ids: list[str]) -> list[str]:
    """Map an executed SQL statement to the source document ids whose tables it
    references. Parses table names via _referenced_tables(sql); for each, returns
    the live doc whose ``duckdb_table_name(doc_id, "")`` prefix the table name
    starts with. Deduped, order-stable. Tables with no matching live doc (e.g.
    CTE names, or docs since deleted) are skipped. Pure — no I/O."""
```

- Uses the existing `_referenced_tables(sql)` (already excludes CTE names via `_cte_names`)
  so aliases/CTEs are not mis-resolved.
- For each referenced table, computes `prefix = duckdb_table_name(doc_id, "")` for each
  `live_doc_ids` entry and matches `table.startswith(prefix)`; the matching `doc_id` is
  added (deduped).
- Returns `list[str]` of doc_ids in first-seen order.

## Component 2 — SQL-source citations in the synthesizer

In `synthesize_answer`, after the existing chunk-citation construction:

- Read `trace = state.get("structured_trace")`. If the trigger holds, resolve the source
  docs and build citations, all inside a single `try/except` (fail-open).
- **Reuse the existing async metadata helper** (the block that already runs an async
  function to fetch `source_url` per doc, lines 174-192). Extend that path to also call
  `metadata_store.list_documents()` (for the live doc-id list) and `get_document(doc_id)`
  (for filename/doc_type). Resolution = `referenced_source_docs(trace["sql"], live_doc_ids)`.
- For each resolved doc not already present in the chunk-derived citations (dedup by
  `doc_id`), append:

```python
Citation(
    doc_id=doc_id,
    filename=doc_rec.filename,
    doc_type=getattr(doc_rec, "doc_type", "") or "",
    chunk_index=0,
    page=None,
    snippet=f"Structured query returned {row_count} rows from this table.",
    relevance=1.0,
)
```

- **Dedup/merge by `doc_id`:** if a doc is already cited from a narrative chunk (SWEEP),
  keep the existing citation; SQL citations only *add* docs not already represented.

## Data flow

`structured_trace` (already on `AgentState`, carries `sql`/`status`/`row_count`) → synthesizer
trigger check → `referenced_source_docs(sql, live_doc_ids)` → metadata-store lookup for
filename/doc_type → `Citation` objects appended to the existing citation list (deduped). No
change to retrieval, the answer text, or the `Citation` model.

## Error handling

Fully fail-open and additive. The SQL-citation block is wrapped in `try/except`; any failure
(SQL parse error, no `structured_trace`, metadata-store error, deleted doc) results in no SQL
citations and leaves the answer and chunk-derived citations untouched. Mirrors the existing
`source_url` lookup's defensive style (`logger.debug` on failure).

## Testing (TDD, pytest)

- **`referenced_source_docs`** (`tests/test_ingestion/test_tabular_store*.py` or a new test):
  - SQL `SELECT ... FROM doc_<live1>_pay` with `live_doc_ids=["live1","other"]` → `["live1"]`.
  - SQL joining two doc tables → both doc_ids, order-stable.
  - SQL referencing a table whose doc isn't in `live_doc_ids` → `[]` (skipped).
  - SQL with a CTE (`WITH x AS (...) SELECT ... FROM doc_<live1>_pay`) → `["live1"]` only
    (CTE name not mis-resolved).
- **Synthesizer** (`tests/test_agent/test_synthesizer.py`):
  - `structured_trace` status="ran", row_count=15, sql referencing a doc table, `chunks=[]`,
    `sql_results` non-empty → citations contain that doc with the summary snippet and
    relevance 1.0. (metadata store mocked to return a doc record + list_documents.)
  - same doc also present as a narrative chunk → single deduped citation (no double).
  - trace status in {skipped, error} or row_count==0 → no SQL citation added.

## Out of scope / deferred

- Multi-table/year disambiguation (the 3 indistinguishable `all_gs` tables) — a separate
  retrieval-quality concern; this feature *exposes* which table answered (good) but does not
  fix which one gets picked.
- Per-row / cell-level citation granularity — citation is at the document level.
- Putting the SQL text in the citation snippet (decided: short summary only; the SQL is
  already shown in the playground Structured Lookup step).
