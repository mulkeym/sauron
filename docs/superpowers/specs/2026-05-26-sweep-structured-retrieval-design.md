# Structured Retrieval in SWEEP (and shared SQL core)

## Overview

Make the structured (DuckDB) store and the row narratives usable by the **SWEEP** strategy — not just `ANALYTICAL` — so that the common path for spreadsheet questions can blend **exact SQL answers** with the existing document RAG. A cheap, no-LLM relevance gate decides per query whether any ACL-visible registered table is relevant; if so, a shared structured retriever returns exact SQL rows **and** top-k row-narrative chunks. Everything merges into one answer. Strictly additive and fail-open.

## Problem

The tabular pipeline (DuckDB tables + registered schemas + row narratives + hardened text-to-SQL) is live, but its payoff is gated entirely behind the `ANALYTICAL` query type:

- `SWEEP` → `retrieve_sweep` + `retrieve_map_reduce` — vector RAG over text chunks; **never touches DuckDB**.
- `LOOKUP`/`TEMPORAL` → vector RAG — also no DuckDB.
- `ANALYTICAL` → `retrieve_analytical` → DuckDB SQL.

Two consequences, both confirmed in code/testing (2026-05-26):
1. **The exact-SQL capability is bypassed for most real queries.** The original 2600s incident query ("pay scales for an engineer… Tampa") classifies as **SWEEP**, and the user expects most traffic to be sweeps. So as built, the very questions we set out to answer never reach SQL.
2. **The row narratives are inert.** They are stored at `chunk_size_tier="table_row"`, but every strategy searches only `summary`/`xlarge`/`large`/`medium` tiers — `grep table_row` across `src/agent` and `src/retrieval` returns nothing. The narratives are embedded but never retrieved.

`SWEEP` is in fact the *ideal* place for SQL: exhaustive/aggregate questions ("all", "every", "total", "average across localities") are exactly `GROUP BY`/`SUM`/`WHERE` — exact and instant in SQL, slow and lossy in map-reduce over number grids.

## Solution: a gated structured branch in SWEEP + a shared SQL core

`SWEEP` already runs multiple retrievers in parallel and merges. We add a third, **gated** branch that contributes structured results, and we make the narratives retrievable through it.

### Component 1 — Shared SQL core: `structured_sql_rows`

Extract the SQL logic currently inside `retrieve_analytical` into a reusable function so both `ANALYTICAL` and the SWEEP branch use one implementation (DRY):

```
structured_sql_rows(question: str, schemas: list[TableSchema]) -> list[dict]
```
- Opens `connect_tabular(read_only=True)`, builds the value-enriched prompt via `schema_prompt_with_values(schemas, con)`, calls `generate` for SQL, and runs it through `execute_duckdb_sql(con, sql, allowed_tables={s.table for s in schemas})`.
- Runs entirely inside one connection, off the event loop (`asyncio.to_thread`).
- Raises on any failure (LLM error, blocked/empty SQL, execution error); callers decide fallback.

`retrieve_analytical` is refactored to call this core (its no-schemas / error path still falls back to map-reduce, unchanged in behavior).

### Component 2 — Cheap relevance gate: `tables_relevant_to`

```
tables_relevant_to(question: str, schemas: list[TableSchema], threshold: float) -> list[TableSchema]
```
- **No LLM.** Embeds the question (`embed_query`) and compares (cosine) against an embedding of each table's `description + column names`; keeps tables scoring above a **permissive** threshold.
- Bias toward inclusion: a false positive costs one SQL attempt (which fails open); a false negative silently loses the SQL benefit. Threshold is a tunable constant.
- Operates only on the **ACL-filtered** schema list (`schema_registry.list_for_user(user_groups)`), so the gate can never surface another group's table.

### Component 3 — Structured retriever: `retrieve_structured`

```
retrieve_structured(question, user_groups, schema_registry, vector_store) -> dict
   # {"sql_results": list[dict], "retrieved_chunks": list[RetrievedChunk]}
```
- `schemas = schema_registry.list_for_user(user_groups)`; `relevant = tables_relevant_to(question, schemas)`.
- If no relevant tables → returns empty (`{}`), the sweep proceeds RAG-only.
- Otherwise, concurrently:
  - `sql_results = structured_sql_rows(question, relevant)` (fail-open → `[]` on error),
  - `retrieved_chunks` = top-k vector search over `tier="table_row"` (ACL-filtered; scoped to the relevant tables' doc_ids), turning the inert narratives into retrieved context.
- Fail-open throughout: any error yields whatever succeeded (possibly empty), never raises.

### Component 4 — Wire into the SWEEP branch (`graph.py`)

In the `QueryType.SWEEP` branch, add `retrieve_structured(...)` to the existing `gather(retrieve_sweep, retrieve_map_reduce)` (the retrieve node already closes over `schema_registry`). Merge its `retrieved_chunks` into the deduplicated chunk set and attach `sql_results` to the result dict. The synthesizer already formats `sql_results` ("…SQL Database Results…") alongside chunks, so exact pay numbers and RAG context appear in one answer.

### Component 5 — Narratives become retrievable

No storage change. `retrieve_structured` searches `tier="table_row"` (the tier Plan 2c already wrote to), so the row narratives are finally pulled into results — for the fuzzy questions where SQL is hard.

## Data flow

```
classify (table-aware) → SWEEP:
    gather(
        retrieve_sweep,           # existing RAG chunks
        retrieve_map_reduce,      # existing per-doc extraction (Phase-0 bounded)
        retrieve_structured,      # NEW, gated: SQL rows + table_row narratives
    )
    → merge retrieved_chunks (dedup) + attach sql_results
  ‖ enrich_with_graph (KG, parallel)
    → merge → synthesize (renders sql_results + chunks)
```

## Error handling

Strictly additive / fail-open — a structured miss can never break or slow-fail the sweep:
| Failure | Handling |
|---|---|
| Gate (embedding) errors | Treat as "no relevant table" → RAG-only sweep |
| `structured_sql_rows` raises (LLM/blocked/exec) | `sql_results = []`; narratives + RAG still returned |
| `table_row` narrative search errors | Skipped; SQL rows + RAG still returned |
| No registered tables / none relevant | `retrieve_structured` returns empty; sweep unchanged |
| ACL | Gate + SQL allowlist both operate on `list_for_user(user_groups)` only |

## Testing

- **Unit:** `tables_relevant_to` (a pay question matches the pay table; an unrelated question returns none); `structured_sql_rows` (mock `generate` + temp DuckDB → rows; raises on bad SQL); `retrieve_structured` (gate fires → merges SQL rows + narrative chunks; gate misses → empty; SQL raises → narratives still returned, no exception).
- **Integration:** SWEEP branch invokes the structured task **only when** the gate fires, and merges `sql_results` + narrative chunks with sweep/map-reduce output; SWEEP with no relevant table behaves exactly as today.
- **Regression:** `retrieve_analytical` still works through the shared `structured_sql_rows` core; existing sweep/map_reduce/classifier/analytical tests unchanged.

## Scope / explicitly deferred

In scope: gated structured branch in SWEEP, shared SQL core, retrievable narratives, fail-open merge.

Deferred (separate follow-ups, surfaced during testing):
- **Locality code→name mapping** (cryptic `locname` codes like `RUS`); the value-in-prompt fix helps but full natural-language locality resolution needs a reference table.
- **Multi-table disambiguation** (the model sometimes picks the LEO table for a "GS-12" question).
- **SQL-generation non-determinism / validate-and-repair** on the quantized local model.
- Extending structured retrieval to `LOOKUP`/`CROSS_REFERENCE` (Approach B's fuller cross-strategy layer) — natural future growth; not needed now.
- Classifier routing changes — SWEEP stays SWEEP; this design enriches it rather than re-routing.
