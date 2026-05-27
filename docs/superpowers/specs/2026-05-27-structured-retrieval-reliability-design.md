# Structured-Retrieval Reliability Design

**Date:** 2026-05-27
**Status:** Approved (brainstorming) — pending spec review before implementation planning

## Goal

A tabular/structured question (e.g. "What are the GS salary rates in Tampa?") returns
the correct, complete answer **regardless of which strategy classifies it**, and the
**same question classifies consistently across runs**.

Today it does neither:
- The same question classified `sweep` on one run and `analytical` on the next, at
  `temperature=0`.
- When it lands on `sweep`, the answer is often "no information" even though structured
  retrieval found the rows — because the synthesizer discards them.
- When it lands on `analytical`, it returns the correct rows.

This is two separable defects. We fix both in one spec, Phase 1 first.

## Background — observed evidence

From the live `api` logs (observability added in commit `3108f62`):

```
Classified 'What are the GS salary rates in Tampa?' -> analytical (tables_available=True)
Text-to-SQL for 'What are the GS salary rates in Tampa?'
  -> SELECT * FROM doc_ed511620..._all_gs WHERE locname = 'TU'
Text-to-SQL returned 15 row(s)
```

The SQL layer works: it maps Tampa→`TU` from the schema prompt's distinct values and
returns 15 rows (all GS grades). The earlier `sweep`/"no information" outcome was **not**
a SQL problem — it was the downstream synthesizer discarding retrieved data, combined with
run-to-run classification flips.

### Root-cause map (all confirmed in code)

**Defect B — SWEEP discards good results (`src/agent/synthesizer.py:104-117`):**
```python
SYNTHETIC_IDS = {"map-reduce", "knowledge-graph", "metadata-context"}
has_map_reduce = any(c.metadata.doc_id == "map-reduce" for c in chunks)
if has_map_reduce:
    regular_chunks = []   # drops EVERYTHING non-synthetic
```
In the SWEEP branch (`graph.py:41-111`), `retrieve_structured` contributes precise
`table_row` narratives (real `doc_id`s). Map-reduce almost always emits a synthetic
chunk (`doc_id="map-reduce"`), so `has_map_reduce` is true and **all** structured
narratives + raw sweep chunks are dropped. If map-reduce itself found nothing, the
synthesizer returns "no information." Note: `sql_results` is forwarded separately
(`graph.py:74-75`) and appended unconditionally (`synthesizer.py:140`), so exact SQL
rows survive — but the narratives do not, and in sweep runs where the structured gate
blocks SQL there is nothing left.

**Defect A — classification is non-deterministic:**
- The classifier prompt embeds the registered-table list (`classifier._classify_node_factory`
  → `format_available_tables`). Re-ingestion changes that list, and iteration order is
  not stabilized → different prompt input → different output.
- `classifier.classify_query` passes `temperature=0.0` but `llm_client._call_llm` sends
  **no `seed`** → not guaranteed deterministic even for identical input.
- The schema registry / DuckDB store is polluted with stale **test-fixture** tables
  (toy `Tampa/Boston/Denver` 3-row tables, `doc1_*`) whose `doc_id`s have no live
  document — these inflate and shift the table list.

(The `strategy_memory` subsystem logs which strategy worked per normalized question
pattern but `get_best_strategy()` is never read — out of scope here; see Deferred.)

## Phase 1 — Keep structured results in SWEEP

**File:** `src/agent/synthesizer.py`, function `synthesize_answer` (the `has_map_reduce`
branch, lines ~108-112).

**Change:** when `has_map_reduce` is true, do not empty `regular_chunks`. Keep the
structured `table_row` narratives (precise, compact, authoritative) and drop only the
bulky raw sweep chunks (tiers `small/medium/large/xlarge/summary`):

```python
if has_map_reduce:
    # Map-reduce distilled the prose docs; raw doc chunks are redundant.
    # Structured table_row narratives are precise and compact — keep them so
    # SWEEP never discards retrieved structured data.
    regular_chunks = sorted(
        [c for c in chunks
         if c.metadata.doc_id not in SYNTHETIC_IDS
         and c.metadata.chunk_size_tier == "table_row"],
        key=lambda c: c.score, reverse=True,
    )
    logger.info(
        "Synthesizer: map-reduce synthesis + %d structured narratives "
        "(raw sweep chunks skipped)", len(regular_chunks))
else:
    regular_chunks = sorted(
        [c for c in chunks if c.metadata.doc_id not in SYNTHETIC_IDS],
        key=lambda c: c.score, reverse=True,
    )
```

**Why this is safe against the flood problem** (the original symptom that started this
work): structured narratives are emitted right after the map-reduce synthesis and before
the `MAX_CONTEXT_CHARS` (`llm_max_context`, 200K) cap loop bites; the bulky raw chunks
stay dropped. `sql_results` appending (line 140) is unchanged. So precise data survives
without reintroducing the giant raw-chunk dumps.

**Discriminator:** `metadata.chunk_size_tier == "table_row"` is the existing tier for
all row/region narratives (clean rows + messy region narratives). Raw text chunks use
the other tiers, so this cleanly separates the two without new metadata.

## Phase 2 — Classification determinism hygiene

Three independent, low-risk changes.

### 2.1 Fixed LLM seed
**File:** `src/generation/llm_client.py::_call_llm`. Add `seed` to the request payload,
sourced from a new config setting `llm_seed` (default `0`). gpt-4.1-mini
and vLLM both honor `seed`. Additive; no behavior change beyond determinism.

### 2.2 Stable table ordering
**File:** `src/agent/classifier.py::format_available_tables`. Sort schemas by table name
before rendering, so the prompt text is byte-identical run-to-run regardless of registry
iteration order.

### 2.3 Purge orphaned/test schemas
**File:** `src/ingestion/tabular_ingest.py` — add a `purge_orphan_schemas(metadata_store)`
function (alongside the existing `cleanup_spreadsheet_tables`) that removes any registered
schema **and** its DuckDB table whose `doc_id`
has no corresponding document in the metadata store. This removes the toy `Tampa/Boston/
Denver` and `doc1_*` fixtures by **orphan status**, not hardcoded names, and prevents a
polluted/shifting table list from flipping the classifier. Runs only via the existing
cleanup entry point (not on the request hot path); per-table errors are logged and skipped.

## Data flow

Unchanged except: (1) the synthesizer's keep/drop decision now retains `table_row`
narratives under `has_map_reduce`; (2) the classifier sees a stable prompt and the LLM
runs with a seed. No new request-path stages, no new components.

## Error handling

All changes preserve existing fail-open behavior:
- Synthesizer: still returns "no information" only when there is genuinely no chunk and no
  `sql_results`; the change strictly *adds* surviving content.
- Seed: additive payload field; if the endpoint ignores it, behavior is today's behavior.
- Purge: off the hot path; wrapped per-table; a failure to drop one table does not abort
  the rest and never affects live ingestion or query.

## Testing (TDD, pytest)

- **Synthesizer (`tests/test_agent/`):** with a `map-reduce` synthetic chunk + a `table_row`
  narrative chunk + a raw `large` chunk in `retrieved_chunks`, assert the built context
  contains the map-reduce synthesis and the narrative text, and excludes the raw `large`
  chunk text. Regression: a non-sweep call (no map-reduce chunk) keeps existing behavior.
- **Classifier:** `format_available_tables` returns identical sorted output for inputs that
  differ only in order.
- **Purge:** a schema whose `doc_id` is absent from the metadata store is removed (schema +
  DuckDB table); a schema with a live `doc_id` is retained.
- **Determinism (light):** unit-assert the `seed` is present in the `_call_llm` payload
  (mock the HTTP call); do not attempt to assert end-to-end LLM determinism.

## Verification

Re-run "What are the GS salary rates in Tampa?" in the playground. Using the existing
observability logs, confirm it returns the GS rows whether it classifies `sweep` or
`analytical`, and that repeated runs classify consistently.

## Out of scope / deferred

- The `tables_relevant_to` ≥ 0.30 gate that sometimes prevents SQL from firing inside
  SWEEP (would make structured retrieval fire more often; separate lever).
- Wiring `strategy_memory.get_best_strategy()` into routing (a feedback loop with its own
  tuning).
- Query-cache behavior around reusing a cached `query_type` across invalidations.
- Locality code↔name robustness for genuinely cryptic codes (worked for Tampa→TU here).
