# Adaptive SQL Consolidation + Bounded Repair Loop

**Date:** 2026-05-30
**Status:** Design approved, pending spec review

## Problem

A broad question against a wide structured table overflows the model context. Concretely, "what are the pay rates?" generated `SELECT *` against the OPM General Schedule locality pay table (`doc_39364397..._all_gs`: 885 rows × 32 cols = ~28,000 cells), which serialized to ~893k chars / ~223k tokens and exceeded vLLM's 256k window (`VLLMValidationError`, 223,233 input + 32,768 output > 256,000).

Two root causes in the text-to-SQL path (`src/agent/strategies/structured.py`):

1. The text-to-SQL system prompt **explicitly instructs** the model toward the overflow — `structured.py:45`: *"When in doubt, prefer `SELECT *` over a hand-picked subset of columns."*
2. There is **no volume gate**: `generate_sql` → `run_sql` → `row_count = len(rows)` runs unconditionally. The only existing reaction is post-hoc (`analytical.py:39`: `row_count == 0` → fallback).

A row-cap truncation already shipped in the synthesizer (`SQL_RESULT_MAX_ROWS = 100` + char-budget loop) prevents the crash, but truncation **silently biases the answer** — capping an 885-row alphabetically-ordered table to 100 rows returns only the first localities (AK…) and stops. The cap should be the last safety net, not the primary mechanism.

## Goals

- Stop wide tables from overflowing the synthesis context *at the query-generation layer*, not by truncation.
- Give the model bounded "space to improve" a query that comes back unsatisfactory (too large, empty, degenerate, or errored).
- Keep the common-case cost flat: a clean result costs exactly one LLM generation, no judge call.
- Never end up worse than the original query; never crash.

## Non-Goals

- No change to the unstructured prose / SWEEP / map-reduce path, the playground, or the metadata-catalog path (they consume the now-smaller result unchanged).
- No always-on LLM relevance grader on the happy path.
- No attempt to retry-our-way out of genuine question ambiguity (handled by summarize-and-offer-drill-down, not blind retries).

## Design

### Where it lives

All logic lives in the structured/SQL path, `src/agent/strategies/structured.py`. Two functions generate+run SQL today and **both must get identical behavior** (we have been bitten before fixing one path and missing its twin):

- `run_structured_lookup` — full `StructuredLookupTrace`, never raises.
- `structured_sql_rows` — thin variant, raises on failure.

A single shared internal helper, `_generate_run_fit`, encapsulates the gate + loop; both functions call it.

### 1. Proactive pre-flight gate

Before generating SQL, estimate the candidate table's payload:

- rows: cheap `SELECT COUNT(*)` per candidate table (DuckDB, read-only).
- cols: column count from the schema.

If `rows × cols` exceeds a threshold (`sql_wide_table_cell_threshold`, default 5,000 cells), inject a steering block into `TEXT_TO_SQL_PROMPT` that **overrides** the `SELECT *` preference for that table:

> "Table {name} has ~{rows} rows × {cols} cols. Returning every row is unhelpful and will be truncated. Prefer aggregation (MIN/MAX/AVG with GROUP BY on low-cardinality columns) or scope with WHERE/LIMIT to directly answer the question."

Small tables: unchanged, `SELECT *` remains acceptable.

Additionally, soften the global instruction at `structured.py:45` from *"prefer `SELECT *`"* to *"prefer the narrowest set of columns that answers the question."*

### 2. Shared budget signal

New config knob `sql_result_budget_chars`, default ≈ 65% of `llm_max_context` (~130k at the current 200k setting). "Too large" = `len(json.dumps(rows)) > sql_result_budget_chars`. Width-aware (it measures the actual serialized payload, so it catches wide-but-short results), and stays under the synthesizer's hard cap so the rest of the context still fits.

### 3. Satisfaction check (unified retry trigger)

After each query runs, classify the result:

- **too_large** — `len(json.dumps(rows)) > sql_result_budget_chars`
- **empty** — zero rows
- **degenerate** — all returned columns are NULL across all rows
- **error** — SQL invalid/blocked/failed to execute

The four conditions above are the **only retry triggers**, and they are all free/deterministic. The LLM relevance judge is **not** an independent trigger — it is an *enrichment of the retry feedback*: when a query has already been flagged unsatisfactory by one of the four conditions and `sql_relevance_judge_enabled` is set, one judge call asks *"Do these rows answer '{question}'? If not, what is wrong with the query?"* and its reason is woven into the next retry's feedback so the model fixes the right thing. It never fires on a result that passed all four checks.

A result that is non-empty, valid, not degenerate, and within budget is **satisfactory** → return immediately, no judge call, one generation total. This means the "fits-but-maybe-not-what-they-meant" case is intentionally *not* retried here — it is left to the synthesizer's own "context doesn't contain enough information" gate and to the summarize-and-drill-down answer shape, because blind retries cannot recover question intent. This also absorbs the existing `analytical.py:39` zero-row fallback into the loop.

### 4. Bounded repair loop

On an unsatisfactory result, feed the model its own SQL plus a failure-specific message and regenerate. **Max 2 retries (3 generations total).** Temperature 0.0 on attempt 1, 0.3 on retries (so it does not deterministically reproduce the same query).

Failure-specific feedback:

| Failure | Feedback to model |
|---|---|
| too_large | "Returned {rows} rows (~{chars} chars), too large. Aggregate (MIN/MAX/AVG + GROUP BY) or scope with WHERE/LIMIT." |
| empty | "Returned no rows. Your filter or column name may be wrong — loosen the filter or check column names." |
| degenerate | "Returned only NULLs. The selected columns may be wrong for this question." |
| error | "Query failed: {error}. Fix the SQL." |
| judge-unhelpful | "These rows don't answer the question because {reason}. Try a different approach." |

The loop keeps the **best valid result seen so far**. A retry that errors or produces invalid SQL is treated as "still unsatisfactory" and does not discard a prior valid (if oversized) result.

### 5. Final fallback

If all 3 generations remain unsatisfactory, return the best valid result and let the **existing synthesizer row-cap** truncate it (`showing N of M` note). No crash under any path.

## Error handling

`run_structured_lookup` preserves its current contract: never raises; on total failure records `status="error"`, `fell_back=True`. `structured_sql_rows` preserves its raising contract for total failure, but returns the best valid result if any attempt produced one. The loop never produces a result worse than the original query.

## Data flow

```
question
  -> pre-flight: COUNT(*) × cols per candidate table
       wide? -> inject aggregate/scope steering block
  -> generate_sql (attempt 1, temp 0.0)
  -> run_sql -> satisfaction check
       satisfactory? -> return (1 generation, no judge)
       unsatisfactory? -> [optional relevance judge] -> feedback
            -> generate_sql (retry, temp 0.3) -> run_sql -> check
            -> (max 2 retries)
  -> still unsatisfactory? -> best valid result -> synthesizer row-cap
  -> synthesize
```

## Config knobs

| Knob | Default | Meaning |
|---|---|---|
| `sql_result_budget_chars` | `int(llm_max_context * 0.65)` | Serialized-result size that counts as "too large" |
| `sql_wide_table_cell_threshold` | 5000 | `rows × cols` above which the pre-flight steering block fires |
| `sql_repair_max_retries` | 2 | Max retries after the first generation (3 generations total) |
| `sql_relevance_judge_enabled` | True | Whether to make the LLM relevance call on suspect queries |

## Testing

Unit tests use a stubbed `generate_fn` (no live LLM), one scenario each, exercising **both** `run_structured_lookup` and `structured_sql_rows`:

1. Clean small result → exactly 1 generation, no judge call, `SELECT *` preserved.
2. Wide table (pre-flight) → steering block present in the rendered prompt; `SELECT *` instruction overridden.
3. Oversized result → loop retries; stops as soon as a stubbed "fitting" query returns.
4. Empty result → loop retries with the loosen-filter feedback.
5. Degenerate (all-NULL) → loop retries.
6. Already-flagged result (e.g. too_large) with `sql_relevance_judge_enabled=True` → judge runs once and its reason appears in the retry feedback; with the flag False → no judge call is made; clean result with the flag True → still no judge call.
7. All attempts unsatisfactory → best valid result returned, synthesizer cap engages (`showing N of M`).

Plus a **realistic wide-table regression test** for the synthesizer cap: build an 885-row × 32-col result and assert the synthesis context fits within `MAX_CONTEXT_CHARS` and carries the truncation note (this is the missing regression test identified during the cap fix verification).

## Risks / open considerations

- A 26B MoE (Gemma4) reliably writes mechanical aggregation SQL when told to, but is less reliable at *choosing the right consolidation dimension* for an ambiguous question. Mitigated by explicit "group by low-cardinality columns" guidance and the summarize-and-drill-down answer shape — not by additional retries.
- The relevance judge shares the generator's blind spot; it is therefore scoped to detecting *unambiguous* misfires (empty/degenerate/clearly-off), never to recovering user intent.
- Worst-case latency rises to 3 generations + 1 judge call, but only on genuinely problematic queries; normal queries stay at 1 call.
