# Reasoning-Assisted SQL Generation on Wide Tables

**Date:** 2026-05-30
**Status:** Design approved, pending spec review

## Problem

"what are the pay rates?" against the 885-row × 32-col GS locality pay table produced `SELECT locname, grade, annual1, hourly1, overtime1 FROM … LIMIT 100` — the first 100 rows (alphabetically, AK…), a biased partial sample rather than a consolidation. The user wants a *smarter* query: aggregated ranges (e.g. `MIN`/`MAX` per grade), not a truncation.

Two causes conspired:

1. The wide-table gate's steering offered *"prefer aggregation **or** scope with WHERE/LIMIT"* — LIMIT was an equal, easier option, and the model took it.
2. A 100-row result fits the synthesis budget, so the repair loop (`_generate_run_fit`) marked it **satisfactory** and never pushed back. "Fits budget" ≠ "actually consolidated."

The model (`cyankiwi/gemma-4-26B-A4B-it-AWQ-4bit`, served with `--reasoning-parser gemma4`) currently runs with **thinking disabled** for speed. Generating a good query for a vague, broad question is a *judgment* task — exactly what reasoning helps with.

## Goal

Make the SQL generated for a broad question against a wide table a sensible **aggregation** rather than a truncated `LIMIT`, by enabling the model's reasoning for that specific generation and strengthening the steering suggestion — without hard rules, SQL-shape inspection, or slowing the rest of the pipeline.

## Non-Goals

- No hard "reject LIMIT" rule or SQL-shape regex. Steering stays a *suggestion*; the model decides.
- No thinking on classification, synthesis, the relevance judge, or normal/small-table SQL.
- No classifier/routing change — "what are the pay rates?" still routes to SWEEP (which runs structured SQL anyway via the parallel `retrieve_structured`).
- No guarantee against an occasional lazy LIMIT. Accepted trade for trusting the reasoning model; a lazy-LIMIT backstop is a deferred follow-up if the live smoke shows inconsistency.

## Design

### 1. Thinking is tied to the wide-table gate

`_generate_run_fit` (`src/agent/strategies/structured.py`) already computes `steering = _wide_table_steering(con, schemas)` (non-empty only when a candidate table exceeds `sql_wide_table_cell_threshold`). Set:

```python
thinking = bool(steering) and settings.sql_thinking_on_wide_table
```

and thread it into `generate_sql(..., thinking=thinking)`. Because `steering` is constant across a query's attempts, thinking stays on for any repair-loop retries of a wide-table query. Small/normal tables → `steering == ""` → `thinking == False` → fast, non-thinking generation, unchanged.

The `_relevance_judge` call passes no flag → stays non-thinking.

### 2. LLM client plumbing

`generate()` and `_call_llm()` (`src/generation/llm_client.py`) gain a keyword-only `thinking: bool = False`:

- When `thinking=True`, `_call_llm` adds the reasoning toggle to the request payload and raises `max_tokens` to `settings.sql_thinking_max_tokens` (reasoning + SQL needs more than the current 2048).
- The exact toggle param for this gemma-4 build is confirmed empirically at implementation time (likely `chat_template_kwargs={"enable_thinking": true}` in the OpenAI-compatible payload). **Fail safe:** if the param is unrecognized/ineffective, generation still returns SQL (just non-thinking) — never an error.
- Output handling is unchanged: the client already separates `reasoning_content` and strips `<think>` blocks, so `_extract_sql` keeps receiving clean SQL. The thinking path is verified not to break extraction during implementation.

`generate_sql` (`src/agent/strategies/structured.py`) gains keyword-only `thinking: bool = False` and forwards it to `generate(..., thinking=thinking)`. The two existing extra params (`extra_user_context`, `temperature`) are unchanged.

### 3. Steering rewrite (suggestion, leads with aggregation)

Replace the current `_wide_table_steering` body text — currently *"… Prefer aggregation (MIN/MAX/AVG with GROUP BY on low-cardinality columns such as locality/grade) or scope with WHERE/LIMIT to directly answer the question."* — with a suggestion that leads with aggregation, includes a worked example, and de-emphasizes LIMIT:

> "{table} is wide (~{nrows} rows × {ncols} cols). Returning every row is rarely what's wanted. The most useful answer is usually an **aggregation** — for example `SELECT grade, MIN(annual1), MAX(annual10) FROM {table} GROUP BY grade`. Aggregate the measure columns over the low-cardinality identifying columns (e.g. grade, locality). Only return raw rows if the question genuinely asks for specific records."

Still a suggestion; with thinking on, the model reasons about it and decides.

### 4. Config

| Knob | Default | Meaning |
|---|---|---|
| `sql_thinking_on_wide_table` | `True` | Master switch: enable reasoning for SQL generation when the wide-table gate fires. Set `False` to revert to fast non-thinking SQL everywhere. |
| `sql_thinking_max_tokens` | `4096` | `max_tokens` for a thinking SQL-generation call (reasoning + SQL). |

## Data flow

```
question -> _generate_run_fit
  steering = _wide_table_steering(con, schemas)
  thinking = bool(steering) and settings.sql_thinking_on_wide_table
  loop:
    generate_sql(prompt, question, extra_user_context=steering/feedback,
                 temperature=..., thinking=thinking)
       -> generate(..., thinking=thinking)
          -> _call_llm(..., thinking=thinking)
             thinking? add reasoning toggle + max_tokens=sql_thinking_max_tokens
    run_sql -> classify (too_large/empty/degenerate/error) -> retry/accept
  judge call: generate (thinking defaults False)
```

## Error handling

- Thinking is fail-safe: an unrecognized/ineffective toggle param degrades to non-thinking generation, never an error. SQL extraction is unchanged and already handles reasoning/`<think>` output.
- All existing repair-loop semantics (best-valid retained, raise only on all-errors, the synthesizer row-cap as final fallback) are unchanged.

## Testing

**Unit (deterministic, stubbed `generate_fn`/`gen` — no live LLM):**

1. `generate_sql` forwards `thinking=True` to its `gen` callable (stub captures the kwarg).
2. `_generate_run_fit` sets `thinking=True` when a wide schema makes the gate fire, and `thinking=False` for a small table.
3. With `sql_thinking_on_wide_table=False`, `_generate_run_fit` never requests thinking even on a wide table.
4. The `_relevance_judge` call never requests thinking.

(The LLM-client `_call_llm` payload assembly for `thinking=True` is verified by a small unit test that inspects the constructed payload via a stubbed transport, asserting the reasoning toggle and `sql_thinking_max_tokens` are present.)

**Manual smoke (live, required acceptance test):** replay "what are the pay rates?" against the deployed container and confirm the generated SQL now aggregates (`GROUP BY` + `MIN`/`MAX`) instead of `LIMIT 100`. Reasoning quality cannot be unit-tested; this is the real acceptance criterion.

## Risks / open considerations

- **No hard LIMIT guarantee.** If the live smoke shows the model still sometimes emits a bare LIMIT, the follow-ups (not in this change) are: strengthen the suggestion wording, or add an opt-in lazy-LIMIT retry verdict (the hard rule deliberately avoided here).
- **Latency** rises for wide-table SQL generation only (one bounded sub-step, up to 3× if the loop retries). The rest of the pipeline is unaffected. `sql_thinking_on_wide_table=False` is the escape hatch.
- **Toggle-param uncertainty** is resolved at implementation time against the live vLLM, with the fail-safe degrade-to-non-thinking behavior.
