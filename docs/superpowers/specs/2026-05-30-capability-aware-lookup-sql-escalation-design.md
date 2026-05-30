# Capability-Aware LOOKUP → SQL Escalation

**Date:** 2026-05-30
**Status:** Design approved, pending spec review

## Problem

"what are the pay rates for florida?" returns "the provided context does not contain information regarding pay rates for Florida (2025_April_Dec_AD_Pay.pdf)" even though the federal GS pay table answers it (florida → locality codes `RUS`/`MFL`/`PB`).

Verified root cause (traced end-to-end through `run_agent`): the question classifies as **LOOKUP**, and the LOOKUP path runs only `retrieve_lookup` (vector search) — it **never runs the structured/SQL path**. The SQL path runs only for `ANALYTICAL` or `SWEEP`. The agent graph is a single pass:

```
classify → [ retrieve ∥ enrich ] → merge → synthesize → END
```

There is **no `evaluate` node and no escalation/retry loop**. `evaluate_context` (evaluator.py), which contains lookup→sweep escalation logic, is **dead code — never called anywhere in `src/`**. So a LOOKUP classification is final, and any question the classifier under-routes to LOOKUP can never reach the SQL table. "alaska" works only because it occasionally tips to ANALYTICAL; "florida" consistently lands LOOKUP.

(Prior fixes this session — SWEEP path hints, `_extract_sql` reasoning-leak — are correct and retained, but live downstream of the SQL path actually running, so they don't fix Florida on their own.)

## Goal

When a question classified LOOKUP could be answered by a registered structured table, run that table's SQL and include the results — so "pay rate for `<location>`" gets a real answer instead of "no data" — without adding LangGraph loop machinery.

## Non-Goals

- No general multi-step escalation ladder (lookup→sweep→cross_reference). Just the one capability-aware step.
- No LLM sufficiency judge (rejected: risks false "sufficient" on topically-relevant-but-useless chunks — the trap that would leave Florida broken).
- No classifier change. No query reformulation (reformulation drifted "florida" → "florida minimum wage"; we keep the original question).
- No new graph nodes or conditional edges.

## Design

### Inline escalation in the `retrieve` node's LOOKUP branch

`src/agent/graph.py`, the `retrieve` node, LOOKUP branch. After `retrieve_lookup` runs, if no SQL was produced, also run the gated structured-SQL path and merge its results:

```python
if query_type == QueryType.LOOKUP:
    result = await _asyncio_lookup.to_thread(retrieve_lookup, retry_state, vector_store=vector_store)
    # Capability-aware escalation: LOOKUP never queries structured tables. If a
    # registered table is relevant to the question, also run the gated SQL path
    # (its relevance gate is the capability check; it uses the original question
    # and carries domain hints). Lookup alone could never answer e.g. "pay rate
    # for florida" from the GS pay table.
    if not result.get("sql_results"):
        struct = await retrieve_structured(retry_state, vector_store, schema_registry)
        if struct.get("sql_results"):
            result["sql_results"] = struct["sql_results"]
            result["structured_trace"] = struct.get("structured_trace")
            result.setdefault("retrieved_chunks", []).extend(struct.get("retrieved_chunks", []))
```

`retry_state` is the existing per-attempt state (carries the original question). `retrieve_structured` and `QueryType` are already imported in graph.py.

### Why `retrieve_structured` is the SQL step

It is the precise realization of the two approved choices (capability-aware trigger + focused SQL):

- **Capability check = its relevance gate.** `retrieve_structured` runs `tables_relevant_scored`; it produces `sql_results` only when a registered table clears the relevance threshold. No relevant table → it returns `{}` and lookup is unchanged. (Florida's GS table scores ~0.5 → passes.)
- **Focused:** passes only the gated/relevant tables to the SQL generator (not all ~80), keeping the prompt small.
- **Original question:** called with `retry_state` (the original question) — no drifting reformulation.
- **Hints:** carries the locality glossary (this session's fix) so "florida" → `RUS`/`MFL`/`PB`.

`retrieve_analytical` is not used: it is ungated (would fire SQL on every lookup) and passes all visible schemas to the generator.

### Dead-code cleanup

`evaluate_context` (and the `EVALUATION_PROMPT`, `MAX_RETRIEVAL_ATTEMPTS`) in `src/agent/evaluator.py` are never called. They caused a multi-step misdiagnosis this session (assumed to be live escalation). Mark the module clearly as unused, or remove it, so it stops being mistaken for live behavior. (Confirm no imports before removing: `grep -rn "evaluator\|evaluate_context" src/`.)

## Data flow

```
classify=LOOKUP
  -> retrieve (LOOKUP branch):
       retrieve_lookup -> chunks
       no sql_results AND retrieve_structured gate passes (table relevant)?
          -> retrieve_structured (gated SQL, original question, hints)
          -> merge sql_results + structured_trace + table_row chunks
  -> enrich (parallel, unchanged)
  -> merge -> synthesize (now sees SQL rows -> real answer)
```

## Error handling

`retrieve_structured` is already fail-open (gate/registry errors → `{}`). The escalation is guarded by `if not result.get("sql_results")` and `if struct.get("sql_results")`, so any failure leaves the lookup result exactly as today. No new failure modes.

## Testing

**Unit (graph `retrieve` node, mocked deps):**
1. LOOKUP + `retrieve_structured` returns `sql_results` → the result carries those `sql_results`, `structured_trace`, and the structured chunks appended to the lookup chunks.
2. LOOKUP + `retrieve_structured` returns `{}` (no relevant table) → result equals the plain `retrieve_lookup` result (no regression).
3. A LOOKUP result that already has `sql_results` (shouldn't normally happen) → `retrieve_structured` is not called twice (guard).
4. Non-LOOKUP query types are unaffected (SWEEP/ANALYTICAL branches unchanged).

**Live acceptance (post-deploy, required):** "what are the pay rates for florida?" through full `run_agent` returns real GS pay data (grade/min/max), not "no data." Spot-check: a relevant alaska question still answers; a no-table lookup ("what did X say?") still behaves as before.

## Risks / considerations

- **Latency:** a LOOKUP question now also runs the embedding relevance gate; only when a table is relevant does it pay the (~11s thinking) SQL cost. Acceptable — correctness over a few seconds, only on table-answerable questions.
- **Scope:** this is a single capability-aware step, not a general retry loop. A full lookup→sweep→cross_reference ladder remains a separate, larger change if ever wanted.
- **Synthesis:** the merged result includes both the (often unhelpful) lookup chunk and the SQL rows; the synthesizer already prioritizes/handles SQL blocks. If the lookup chunk misleads synthesis, that's a separate tuning concern, not part of this change.
