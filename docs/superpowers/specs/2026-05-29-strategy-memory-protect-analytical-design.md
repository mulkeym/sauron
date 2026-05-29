# Strategy Memory must not override capability-gated ANALYTICAL routing

**Date:** 2026-05-29
**Status:** Design — approved for planning

## Problem

The query classifier (`src/agent/classifier.py`) picks a retrieval strategy in two
stages inside `classify_node`:

1. An LLM classification (`classify_query`) that is **capability-aware**: it is shown
   the registered, ACL-filtered structured tables (now hint-enriched) and chooses
   ANALYTICAL when a relevant table can answer the question.
2. A confidence-gated **Strategy Memory** soft override (`get_best_strategy`): if a
   learned record for the query pattern clears the gates
   (`strategy_memory_min_runs=3`, `strategy_memory_margin=0.15`), it replaces the
   LLM's pick.

These fight each other. Verified live for "What is the pay range for an officer?":

```
Classified ... -> analytical (tables_available=True)
Strategy memory: pattern='pay range officer' -> best=lookup (cited=1.0, margin=100%, n=3)
Strategy memory override: analytical -> lookup (n=3, margin=100%)
```

The memory had been trained by the user's earlier *failed* attempts: those answers
were unhelpful ("the document does not contain the amounts") yet still **cited** the
PDF, so the memory scored LOOKUP a success (`docs_cited=1`). A fuzzy, noisily-trained
prior thus vetoes a deterministic capability match — and ANALYTICAL never runs, so it
can never relearn. Chicken-and-egg.

(The poisoned rows for this one pattern were cleared manually to unblock testing;
this design prevents recurrence for every other pattern.)

## Goal

Make capability-gated ANALYTICAL routing authoritative: Strategy Memory may not
override the LLM's pick when that pick is ANALYTICAL. This aligns with how
tool/function-calling routers treat a capability match (the learned layer nudges,
it does not veto) and generalizes to any dataset that registers a structured table.

Non-goals (noted for later, not built here):
- Fixing the memory success metric so a "cited but unhelpful" answer is not counted
  as a win.
- Multi-retriever fan-out + fusion.
- Extending the protection to CROSS_REFERENCE (also reads structured tables) — a
  natural follow-up, deliberately deferred to keep this change minimal and matching
  the chosen scope (ANALYTICAL only).

## Design

Single point of change in `classify_node` (the async node built by
`_classify_node_factory` in `src/agent/classifier.py`), inside the existing
`if settings.strategy_memory_enabled:` block.

Today the override fires when:
- `best` exists, and
- `mem_type` is valid and differs from `llm_pick`, and
- `best["count"] >= strategy_memory_min_runs`, and
- `best["margin"] >= strategy_memory_margin`.

Add one guard: when `llm_pick == QueryType.ANALYTICAL`, the override is suppressed
regardless of the gates.

Behavior table:

| LLM pick | Memory wants | Gates pass | Result |
|---|---|---|---|
| ANALYTICAL | lookup/sweep/… | yes | **No override** (NEW). `query_type` stays ANALYTICAL. |
| lookup | sweep | yes | Override → sweep (unchanged) |
| any | same as LLM | — | "agreed" (unchanged) |
| any | differs | no | "below gate" (unchanged) |

Telemetry: the existing `memory_decision` dict is preserved and made visible:
- `overrode = False`
- `reason = "protected"`
- `memory_best`, `count`, `margin` still populated (so the playground shows that
  memory *wanted* a different strategy but was blocked).
- Log once: `logger.info("Strategy memory suppressed: analytical capability pick protected (memory wanted %s, n=%d, margin=%.0f%%)", mem_type, best["count"], best["margin"] * 100)`.

No change to `get_best_strategy`, `normalize_query_pattern`, the gate settings, or
`log_strategy_result`. Memory continues to record analytical runs as usual, so it
self-corrects from real results once analytical is allowed to run.

## Testing

Unit tests on `_classify_node_factory(...)` node (mirroring existing classifier
tests that monkeypatch `classifier.generate` and `classifier.get_best_strategy`):

1. **Protection:** LLM returns ANALYTICAL, a table is registered (visible to the
   user), memory's `get_best_strategy` returns `lookup` with `count >= 3` and
   `margin >= 0.15`. Assert the node returns `query_type == ANALYTICAL`,
   `strategy_memory["overrode"] is False`, `strategy_memory["reason"] == "protected"`.
2. **Regression — non-structured override still works:** LLM returns `lookup`,
   memory returns `sweep` past the gates. Assert override applies
   (`query_type == SWEEP`, `overrode is True`, `reason == "override"`).
3. **Agreement unchanged:** LLM ANALYTICAL, memory `analytical` → `reason == "agreed"`,
   no override, `query_type == ANALYTICAL`.
4. **Below-gate unchanged:** LLM `lookup`, memory `sweep` but `count < min_runs` →
   `reason == "below gate"`, no override.

## Risks

- A genuinely mis-classified ANALYTICAL pick can no longer be corrected by memory.
  Acceptable: ANALYTICAL is only chosen when a relevant table is registered and
  ACL-visible, and the Part B fallback (zero-row SQL → structured narratives →
  map-reduce, already shipped) recovers when the structured path finds nothing.
