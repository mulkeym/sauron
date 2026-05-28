# Search Quality: Wire In Dormant Learning + Final-N Reranking

**Date:** 2026-05-28
**Status:** Approved (brainstorm); pending implementation plan
**Branch:** `feat/search-quality-rerank-learning`

## Overview

Three independent improvements to the retrieval layer that activate capabilities
the codebase already *has* but does not fully use, plus a consistent final-stage
reranking pass. Each change is **fail-open** (degrades to current behavior on any
error) and **flag-guarded**. No ingestion, embedding, or schema changes.

The three findings, all confirmed against the current `master`:

- **F1 — Strategy Memory is write-only.** `strategy_memory.log_strategy_result()`
  records which strategy worked per normalized query-pattern, but
  `strategy_memory.get_best_strategy()` is **never called** anywhere — the learning
  signal is collected and discarded. (Phase 3 of the 2026-05-19 adaptive-retrieval
  spec, left half-built.)
- **F2 — Relevance-feedback boosts only reach `map_reduce`.** The `query_feedback`
  log and `get_feedback_boosts()` are fully built, but only `map_reduce.py` consumes
  them. `sweep`, `lookup`, `cross_reference`, and `structured` retrieval ignore the
  boosts, so "what worked before" never helps the broad-query paths users hit most.
- **F3 — CrossEncoder reranking only runs in `lookup` + `cross_reference`.**
  `hybrid_search_reranked()` (the highest-quality path) is used by just those two
  strategies. `sweep`, `map_reduce`, and `structured` narrative search return chunks
  with no CrossEncoder rerank — leaving quality on the table for exactly the
  high-recall, many-candidate strategies that benefit most.

## Problem

The retrieval system has accreted three quality mechanisms — strategy learning,
relevance-feedback boosting, and CrossEncoder reranking — but each is only partially
wired. The learning signal that *could* compound over time is either discarded (F1)
or applied to a single strategy (F2), and the highest-quality ranking pass is applied
inconsistently (F3). The net effect: the system does not get measurably better with
use, and broad-query answers (sweep/map-reduce) are ranked with weaker signals than
narrow lookups.

## Non-Goals

- No reranking of the wide candidate-discovery set (sweep/map_reduce search at
  `top_k=500` over summary embeddings). CrossEncoder scores candidates one-by-one;
  reranking 500 per query on the local quantized stack is too slow. Reranking is
  bounded to the final synthesizer-bound chunk set only.
- No new feedback *capture* — `log_feedback`/`log_strategy_result` already run. This
  spec only consumes the signals they produce.
- No user-facing thumbs-up/down (deferred in the original adaptive spec).
- No change to the relevance-gate threshold (`tables_relevant_to` ≥0.30) or the
  query cache; those remain open follow-ups.

## Solution

### F1 — Strategy Memory → routing (confidence-gated soft override)

**Files:** `src/agent/classifier.py`, `src/agent/graph.py`, `src/retrieval/strategy_memory.py`, `src/config.py`.

1. **Fix the selection metric in `get_best_strategy()`.** Today it ranks candidate
   strategies by precision (`avg_relevant / avg_discovered`), which structurally
   favors narrow strategies — a `lookup` finding 3/3 docs (1.0) beats a `sweep`
   finding 40/50 (0.8) even when the sweep produced the better answer. Replace the
   ranking key with a composite:
   - **Primary:** `avg_docs_cited` (docs that actually appeared in the final answer —
     the truest success proxy available).
   - **Tiebreak:** `avg_relevant`.
   - `precision` is retained in the returned dict as reported info, not the ranking key.
   - The returned dict also carries `count` (run-count for the winner) and `margin`
     (winner composite minus runner-up composite, normalized; `1.0` when only one
     strategy has records).

2. **Make the classify node async.** `_classify_node_factory`'s inner `classify_node`
   becomes `async def` (its peers `retrieve`/`enrich` are already async) so it can
   `await get_best_strategy(question)`. `classify_query` stays synchronous for the
   LLM call; the memory lookup happens in the node wrapper after it.

3. **Soft override logic** (in the classify node, after the LLM pick):
   - Skip entirely if `settings.strategy_memory_enabled` is false or the lookup
     returns `None`.
   - Override the LLM's `query_type` with memory's winner **only if all hold:**
     `count >= settings.strategy_memory_min_runs` AND
     `margin >= settings.strategy_memory_margin` AND
     the winner differs from the LLM pick.
   - Otherwise keep the LLM's classification unchanged.
   - The whole block is wrapped fail-open: any exception → keep the LLM pick.

4. **Observability.** The classify node returns a `strategy_memory` decision dict
   alongside `query_type`/`sub_tasks`:
   `{pattern, llm_pick, memory_best, count, margin, overrode: bool, reason}`.
   Threaded through `AgentState`, surfaced in `AgentTrace` and the admin playground
   trace — mirroring the existing `structured_trace` plumbing. `None` when the
   feature is disabled or no record matched.

### F2 — Relevance-feedback boosts → all retrieval strategies

**Files:** `src/retrieval/feedback.py` (new shared helper), `src/agent/strategies/sweep.py`, `lookup.py`, `cross_reference.py`, `structured.py`.

`get_feedback_boosts(query_vector, user_groups) -> {doc_id: boost}` already exists and
is fail-open. Two application shapes:

1. **Doc-selection strategy (`sweep`).** Fetch boosts once after the initial doc
   search. Add `feedback_boosts.get(doc_id, 0.0)` to each doc's score **before** the
   existing 30%-of-top-score cutoff (`sweep.py:44-50`), and drop docs whose boost is
   negative — mirroring `map_reduce.py:326-345`. This influences *which* docs sweep
   pulls chunks from.

2. **Chunk-level strategies (`lookup`, `cross_reference`, `structured` narrative
   search).** New helper in `feedback.py`:
   ```
   def apply_feedback_boosts_to_chunks(
       chunks: list[RetrievedChunk], boosts: dict[str, float]
   ) -> list[RetrievedChunk]:
       # add owning-doc boost to each chunk.score, re-sort desc, return.
       # no-op when boosts is empty.
   ```
   Each strategy fetches boosts (fail-open) and calls this **before** its existing
   30%-of-top chunk filter, so boosts affect what survives the cutoff. Negative-boost
   docs are de-prioritized via the score adjustment (chunk-level strategies do not
   hard-drop, to avoid starving a narrow lookup).

3. Every `get_feedback_boosts` call is wrapped in try/except → `{}`; gated on
   `settings.feedback_enabled` (the function already checks this).

### F3 — Final-N CrossEncoder rerank (single central chokepoint)

**Files:** `src/retrieval/vector_store.py` (new method), `src/agent/graph.py` (`merge_results`), `src/config.py`.

1. **New `VectorStore.rerank_chunks(chunks, text_query, top_n) -> list[RetrievedChunk]`.**
   Scores each `(text_query, chunk.text)` pair with a CrossEncoder model (reuse the
   lazily-cached `_get_cross_encoder` model object; call its underlying
   `sentence_transformers` `CrossEncoder.predict` on the explicit pair list rather
   than driving a LanceDB query — implementation detail for the plan to resolve, with
   a direct `sentence_transformers.CrossEncoder` load as the fallback). Assigns the
   rerank score to each chunk, re-sorts desc, trims to `top_n`. **Fail-open:** any
   error (model load, predict) → return the input list unchanged (logged at warning).

2. **Applied once** in the graph's `merge_results` over the consolidated
   `state["retrieved_chunks"]`, capped at `settings.rerank_final_top_n` (default 50).
   This is the single chokepoint the synthesizer consumes, so it covers *every*
   strategy (sweep/map_reduce/structured included) without editing each one.

3. **Signal combination with F2.** Reranking reorders purely by semantic relevance and
   would otherwise erase F2's doc-level boost. After computing rerank scores, re-add
   the doc-level feedback boost to each chunk's rerank score and do the final sort, so
   both signals survive. To keep `merge_results` **synchronous** (it is today, and a
   CrossEncoder `predict` is CPU-bound, not I/O), the F2 strategies stash the
   `{doc_id: boost}` map they already fetched into a new `AgentState["feedback_boosts"]`
   field; `merge_results` reads it (default `{}`) rather than issuing a second async DB
   read. When feedback is disabled or absent the boost term is simply 0.

4. Guarded by `settings.rerank_final_enabled` (default `True`); when false,
   `merge_results` behaves exactly as today.

### New settings (`src/config.py`)

| Setting | Default | Purpose |
| --- | --- | --- |
| `strategy_memory_min_runs` | `3` | Min recorded runs before memory may override routing |
| `strategy_memory_margin` | `0.15` | Min normalized composite margin (winner vs runner-up) to override |
| `rerank_final_enabled` | `True` | Toggle the final-N CrossEncoder rerank in `merge_results` |
| `rerank_final_top_n` | `50` | Cap on chunks reranked / kept by the final pass |

Reuses existing `strategy_memory_enabled`, `feedback_enabled`, and the feedback
boost/decay settings.

## Data Flow (after changes)

```
classify (async):
  ├─ LLM classify → query_type
  └─ await get_best_strategy(question)  → confidence-gated soft override
                                          → emit strategy_memory trace
retrieve (per strategy):
  ├─ doc search
  ├─ F2: + get_feedback_boosts → boost/de-prioritize docs (sweep) or chunks (lookup/xref/structured)
  └─ strategy's existing cutoff over boosted scores
merge_results:
  └─ F3: rerank_chunks(retrieved_chunks, question, top_n)
         final score = crossencoder_score + doc_feedback_boost
synthesize → answer
(post): log_feedback + log_strategy_result  (already wired)
```

## Error Handling

Every new path is fail-open and individually flag-guarded:

- F1: memory lookup or override logic throws → keep the LLM classification.
- F2: `get_feedback_boosts` throws → `{}` → strategies behave as today.
- F3: CrossEncoder load/predict throws → return chunks in their pre-rerank order.

Disabling all four new flags reproduces current behavior byte-for-byte (regression
guard test).

## Testing

- **F1:** `get_best_strategy` composite ranking (cited beats precision-only winner);
  min-runs gate; margin gate; soft-override applies only when all gates pass;
  fail-open when the metadata store is down; `strategy_memory` trace shape.
- **F2:** `apply_feedback_boosts_to_chunks` (boost added, re-sorted, empty-boost
  no-op); sweep doc-level boost + negative-doc drop; fail-open per strategy.
- **F3:** `rerank_chunks` reorders by score, trims to `top_n`, fail-open on reranker
  error; `merge_results` applies it under the flag and combines with feedback boost;
  disabled-flag is a no-op.
- **Regression:** all four flags off → identical behavior to current `merge_results`
  / classify / strategy outputs.
- **Baseline caveat:** `tests/test_agent/` carries ~8 pre-existing failures
  (test_graph/lookup/sweep/cross_reference — harness/mocking issues that fail at base
  too) and the environmental `test_ingestion` failures; success is measured as the
  same pre-existing set, not zero failures.

## Build Order

1. **F3** — most self-contained, immediate quality win, no behavioral routing change.
2. **F2** — shared helper + per-strategy boost application.
3. **F1** — routing + metric change; highest behavioral risk, so last and most
   heavily gated.

## Open Follow-Ups (out of scope, noted for later)

- The `tables_relevant_to` ≥0.30 gate occasionally blocking SQL in sweep.
- Query cache reusing a stale `query_type` across invalidations.
- A benchmark to decide whether reranking the wide discovery set is ever worth it.
- User thumbs-up/down feedback capture (Phase "Manual Feedback" from the 2026-05-19 spec).
