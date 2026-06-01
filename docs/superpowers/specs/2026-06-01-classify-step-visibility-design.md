# Classify-step visibility in the async query status

**Date:** 2026-06-01
**Status:** Approved design, pending implementation
**Area:** `src/agent/state.py`, `src/agent/classifier.py`, `src/agent/graph.py`,
`src/api/query_jobs.py`, `src/api/models.py`, `src/api/routes_query.py`,
`examples/query_async.py`

## Problem

When polling an async query, the `classify` step appears to hang for 30+
seconds and the caller sees only the label `"classifying question"` and, at the
end, the answer + citations. The classify node actually does substantial work —
resolve table hints, an LLM classification call, and a strategy-memory lookup
with override logic — and produces useful structured output (query type,
sub-tasks, strategy-memory decision) that is never surfaced.

Two compounding causes:
1. The graph streams with `stream_mode="updates"`, so a node's step label is
   emitted only AFTER the node finishes — labels lag. While `classify` runs, the
   displayed step is still the previous one ("checking cache").
2. The status payload only carries a single `step` label, never the classify
   node's structured results.

## Goal

Surface, in the async poll status (`GET /query/async/{token}`):
- **Live sub-steps + timing** within classify (so it's clearly working and you
  can see where the time goes).
- **Classification result** — query type + a one-line reason.
- **Sub-task decomposition** — the sub-queries the classifier produced.
- **Strategy-memory decision** — override/suppress/agree, chosen strategy, run
  count, margin.

## Approach (chosen: A — state-injected progress callback)

Inject an optional `progress(name, detail=None)` callback into `AgentState`.
Nodes call it synchronously mid-execution, so progress is reported as work
happens (not after the node finishes), which fixes the perceived freeze. The
async worker records each call on the job.

Rejected:
- **B — LangGraph custom StreamWriter:** idiomatic but couples to LangGraph
  streaming internals/version for no added capability here.
- **C — post-hoc extraction:** attach classify outputs after the node completes;
  drops the requested live sub-steps (still a 30s freeze).

## Design

### 1. Progress plumbing

- `src/agent/state.py`: add optional `progress` field to `AgentState`
  (`Callable[[str, dict | None], None]`).
- `src/agent/graph.py` `run_agent_streamed`: the per-node `step_callback` is
  extended to `step_callback(name, detail=None)`. When a `step_callback` is
  supplied, inject `progress=lambda name, detail=None: step_callback(name, detail)`
  into `initial_state`. The existing per-node update still calls
  `step_callback(node_name)`. When no `step_callback`, no `progress` is injected.
- Nodes read `progress = state.get("progress") or (lambda *a, **k: None)`, so the
  sync path / tests get a no-op and behavior is unchanged.

### 2. Classify node sub-steps (`src/agent/classifier.py`)

`classify_node` emits, in order:
- `progress("classify.hints")` before hint resolution + `format_available_tables`.
- `progress("classify.llm")` before the `classify_query` LLM call.
- `progress("classify.strategy")` before the strategy-memory lookup.
- `progress("classify.done", {"kind": "classification", "data": {...}})` after
  computing the result, where `data` = `{query_type, reason, sub_tasks,
  strategy_memory}` (`strategy_memory` is the existing `memory_decision` dict).

`classify_query` (and its prompt) additionally returns a one-line `reason`
(fail-open: default `""` when absent/unparseable). The classifier prompt stays
otherwise unchanged (still deterministic under the fixed seed).

### 3. Step labels (`src/api/query_jobs.py`)

Extend `STEP_LABELS` with the sub-steps:
- `classify.hints` → "reading available data tables"
- `classify.llm` → "classifying question"
- `classify.strategy` → "checking strategy memory"
- `classify.done` → "classification complete"

`update_step` already falls back to the raw name for unmapped keys.

### 4. Job + response shape

- `QueryJob` gains:
  - `steps: list[dict]` — appended on every `update_step` as
    `{"step": <label>, "at": round(now - created_at, 2)}` (a timeline across the
    WHOLE pipeline, not just classify).
  - `classification: dict | None` — set from a `detail` whose `kind ==
    "classification"`.
- `update_step(token, node, detail=None)`: set `job.step`; append to `job.steps`;
  if `detail` is a classification, set `job.classification = detail["data"]`.
- `AsyncQueryStatusResponse` (`src/api/models.py`) gains
  `steps: list[dict] = []` and `classification: dict | None = None`.
- `routes_query.py` poll handler populates both from the job.

### 5. Example script (`examples/query_async.py`)

Print each new step from the `steps` timeline with its elapsed time, and when
`classification` is present, print a short block: query type + reason, sub-tasks,
and the strategy-memory decision.

## Testing

- **classify node** (`tests/test_agent/...`): inject a recording `progress`;
  assert it is called with `classify.hints`, `classify.llm`, `classify.strategy`,
  and a final `classify.done` carrying `kind="classification"` and a `data` dict
  with `query_type`, `sub_tasks`, `strategy_memory`. No-op when `progress` absent.
- **classifier reason**: `classify_query` returns a `reason` string; fail-open to
  `""` on malformed JSON.
- **query_jobs**: `update_step` appends to `steps` with an `at` time and stores
  `classification` from a classification detail.
- **response model**: `AsyncQueryStatusResponse` serializes `steps` +
  `classification`.
- **Live verification (deployed):** poll a real query (e.g. the GS-13 pay
  question) and confirm the classify sub-steps stream with timing and the
  classification block (type, reason, sub-tasks, strategy decision) appears.
- Full suite: no new failures vs the 15-failure baseline.

## Out of scope

- Sub-steps for other nodes (retrieve / enrich / merge / synthesize) beyond the
  whole-pipeline `steps` timeline.
- Playground (admin UI) classification card — async status only.
- Any change to classification logic or strategy-memory behavior (observability
  only).
