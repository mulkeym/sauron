# Live classify sub-progress in the admin playground

**Date:** 2026-06-01
**Status:** Approved design, pending implementation
**Area:** `src/admin/routes.py` (playground `run_query`, `_format_live_step`,
`/api/playground/status`), `src/admin/templates/playground.html`

## Problem

In the web playground, "Step 2 of 4: Classify Query" sits unchanged for the
whole time the classify node runs (~38s observed for "list all contracts
awarded by the DHA"). The active step's elapsed timer ticks, but the label,
icon, and content never change — there is no indication of what classify is
doing.

The classify sub-step visibility added earlier (sub-steps + classification
detail via the injected `progress` reporter) flows only through the **async
API** path (`/query/async/...`) and the CLI example. The playground is a
**separate** code path: `playground_start` launches a background `run_query()`
that drives its own `graph.astream(stream_mode="updates")` loop and records a
step into `completed_steps` only AFTER each node finishes. It never calls the
`progress` reporter, so no classify sub-progress reaches the UI.

## Goal

In the playground, while `classify` is the active step, show its current
sub-step inline on the row, and when it completes show the full classification
decision in the step's drill-down.

## Approach (chosen)

Reuse the existing `progress` reporter mechanism (already injected as an
`AgentState` channel) inside the playground's own `run_query` loop, and render
the result with the playground's existing active-step / drill-down UI. No new
endpoint, no second runner.

## Design

### 1. Backend — feed sub-steps into the job (`run_query`, `admin/routes.py`)

- Inject a reporter into the playground's `initial_state` before the astream
  loop:
  `initial_state["progress"] = lambda name, detail=None: _record_substep(query_id, name)`.
- `_record_substep(query_id, name)`: for `classify.*` step names, set
  `_playground_jobs[query_id]["active_substep"]` to the friendly label, reusing
  `src.api.query_jobs.STEP_LABELS` (single source of wording: `classify.hints` →
  "reading available data tables", `classify.llm` → "classifying question",
  `classify.strategy` → "checking strategy memory"). `classify.done` clears
  `active_substep` (the decision then shows in the completed-step drill-down).
- When the `classify` node completes in the existing astream loop, clear
  `active_substep` (defensive — covers the case where `classify.done` wasn't
  emitted).
- `active_substep` is initialized to `""` in the job dict.

`progress()` fires synchronously inside the classify node; the JS polls every
500ms, so the active sub-step updates live.

### 2. Status endpoint

`/api/playground/status/{query_id}` JSON gains `active_substep` (default `""`)
alongside `step` and `completed_steps`.

### 3. Frontend (`playground.html`)

In the poll loop's active-step branch, when the active step is `classify` and
`status.active_substep` is non-empty, render the row label as
`Step 2 of 4: Classify Query — {active_substep}` (the inline sub-label). When
`classify` is no longer active (completed), the sub-label is gone and the
completed row carries the drill-down. `markStepActive` is extended to accept an
optional sub-label; other steps pass none (unchanged).

### 4. Drill-down content (`_format_live_step`, classify branch)

Enhance the `classify` branch so the completed step's expandable detail shows
the full decision (node output already carries all fields, including the new
`reason`):
- query type + reason
- sub-tasks
- strategy-memory decision (override/kept, chosen strategy, count, margin)

Today it shows only `Strategy: <type>` + sub-tasks.

## Testing

- **Backend unit** (`tests/test_admin/...`):
  - `_record_substep` sets `active_substep` to the mapped label for a
    `classify.*` name and clears it on `classify.done`.
  - the status endpoint returns `active_substep`.
  - `_format_live_step` classify branch renders reason + the strategy-memory
    decision (not just the type).
- **Live verification (deployed):** run the contracts query in the playground;
  the "Classify Query" row's sub-label changes during processing
  (reading data tables → classifying → checking strategy memory), and the
  completed step expands to the full classification decision.
- Full suite: no new failures vs the 15-failure baseline.

## Out of scope

- Live sub-steps for other long nodes (retrieve / enrich) — they emit no
  sub-steps yet; a future iteration can instrument them with the same reporter.
- Any change to the async API path or the CLI example (already shipped).
- Nested sub-step rows or a separate classification card (chose the inline
  sub-label).
