# Classify-step Visibility Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or superpowers:executing-plans. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Surface live classify sub-steps (with timing) and the classify decision (query type, reason, sub-tasks, strategy memory) in the async query poll status.

**Architecture:** A `progress(name, detail)` callback injected into `AgentState` by `run_agent_streamed`; the classify node calls it mid-execution; the async worker records a `steps` timeline + a `classification` detail on the job and exposes them in the status response. Observability only — no logic change.

**Tech Stack:** Python, FastAPI/pydantic, LangGraph, pytest.

---

### Task 1: Classifier returns a one-line `reason`

**Files:** Modify `src/agent/classifier.py` (prompt + `classify_query`). Test: `tests/test_agent/test_classifier.py`.

- [ ] Step 1: Write failing test — `classify_query` returns `reason` from JSON, fail-open to `""`.
- [ ] Step 2: Run it, verify FAIL.
- [ ] Step 3: Add `reason` to the classifier prompt JSON shape and parse it (`parsed.get("reason", "")`), include in the returned dict.
- [ ] Step 4: Run classifier tests, verify PASS.
- [ ] Step 5: Commit.

### Task 2: AgentState `progress` field + graph injection

**Files:** Modify `src/agent/state.py` (add field), `src/agent/graph.py` (`run_agent_streamed`). Test: `tests/test_agent/test_graph.py`.

- [ ] Step 1: Write failing test — when `run_agent_streamed` is given a `step_callback`, the classify node receives a callable `progress` in state (assert via a stub callback capturing names).
- [ ] Step 2: Verify FAIL.
- [ ] Step 3: Add `progress` to `AgentState`; in `run_agent_streamed`, when `step_callback` provided, set `initial_state["progress"] = lambda name, detail=None: step_callback(name, detail)`; keep per-node `step_callback(node_name)`.
- [ ] Step 4: Verify PASS.
- [ ] Step 5: Commit.

### Task 3: Classify node emits sub-steps + classification detail

**Files:** Modify `src/agent/classifier.py` (`classify_node`). Test: `tests/test_agent/test_classifier.py`.

- [ ] Step 1: Write failing test — inject a recording `progress`; assert calls `classify.hints`, `classify.llm`, `classify.strategy`, then `classify.done` with `detail={"kind":"classification","data":{query_type, reason, sub_tasks, strategy_memory}}`. No-op when `progress` absent.
- [ ] Step 2: Verify FAIL.
- [ ] Step 3: In `classify_node`, read `progress = state.get("progress") or (lambda *a, **k: None)`; call it before hint resolution, before `classify_query`, before strategy memory, and at the end with the classification detail.
- [ ] Step 4: Verify PASS.
- [ ] Step 5: Commit.

### Task 4: QueryJob steps timeline + classification + labels

**Files:** Modify `src/api/query_jobs.py`. Test: `tests/test_api/test_query_jobs.py`.

- [ ] Step 1: Write failing test — `update_step(token, "classify.llm")` appends `{"step": "classifying question", "at": <float>}` to `job.steps`; `update_step(token, "classify.done", {"kind":"classification","data":{...}})` sets `job.classification`.
- [ ] Step 2: Verify FAIL.
- [ ] Step 3: Add `steps: list` + `classification: dict|None` to `QueryJob`; extend `STEP_LABELS` with the four `classify.*` sub-steps; `update_step(token, node, detail=None)` appends to `steps` and stores classification detail.
- [ ] Step 4: Verify PASS.
- [ ] Step 5: Commit.

### Task 5: Status response exposes steps + classification

**Files:** Modify `src/api/models.py` (`AsyncQueryStatusResponse`), `src/api/routes_query.py` (poll handler). Test: `tests/test_api/test_query_async.py` (or existing async route test).

- [ ] Step 1: Write failing test — the poll handler returns `steps` and `classification` from the job.
- [ ] Step 2: Verify FAIL.
- [ ] Step 3: Add `steps: list[dict] = []` and `classification: dict | None = None` to `AsyncQueryStatusResponse`; populate both in the handler.
- [ ] Step 4: Verify PASS.
- [ ] Step 5: Commit.

### Task 6: Example script prints timeline + classification

**Files:** Modify `examples/query_async.py`.

- [ ] Step 1: Update the poll loop to print new entries from `steps` (with elapsed `at`), and after completion (or when present) print a classification block: type + reason, sub-tasks, strategy-memory decision. (Manual/script change — verified live in Task 7.)
- [ ] Step 2: Commit.

### Task 7: Deploy + live verify

- [ ] Step 1: `docker compose up -d --build api` ; wait healthy.
- [ ] Step 2: Run `python examples/query_async.py "what are the pay rates for a gs-13?" --groups executives` (reworded to avoid cache); confirm classify sub-steps stream with timing and the classification block prints.
- [ ] Step 3: `python -m pytest -q` — no new failures vs the 15-failure baseline.

## Self-Review
- Spec coverage: progress plumbing (T2), classify sub-steps + detail (T3), reason (T1), job timeline + classification + labels (T4), response fields (T5), example script (T6), live verify (T7). All covered.
- Placeholders: none — each task names exact files, behaviors, and step names.
- Consistency: sub-step names (`classify.hints/llm/strategy/done`), detail shape (`{"kind":"classification","data":{...}}`), and field names (`steps`, `classification`) match across tasks.
