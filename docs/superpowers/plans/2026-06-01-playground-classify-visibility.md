# Playground Classify Sub-Progress Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:subagent-driven-development or superpowers:executing-plans. Steps use checkbox (`- [ ]`) syntax.

**Goal:** In the admin playground, show the active "Classify Query" step's live sub-step inline, and the full classification decision in its drill-down.

**Architecture:** Reuse the `progress` reporter (already an `AgentState` channel) inside the playground's own `run_query` astream loop. A module-level `_record_substep` writes `active_substep` into the job dict (which the status endpoint already returns whole). Two small module-level formatters make the logic testable. Frontend renders the inline sub-label.

**Tech Stack:** Python/FastAPI, vanilla JS, pytest.

---

### Task 1: `_classify_substep_label` (module-level mapping)

**Files:** Modify `src/admin/routes.py`. Test: `tests/test_admin/test_playground_substeps.py` (new).

- [ ] Step 1: Failing test — `_classify_substep_label("classify.strategy") == "checking strategy memory"`; `_classify_substep_label("classify.done") == ""`; `_classify_substep_label("retrieve") == ""`.
- [ ] Step 2: Run, verify FAIL (import error).
- [ ] Step 3: Add near `_playground_jobs`:
```python
from src.api.query_jobs import STEP_LABELS as _STEP_LABELS

def _classify_substep_label(name: str) -> str:
    """Friendly label for a live classify sub-step; '' for anything else
    (including classify.done, which clears the active sub-step)."""
    if name in ("classify.hints", "classify.llm", "classify.strategy"):
        return _STEP_LABELS.get(name, "")
    return ""
```
- [ ] Step 4: Run, verify PASS.
- [ ] Step 5: Commit.

### Task 2: `_record_substep` (module-level job updater)

**Files:** Modify `src/admin/routes.py`. Test: same new test file.

- [ ] Step 1: Failing test — seed `routes._playground_jobs["q"] = {"active_substep": ""}`; `_record_substep("q", "classify.strategy")` sets it to "checking strategy memory"; `_record_substep("q", "classify.done")` clears it to ""; unknown query id is a no-op.
- [ ] Step 2: Verify FAIL.
- [ ] Step 3: Add:
```python
def _record_substep(query_id: str, name: str) -> None:
    """Live progress reporter target for the playground: record the current
    classify sub-step on the job so the status poll can show it."""
    job = _playground_jobs.get(query_id)
    if job is None:
        return
    if name == "classify.done":
        job["active_substep"] = ""
    else:
        label = _classify_substep_label(name)
        if label:
            job["active_substep"] = label
```
- [ ] Step 4: Verify PASS.
- [ ] Step 5: Commit.

### Task 3: `_format_classify_detail` + wire into `_format_live_step`

**Files:** Modify `src/admin/routes.py`. Test: same new test file.

- [ ] Step 1: Failing test — `_format_classify_detail({"query_type": "analytical", "reason": "asks for pay by grade", "sub_tasks": ["x"], "strategy_memory": {"overrode": True, "llm_pick": "lookup", "memory_best": "analytical", "count": 7, "margin": 0.8}})` returns HTML containing "analytical", "asks for pay by grade", "x", and "override".
- [ ] Step 2: Verify FAIL.
- [ ] Step 3: Add module-level helper:
```python
def _format_classify_detail(output: dict) -> str:
    import html as _h
    qt = output.get("query_type", "")
    detail = f"<strong>Type:</strong> {_h.escape(str(qt))}"
    reason = output.get("reason") or ""
    if reason:
        detail += f"<br><strong>Reason:</strong> {_h.escape(str(reason))}"
    subs = output.get("sub_tasks") or []
    if subs:
        detail += "<br><strong>Sub-tasks:</strong> " + _h.escape(", ".join(str(s) for s in subs[:5]))
    sm = output.get("strategy_memory") or {}
    if sm.get("overrode"):
        detail += (f"<br><strong>Strategy memory:</strong> override {_h.escape(str(sm.get('llm_pick')))} "
                   f"&rarr; {_h.escape(str(sm.get('memory_best')))} (n={sm.get('count')}, "
                   f"margin={sm.get('margin')})")
    elif sm and sm.get("reason") not in (None, "disabled"):
        detail += f"<br><strong>Strategy memory:</strong> kept {_h.escape(str(sm.get('llm_pick')))} ({_h.escape(str(sm.get('reason')))})"
    return detail
```
Then in `_format_live_step`, replace the `classify` branch body with `return _format_classify_detail(output)`.
- [ ] Step 4: Verify PASS.
- [ ] Step 5: Commit.

### Task 4: Wire progress reporter + active_substep into run_query

**Files:** Modify `src/admin/routes.py` (`playground_start`/`run_query`).

- [ ] Step 1: In the job-dict init (`_playground_jobs[query_id] = {...}`), add `"active_substep": ""`.
- [ ] Step 2: Where the playground builds `initial_state` for the astream loop, add:
```python
initial_state["progress"] = lambda name, detail=None: _record_substep(query_id, name)
```
- [ ] Step 3: In the astream loop, when `node_name == "classify"` completes, clear it: `_playground_jobs[query_id]["active_substep"] = ""` (defensive).
- [ ] Step 4: Run `python -m pytest tests/test_admin -q` — existing admin tests still pass.
- [ ] Step 5: Commit.

### Task 5: Frontend inline sub-label (`playground.html`)

**Files:** Modify `src/admin/templates/playground.html`.

- [ ] Step 1: Give the label span an id: in `initProgressPanel`, change `<span> Step ${step.num} of 4: ${step.label}</span>` to `<span id="step-label-${step.id}"> Step ${step.num} of 4: ${step.label}</span>`.
- [ ] Step 2: Extend `markStepActive(stepId, elapsed, subLabel)` to update the label span:
```javascript
const labelEl = document.getElementById(`step-label-${stepId}`);
const base = ` Step ${STEPS.find(s=>s.id===stepId).num} of 4: ${STEPS.find(s=>s.id===stepId).label}`;
if (labelEl) labelEl.textContent = subLabel ? `${base} — ${subLabel}` : base;
```
- [ ] Step 3: In the poll loop active-step branch, pass the sub-label for classify:
`markStepActive(step.id, stepElapsed, step.id === 'classify' ? (status.active_substep || '') : '');`
- [ ] Step 4: Manual/live check in Task 6.
- [ ] Step 5: Commit.

### Task 6: Deploy + live verify

- [ ] Step 1: `docker compose up -d --build api`; wait healthy.
- [ ] Step 2: In the playground (or via the async path for the data), run "list all contracts awarded by the DHA"; watch the "Classify Query" row sub-label change during classify and expand the completed step to see type + reason + sub-tasks + strategy decision.
- [ ] Step 3: `python -m pytest -q` — no new failures vs the 15-failure baseline.

## Self-Review
- Spec coverage: label mapping (T1), job recording (T2), drill-down detail (T3), reporter wiring + active_substep + status (T4, auto via whole-dict return), frontend sub-label (T5), live verify (T6). Covered.
- Placeholders: none — full code inline.
- Consistency: sub-step names (`classify.hints/llm/strategy/done`), `active_substep` key, and `_classify_substep_label`/`_record_substep`/`_format_classify_detail` names consistent across tasks.
