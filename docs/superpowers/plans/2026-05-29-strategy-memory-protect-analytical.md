# Protect ANALYTICAL routing from Strategy Memory override — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** When the LLM classifier picks ANALYTICAL (a capability-gated pick made only when a relevant structured table is registered + ACL-visible), Strategy Memory must not override it back to another strategy.

**Architecture:** One guard inside the existing Strategy Memory override branch in `classify_node` (`src/agent/classifier.py`): if `llm_pick == QueryType.ANALYTICAL`, suppress the override and record `reason="protected"` in the existing `memory_decision` telemetry. All other override paths (lookup↔sweep/temporal, agreement, below-gate) are untouched.

**Tech Stack:** Python, pytest (`pytest-asyncio`). Spec: `docs/superpowers/specs/2026-05-29-strategy-memory-protect-analytical-design.md`.

**Reference facts (verified in current code):**
- `classify_node` is built by `_classify_node_factory(schema_registry)` in `src/agent/classifier.py` (async node).
- The override branch is at `src/agent/classifier.py:137-144`:
  ```python
  elif (mem_type is not None
          and best["count"] >= settings.strategy_memory_min_runs
          and best["margin"] >= settings.strategy_memory_margin):
      result["query_type"] = mem_type
      memory_decision["overrode"] = True
      memory_decision["reason"] = "override"
      logger.info("Strategy memory override: %s -> %s (n=%d, margin=%.0f%%)",
                  llm_pick, mem_type, best["count"], best["margin"] * 100)
  ```
- `memory_decision` is initialized `{"llm_pick": str(llm_pick), "overrode": False, "reason": "disabled"}` and, when `best` exists, updated with `memory_best`, `count`, `margin` (lines 127-130).
- `get_best_strategy(question)` is async and returns either `None` or a dict with keys `strategy`, `count`, `margin` (plus others). Existing tests monkeypatch `classifier.get_best_strategy`.
- `QueryType` is imported in `classifier.py`. Existing classifier tests monkeypatch `classifier.generate` (the LLM call) and use `_schema(...)`, `SchemaRegistry`, `_classify_node_factory`.
- `settings`: `strategy_memory_min_runs=3`, `strategy_memory_margin=0.15`, `strategy_memory_enabled=True` (defaults).

---

## File Structure

- Modify `src/agent/classifier.py` — add the ANALYTICAL guard inside the override branch of `classify_node`.
- Test `tests/test_agent/test_classifier.py` — add 4 node-level tests.

---

## Task 1: Protect ANALYTICAL picks from memory override

**Files:**
- Modify: `src/agent/classifier.py` (the override branch in `classify_node`, currently lines 137-144)
- Test: `tests/test_agent/test_classifier.py`

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_agent/test_classifier.py` (the helpers `_schema`, `SchemaRegistry`, `_classify_node_factory`, `classifier`, `QueryType`, and `pytest` are already imported/defined in the file):

```python
@pytest.mark.asyncio
async def test_memory_does_not_override_analytical(monkeypatch):
    # LLM picks ANALYTICAL (capability-gated); memory wants lookup past the gates.
    # The override must be suppressed and recorded as "protected".
    def fake_generate(system_prompt, user_prompt, **kwargs):
        return '{"query_type": "analytical", "sub_tasks": []}'
    monkeypatch.setattr(classifier, "generate", fake_generate)
    async def _best(q):
        return {"strategy": "lookup", "count": 3, "margin": 1.0}
    monkeypatch.setattr(classifier, "get_best_strategy", _best, raising=False)
    async def _no_hints(schemas):
        return {}
    monkeypatch.setattr(classifier, "_resolve_hints_for_classifier", _no_hints)

    reg = SchemaRegistry()
    reg.register(_schema(table="doc_pay", desc="military pay", acl=["ALL"]))
    node = _classify_node_factory(reg)
    out = await node({"question": "pay range for an officer?", "user_groups": ["ALL"]})

    assert out["query_type"] == QueryType.ANALYTICAL
    assert out["strategy_memory"]["overrode"] is False
    assert out["strategy_memory"]["reason"] == "protected"
    assert out["strategy_memory"]["memory_best"] == "lookup"


@pytest.mark.asyncio
async def test_memory_still_overrides_non_analytical(monkeypatch):
    # Regression: a learned override among non-structured strategies still applies.
    def fake_generate(system_prompt, user_prompt, **kwargs):
        return '{"query_type": "lookup", "sub_tasks": []}'
    monkeypatch.setattr(classifier, "generate", fake_generate)
    async def _best(q):
        return {"strategy": "sweep", "count": 5, "margin": 0.5}
    monkeypatch.setattr(classifier, "get_best_strategy", _best, raising=False)

    node = _classify_node_factory(None)
    out = await node({"question": "list all contracts", "user_groups": ["ALL"]})

    assert out["query_type"] == QueryType.SWEEP
    assert out["strategy_memory"]["overrode"] is True
    assert out["strategy_memory"]["reason"] == "override"


@pytest.mark.asyncio
async def test_memory_agreement_on_analytical_unchanged(monkeypatch):
    # When memory agrees with an ANALYTICAL pick, reason stays "agreed".
    def fake_generate(system_prompt, user_prompt, **kwargs):
        return '{"query_type": "analytical", "sub_tasks": []}'
    monkeypatch.setattr(classifier, "generate", fake_generate)
    async def _best(q):
        return {"strategy": "analytical", "count": 3, "margin": 1.0}
    monkeypatch.setattr(classifier, "get_best_strategy", _best, raising=False)

    node = _classify_node_factory(None)
    out = await node({"question": "pay range for an officer?", "user_groups": ["ALL"]})

    assert out["query_type"] == QueryType.ANALYTICAL
    assert out["strategy_memory"]["overrode"] is False
    assert out["strategy_memory"]["reason"] == "agreed"


@pytest.mark.asyncio
async def test_memory_below_gate_unchanged(monkeypatch):
    # A differing memory pick that fails the count gate is "below gate", no override.
    def fake_generate(system_prompt, user_prompt, **kwargs):
        return '{"query_type": "lookup", "sub_tasks": []}'
    monkeypatch.setattr(classifier, "generate", fake_generate)
    async def _best(q):
        return {"strategy": "sweep", "count": 1, "margin": 1.0}
    monkeypatch.setattr(classifier, "get_best_strategy", _best, raising=False)

    node = _classify_node_factory(None)
    out = await node({"question": "who is John?", "user_groups": ["ALL"]})

    assert out["query_type"] == QueryType.LOOKUP
    assert out["strategy_memory"]["overrode"] is False
    assert out["strategy_memory"]["reason"] == "below gate"
```

- [ ] **Step 2: Run tests to verify they fail**

Run (host python may be too old — if so use the docker form below):
`pytest tests/test_agent/test_classifier.py -k "does_not_override_analytical or agreement_on_analytical" -v`
Docker form: `docker exec -i sauron-api-1 env PYTHONPATH=/app python -m pytest "tests/test_agent/test_classifier.py" -k "does_not_override_analytical or agreement_on_analytical" -v` (if the container lacks the new test file, `docker cp tests/test_agent/test_classifier.py sauron-api-1:/app/tests/test_agent/test_classifier.py` first).
Expected: `test_memory_does_not_override_analytical` FAILS — current code overrides ANALYTICAL→lookup, so `query_type` becomes LOOKUP and `reason` is `"override"`, not `"protected"`. (`test_memory_agreement_on_analytical_unchanged` already passes — it's a guard against regressions.)

- [ ] **Step 3: Add the guard**

In `src/agent/classifier.py`, replace the override branch (currently lines 137-144):

```python
                    elif (mem_type is not None
                            and best["count"] >= settings.strategy_memory_min_runs
                            and best["margin"] >= settings.strategy_memory_margin):
                        result["query_type"] = mem_type
                        memory_decision["overrode"] = True
                        memory_decision["reason"] = "override"
                        logger.info("Strategy memory override: %s -> %s (n=%d, margin=%.0f%%)",
                                    llm_pick, mem_type, best["count"], best["margin"] * 100)
```

with:

```python
                    elif (mem_type is not None
                            and best["count"] >= settings.strategy_memory_min_runs
                            and best["margin"] >= settings.strategy_memory_margin):
                        if llm_pick == QueryType.ANALYTICAL:
                            # Capability-gated pick: ANALYTICAL is chosen only when a
                            # relevant structured table is registered + ACL-visible. A
                            # learned prior (trainable by cited-but-unhelpful answers)
                            # must not veto it. Memory relearns once analytical runs.
                            memory_decision["reason"] = "protected"
                            logger.info("Strategy memory suppressed: analytical capability "
                                        "pick protected (memory wanted %s, n=%d, margin=%.0f%%)",
                                        mem_type, best["count"], best["margin"] * 100)
                        else:
                            result["query_type"] = mem_type
                            memory_decision["overrode"] = True
                            memory_decision["reason"] = "override"
                            logger.info("Strategy memory override: %s -> %s (n=%d, margin=%.0f%%)",
                                        llm_pick, mem_type, best["count"], best["margin"] * 100)
```

(`memory_decision["overrode"]` stays `False` from its initialization in the protected case; `memory_best`/`count`/`margin` were already set when `best` was found.)

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_agent/test_classifier.py -v` (or the docker form from Step 2).
Expected: PASS — all four new tests plus every pre-existing classifier test.

- [ ] **Step 5: Commit**

```bash
git add src/agent/classifier.py tests/test_agent/test_classifier.py
git commit -m "fix: protect capability-gated ANALYTICAL routing from strategy-memory override"
```

---

## Task 2: Deploy + live verification

**Files:** none (verification only)

- [ ] **Step 1: Rebuild and restart the api container**

Run: `docker compose build api && docker compose up -d api`
Expected: `api` recreated and healthy (`docker inspect -f '{{.State.Health.Status}}' sauron-api-1` → `healthy`).

- [ ] **Step 2: Confirm no live override on the officer query**

Drive the live query in-container (the table is ACL-scoped to `executives`):
```bash
docker exec -i sauron-api-1 env PYTHONPATH=/app python - <<'PY'
import json, urllib.request
from src.auth.jwt import create_token
tok = create_token("verify", ["executives"])
req = urllib.request.Request("http://localhost:8080/api/v1/query",
    data=json.dumps({"question": "What is the pay range for an officer?"}).encode(),
    headers={"Content-Type":"application/json","X-API-Key":"dev-key-1","Authorization":f"Bearer {tok}"},
    method="POST")
print(json.load(urllib.request.urlopen(req, timeout=300))["answer"][:300])
PY
docker logs sauron-api-1 2>&1 | grep -iE "Classified|Strategy memory|Text-to-SQL returned" | tail -5
```
Expected: logs show `Classified ... -> analytical (tables_available=True)` followed by `Text-to-SQL returned N row(s)` and **no** `Strategy memory override: analytical -> ...` line (a `Strategy memory suppressed: analytical capability pick protected` line is fine). The answer is sourced from the PDF table.

- [ ] **Step 3: Report**

Report the routing logs + answer as evidence.

---

## Notes for the implementer

- YAGNI: scope is ANALYTICAL only. Do not add CROSS_REFERENCE protection or a config flag — both are explicitly out of scope in the spec.
- Do not touch `get_best_strategy`, `normalize_query_pattern`, the gate settings, or `log_strategy_result`.
- Container source is baked at build time (no src volume mount), so live verification requires the rebuild in Task 2. To run tests against the container before rebuild, `docker cp` the changed files in (as prior tasks did).
