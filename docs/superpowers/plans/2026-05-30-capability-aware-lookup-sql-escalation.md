# Capability-Aware LOOKUP → SQL Escalation — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make a LOOKUP-classified question that a registered structured table can answer actually run that table's SQL — so "pay rate for `<location>`" returns real data instead of "no data."

**Architecture:** Extract a small async helper `_lookup_then_structured` in `src/agent/graph.py` that runs `retrieve_lookup` and, if no SQL was produced, runs the gated `retrieve_structured` (its relevance gate = the capability check; original question; hints) and merges the results. Wire it into the retrieve node's LOOKUP branch. No new graph nodes, no loop, no LLM judge. Plus delete the dead `evaluate_context`.

**Tech Stack:** Python 3.11, LangGraph, pytest/pytest-asyncio.

**Spec:** `docs/superpowers/specs/2026-05-30-capability-aware-lookup-sql-escalation-design.md`

---

## File Structure

- `src/agent/graph.py` — add `_lookup_then_structured` helper; call it from the retrieve node's LOOKUP branch (modify).
- `tests/test_agent/test_graph.py` — unit tests for the helper; remove the stray `evaluator.generate` patch (modify).
- `src/agent/evaluator.py` — delete (dead code).
- `tests/test_agent/test_evaluator.py` — delete (tests the dead module).

---

## Task 1: `_lookup_then_structured` helper + wire into retrieve node

**Files:**
- Modify: `src/agent/graph.py` (add helper near line 47; change retrieve LOOKUP branch lines 67-69)
- Test: `tests/test_agent/test_graph.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_agent/test_graph.py`:

```python
@pytest.mark.asyncio
async def test_lookup_then_structured_merges_sql(monkeypatch):
    from src.agent import graph as g
    def fake_lookup(state, vector_store=None):
        return {"retrieved_chunks": ["L"], "retrieval_attempts": 1}
    async def fake_struct(state, vector_store, schema_registry):
        return {"sql_results": [{"x": 1}], "structured_trace": {"status": "ran"},
                "retrieved_chunks": ["S"]}
    monkeypatch.setattr(g, "retrieve_lookup", fake_lookup)
    monkeypatch.setattr(g, "retrieve_structured", fake_struct)
    out = await g._lookup_then_structured({"question": "q", "user_groups": ["x"]},
                                          vector_store=None, schema_registry=None)
    assert out["sql_results"] == [{"x": 1}]
    assert out["structured_trace"] == {"status": "ran"}
    assert out["retrieved_chunks"] == ["L", "S"]


@pytest.mark.asyncio
async def test_lookup_then_structured_no_table_unchanged(monkeypatch):
    from src.agent import graph as g
    def fake_lookup(state, vector_store=None):
        return {"retrieved_chunks": ["L"], "retrieval_attempts": 1}
    async def fake_struct(state, vector_store, schema_registry):
        return {}   # no relevant table -> gate miss
    monkeypatch.setattr(g, "retrieve_lookup", fake_lookup)
    monkeypatch.setattr(g, "retrieve_structured", fake_struct)
    out = await g._lookup_then_structured({"question": "q", "user_groups": ["x"]},
                                          vector_store=None, schema_registry=None)
    assert "sql_results" not in out
    assert out["retrieved_chunks"] == ["L"]


@pytest.mark.asyncio
async def test_lookup_then_structured_skips_when_sql_present(monkeypatch):
    from src.agent import graph as g
    calls = {"struct": 0}
    def fake_lookup(state, vector_store=None):
        return {"retrieved_chunks": ["L"], "sql_results": [{"y": 2}]}
    async def fake_struct(state, vector_store, schema_registry):
        calls["struct"] += 1
        return {"sql_results": [{"z": 3}]}
    monkeypatch.setattr(g, "retrieve_lookup", fake_lookup)
    monkeypatch.setattr(g, "retrieve_structured", fake_struct)
    out = await g._lookup_then_structured({"question": "q", "user_groups": ["x"]},
                                          vector_store=None, schema_registry=None)
    assert out["sql_results"] == [{"y": 2}]
    assert calls["struct"] == 0
```

- [ ] **Step 2: Run to verify failure**

Run: `python3 -m pytest tests/test_agent/test_graph.py -k lookup_then_structured -v`
Expected: FAIL with `AttributeError: module 'src.agent.graph' has no attribute '_lookup_then_structured'`.

- [ ] **Step 3: Add the helper**

In `src/agent/graph.py`, add this function immediately after `_rerank_merge` (it ends near line 46, before `def create_agent_graph`). It references the module-level `retrieve_lookup` and `retrieve_structured` (already imported at the top of the file, lines 7 and 11):

```python
async def _lookup_then_structured(retry_state, vector_store, schema_registry) -> dict:
    """LOOKUP retrieval plus a capability-aware escalation. LOOKUP never queries
    structured tables, so if no SQL was produced and a registered table is relevant
    to the question, also run the gated SQL path (retrieve_structured) and merge its
    results. The relevance gate inside retrieve_structured IS the capability check;
    it uses the original question and carries domain hints. retrieve_structured
    returns {} (no-op) when no table is relevant, so plain lookups are unchanged."""
    import asyncio
    result = await asyncio.to_thread(retrieve_lookup, retry_state, vector_store=vector_store)
    if not result.get("sql_results"):
        struct = await retrieve_structured(retry_state, vector_store, schema_registry)
        if struct.get("sql_results"):
            result["sql_results"] = struct["sql_results"]
            result["structured_trace"] = struct.get("structured_trace")
            result.setdefault("retrieved_chunks", []).extend(struct.get("retrieved_chunks", []))
    return result
```

- [ ] **Step 4: Run the helper tests to verify they pass**

Run: `python3 -m pytest tests/test_agent/test_graph.py -k lookup_then_structured -v`
Expected: 3 passed.

- [ ] **Step 5: Wire the helper into the retrieve node**

In `src/agent/graph.py`, the retrieve node's LOOKUP branch currently reads (lines 67-69):

```python
        if query_type == QueryType.LOOKUP:
            import asyncio as _asyncio_lookup
            result = await _asyncio_lookup.to_thread(retrieve_lookup, retry_state, vector_store=vector_store)
```

Replace those three lines with:

```python
        if query_type == QueryType.LOOKUP:
            result = await _lookup_then_structured(retry_state, vector_store, schema_registry)
```

(The `import asyncio as _asyncio_lookup` line is removed here because the helper does its own `import asyncio`. Leave the separate `import asyncio as _asyncio` in the SWEEP branch untouched.)

- [ ] **Step 6: Run the existing graph tests to confirm no regression**

Run: `python3 -m pytest tests/test_agent/test_graph.py -v`
Expected: all pass. In particular `test_run_agent_lookup` and `test_run_agent_no_results` still pass — they pass a bare `SchemaRegistry()` (empty), so `retrieve_structured`'s gate finds no tables, returns `{}`, and the escalation no-ops.

- [ ] **Step 7: Commit**

```bash
git add src/agent/graph.py tests/test_agent/test_graph.py
git commit -m "feat: LOOKUP escalates to gated SQL when a structured table is relevant"
```

---

## Task 2: Delete the dead `evaluate_context`

`evaluate_context` (evaluator.py) is never called in `src/` (confirmed: `grep -rn "evaluate_context" src/` returns only its own definition). It contains a lookup→sweep escalation that is NOT wired into the graph, and mistaking it for live behavior caused a multi-step misdiagnosis. Remove it and its now-orphaned references.

**Files:**
- Delete: `src/agent/evaluator.py`
- Delete: `tests/test_agent/test_evaluator.py`
- Modify: `tests/test_agent/test_graph.py` (remove the stray `evaluator.generate` patch in `test_run_agent_lookup`)

- [ ] **Step 1: Confirm nothing in `src/` imports it**

Run: `grep -rn "evaluator\|evaluate_context" src/`
Expected: matches ONLY inside `src/agent/evaluator.py` itself (no other `src/` file imports it). If any other `src/` file imports it, STOP and report — the deletion premise is wrong.

- [ ] **Step 2: Remove the stray patch in `test_run_agent_lookup`**

In `tests/test_agent/test_graph.py`, `test_run_agent_lookup` currently nests a patch of `src.agent.evaluator.generate`:

```python
            with patch("src.agent.evaluator.generate", return_value='{"sufficient": true, "reason": "ok"}'):
                with patch("src.agent.synthesizer.generate", return_value="Policy 4.2 requires approval for expenses over $500 [1]."):
                    from src.db.schema_registry import SchemaRegistry
                    result = await run_agent(question="What is policy 4.2?", user_groups=["finance"], vector_store=mock_store, schema_registry=SchemaRegistry())
```

Replace that block with (drop the evaluator patch, de-indent its body by 4 spaces):

```python
            with patch("src.agent.synthesizer.generate", return_value="Policy 4.2 requires approval for expenses over $500 [1]."):
                from src.db.schema_registry import SchemaRegistry
                result = await run_agent(question="What is policy 4.2?", user_groups=["finance"], vector_store=mock_store, schema_registry=SchemaRegistry())
```

- [ ] **Step 3: Delete the dead module and its test**

```bash
git rm src/agent/evaluator.py tests/test_agent/test_evaluator.py
```

- [ ] **Step 4: Run the graph tests + a full import check**

Run: `python3 -m pytest tests/test_agent/test_graph.py -v`
Expected: all pass.
Run: `python3 -c "import src.agent.graph, src.main"`
Expected: no ImportError (nothing imported the deleted module).

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "chore: delete dead evaluate_context (never wired into the graph)"
```

---

## Task 3: Full suite + live verification

**Files:** none (verification only)

- [ ] **Step 1: Run the full agent suite**

Run: `python3 -m pytest tests/test_agent/ -q`
Expected: all pass, 0 failures (the 8 historical failures were fixed earlier this session; confirm none reappear).

- [ ] **Step 2: Live acceptance — the original failing question (post-deploy)**

After rebuild + recreate of `sauron-api-1`, run the FULL pipeline (this is the end-to-end test that the earlier component-only verification missed):

```bash
docker exec sauron-api-1 python3 -c "
import asyncio
from src.api.routes_ingest import get_metadata_store, get_hint_store, get_vector_store, get_schema_registry
from src.ingestion.tabular_ingest import populate_hint_store, populate_schema_registry
from src.agent.graph import run_agent
store=get_metadata_store()
asyncio.run(populate_hint_store(store,get_hint_store()))
asyncio.run(populate_schema_registry(store,get_schema_registry()))
reg=get_schema_registry(); vs=get_vector_store()
res=asyncio.run(run_agent(question='what are the pay rates for florida?', user_groups=['executives'], vector_store=vs, schema_registry=reg, metadata_store=store))
print('ANSWER:', res.answer[:400])
print('CITATIONS:', [c.filename for c in res.citations][:5])
"
```

Expected: the answer now contains real GS pay numbers for Florida (e.g. per-grade min/max dollar figures), NOT "the provided context does not contain information regarding pay rates for Florida." Record the actual answer in the completion notes.

- [ ] **Step 3: Spot-check no regression on a non-table lookup**

```bash
docker exec sauron-api-1 python3 -c "
import asyncio
from src.api.routes_ingest import get_metadata_store, get_hint_store, get_vector_store, get_schema_registry
from src.ingestion.tabular_ingest import populate_hint_store, populate_schema_registry
from src.agent.graph import run_agent
store=get_metadata_store()
asyncio.run(populate_hint_store(store,get_hint_store()))
asyncio.run(populate_schema_registry(store,get_schema_registry()))
reg=get_schema_registry(); vs=get_vector_store()
res=asyncio.run(run_agent(question='what does the document say about supervisory positions?', user_groups=['executives'], vector_store=vs, schema_registry=reg, metadata_store=store))
print('ANSWER:', res.answer[:300])
"
```

Expected: a normal prose answer (a non-pay lookup still works; escalation gate either misses or adds harmless SQL). Confirm it doesn't error.

- [ ] **Step 4: Final status**

```bash
git status   # clean
```

---

## Self-Review Notes (completed by plan author)

- **Spec coverage:** inline helper in the retrieve LOOKUP branch (Task 1); uses `retrieve_structured` = gated capability check + original question + hints (Task 1 helper); fail-open / guarded so plain lookups unchanged (Task 1 tests 2 & 3); dead `evaluate_context` removed (Task 2); live florida acceptance + non-table spot-check (Task 3). No spec gaps.
- **Type/signature consistency:** `_lookup_then_structured(retry_state, vector_store, schema_registry)` is defined in Task 1 Step 3 and called identically in Step 5 and the tests in Step 1. It reads `result.get("sql_results")`, `struct.get("sql_results"/"structured_trace"/"retrieved_chunks")` — matching the dict shapes `retrieve_structured` returns (`sql_results`, `structured_trace`, `retrieved_chunks`).
- **Placeholder scan:** none — every code/edit step shows complete code; the `grep` in Task 2 Step 1 is a real safety pre-check with a defined STOP condition.
