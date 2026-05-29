# ANALYTICAL routing + resilient structured fallback — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Route value/range questions over registered tables (e.g. "pay range for an officer") to ANALYTICAL, and when the model's SQL returns no rows, fall back to glossary-annotated `table_row` narratives instead of answering "no data."

**Architecture:** Two contained changes. (A) Enrich the classifier's table view with resolved, ACL-filtered hint notes so the LLM can recognize a table's domain. (B) In `retrieve_analytical`, treat a zero-row SQL result as a miss and fall back to `retrieve_structured` (SQL + `tier="table_row"` narratives), then to map-reduce only if structured gates out.

**Tech Stack:** Python, pytest (`pytest-asyncio`), LangGraph nodes, DuckDB-backed structured store. Spec: `docs/superpowers/specs/2026-05-29-analytical-routing-structured-fallback-design.md`.

**Reference facts (verified in code):**
- `ResolvedHints` (`src/agent/strategies/hint_resolver.py:8`): `column_glossaries: dict[str, dict[str, str]]` (col → {code: meaning}), `column_notes: dict[str, str]`, `table_notes: list[str]`.
- `format_available_tables(schemas)` (`src/agent/classifier.py:30`) emits `- <table>: <description>`, sorted by table name.
- `classify_node` (`src/agent/classifier.py:_classify_node_factory`) is async; builds `available` from `schema_registry.list_for_user(...)`.
- `resolve_hints_for_schemas(schemas, hint_store, metadata_store)` (`src/agent/strategies/structured.py:219`) → `dict[table -> ResolvedHints]`, fail-open `{}`.
- `retrieve_analytical` (`src/agent/strategies/analytical.py`) falls back to map-reduce only on `trace.status == "error"`.
- `StructuredLookupTrace` (`src/agent/strategies/structured.py:76`) has `.status` (`"ran"|"skipped"|"error"`), `.row_count`, `.fell_back`, `.to_dict()`.
- `retrieve_structured(state, vector_store, schema_registry)` (`src/agent/strategies/structured.py:240`) → `{"sql_results", "retrieved_chunks", "structured_trace", ...}` or `{}` / `{"structured_trace": ...}` when it gates out.

---

## File Structure

- Modify `src/agent/classifier.py` — add `_hint_note` helper, add `hints` param to `format_available_tables`, add `_resolve_hints_for_classifier`, wire into `classify_node`.
- Modify `src/agent/strategies/analytical.py` — add zero-row fallback to `retrieve_structured`, then map-reduce.
- Test `tests/test_agent/test_classifier.py` — extend (Part A).
- Test `tests/test_agent/test_analytical_fallback.py` — create (Part B).

---

## Task 1: Hint-enriched classifier table view (Part A)

**Files:**
- Modify: `src/agent/classifier.py`
- Test: `tests/test_agent/test_classifier.py`

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_agent/test_classifier.py` (the existing `_schema` helper and imports are already present):

```python
from src.agent.classifier import _hint_note
from src.agent.strategies.hint_resolver import ResolvedHints


def test_hint_note_combines_table_notes_and_glossary_meanings():
    rh = ResolvedHints(
        column_glossaries={"col_0": {"O-1": "Commissioned Officer", "E-1": "Enlisted Member"}},
        column_notes={},
        table_notes=["U.S. military active-duty basic pay"],
    )
    note = _hint_note(rh)
    assert "U.S. military active-duty basic pay" in note
    assert "Commissioned Officer" in note
    assert "Enlisted Member" in note


def test_hint_note_is_length_capped():
    rh = ResolvedHints(table_notes=["x" * 500])
    assert len(_hint_note(rh)) <= 200


def test_format_available_tables_appends_note_when_hint_present():
    s = _schema(table="doc_pay", desc="financial values indexed by col_0")
    hints = {"doc_pay": ResolvedHints(table_notes=["U.S. military active-duty basic pay"])}
    line = format_available_tables([s], hints)
    assert line.startswith("- doc_pay: financial values indexed by col_0")
    assert "U.S. military active-duty basic pay" in line


def test_format_available_tables_unchanged_without_hints():
    # Byte-identical to pre-change behavior when no hints are supplied.
    assert format_available_tables([_schema()]) == "- doc_x_pay: GS pay by grade and step"
    assert format_available_tables([_schema()], {}) == "- doc_x_pay: GS pay by grade and step"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_agent/test_classifier.py -k "hint_note or appends_note or unchanged_without_hints" -v`
Expected: FAIL — `ImportError: cannot import name '_hint_note'`.

- [ ] **Step 3: Implement the helper and extend `format_available_tables`**

In `src/agent/classifier.py`, replace the existing `format_available_tables` function with:

```python
_MAX_NOTE_CHARS = 200


def _hint_note(rh) -> str:
    """Compact, length-capped domain note for a table, built from its resolved
    hints: table notes first, then the distinct glossary meanings (e.g. the human
    labels behind coded values). Lets the classifier recognize what a generically
    profiled table actually holds."""
    parts = [n for n in rh.table_notes if n]
    meanings: list[str] = []
    for col_map in rh.column_glossaries.values():
        for meaning in col_map.values():
            if meaning and meaning not in meanings:
                meanings.append(meaning)
    if meanings:
        parts.append(", ".join(meanings))
    return "; ".join(parts)[:_MAX_NOTE_CHARS]


def format_available_tables(schemas, hints=None) -> str:
    """One '- <table>: <description>' line per schema, sorted by table name for a
    stable (run-to-run identical) classifier prompt. When ``hints`` (table ->
    ResolvedHints) supplies a note for a table, it is appended after an em dash.
    With ``hints`` None/empty the output is byte-identical to before."""
    hints = hints or {}
    lines = []
    for s in sorted(schemas, key=lambda s: s.table):
        line = f"- {s.table}: {s.description}"
        rh = hints.get(s.table)
        note = _hint_note(rh) if rh is not None else ""
        if note:
            line += f" — {note}"
        lines.append(line)
    return "\n".join(lines)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_agent/test_classifier.py -v`
Expected: PASS (new tests + all existing classifier tests, including `test_format_available_tables`).

- [ ] **Step 5: Commit**

```bash
git add src/agent/classifier.py tests/test_agent/test_classifier.py
git commit -m "feat: hint-enriched classifier table view for ANALYTICAL routing"
```

---

## Task 2: Resolve hints in the classify node (Part A wiring)

**Files:**
- Modify: `src/agent/classifier.py`
- Test: `tests/test_agent/test_classifier.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_agent/test_classifier.py`:

```python
@pytest.mark.asyncio
async def test_node_factory_injects_resolved_hint_notes(monkeypatch):
    captured = {}
    def fake_generate(system_prompt, user_prompt, **kwargs):
        captured["system"] = system_prompt
        return '{"query_type": "analytical", "sub_tasks": []}'
    monkeypatch.setattr(classifier, "generate", fake_generate)
    async def _no_memory(q):
        return None
    monkeypatch.setattr(classifier, "get_best_strategy", _no_memory, raising=False)

    from src.agent.strategies.hint_resolver import ResolvedHints
    async def _fake_hints(schemas):
        return {"doc_pay": ResolvedHints(table_notes=["U.S. military active-duty basic pay"])}
    monkeypatch.setattr(classifier, "_resolve_hints_for_classifier", _fake_hints)

    reg = SchemaRegistry()
    reg.register(_schema(table="doc_pay", desc="financial values indexed by col_0", acl=["ALL"]))

    node = _classify_node_factory(reg)
    out = await node({"question": "pay range for an officer?", "user_groups": ["ALL"]})

    assert "U.S. military active-duty basic pay" in captured["system"]
    assert out["query_type"] == QueryType.ANALYTICAL


@pytest.mark.asyncio
async def test_resolve_hints_for_classifier_fails_open(monkeypatch):
    # Any error resolving hints must yield {} (never break classification).
    import src.agent.strategies.structured as structured
    async def _boom(*a, **k):
        raise RuntimeError("store down")
    monkeypatch.setattr(structured, "resolve_hints_for_schemas", _boom)
    assert await classifier._resolve_hints_for_classifier([_schema()]) == {}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_agent/test_classifier.py -k "injects_resolved_hint_notes or fails_open" -v`
Expected: FAIL — `AttributeError: module 'src.agent.classifier' has no attribute '_resolve_hints_for_classifier'`.

- [ ] **Step 3: Add the resolver helper and wire it into the node**

In `src/agent/classifier.py`, add the module-level helper (place it just above `_classify_node_factory`):

```python
async def _resolve_hints_for_classifier(schemas) -> dict:
    """Fail-open hint resolution for the classifier table view. Mirrors the call
    retrieve_analytical uses; returns {} on any error so classification never breaks."""
    try:
        from src.agent.strategies.structured import resolve_hints_for_schemas
        from src.api.routes_ingest import get_hint_store, get_metadata_store
        return await resolve_hints_for_schemas(schemas, get_hint_store(), get_metadata_store())
    except Exception:
        logger.warning("Classifier hint resolution failed; using bare table descriptions", exc_info=True)
        return {}
```

Then in `classify_node` (inside `_classify_node_factory`), replace:

```python
        available = ""
        if schema_registry is not None:
            schemas = schema_registry.list_for_user(state.get("user_groups", ["ALL"]))
            available = format_available_tables(schemas)
```

with:

```python
        available = ""
        if schema_registry is not None:
            schemas = schema_registry.list_for_user(state.get("user_groups", ["ALL"]))
            hints = await _resolve_hints_for_classifier(schemas)
            available = format_available_tables(schemas, hints)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_agent/test_classifier.py -v`
Expected: PASS (all classifier tests).

- [ ] **Step 5: Commit**

```bash
git add src/agent/classifier.py tests/test_agent/test_classifier.py
git commit -m "feat: resolve ACL-filtered hint notes in the classify node"
```

---

## Task 3: Zero-row fallback in `retrieve_analytical` (Part B)

**Files:**
- Modify: `src/agent/strategies/analytical.py`
- Test: `tests/test_agent/test_analytical_fallback.py` (create)

- [ ] **Step 1: Write the failing tests**

Create `tests/test_agent/test_analytical_fallback.py`:

```python
"""retrieve_analytical: zero-row SQL must fall back to structured narratives,
then to map-reduce only when structured gates out."""
import pytest

import src.agent.strategies.analytical as analytical
import src.agent.strategies.structured as structured
import src.agent.strategies.map_reduce as map_reduce
from src.agent.strategies.structured import StructuredLookupTrace


class _Schema:
    def __init__(self, table="doc_pay"):
        self.table = table


class _Registry:
    def list_for_user(self, groups):
        return [_Schema()]


def _state():
    return {"question": "pay range for an officer?", "user_groups": ["ALL"],
            "retrieval_attempts": 0}


@pytest.fixture(autouse=True)
def _stub_hints(monkeypatch):
    async def _no_hints(schemas, hint_store, metadata_store):
        return {}
    monkeypatch.setattr(structured, "resolve_hints_for_schemas", _no_hints)
    monkeypatch.setattr("src.api.routes_ingest.get_hint_store", lambda: object(), raising=False)
    monkeypatch.setattr("src.api.routes_ingest.get_metadata_store", lambda: object(), raising=False)


@pytest.mark.asyncio
async def test_rows_returned_unchanged(monkeypatch):
    trace = StructuredLookupTrace(query_type="analytical", status="ran", row_count=2,
                                  rows=[{"a": 1}, {"a": 2}])
    monkeypatch.setattr(structured, "run_structured_lookup", lambda *a, **k: trace)
    called = {"structured": False}
    async def _should_not_run(*a, **k):
        called["structured"] = True
        return {}
    monkeypatch.setattr(structured, "retrieve_structured", _should_not_run)

    out = await analytical.retrieve_analytical(_state(), vector_store=object(), schema_registry=_Registry())
    assert out["sql_results"] == [{"a": 1}, {"a": 2}]
    assert called["structured"] is False


@pytest.mark.asyncio
async def test_zero_rows_falls_back_to_structured(monkeypatch):
    trace = StructuredLookupTrace(query_type="analytical", status="ran", row_count=0, rows=[])
    monkeypatch.setattr(structured, "run_structured_lookup", lambda *a, **k: trace)
    async def _structured(state, vector_store, schema_registry):
        return {"sql_results": [], "retrieved_chunks": [{"id": "row1"}],
                "structured_trace": {"query_type": "sweep"}}
    monkeypatch.setattr(structured, "retrieve_structured", _structured)

    out = await analytical.retrieve_analytical(_state(), vector_store=object(), schema_registry=_Registry())
    assert out["retrieved_chunks"] == [{"id": "row1"}]
    assert out["structured_trace"]["fell_back"] is True          # analytical trace preserved + flagged
    assert out["structured_trace"]["query_type"] == "analytical"


@pytest.mark.asyncio
async def test_zero_rows_then_structured_empty_falls_back_to_map_reduce(monkeypatch):
    trace = StructuredLookupTrace(query_type="analytical", status="ran", row_count=0, rows=[])
    monkeypatch.setattr(structured, "run_structured_lookup", lambda *a, **k: trace)
    async def _structured(state, vector_store, schema_registry):
        return {}  # table not relevant / gated out
    monkeypatch.setattr(structured, "retrieve_structured", _structured)
    async def _map_reduce(state, vector_store):
        return {"retrieved_chunks": [{"id": "mr"}]}
    monkeypatch.setattr(map_reduce, "retrieve_map_reduce", _map_reduce)

    out = await analytical.retrieve_analytical(_state(), vector_store=object(), schema_registry=_Registry())
    assert out["retrieved_chunks"] == [{"id": "mr"}]
    assert out["structured_trace"]["fell_back"] is True


@pytest.mark.asyncio
async def test_hard_error_still_falls_back_to_map_reduce(monkeypatch):
    trace = StructuredLookupTrace(query_type="analytical", status="error", error="bad sql")
    monkeypatch.setattr(structured, "run_structured_lookup", lambda *a, **k: trace)
    async def _map_reduce(state, vector_store):
        return {"retrieved_chunks": [{"id": "mr"}]}
    monkeypatch.setattr(map_reduce, "retrieve_map_reduce", _map_reduce)

    out = await analytical.retrieve_analytical(_state(), vector_store=object(), schema_registry=_Registry())
    assert out["retrieved_chunks"] == [{"id": "mr"}]
    assert out["structured_trace"]["status"] == "error"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_agent/test_analytical_fallback.py -v`
Expected: FAIL — `test_zero_rows_*` assert wrong values because current code returns `sql_results: []` with no fallback (`retrieve_structured`/map-reduce never invoked on zero rows).

- [ ] **Step 3: Implement the zero-row fallback**

In `src/agent/strategies/analytical.py`, replace the tail of `retrieve_analytical` — from the `trace = await asyncio.to_thread(...)` line through the final `return {...}` — with:

```python
    trace = await asyncio.to_thread(run_structured_lookup, question, schemas, "analytical", None, None, hints)

    if trace.status == "error":
        from src.agent.strategies.map_reduce import retrieve_map_reduce
        result = await retrieve_map_reduce(state, vector_store=vector_store)
        result["structured_trace"] = trace.to_dict()
        return result

    # Runnable-but-empty SQL (e.g. WHERE col_0='officer' when values are O-1..O-10)
    # is a miss, not an answer. Fall back to the structured row-narrative path so the
    # glossary-annotated table_row chunks reach the synthesizer; then map-reduce.
    if trace.status == "ran" and trace.row_count == 0:
        trace.fell_back = True
        from src.agent.strategies.structured import retrieve_structured
        structured = await retrieve_structured(state, vector_store, schema_registry)
        if structured.get("retrieved_chunks") or structured.get("sql_results"):
            structured["structured_trace"] = trace.to_dict()
            structured["retrieval_attempts"] = state.get("retrieval_attempts", 0) + 1
            return structured
        from src.agent.strategies.map_reduce import retrieve_map_reduce
        result = await retrieve_map_reduce(state, vector_store=vector_store)
        result["structured_trace"] = trace.to_dict()
        return result

    return {
        "retrieved_chunks": [],
        "sql_results": trace.rows,
        "structured_trace": trace.to_dict(),
        "retrieval_attempts": state.get("retrieval_attempts", 0) + 1,
    }
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_agent/test_analytical_fallback.py -v`
Expected: PASS (all four tests).

- [ ] **Step 5: Commit**

```bash
git add src/agent/strategies/analytical.py tests/test_agent/test_analytical_fallback.py
git commit -m "feat: fall back to structured narratives when ANALYTICAL SQL returns no rows"
```

---

## Task 4: Full suite + end-to-end verification

**Files:** none (verification only)

- [ ] **Step 1: Run the full test suite**

Run: `pytest -q`
Expected: PASS — no regressions (baseline was 165 passing before this work).

- [ ] **Step 2: Rebuild/restart the API container with the changes**

Run: `docker compose up -d --build api`
Expected: `api` container recreated and healthy.

- [ ] **Step 3: End-to-end check on the real PDF**

Ask, via the deployed app/playground: **"What is the pay range for an officer?"**
Expected: the query classifies **ANALYTICAL** (not LOOKUP); the answer contains real dollar figures for Commissioned Officers and cites `2025_April_Dec_AD_Pay.pdf`. If SQL returns no rows, the answer is still populated from the annotated `table_row` narratives (not "the document does not contain the amounts").

- [ ] **Step 4: Confirm and report**

Confirm routing + answer with evidence (the playground's Structured Lookup step shows the SQL/trace and `fell_back` flag). Report the result.

---

## Notes for the implementer

- DRY: Part A's `_resolve_hints_for_classifier` deliberately mirrors the resolve call in `retrieve_analytical`; don't duplicate the store-wiring logic elsewhere.
- The zero-row branch keeps the **analytical** trace (with `fell_back=True`) as `structured_trace` so the playground shows the original SQL attempt, while surfacing `retrieve_structured`'s chunks/rows. This is intentional — the SQL attempt is the more informative trace.
- `retrieve_structured` re-runs SQL under a relevance gate (a second, cheap execution). Accepted for reuse; do not refactor it in this plan.
