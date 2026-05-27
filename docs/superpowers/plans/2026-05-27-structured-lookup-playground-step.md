# Structured Lookup Playground Step Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a dedicated, collapsible "Structured Lookup" step to the admin playground trace that surfaces the structured/SQL retrieval decision (query_type + relevance-gate scores), the generated SQL, and the result (rows, or the skip/error/0-row reason).

**Architecture:** A `StructuredLookupTrace` dataclass is built by the structured strategies via a new sync core `run_structured_lookup` (which captures the SQL even when execution errors, via `generate_sql`/`run_sql` seams). The trace dict is returned under a `structured_trace` key, threaded through the `retrieve` graph node into `AgentState`, and rendered by a shared `_format_structured_lookup` formatter used by both the live-polling and final-trace playground paths. The frontend adds one `STEPS` row that stays hidden until the step actually fires.

**Tech Stack:** Python 3, pytest, FastAPI/Jinja admin UI, vanilla JS, DuckDB. No new dependencies.

Spec: `docs/superpowers/specs/2026-05-27-structured-lookup-playground-step-design.md`.

---

## Task 1: Trace core in `structured.py` (`StructuredLookupTrace`, seams, `run_structured_lookup`)

**Files:**
- Modify: `src/agent/strategies/structured.py`
- Test: `tests/test_agent/test_strategies/test_structured.py`

Introduce the data model and a sync core that generates + runs SQL and captures a full trace (SQL recorded even on error). Split the existing `structured_sql_rows` into reusable `generate_sql`/`run_sql` seams; keep `structured_sql_rows` behavior-identical (still raises on failure) so its existing tests stay green.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_agent/test_strategies/test_structured.py`:

```python
from src.agent.strategies.structured import (
    StructuredLookupTrace, run_structured_lookup, tables_relevant_scored,
)


def test_run_structured_lookup_success(tmp_path, monkeypatch):
    schema, _ = _pay_schema_and_db(tmp_path, monkeypatch)
    monkeypatch.setattr(
        structured, "generate",
        lambda **kw: f'SELECT grade, salary FROM "{schema.table}" ORDER BY grade',
    )
    trace = run_structured_lookup("engineer pay", [schema], query_type="analytical")
    assert trace.status == "ran"
    assert trace.query_type == "analytical"
    assert trace.gate is None
    assert trace.row_count == 4
    assert trace.sample_rows == trace.rows[:5]
    assert "SELECT" in trace.sql
    d = trace.to_dict()
    assert d["status"] == "ran" and d["row_count"] == 4 and "rows" not in d


def test_run_structured_lookup_error_captures_sql(tmp_path, monkeypatch):
    schema, _ = _pay_schema_and_db(tmp_path, monkeypatch)
    # Disallowed table -> execute_duckdb_sql rejects -> error, but SQL is still captured.
    monkeypatch.setattr(structured, "generate", lambda **kw: 'SELECT * FROM "doc_other_secret"')
    trace = run_structured_lookup("x", [schema], query_type="analytical")
    assert trace.status == "error"
    assert trace.error
    assert trace.fell_back is True
    assert 'doc_other_secret' in trace.sql   # SQL captured despite the failure


def test_run_structured_lookup_zero_rows(tmp_path, monkeypatch):
    schema, _ = _pay_schema_and_db(tmp_path, monkeypatch)
    monkeypatch.setattr(
        structured, "generate",
        lambda **kw: f'SELECT * FROM "{schema.table}" WHERE grade = \'NOPE\'',
    )
    trace = run_structured_lookup("x", [schema], query_type="sweep", gate=[["t", 0.5, True]])
    assert trace.status == "ran"
    assert trace.row_count == 0
    assert trace.sample_rows == []
    assert trace.gate == [["t", 0.5, True]]


def test_tables_relevant_scored_reports_all_scores(monkeypatch):
    from types import SimpleNamespace
    schemas = [
        SimpleNamespace(table="t_hi", description="pay", columns=[SimpleNamespace(name="salary")]),
        SimpleNamespace(table="t_lo", description="weather", columns=[SimpleNamespace(name="temp")]),
    ]
    monkeypatch.setattr(structured, "embed_query", lambda q: [1.0, 0.0])
    monkeypatch.setattr(structured, "embed_texts", lambda texts: [[1.0, 0.0], [0.0, 1.0]])
    scored = tables_relevant_scored("pay?", schemas)
    assert scored[0][0].table == "t_hi" and scored[0][2] is True       # passes (cos=1.0)
    assert scored[1][0].table == "t_lo" and scored[1][2] is False      # fails (cos=0.0)
    assert scored[0][1] > scored[1][1]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_agent/test_strategies/test_structured.py -k "run_structured_lookup or tables_relevant_scored" -v`
Expected: FAIL — `ImportError: cannot import name 'StructuredLookupTrace'`.

- [ ] **Step 3: Implement the data model + seams + core**

In `src/agent/strategies/structured.py`, add `from dataclasses import dataclass, field` to the imports at the top (after `import re`). Then add this block immediately AFTER `_extract_sql` (after its `return` on line ~49) and BEFORE `structured_sql_rows`:

```python
@dataclass
class StructuredLookupTrace:
    """Per-query record of the structured/SQL retrieval attempt, for the
    playground 'Structured Lookup' step. ``rows`` is the transient full result
    set (used to populate sql_results); it is excluded from ``to_dict``."""
    query_type: str
    gate: list | None = None            # list of [table, score, passed]; None when no gate (analytical)
    sql: str = ""
    status: str = "ran"                 # "ran" | "skipped" | "error"
    skip_reason: str = ""
    error: str = ""
    row_count: int = 0
    sample_rows: list = field(default_factory=list)
    fell_back: bool = False
    rows: list = field(default_factory=list, repr=False)

    def to_dict(self) -> dict:
        return {
            "query_type": self.query_type,
            "gate": self.gate,
            "sql": self.sql,
            "status": self.status,
            "skip_reason": self.skip_reason,
            "error": self.error,
            "row_count": self.row_count,
            "sample_rows": self.sample_rows,
            "fell_back": self.fell_back,
        }


def generate_sql(schema_prompt: str, question: str, generate_fn=None) -> str:
    """LLM text-to-SQL for one question + rendered schema prompt; returns the
    extracted SQL string (robust to prose/code-fence wrapping)."""
    gen = generate_fn or generate
    raw = gen(
        system_prompt=TEXT_TO_SQL_PROMPT.format(schema=schema_prompt),
        user_prompt=f"Question: {question}",
        temperature=0.0,
        max_tokens=2048,
    )
    sql = _extract_sql(raw)
    logger.info("Text-to-SQL for %r -> %s", question, sql)
    return sql


def run_sql(con, sql: str, allowed_tables: set) -> list[dict]:
    """Execute SELECT-only SQL against the tabular DuckDB, restricted to
    ``allowed_tables``. Raises on blocked/invalid SQL or execution error."""
    from src.ingestion.tabular_store import execute_duckdb_sql
    rows = execute_duckdb_sql(con, sql, allowed_tables=allowed_tables)
    logger.info("Text-to-SQL returned %d row(s)", len(rows))
    return rows


def run_structured_lookup(question: str, schemas, query_type: str,
                          gate: list | None = None, generate_fn=None) -> StructuredLookupTrace:
    """Generate + run SQL and capture a full trace. Never raises: a failure is
    recorded as status='error' (with the SQL, if generated) and fell_back=True so
    the caller can fall back. Sync (run via asyncio.to_thread from async callers)."""
    from src.ingestion.tabular_store import connect_tabular, schema_prompt_with_values
    trace = StructuredLookupTrace(query_type=query_type, gate=gate)
    con = connect_tabular(read_only=True)
    try:
        trace.sql = generate_sql(schema_prompt_with_values(schemas, con), question,
                                 generate_fn=generate_fn)
        rows = run_sql(con, trace.sql, {s.table for s in schemas})
        trace.status = "ran"
        trace.rows = rows
        trace.row_count = len(rows)
        trace.sample_rows = rows[:5]
    except Exception as e:
        trace.status = "error"
        trace.error = str(e)
        trace.fell_back = True
    finally:
        con.close()
    return trace
```

Then REPLACE the body of `structured_sql_rows` (lines ~60-80, the `from ... import` through the `finally`) so it reuses the seams but keeps its raising contract:

```python
    from src.ingestion.tabular_store import connect_tabular, schema_prompt_with_values
    con = connect_tabular(read_only=True)
    try:
        sql = generate_sql(schema_prompt_with_values(schemas, con), question,
                           generate_fn=generate_fn)
        return run_sql(con, sql, {s.table for s in schemas})
    finally:
        con.close()
```

Finally, ADD `tables_relevant_scored` and make `tables_relevant_to` delegate. REPLACE the existing `tables_relevant_to` (lines ~100-112) with:

```python
def tables_relevant_scored(question: str, schemas, threshold: float = RELEVANCE_THRESHOLD,
                           embed_query_fn=None, embed_texts_fn=None) -> list:
    """Score every ACL-filtered table against the question. Returns
    ``[(schema, score, passed), ...]`` so callers can show all scores (the gate)
    and pick the passers. No-LLM; ``embed_*_fn`` injectable for tests."""
    if not schemas:
        return []
    eq = embed_query_fn or embed_query
    et = embed_texts_fn or embed_texts
    qv = eq(question)
    tvs = et([_table_text(s) for s in schemas])
    return [(s, _cosine(qv, tv), _cosine(qv, tv) >= threshold) for s, tv in zip(schemas, tvs)]


def tables_relevant_to(question: str, schemas, threshold: float = RELEVANCE_THRESHOLD,
                       embed_query_fn=None, embed_texts_fn=None) -> list:
    """Cheap, no-LLM gate: tables whose text is embedding-similar above
    ``threshold``. Thin wrapper over ``tables_relevant_scored``."""
    return [s for s, _score, passed in tables_relevant_scored(
        question, schemas, threshold, embed_query_fn, embed_texts_fn) if passed]
```

> NOTE: `RELEVANCE_THRESHOLD`, `_cosine`, `_table_text`, `embed_query`, `embed_texts` are already defined/imported in this file. `tables_relevant_scored` must be defined AFTER `_cosine`/`_table_text` (i.e., in the same region as the current `tables_relevant_to`).

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_agent/test_strategies/test_structured.py -v`
Expected: PASS — the 4 new tests plus all pre-existing structured tests (including `test_structured_sql_rows_generates_and_runs`, `test_structured_sql_rows_raises_on_disallowed_table`, the gate tests, and the `_extract_sql` tests) stay green.

- [ ] **Step 5: Commit**

```bash
git add src/agent/strategies/structured.py tests/test_agent/test_strategies/test_structured.py
git commit -m "feat: StructuredLookupTrace + run_structured_lookup core (captures SQL on error)"
```

End the commit message with:
```
Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
```

---

## Task 2: Strategies populate `structured_trace`

**Files:**
- Modify: `src/agent/strategies/structured.py` (`retrieve_structured`)
- Modify: `src/agent/strategies/analytical.py` (`retrieve_analytical`)
- Modify: `src/agent/strategies/cross_reference.py` (`retrieve_cross_reference`)
- Test: `tests/test_agent/test_strategies/test_structured.py`, `tests/test_agent/test_strategies/test_analytical.py`

- [ ] **Step 1: Update + add tests (the existing retrieve_structured tests patch `structured_sql_rows`, which the refactor no longer calls)**

In `tests/test_agent/test_strategies/test_structured.py`, REPLACE `test_retrieve_structured_returns_sql_and_narratives` and `test_retrieve_structured_fail_open_on_sql_error` with these (they now patch `run_structured_lookup`) and add two trace tests:

```python
@pytest.mark.asyncio
async def test_retrieve_structured_returns_sql_and_narratives(monkeypatch):
    schemas = [SimpleNamespace(table="t_pay", description="pay", columns=[SimpleNamespace(name="salary")])]

    class _Reg:
        def list_for_user(self, g): return schemas

    monkeypatch.setattr(structured, "tables_relevant_scored",
                        lambda q, s: [(schemas[0], 0.71, True)])
    monkeypatch.setattr(structured, "run_structured_lookup",
                        lambda q, s, query_type, gate=None: StructuredLookupTrace(
                            query_type="sweep", gate=gate, status="ran", sql="SELECT 1",
                            row_count=1, sample_rows=[{"salary": 86415.0}], rows=[{"salary": 86415.0}]))
    monkeypatch.setattr(structured, "embed_query", lambda q: [0.0])

    class _VS:
        def search(self, **kw): return [SimpleNamespace(text="narr", metadata=SimpleNamespace(doc_id="d", chunk_index=0))]

    out = await structured.retrieve_structured(
        {"question": "gs pay", "user_groups": ["ALL"]}, vector_store=_VS(), schema_registry=_Reg())
    assert out["sql_results"] == [{"salary": 86415.0}]
    assert out["structured_trace"]["status"] == "ran"
    assert out["structured_trace"]["gate"] == [["t_pay", 0.71, True]]


@pytest.mark.asyncio
async def test_retrieve_structured_skipped_trace_when_gate_misses(monkeypatch):
    schemas = [SimpleNamespace(table="t_pay", description="pay", columns=[SimpleNamespace(name="salary")])]

    class _Reg:
        def list_for_user(self, g): return schemas

    monkeypatch.setattr(structured, "tables_relevant_scored",
                        lambda q, s: [(schemas[0], 0.10, False)])

    class _VS:
        def search(self, **kw): return []

    out = await structured.retrieve_structured(
        {"question": "weather", "user_groups": ["ALL"]}, vector_store=_VS(), schema_registry=_Reg())
    assert "sql_results" not in out
    assert out["structured_trace"]["status"] == "skipped"
    assert out["structured_trace"]["gate"] == [["t_pay", 0.1, False]]


@pytest.mark.asyncio
async def test_retrieve_structured_fail_open_on_gate_error(monkeypatch):
    class _Reg:
        def list_for_user(self, g): raise RuntimeError("embeddings down")

    out = await structured.retrieve_structured(
        {"question": "x", "user_groups": ["ALL"]}, vector_store=None, schema_registry=_Reg())
    assert out == {}
```

In `tests/test_agent/test_strategies/test_analytical.py`, append:

```python
@pytest.mark.asyncio
async def test_analytical_emits_structured_trace(tmp_path, monkeypatch):
    from src.agent.strategies import structured
    schema, _ = _pay_schema_and_db(tmp_path, monkeypatch)   # reuse this file's existing fixture
    monkeypatch.setattr(structured, "generate",
                        lambda **kw: f'SELECT grade, salary FROM "{schema.table}"')

    class _Reg:
        def list_for_user(self, g): return [schema]

    result = await retrieve_analytical(
        {"question": "pay", "user_groups": ["ALL"], "retrieval_attempts": 0},
        vector_store=None, schema_registry=_Reg())
    assert result["structured_trace"]["query_type"] == "analytical"
    assert result["structured_trace"]["gate"] is None
    assert result["structured_trace"]["status"] == "ran"
    assert result["structured_trace"]["row_count"] == 4
```

> NOTE: confirm the top of `test_structured.py` imports `from types import SimpleNamespace` (added in Task 1's region or already present from earlier tests). If absent, add it. `test_analytical.py` already defines `_pay_schema_and_db` and imports `retrieve_analytical`; reuse them.

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_agent/test_strategies/test_structured.py -k retrieve_structured -v`
Expected: FAIL — `retrieve_structured` doesn't return `structured_trace` yet (KeyError on `out["structured_trace"]`).

- [ ] **Step 3: Implement `retrieve_structured`**

In `src/agent/strategies/structured.py`, REPLACE `retrieve_structured` (the whole function) with:

```python
async def retrieve_structured(state, vector_store, schema_registry) -> dict:
    """Gated structured retrieval for the SWEEP branch. Returns exact SQL rows +
    top-k row-narrative chunks when a registered table is relevant, plus a
    ``structured_trace`` describing the decision/SQL/result. Fail-open: gate or
    registry errors yield {} (RAG-only sweep)."""
    question = state["question"]
    user_groups = state["user_groups"]

    try:
        schemas = schema_registry.list_for_user(user_groups)
        scored = tables_relevant_scored(question, schemas)
    except Exception:
        return {}   # gate/registry error -> RAG-only sweep (fail-open)

    gate = [[s.table, round(score, 3), passed] for s, score, passed in scored]
    relevant = [s for s, _score, passed in scored if passed]

    if not relevant:
        if gate:  # tables existed but none cleared the threshold -> a visible "skipped" decision
            trace = StructuredLookupTrace(
                query_type="sweep", gate=gate, status="skipped",
                skip_reason=f"no table >= {RELEVANCE_THRESHOLD} relevance")
            return {"structured_trace": trace.to_dict()}
        return {}

    trace = await asyncio.to_thread(run_structured_lookup, question, relevant, "sweep", gate)

    chunks: list = []
    try:
        qv = await asyncio.to_thread(embed_query, question)
        chunks = vector_store.search(
            vector=qv, user_groups=user_groups, top_k=20, tier="table_row",
        )
    except Exception:
        chunks = []

    return {"sql_results": trace.rows, "retrieved_chunks": chunks,
            "structured_trace": trace.to_dict()}
```

- [ ] **Step 4: Implement `retrieve_analytical`**

In `src/agent/strategies/analytical.py`, REPLACE the whole function body with:

```python
async def retrieve_analytical(state: AgentState, vector_store, schema_registry: SchemaRegistry) -> dict:
    question = state["question"]
    user_groups = state["user_groups"]

    from src.agent.strategies.structured import run_structured_lookup, StructuredLookupTrace

    schemas = schema_registry.list_for_user(user_groups)
    if not schemas:
        from src.agent.strategies.map_reduce import retrieve_map_reduce
        result = await retrieve_map_reduce(state, vector_store=vector_store)
        result["structured_trace"] = StructuredLookupTrace(
            query_type="analytical", gate=None, status="skipped",
            skip_reason="no registered tables").to_dict()
        return result

    trace = await asyncio.to_thread(run_structured_lookup, question, schemas, "analytical")
    if trace.status == "error":
        # No usable SQL (LLM error), blocked SQL, or execution error -> comprehensive retrieval.
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

(The existing imports `import asyncio`, `from src.agent.state import AgentState`, `from src.db.schema_registry import SchemaRegistry` remain; the old `from src.agent.strategies.structured import structured_sql_rows` import may be removed if now unused.)

- [ ] **Step 5: Forward the trace in `retrieve_cross_reference`**

In `src/agent/strategies/cross_reference.py`, in the `if has_schemas:` block, capture and forward the trace. REPLACE:

```python
    sql_results = []
    has_schemas = len(schema_registry.list_for_user(user_groups)) > 0
    if has_schemas:
        analytical_result = await retrieve_analytical(state, vector_store=vector_store, schema_registry=schema_registry)
        sql_results = analytical_result.get("sql_results", [])
```

with:

```python
    sql_results = []
    structured_trace = None
    has_schemas = len(schema_registry.list_for_user(user_groups)) > 0
    if has_schemas:
        analytical_result = await retrieve_analytical(state, vector_store=vector_store, schema_registry=schema_registry)
        sql_results = analytical_result.get("sql_results", [])
        structured_trace = analytical_result.get("structured_trace")
```

and REPLACE the `return {...}` at the end with:

```python
    unique_chunks = vector_store.expand_window(unique_chunks, window=2)
    result = {
        "retrieved_chunks": unique_chunks,
        "sql_results": sql_results,
        "retrieval_attempts": state.get("retrieval_attempts", 0) + 1,
    }
    if structured_trace:
        result["structured_trace"] = structured_trace
    return result
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `python -m pytest tests/test_agent/test_strategies/test_structured.py tests/test_agent/test_strategies/test_analytical.py tests/test_agent/test_strategies/test_cross_reference.py -v`
Expected: PASS — new trace tests pass; `test_analytical.py` (patches `structured.generate`, not `structured_sql_rows`) and `test_cross_reference.py` (mocks `retrieve_analytical`) stay green.

- [ ] **Step 7: Commit**

```bash
git add src/agent/strategies/structured.py src/agent/strategies/analytical.py src/agent/strategies/cross_reference.py tests/test_agent/test_strategies/test_structured.py tests/test_agent/test_strategies/test_analytical.py
git commit -m "feat: structured strategies emit structured_trace (gate/SQL/result decision)"
```

End with the `Co-Authored-By` trailer as in Task 1.

---

## Task 3: Thread `structured_trace` through state + graph

**Files:**
- Modify: `src/agent/state.py:42` (add field)
- Modify: `src/agent/graph.py` (SWEEP branch ~line 74)
- Test: `tests/test_agent/test_state.py`

The analytical/cross_reference branches return their strategy dict directly (already carries `structured_trace`); the SWEEP branch builds `result` manually and must copy the key. `AgentState` needs the channel declared so LangGraph keeps it.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_agent/test_state.py`:

```python
def test_agent_state_accepts_structured_trace():
    from src.agent.state import AgentState
    st: AgentState = {"question": "q", "structured_trace": {"status": "ran", "query_type": "sweep"}}
    assert st["structured_trace"]["status"] == "ran"
    # The field is a declared channel (present in the TypedDict annotations).
    assert "structured_trace" in AgentState.__annotations__
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_agent/test_state.py::test_agent_state_accepts_structured_trace -v`
Expected: FAIL — `assert "structured_trace" in AgentState.__annotations__` is False.

- [ ] **Step 3: Add the state field**

In `src/agent/state.py`, inside the `AgentState` TypedDict, add after the `dataset_id: int` line (line 42):

```python
    structured_trace: dict  # playground: structured/SQL lookup decision + result
```

- [ ] **Step 4: Copy the trace in the SWEEP branch**

In `src/agent/graph.py`, in the SWEEP branch, immediately AFTER the existing:

```python
            if struct_result.get("sql_results"):
                result["sql_results"] = struct_result["sql_results"]
```

add:

```python
            if struct_result.get("structured_trace"):
                result["structured_trace"] = struct_result["structured_trace"]
```

(The ANALYTICAL and CROSS_REFERENCE branches assign `result = await retrieve_...(...)`, so their `structured_trace` already flows through unchanged.)

- [ ] **Step 5: Run tests + import check**

Run: `python -m pytest tests/test_agent/test_state.py -v && python -c "import src.agent.graph; print('graph import OK')"`
Expected: PASS and `graph import OK`.

- [ ] **Step 6: Commit**

```bash
git add src/agent/state.py src/agent/graph.py tests/test_agent/test_state.py
git commit -m "feat: thread structured_trace through AgentState + SWEEP retrieve node"
```

End with the `Co-Authored-By` trailer.

---

## Task 4: `_format_structured_lookup` formatter + wire into the playground

**Files:**
- Modify: `src/admin/routes.py` (add module-level formatter; emit step in `run_query`; add `format_step_detail` branch + `step_labels`)
- Test: `tests/test_admin/test_structured_lookup_format.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_admin/test_structured_lookup_format.py`:

```python
"""Tests for the playground Structured Lookup step formatter."""
from src.admin.routes import _format_structured_lookup


def test_format_ran_includes_sql_gate_and_sample():
    trace = {
        "query_type": "sweep", "gate": [["all_gs", 0.71, True], ["leo", 0.18, False]],
        "sql": "SELECT * FROM all_gs WHERE locname = 'TU'", "status": "ran",
        "skip_reason": "", "error": "", "row_count": 15,
        "sample_rows": [{"grade": "GS-12", "salary": 91162}], "fell_back": False,
    }
    html = _format_structured_lookup(trace)
    assert "sweep" in html
    assert "all_gs" in html and "0.71" in html
    assert "SELECT * FROM all_gs" in html
    assert "15 rows" in html
    assert "view sample" in html


def test_format_skipped_shows_reason_no_sql():
    trace = {"query_type": "sweep", "gate": [["t", 0.1, False]], "sql": "",
             "status": "skipped", "skip_reason": "no table >= 0.3 relevance",
             "error": "", "row_count": 0, "sample_rows": [], "fell_back": False}
    html = _format_structured_lookup(trace)
    assert "skipped" in html and "no table" in html
    assert "SELECT" not in html


def test_format_error_shows_message_and_fallback():
    trace = {"query_type": "analytical", "gate": None, "sql": "SELECT bad",
             "status": "error", "skip_reason": "", "error": "Parser Error",
             "row_count": 0, "sample_rows": [], "fell_back": True}
    html = _format_structured_lookup(trace)
    assert "error" in html and "Parser Error" in html
    assert "map-reduce" in html
    assert "no gate" in html  # analytical -> gate None


def test_format_zero_rows():
    trace = {"query_type": "analytical", "gate": None, "sql": "SELECT 1 WHERE 1=0",
             "status": "ran", "skip_reason": "", "error": "", "row_count": 0,
             "sample_rows": [], "fell_back": False}
    html = _format_structured_lookup(trace)
    assert "0 rows" in html
    assert "view sample" not in html


def test_format_empty_trace():
    assert "No structured lookup" in _format_structured_lookup({})
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_admin/test_structured_lookup_format.py -v`
Expected: FAIL — `ImportError: cannot import name '_format_structured_lookup'`.

- [ ] **Step 3: Add the module-level formatter**

In `src/admin/routes.py`, add this function at module level (near the other top-level helpers, before the route handlers that use it):

```python
def _format_structured_lookup(trace: dict) -> str:
    """Render a StructuredLookupTrace dict as playground step-detail HTML.
    Shared by the live-polling and final-trace render paths."""
    import html as _h, json as _json
    if not trace:
        return "<em>No structured lookup</em>"
    gate = trace.get("gate")
    parts = [f"<strong>Decision:</strong> {_h.escape(str(trace.get('query_type', '')))} → "
             + ("gate ran" if gate else "no gate (analytical)")]
    if gate:
        rows = "".join(
            f"<div>{_h.escape(str(t))} &nbsp; {float(score):.2f} {'&#10003;' if passed else '&#10007;'}</div>"
            for t, score, passed in gate)
        parts.append(f"<strong>Tables:</strong>{rows}")
    sql = trace.get("sql", "")
    if sql:
        parts.append("<strong>SQL:</strong><pre style=\"white-space:pre-wrap; background:#0f172a; "
                     f"padding:0.5rem; border-radius:4px;\">{_h.escape(sql)}</pre>")
    status = trace.get("status", "ran")
    if status == "skipped":
        parts.append(f"<strong>Result:</strong> skipped — {_h.escape(trace.get('skip_reason', ''))}")
    elif status == "error":
        fb = " (fell back to map-reduce)" if trace.get("fell_back") else ""
        parts.append(f"<strong>Result:</strong> error — {_h.escape(trace.get('error', ''))}{fb}")
    else:
        rc = trace.get("row_count", 0)
        if not rc:
            parts.append("<strong>Result:</strong> 0 rows (filter matched nothing)")
        else:
            sample = _h.escape(_json.dumps(trace.get("sample_rows", []), indent=2, default=str))
            parts.append(
                f"<strong>Result:</strong> {rc} rows"
                "<details style=\"margin-top:0.3rem;\"><summary style=\"cursor:pointer;\">view sample</summary>"
                f"<pre style=\"white-space:pre-wrap; background:#0f172a; padding:0.5rem; border-radius:4px;\">{sample}</pre></details>")
    return "<br>".join(parts)
```

- [ ] **Step 4: Run formatter tests to verify they pass**

Run: `python -m pytest tests/test_admin/test_structured_lookup_format.py -v`
Expected: PASS (5 tests).

- [ ] **Step 5: Emit the step in the live loop**

In `src/admin/routes.py`, inside `run_query`'s `async for event in graph.astream(...)` loop, find the block `if node_name != "merge":` (line ~855) that appends the step entry. REPLACE:

```python
                    if node_name != "merge":
                        step_entry = {"step": node_name, "time": node_elapsed, "detail": _format_live_step(node_name, node_output, final_state)}
                        steps_data.append({"step": node_name, "time": node_elapsed, "output": output})
                        _playground_jobs[query_id]["completed_steps"].append(step_entry)
```

with:

```python
                    if node_name != "merge":
                        # Structured lookup is computed inside the retrieve node; emit it as
                        # its own step (displayed before Retrieve) when a trace is present.
                        if node_name == "retrieve" and output.get("structured_trace"):
                            st = output["structured_trace"]
                            sl_detail = _format_structured_lookup(st)
                            steps_data.append({"step": "structured_lookup", "time": 0.0, "output": {"structured_trace": st}})
                            _playground_jobs[query_id]["completed_steps"].append(
                                {"step": "structured_lookup", "time": 0.0, "detail": sl_detail})
                        step_entry = {"step": node_name, "time": node_elapsed, "detail": _format_live_step(node_name, node_output, final_state)}
                        steps_data.append({"step": node_name, "time": node_elapsed, "output": output})
                        _playground_jobs[query_id]["completed_steps"].append(step_entry)
```

- [ ] **Step 6: Add the final-trace branch + label**

In `src/admin/routes.py`, in `format_step_detail` (the nested function ~line 919), add this branch immediately after the `if step_name == "cache_check":` ... before `elif step_name == "classify":` (order among elifs does not matter; place it adjacent to classify):

```python
                elif step_name == "structured_lookup":
                    return _format_structured_lookup(output.get("structured_trace", {}))
```

And in the `step_labels` dict (line ~917), add the `structured_lookup` label:

```python
            step_labels = {"cache_check": "Check Cache", "classify": "Classify Query", "structured_lookup": "Structured Lookup", "retrieve": "Retrieve Documents", "enrich": "Knowledge Graph", "synthesize": "Generate Answer"}
```

- [ ] **Step 7: Verify import + full admin tests**

Run: `python -c "import src.admin.routes; print('OK')" && python -m pytest tests/test_admin/ -q`
Expected: `OK` and admin tests pass (no NEW failures vs. before this branch).

- [ ] **Step 8: Commit**

```bash
git add src/admin/routes.py tests/test_admin/test_structured_lookup_format.py
git commit -m "feat: Structured Lookup playground step formatter + wiring (live + final trace)"
```

End with the `Co-Authored-By` trailer.

---

## Task 5: Frontend — add the (hidden-until-fired) step row

**Files:**
- Modify: `src/admin/templates/playground.html` (`STEPS` array + `initProgressPanel`)

The step row must exist so `markStepCompleted` can fill it, but stay hidden for queries that never do a structured lookup. `markStepCompleted` already resets `row.style = ''` (clearing any inline `display:none`), so rendering the row hidden and letting completion reveal it requires no change to the polling loop.

- [ ] **Step 1: Add the step to the `STEPS` array**

In `src/admin/templates/playground.html`, REPLACE the `STEPS` array (lines 48-54) with:

```javascript
const STEPS = [
    {id: "cache_check", label: "Check Cache", num: 1},
    {id: "classify", label: "Classify Query", num: 2},
    {id: "structured_lookup", label: "Structured Lookup", num: 3},
    {id: "retrieve", label: "Retrieve Documents", num: 3},
    {id: "enrich", label: "Knowledge Graph", num: 3},
    {id: "synthesize", label: "Generate Answer", num: 4},
];
```

- [ ] **Step 2: Render the structured_lookup row hidden initially**

In `initProgressPanel` (lines 57-70), REPLACE the loop body so the structured_lookup row starts hidden:

```javascript
    for (const step of STEPS) {
        const hidden = step.id === 'structured_lookup' ? ' display:none;' : '';
        html += `<div id="step-row-${step.id}" class="trace-step" style="opacity:0.4;${hidden}">
            <span id="step-icon-${step.id}">&#11036;</span>
            <span> Step ${step.num} of 4: ${step.label}</span>
            <span id="step-arrow-${step.id}" style="display:none;"> &#9660;</span>
            <span class="trace-time" id="step-time-${step.id}"></span>
        </div>
        <div id="step-detail-${step.id}" class="trace-detail"></div>`;
    }
```

> NOTE: `markStepCompleted` already does `row.style = ''` (line 77), which clears the inline `display:none`, so the row appears only when the backend emits a `structured_lookup` completed step. No change to the polling loop or `markStepActive` is needed — `structured_lookup` is never passed to `markStepActive`.

- [ ] **Step 3: Manual verification (template render)**

Run: `python -c "import src.admin.routes; print('routes import OK')"`
Expected: `routes import OK` (the template is served, not imported; this confirms the backend half still loads). The visual check happens in Task 6 Step 2.

- [ ] **Step 4: Commit**

```bash
git add src/admin/templates/playground.html
git commit -m "feat: hidden-until-fired Structured Lookup step row in playground UI"
```

End with the `Co-Authored-By` trailer.

---

## Task 6: Regression + manual smoke + memory

**Files:** none (verification); then memory update

- [ ] **Step 1: Run the affected suites**

Run: `python -m pytest tests/test_agent/ tests/test_admin/ -q`
Expected: new tests pass; no NEW failures. Known pre-existing/environmental failures remain (the ~8 in `tests/test_agent/` — test_graph/lookup/sweep/cross_reference harness/mocking; verify the failing set is unchanged from before this branch via `git stash` + rerun if unsure). The structured/analytical/cross_reference/state/admin-format tests added here must all PASS.

- [ ] **Step 2: Manual smoke in the deployed playground**

Rebuild + redeploy the api container, then run a structured query and confirm the new step:

```bash
docker compose up -d --build api
```

In the playground, run "What are the GS salary rates in Tampa?" and confirm a **Structured Lookup** step appears (between Classify and Retrieve) showing: the decision (`sweep`/`analytical`), gate table scores (when sweep), the generated SQL, and the row count with a `view sample` expander. Then run a plain prose question (e.g. "summarize the leave policy") and confirm the Structured Lookup step is **absent** (hidden) for a pure lookup.

- [ ] **Step 3: Update the roadmap memory**

Edit `/home/mike/.claude/projects/-home-mike-sauron/memory/tabular-spreadsheet-ingestion-roadmap.md`: note the playground now has a "Structured Lookup" step (decision/gate/SQL/result incl. skip/error/0-row), backed by `StructuredLookupTrace` + `run_structured_lookup` in `structured.py`, threaded via `AgentState.structured_trace` and rendered by `_format_structured_lookup` in `routes.py`. Note the `generate_sql`/`run_sql` seam split (SQL captured even on error).

- [ ] **Step 4: Commit any remaining changes**

```bash
git add -A && git commit -m "docs: note Structured Lookup playground step in roadmap memory" || echo "nothing to commit"
```

(The memory file lives outside the repo working tree under `.claude/`; if `git add -A` stages nothing in the repo, the `|| echo` keeps the step green.)

---

## Self-Review (completed during authoring)

**Spec coverage:**
- `StructuredLookupTrace` data model (query_type, gate, sql, status, skip_reason, error, row_count, sample_rows, fell_back) → Task 1.
- Capture SQL even on error via `generate_sql`/`run_sql` split + `run_structured_lookup` → Task 1.
- Strategies populate trace: analytical (gate=None), sweep (gate scores + skip), cross_reference (forward) → Task 2.
- Thread to UI via `AgentState.structured_trace` + graph node → Task 3.
- Shared `_format_structured_lookup` used by both live and final render; `step_labels` + branch → Task 4.
- Dedicated step row after Classify, before Retrieve, hidden until fired → Task 5.
- Full transparency incl. skip/error/0-rows → covered by formatter (Task 4) + strategy population (Tasks 1-2).
- Render details (decision, gate ✓/✗ vs 0.30, SQL `<pre>`, result line, first-5 sample behind expander) → Task 4 formatter.
- Testing (trace population, formatter HTML per status, `generate_sql`/`run_sql` keep `structured_sql_rows` green) → Tasks 1, 2, 4.
- Fail-open / additive → `run_structured_lookup` never raises; gate errors → {}; formatter handles empty trace.

**Placeholder scan:** none — every code step shows complete code; commands have expected output.

**Type consistency:** `StructuredLookupTrace.to_dict()` keys (query_type, gate, sql, status, skip_reason, error, row_count, sample_rows, fell_back) match exactly what `_format_structured_lookup` reads (Task 4) and what the tests assert (Tasks 1, 2, 4). `gate` is `list[[table, score, passed]]` everywhere (built in `retrieve_structured` Task 2, read in the formatter Task 4, asserted in tests). `structured_trace` is the dict key used in strategies (Task 2), graph/state (Task 3), and the playground emit + formatter (Task 4). `run_structured_lookup(question, schemas, query_type, gate=None, generate_fn=None)` signature matches all call sites (Task 2 analytical/sweep, Task 1 tests). `tables_relevant_scored` returns `[(schema, score, passed)]` consumed in `retrieve_structured` (Task 2) and asserted in Task 1.
