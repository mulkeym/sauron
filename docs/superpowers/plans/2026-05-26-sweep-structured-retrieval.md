# Structured Retrieval in SWEEP — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let the SWEEP strategy blend exact DuckDB SQL answers and retrievable row narratives with its existing document RAG, gated by a cheap no-LLM relevance check.

**Architecture:** A new `src/agent/strategies/structured.py` holds a shared SQL core (`structured_sql_rows`), a cheap embedding-similarity gate (`tables_relevant_to`), and `retrieve_structured` (gate → SQL rows + `table_row` narrative chunks, fail-open). `retrieve_analytical` is refactored to reuse the SQL core. The SWEEP branch in `graph.py` adds `retrieve_structured` to its existing parallel `gather(...)` and merges. Strictly additive and fail-open.

**Tech Stack:** Python 3.11, DuckDB, LanceDB vector store, the local LLM client, pytest/pytest-asyncio. Tests run inside the app image: `docker run --rm -v "$(pwd)":/app -w /app sauron-api python -m pytest <path> -q`.

**Spec:** `docs/superpowers/specs/2026-05-26-sweep-structured-retrieval-design.md`. **Depends on (merged + deployed):** the tabular pipeline (`tabular_store.py` with `connect_tabular`, `execute_duckdb_sql`, `schema_prompt_with_values`), `schema_registry.list_for_user`, and the row narratives stored at `chunk_size_tier="table_row"`.

> ⚠️ Changes live retrieval behavior → rebuild + redeploy after merge.

---

## File Structure

- `src/agent/strategies/structured.py` — **create**. Owns `TEXT_TO_SQL_PROMPT` (moved here), `structured_sql_rows`, `_table_text`, `_cosine`, `tables_relevant_to`, `retrieve_structured`. One responsibility: structured (SQL + narrative) retrieval shared across strategies.
- `src/agent/strategies/analytical.py` — **modify**: drop its local `TEXT_TO_SQL_PROMPT` + inline SQL logic; call `structured_sql_rows`. Keep the map-reduce fallback.
- `src/agent/graph.py` — **modify**: add `retrieve_structured` to the SWEEP branch `gather`, merge its chunks + `sql_results`.
- `tests/test_agent/test_strategies/test_structured.py` — **create**.
- `tests/test_agent/test_strategies/test_analytical.py` — **modify**: patch `generate` where it now lives (in `structured`).

## Key existing signatures consumed

- `tabular_store.connect_tabular(read_only=False)`, `execute_duckdb_sql(con, sql, allowed_tables=None)`, `schema_prompt_with_values(schemas, con)`.
- `schema_registry.list_for_user(user_groups) -> list[TableSchema]`; `TableSchema.table`, `.description`, `.columns` (each `ColumnSchema.name`).
- `embedder.embed_query(text) -> list[float]`.
- `vector_store.search(vector, user_groups, top_k, tier, doc_ids=None) -> list[RetrievedChunk]`.
- `generate(system_prompt, user_prompt, temperature, max_tokens) -> str`.

---

### Task 1: Shared SQL core — `structured_sql_rows`

**Files:**
- Create: `src/agent/strategies/structured.py`
- Test: `tests/test_agent/test_strategies/test_structured.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_agent/test_strategies/test_structured.py`:

```python
"""Tests for shared structured retrieval (SQL core + gate + retrieve_structured)."""
import duckdb
import pytest

from src.ingestion.tabular import SheetGrid, SheetClassification
from src.ingestion.tabular_store import load_sheet_to_duckdb, schema_from_sheet
from src.agent.strategies import structured
from src.agent.strategies.structured import structured_sql_rows


def _pay_schema_and_db(tmp_path, monkeypatch):
    from src.config import settings
    monkeypatch.setattr(settings, "tabular_duckdb_path", str(tmp_path / "t.duckdb"))
    grid = SheetGrid("Pay", [["grade", "step", "salary"]] + [[f"GS-{g}", 5, 80000 + g] for g in range(10, 14)])
    cls = SheetClassification("Pay", "clean", 0, ["text", "number", "number"], "clean table")
    from src.ingestion.tabular_store import connect_tabular
    con = connect_tabular(read_only=False)
    load_sheet_to_duckdb(con, "doc1", "Pay", cls, grid)
    con.close()
    return schema_from_sheet("doc1", "Pay", cls, grid, acl_groups=["ALL"])


def test_structured_sql_rows_generates_and_runs(tmp_path, monkeypatch):
    schema = _pay_schema_and_db(tmp_path, monkeypatch)
    monkeypatch.setattr(structured, "generate",
                        lambda **kw: f'SELECT grade, salary FROM "{schema.table}" WHERE step = 5 ORDER BY salary')
    rows = structured_sql_rows("engineer pay", [schema])
    assert rows[0] == {"grade": "GS-10", "salary": 80010.0}
    assert len(rows) == 4


def test_structured_sql_rows_raises_on_disallowed_table(tmp_path, monkeypatch):
    schema = _pay_schema_and_db(tmp_path, monkeypatch)
    monkeypatch.setattr(structured, "generate", lambda **kw: 'SELECT * FROM "doc_other_secret"')
    with pytest.raises(Exception):
        structured_sql_rows("x", [schema])  # allowlist rejects -> raises (caller falls back)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `docker run --rm -v "$(pwd)":/app -w /app sauron-api python -m pytest tests/test_agent/test_strategies/test_structured.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.agent.strategies.structured'`.

- [ ] **Step 3: Create the module + SQL core**

Create `src/agent/strategies/structured.py`:

```python
"""Structured retrieval shared across strategies: text-to-SQL against the
tabular DuckDB store plus retrievable row narratives.

`structured_sql_rows` is the SQL core used by both `retrieve_analytical` and
the SWEEP branch. `tables_relevant_to` is a cheap (no-LLM) gate. `retrieve_structured`
combines them, fail-open, for the SWEEP strategy.
"""
import asyncio
import math

from src.generation.llm_client import generate

TEXT_TO_SQL_PROMPT = """You are a SQL query generator. Given a natural language question and database schema, generate a single SELECT query.

Rules:
- Output ONLY the SQL query, no explanation
- Only use tables and columns from the provided schema
- Use the table name exactly as given, with no database/schema prefix
- Always use SELECT (never INSERT, UPDATE, DELETE, DROP, etc.)
- Keep queries simple and correct

Schema:
{schema}"""


def structured_sql_rows(question: str, schemas, generate_fn=None) -> list[dict]:
    """Generate SQL from the (value-enriched) schema prompt and run it against
    the tabular DuckDB, restricted to ``schemas`` as the allowlist.

    One read-only connection; raises on any failure (LLM, blocked/empty SQL,
    execution) — callers decide the fallback. Synchronous (run via
    ``asyncio.to_thread`` from async callers).
    """
    from src.ingestion.tabular_store import (
        connect_tabular, execute_duckdb_sql, schema_prompt_with_values,
    )
    gen = generate_fn or generate
    allowed_tables = {s.table for s in schemas}
    con = connect_tabular(read_only=True)
    try:
        schema_prompt = schema_prompt_with_values(schemas, con)
        sql = gen(
            system_prompt=TEXT_TO_SQL_PROMPT.format(schema=schema_prompt),
            user_prompt=f"Question: {question}",
            temperature=0.0,
            max_tokens=2048,
        )
        sql = sql.strip().strip("`").removeprefix("sql\n").removeprefix("sql").strip()
        return execute_duckdb_sql(con, sql, allowed_tables=allowed_tables)
    finally:
        con.close()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `docker run --rm -v "$(pwd)":/app -w /app sauron-api python -m pytest tests/test_agent/test_strategies/test_structured.py -q`
Expected: PASS (2 passed).

- [ ] **Step 5: Commit**

```bash
git add src/agent/strategies/structured.py tests/test_agent/test_strategies/test_structured.py
git commit -m "feat: shared text-to-SQL core for structured retrieval

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task 2: Cheap relevance gate — `tables_relevant_to`

**Files:**
- Modify: `src/agent/strategies/structured.py`
- Test: `tests/test_agent/test_strategies/test_structured.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_agent/test_strategies/test_structured.py` (extend the import to add `tables_relevant_to`):

```python
from src.agent.strategies.structured import tables_relevant_to
from src.db.schema_registry import TableSchema, ColumnSchema


def _schema(table, desc):
    return TableSchema(database="spreadsheets", table=table,
                       columns=[ColumnSchema("grade", "DOUBLE", "")],
                       description=desc, acl_groups=["ALL"])


def test_gate_keeps_only_matching_tables():
    pay = _schema("doc_pay", "GS pay rates by grade and locality")
    weather = _schema("doc_weather", "daily weather observations")
    # Deterministic fake embeddings: question vector == pay-text vector; weather orthogonal.
    eq = lambda q: [1.0, 0.0]
    et = lambda texts: [[1.0, 0.0] if "pay" in t.lower() else [0.0, 1.0] for t in texts]
    out = tables_relevant_to("what is the pay for grade 12", [pay, weather],
                             threshold=0.5, embed_query_fn=eq, embed_texts_fn=et)
    assert [s.table for s in out] == ["doc_pay"]


def test_gate_empty_when_no_schemas():
    assert tables_relevant_to("anything", []) == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `docker run --rm -v "$(pwd)":/app -w /app sauron-api python -m pytest tests/test_agent/test_strategies/test_structured.py -k gate -q`
Expected: FAIL — `ImportError: cannot import name 'tables_relevant_to'`.

- [ ] **Step 3: Implement the gate**

Add to `src/agent/strategies/structured.py`:

```python
RELEVANCE_THRESHOLD = 0.30  # permissive: bias toward attempting SQL (a false positive just costs one try)


def _cosine(a, b) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


def _table_text(schema) -> str:
    cols = ", ".join(c.name for c in schema.columns)
    return f"{schema.table}. {schema.description}. Columns: {cols}"


def tables_relevant_to(question: str, schemas, threshold: float = RELEVANCE_THRESHOLD,
                       embed_query_fn=None, embed_texts_fn=None) -> list:
    """Cheap, no-LLM gate: keep tables whose text is embedding-similar to the
    question above ``threshold``. Operates only on the ACL-filtered schema list
    passed in. ``embed_*_fn`` are injectable for tests.
    """
    if not schemas:
        return []
    from src.ingestion.embedder import embed_query, embed_texts
    eq = embed_query_fn or embed_query
    et = embed_texts_fn or embed_texts
    qv = eq(question)
    tvs = et([_table_text(s) for s in schemas])
    return [s for s, tv in zip(schemas, tvs) if _cosine(qv, tv) >= threshold]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `docker run --rm -v "$(pwd)":/app -w /app sauron-api python -m pytest tests/test_agent/test_strategies/test_structured.py -k gate -q`
Expected: PASS (2 passed).

- [ ] **Step 5: Commit**

```bash
git add src/agent/strategies/structured.py tests/test_agent/test_strategies/test_structured.py
git commit -m "feat: cheap no-LLM relevance gate for structured retrieval

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task 3: `retrieve_structured` (gate → SQL rows + narratives, fail-open)

**Files:**
- Modify: `src/agent/strategies/structured.py`
- Test: `tests/test_agent/test_strategies/test_structured.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_agent/test_strategies/test_structured.py` (extend imports: `from unittest.mock import MagicMock`, and `retrieve_structured`):

```python
from unittest.mock import MagicMock
from src.agent.strategies.structured import retrieve_structured
from src.retrieval.models import RetrievedChunk, ChunkMetadata


def _narr_chunk():
    return RetrievedChunk(text="GS pay grade=GS-12: salary is 86415", score=0.9,
                          metadata=ChunkMetadata(doc_id="d", filename="f", doc_type="xlsx",
                          chunk_index=0, start_char=0, acl_groups=["ALL"], chunk_size_tier="table_row"))


def _reg(schemas):
    r = MagicMock()
    r.list_for_user.return_value = schemas
    return r


@pytest.mark.asyncio
async def test_retrieve_structured_returns_sql_and_narratives(monkeypatch):
    schema = _schema("doc_pay", "GS pay rates")
    monkeypatch.setattr(structured, "tables_relevant_to", lambda q, s, **k: [schema])
    monkeypatch.setattr(structured, "structured_sql_rows", lambda q, s: [{"salary": 86415.0}])
    monkeypatch.setattr(structured, "embed_query", lambda q: [0.1], raising=False)
    vs = MagicMock()
    vs.search.return_value = [_narr_chunk()]
    out = await retrieve_structured({"question": "pay?", "user_groups": ["ALL"]},
                                    vector_store=vs, schema_registry=_reg([schema]))
    assert out["sql_results"] == [{"salary": 86415.0}]
    assert len(out["retrieved_chunks"]) == 1
    # narratives searched on the table_row tier
    assert vs.search.call_args.kwargs.get("tier") == "table_row"


@pytest.mark.asyncio
async def test_retrieve_structured_empty_when_gate_misses(monkeypatch):
    monkeypatch.setattr(structured, "tables_relevant_to", lambda q, s, **k: [])
    out = await retrieve_structured({"question": "weather?", "user_groups": ["ALL"]},
                                    vector_store=MagicMock(), schema_registry=_reg([_schema("doc_pay", "pay")]))
    assert out == {}


@pytest.mark.asyncio
async def test_retrieve_structured_fail_open_on_sql_error(monkeypatch):
    schema = _schema("doc_pay", "GS pay rates")
    monkeypatch.setattr(structured, "tables_relevant_to", lambda q, s, **k: [schema])
    def boom(q, s):
        raise RuntimeError("sql gen down")
    monkeypatch.setattr(structured, "structured_sql_rows", boom)
    monkeypatch.setattr(structured, "embed_query", lambda q: [0.1], raising=False)
    vs = MagicMock(); vs.search.return_value = [_narr_chunk()]
    out = await retrieve_structured({"question": "pay?", "user_groups": ["ALL"]},
                                    vector_store=vs, schema_registry=_reg([schema]))
    assert out["sql_results"] == []                 # SQL failed -> empty, no raise
    assert len(out["retrieved_chunks"]) == 1        # narratives still returned
```

- [ ] **Step 2: Run test to verify it fails**

Run: `docker run --rm -v "$(pwd)":/app -w /app sauron-api python -m pytest tests/test_agent/test_strategies/test_structured.py -k retrieve_structured -q`
Expected: FAIL — `ImportError: cannot import name 'retrieve_structured'`.

- [ ] **Step 3: Implement `retrieve_structured`**

Add to `src/agent/strategies/structured.py` (add `from src.ingestion.embedder import embed_query` near the top-level imports so tests can monkeypatch `structured.embed_query`):

At the top of the file, add to the imports:
```python
from src.ingestion.embedder import embed_query
```

Then add the function at the end:
```python
async def retrieve_structured(state, vector_store, schema_registry) -> dict:
    """Gated structured retrieval for the SWEEP branch.

    If a registered (ACL-visible) table is relevant to the question, return its
    exact SQL rows AND top-k row-narrative chunks. Fail-open: any failure yields
    whatever succeeded; an irrelevant question returns {} (sweep proceeds RAG-only).
    """
    question = state["question"]
    user_groups = state["user_groups"]

    schemas = schema_registry.list_for_user(user_groups)
    relevant = tables_relevant_to(question, schemas)
    if not relevant:
        return {}

    sql_results: list = []
    try:
        sql_results = await asyncio.to_thread(structured_sql_rows, question, relevant)
    except Exception:
        sql_results = []

    chunks: list = []
    try:
        qv = await asyncio.to_thread(embed_query, question)
        chunks = vector_store.search(
            vector=qv, user_groups=user_groups, top_k=20, tier="table_row",
        )
    except Exception:
        chunks = []

    return {"sql_results": sql_results, "retrieved_chunks": chunks}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `docker run --rm -v "$(pwd)":/app -w /app sauron-api python -m pytest tests/test_agent/test_strategies/test_structured.py -k retrieve_structured -q`
Expected: PASS (3 passed).

- [ ] **Step 5: Commit**

```bash
git add src/agent/strategies/structured.py tests/test_agent/test_strategies/test_structured.py
git commit -m "feat: retrieve_structured (gated SQL rows + row narratives, fail-open)

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task 4: Refactor `retrieve_analytical` to reuse the shared core

**Files:**
- Modify: `src/agent/strategies/analytical.py`
- Test: `tests/test_agent/test_strategies/test_analytical.py`

- [ ] **Step 1: Update the existing tests to patch `generate` where it now lives**

In `tests/test_agent/test_strategies/test_analytical.py`, the tests currently `monkeypatch.setattr(analytical, "generate", ...)`. After the refactor, `generate` is called inside `structured.structured_sql_rows`. Change those two patches:

```python
# was: monkeypatch.setattr(analytical, "generate", lambda **kw: '...SQL...')
# now: patch where the SQL core looks it up
import src.agent.strategies.structured as structured
monkeypatch.setattr(structured, "generate", lambda **kw: f'SELECT grade, salary FROM "{table}" WHERE step = 5 ORDER BY salary')
```
(Apply the same change in `test_analytical_runs_sql_against_duckdb` and `test_analytical_falls_back_when_sql_references_disallowed_table` — patch `structured.generate` instead of `analytical.generate`. The no-schemas fallback test is unaffected.)

- [ ] **Step 2: Run test to verify it fails**

Run: `docker run --rm -v "$(pwd)":/app -w /app sauron-api python -m pytest tests/test_agent/test_strategies/test_analytical.py -q`
Expected: FAIL — `test_analytical_runs_sql_against_duckdb` fails because `retrieve_analytical` still calls its own `generate` (patching `structured.generate` has no effect yet).

- [ ] **Step 3: Refactor `retrieve_analytical`**

Replace the entire contents of `src/agent/strategies/analytical.py` with:

```python
import asyncio

from src.agent.state import AgentState
from src.db.schema_registry import SchemaRegistry
from src.agent.strategies.structured import structured_sql_rows


async def retrieve_analytical(state: AgentState, vector_store, schema_registry: SchemaRegistry) -> dict:
    question = state["question"]
    user_groups = state["user_groups"]

    schemas = schema_registry.list_for_user(user_groups)
    if not schemas:
        from src.agent.strategies.map_reduce import retrieve_map_reduce
        return await retrieve_map_reduce(state, vector_store=vector_store)

    try:
        rows = await asyncio.to_thread(structured_sql_rows, question, schemas)
    except Exception:
        # No usable SQL (LLM error), blocked SQL, or execution error -> comprehensive retrieval.
        from src.agent.strategies.map_reduce import retrieve_map_reduce
        return await retrieve_map_reduce(state, vector_store=vector_store)

    return {
        "retrieved_chunks": [],
        "sql_results": rows,
        "retrieval_attempts": state.get("retrieval_attempts", 0) + 1,
    }
```

(`TEXT_TO_SQL_PROMPT` and the value-enriched prompt logic now live in `structured.py`; this file no longer defines them.)

- [ ] **Step 4: Run test to verify it passes**

Run: `docker run --rm -v "$(pwd)":/app -w /app sauron-api python -m pytest tests/test_agent/test_strategies/test_analytical.py tests/test_agent/test_strategies/test_structured.py -q`
Expected: PASS (analytical 3 + structured 7).

- [ ] **Step 5: Commit**

```bash
git add src/agent/strategies/analytical.py tests/test_agent/test_strategies/test_analytical.py
git commit -m "refactor: retrieve_analytical reuses the shared structured SQL core

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task 5: Wire `retrieve_structured` into the SWEEP branch

**Files:**
- Modify: `src/agent/graph.py`
- Test: (graph build smoke + existing strategy tests; the merge mirrors the existing sweep-chunk merge)

- [ ] **Step 1: Add the import**

In `src/agent/graph.py`, next to the other strategy imports (e.g. after `from src.agent.strategies.analytical import retrieve_analytical`), add:

```python
from src.agent.strategies.structured import retrieve_structured
```

- [ ] **Step 2: Add the gated structured task to the SWEEP gather**

In the `QueryType.SWEEP` branch, replace:

```python
            sweep_result, mr_result = await _asyncio.gather(
                retrieve_sweep(retry_state, vector_store=vector_store),
                retrieve_map_reduce(retry_state, vector_store=vector_store),
            )
```

with:

```python
            sweep_result, mr_result, struct_result = await _asyncio.gather(
                retrieve_sweep(retry_state, vector_store=vector_store),
                retrieve_map_reduce(retry_state, vector_store=vector_store),
                retrieve_structured(retry_state, vector_store=vector_store, schema_registry=schema_registry),
            )
```

- [ ] **Step 3: Merge the structured chunks + sql_results**

In the same branch, immediately AFTER the existing loop that merges `sweep_result` chunks into `merged_chunks` (the `for c in sweep_result.get("retrieved_chunks", []):` block ending with `seen_keys.add(key)`), add a matching merge for the structured chunks:

```python
            for c in struct_result.get("retrieved_chunks", []):
                key = (c.metadata.doc_id, c.metadata.chunk_index)
                if key not in seen_keys:
                    merged_chunks.append(c)
                    seen_keys.add(key)
```

Then, where the SWEEP branch builds `result = {"retrieved_chunks": merged_chunks, "retrieval_attempts": ...}`, add the SQL results right after that assignment:

```python
            if struct_result.get("sql_results"):
                result["sql_results"] = struct_result["sql_results"]
```

- [ ] **Step 4: Verify — graph builds, strategies import, full strategy suite passes**

Run: `docker run --rm -v "$(pwd)":/app -w /app sauron-api python -c "import ast; ast.parse(open('src/agent/graph.py').read()); import src.agent.graph as g; print('graph OK', hasattr(g,'create_agent_graph'))"`
Expected: `graph OK True`.

Run: `docker run --rm -v "$(pwd)":/app -w /app sauron-api python -m pytest tests/test_agent/ -q`
Expected: PASS (classifier, analytical, structured, map_reduce all green).

- [ ] **Step 5: Commit**

```bash
git add src/agent/graph.py
git commit -m "feat: blend gated structured retrieval into the SWEEP strategy

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Final verification

- [ ] Run the agent test suite:

```bash
docker run --rm -v "$(pwd)":/app -w /app sauron-api python -m pytest tests/test_agent/ -q
```
Expected: all PASS (structured: 7; analytical: 3; classifier: 11; map_reduce: 18).

- [ ] Confirm changed files are only those intended:

```bash
git diff --stat <first-commit-of-this-plan>^..HEAD
```
Expected: `src/agent/strategies/structured.py` (new), `src/agent/strategies/analytical.py`, `src/agent/graph.py`, and the two test files.

- [ ] **Deploy** (changes live retrieval): `docker compose build && docker compose up -d`; confirm health. Then an end-to-end check: ask a sweep-style pay question (as an `executives` user, given the data's ACL) and confirm the answer includes exact SQL figures (the trace/result carries `sql_results`) blended with RAG context.

## Notes for the implementer

- **The gate is intentionally permissive** (`RELEVANCE_THRESHOLD = 0.30`). A false positive costs one fail-open SQL attempt; a false negative silently loses the SQL benefit. Tune later against real queries.
- **Narrative search is ACL-filtered, not doc_id-scoped.** The spec mentioned scoping to the relevant tables' doc_ids, but the table name (`doc_<safe>_<sheet>`) can't be cleanly reversed to the original UUID `doc_id`. A broad `tier="table_row"` search filtered by `user_groups` is simpler and adequate — similarity surfaces the relevant rows, ACL keeps it safe. This is a deliberate simplification of the spec.
- **Fail-open is load-bearing.** Neither the gate, the SQL attempt, nor the narrative search may raise out of `retrieve_structured`; and adding it to the SWEEP `gather` must not change behavior when no table is relevant (it returns `{}`).
- **DRY win:** `retrieve_analytical` and the SWEEP branch now share one SQL core (`structured_sql_rows`); `TEXT_TO_SQL_PROMPT` lives in exactly one place (`structured.py`).
- **No new storage / no re-ingest.** This only changes retrieval; the existing DuckDB tables, schemas, and `table_row` narratives are consumed as-is.
```
