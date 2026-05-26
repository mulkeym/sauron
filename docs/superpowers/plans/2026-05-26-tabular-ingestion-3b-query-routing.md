# Tabular Ingestion — Plan 3b: Query Routing (exact SQL answers go live)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Route questions that map to a registered spreadsheet table to exact text-to-SQL against DuckDB: make the classifier table-aware (so it picks `ANALYTICAL`), and switch `retrieve_analytical` to run the generated SQL against the tabular DuckDB store with the user's table allowlist.

**Architecture:** The classifier gets the ACL-filtered list of registered table descriptions injected into its prompt via a node factory that closes over the schema registry. `retrieve_analytical` keeps generating SQL from `schema_registry.schemas_to_prompt`, but executes it against `connect_tabular(read_only=True)` + `execute_duckdb_sql(..., allowed_tables=<user's tables>)` (Plan 3a) instead of the sqlite executor — with the existing fall-back to map-reduce on any failure. This is the live payoff: "GS-12 step 5 in Tampa" becomes an exact SQL lookup.

**Tech Stack:** Python 3.11, DuckDB, the existing LLM client, LangGraph, pytest/pytest-asyncio. Tests run inside the app image: `docker run --rm -v "$(pwd)":/app -w /app sauron-api python -m pytest <path> -q`.

**Spec:** `docs/superpowers/specs/2026-05-25-tabular-spreadsheet-ingestion-design.md` (Component 4 — query routing). **Depends on (merged):** Plan 2a/2c (DuckDB store, `connect_tabular`), Plan 3a (`execute_duckdb_sql(con, sql, allowed_tables=...)`), the registry populated at startup (Plan 2c).

> ⚠️ **Changes live behavior** → rebuild + redeploy after merge.

---

## File Structure

- `src/agent/classifier.py` — **modify**: `classify_query(state, available_tables="")` injects available tables into the prompt; new `format_available_tables(schemas)` and `_classify_node_factory(schema_registry)`.
- `src/agent/graph.py` — **modify**: wire the `classify` node via `_classify_node_factory(schema_registry)`.
- `src/agent/strategies/analytical.py` — **modify**: execute generated SQL against DuckDB with the user's `allowed_tables`; drop the sqlite path; add a "bare table name" rule to the SQL prompt.
- `tests/test_agent/test_classifier.py` — **create**.
- `tests/test_agent/test_strategies/test_analytical.py` — **create**.

## Existing signatures this plan consumes

- `schema_registry.list_for_user(user_groups) -> list[TableSchema]`, `.schemas_to_prompt(user_groups) -> str` (returns `"No database schemas available."` when empty); `TableSchema.table`, `.description`.
- `tabular_store.connect_tabular(read_only=False)`, `tabular_store.execute_duckdb_sql(con, sql, allowed_tables=None) -> list[dict]`.
- `classifier.generate(...)`, `classifier.parse_json_response(...)`; `QueryType` enum (`state.py`).
- `create_agent_graph(vector_store, schema_registry, metadata_store)` has `schema_registry` in scope.

---

### Task 1: Classifier table-awareness (`format_available_tables` + `classify_query` injection)

**Files:**
- Modify: `src/agent/classifier.py`
- Test: `tests/test_agent/test_classifier.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_agent/test_classifier.py`:

```python
"""Tests for the table-aware query classifier."""
from src.agent.state import QueryType
from src.agent import classifier
from src.agent.classifier import format_available_tables, classify_query
from src.db.schema_registry import TableSchema, ColumnSchema


def _schema(table="doc_x_pay", desc="GS pay by grade and step", acl=None):
    return TableSchema(database="spreadsheets", table=table,
                       columns=[ColumnSchema("grade", "VARCHAR", "")],
                       description=desc, acl_groups=acl or ["ALL"])


def test_format_available_tables():
    assert format_available_tables([_schema()]) == "- doc_x_pay: GS pay by grade and step"
    assert format_available_tables([]) == ""


def test_classify_injects_tables_and_can_pick_analytical(monkeypatch):
    captured = {}
    def fake_generate(system_prompt, user_prompt, **kwargs):
        captured["system"] = system_prompt
        return '{"query_type": "analytical", "sub_tasks": []}'
    monkeypatch.setattr(classifier, "generate", fake_generate)

    out = classify_query({"question": "pay for GS-12 step 5"},
                         available_tables="- doc_x_pay: GS pay by grade and step")
    assert "doc_x_pay" in captured["system"]            # tables injected into the prompt
    assert "Available structured tables" in captured["system"]
    assert out["query_type"] == QueryType.ANALYTICAL


def test_classify_omits_tables_section_when_none(monkeypatch):
    captured = {}
    def fake_generate(system_prompt, user_prompt, **kwargs):
        captured["system"] = system_prompt
        return '{"query_type": "lookup", "sub_tasks": []}'
    monkeypatch.setattr(classifier, "generate", fake_generate)

    classify_query({"question": "who is John?"})
    assert "Available structured tables" not in captured["system"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `docker run --rm -v "$(pwd)":/app -w /app sauron-api python -m pytest tests/test_agent/test_classifier.py -q`
Expected: FAIL — `ImportError: cannot import name 'format_available_tables'`.

- [ ] **Step 3: Add `format_available_tables` and the `available_tables` parameter**

In `src/agent/classifier.py`, add `format_available_tables` near the top (after `CLASSIFICATION_PROMPT`):

```python
def format_available_tables(schemas) -> str:
    """One '- <table>: <description>' line per schema, for the classifier prompt."""
    return "\n".join(f"- {s.table}: {s.description}" for s in schemas)
```

Then change `classify_query` to accept and inject `available_tables`:

```python
def classify_query(state: AgentState, available_tables: str = "") -> dict:
    question = state["question"]
    system_prompt = CLASSIFICATION_PROMPT
    if available_tables:
        system_prompt += (
            "\n\nAvailable structured tables (queryable with SQL):\n"
            f"{available_tables}\n"
            "If the question asks for specific values, totals, or filtered rows that "
            "these tables contain, classify it as ANALYTICAL."
        )
    response = generate(
        system_prompt=system_prompt,
        user_prompt=f"Question: {question}",
        temperature=0.0,
        max_tokens=1024,
    )
    try:
        parsed = parse_json_response(response)
        query_type = QueryType(parsed["query_type"])
        sub_tasks = parsed.get("sub_tasks", [question])
    except (Exception,):
        query_type = QueryType.LOOKUP
        sub_tasks = [question]
    return {"query_type": query_type, "sub_tasks": sub_tasks}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `docker run --rm -v "$(pwd)":/app -w /app sauron-api python -m pytest tests/test_agent/test_classifier.py -q`
Expected: PASS (3 passed).

- [ ] **Step 5: Commit**

```bash
git add src/agent/classifier.py tests/test_agent/test_classifier.py
git commit -m "feat: make the query classifier aware of registered tables

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task 2: `_classify_node_factory` + graph wiring

**Files:**
- Modify: `src/agent/classifier.py`
- Modify: `src/agent/graph.py`
- Test: `tests/test_agent/test_classifier.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_agent/test_classifier.py` (extend the import to add `_classify_node_factory`):

```python
from src.agent.classifier import _classify_node_factory
from src.db.schema_registry import SchemaRegistry


def test_node_factory_passes_acl_filtered_tables(monkeypatch):
    captured = {}
    def fake_generate(system_prompt, user_prompt, **kwargs):
        captured["system"] = system_prompt
        return '{"query_type": "analytical", "sub_tasks": []}'
    monkeypatch.setattr(classifier, "generate", fake_generate)

    reg = SchemaRegistry()
    reg.register(_schema(table="doc_pay", desc="pay", acl=["ALL"]))
    reg.register(_schema(table="doc_secret", desc="secret", acl=["admins"]))

    node = _classify_node_factory(reg)
    out = node({"question": "pay?", "user_groups": ["ALL"]})

    assert "doc_pay" in captured["system"]          # visible to ALL
    assert "doc_secret" not in captured["system"]   # ACL-filtered out
    assert out["query_type"] == QueryType.ANALYTICAL


def test_node_factory_with_no_registry(monkeypatch):
    captured = {}
    def fake_generate(system_prompt, user_prompt, **kwargs):
        captured["system"] = system_prompt
        return '{"query_type": "lookup", "sub_tasks": []}'
    monkeypatch.setattr(classifier, "generate", fake_generate)

    node = _classify_node_factory(None)
    node({"question": "x", "user_groups": ["ALL"]})
    assert "Available structured tables" not in captured["system"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `docker run --rm -v "$(pwd)":/app -w /app sauron-api python -m pytest tests/test_agent/test_classifier.py -k node_factory -q`
Expected: FAIL — `ImportError: cannot import name '_classify_node_factory'`.

- [ ] **Step 3: Add `_classify_node_factory`**

In `src/agent/classifier.py`, add at the end:

```python
def _classify_node_factory(schema_registry):
    """Build a LangGraph 'classify' node that injects the user's ACL-visible
    registered tables into the classifier prompt."""
    def classify_node(state: AgentState) -> dict:
        available = ""
        if schema_registry is not None:
            schemas = schema_registry.list_for_user(state.get("user_groups", ["ALL"]))
            available = format_available_tables(schemas)
        return classify_query(state, available_tables=available)
    return classify_node
```

- [ ] **Step 4: Wire it into the graph**

In `src/agent/graph.py`, change the classifier import:

```python
from src.agent.classifier import classify_query
```

to (only `_classify_node_factory` is used in `graph.py` now — `classify_query` is used internally by the factory in `classifier.py`):

```python
from src.agent.classifier import _classify_node_factory
```

If anything else in `graph.py` references `classify_query` directly, keep it in the import; otherwise drop it to avoid an unused import.

Then change the classify node registration from:

```python
    graph.add_node("classify", classify_query)
```

to:

```python
    graph.add_node("classify", _classify_node_factory(schema_registry))
```

- [ ] **Step 5: Run tests + a graph-build smoke check**

Run: `docker run --rm -v "$(pwd)":/app -w /app sauron-api python -m pytest tests/test_agent/test_classifier.py -q`
Expected: PASS (5 passed).

Then smoke-check the graph still compiles:
Run: `docker run --rm -v "$(pwd)":/app -w /app sauron-api python -c "import ast; ast.parse(open('src/agent/graph.py').read()); import src.agent.graph as g; print('graph import OK', hasattr(g,'create_agent_graph'))"`
Expected: prints `graph import OK True`.

- [ ] **Step 6: Commit**

```bash
git add src/agent/classifier.py src/agent/graph.py tests/test_agent/test_classifier.py
git commit -m "feat: wire table-aware classify node into the agent graph

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task 3: Route `retrieve_analytical` to DuckDB with the table allowlist

**Files:**
- Modify: `src/agent/strategies/analytical.py`
- Test: `tests/test_agent/test_strategies/test_analytical.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_agent/test_strategies/test_analytical.py`:

```python
"""Tests for analytical retrieval routed to DuckDB."""
from unittest.mock import MagicMock

import pytest

from src.agent.strategies import analytical
from src.agent.strategies.analytical import retrieve_analytical
from src.db.schema_registry import SchemaRegistry, TableSchema, ColumnSchema
from src.ingestion.tabular import SheetGrid, SheetClassification
from src.ingestion.tabular_store import connect_tabular, load_sheet_to_duckdb, schema_from_sheet


def _make_pay_duckdb(tmp_path, monkeypatch):
    """Create a tabular.duckdb with a doc_doc1_pay table; return (registry, table)."""
    from src.config import settings
    monkeypatch.setattr(settings, "tabular_duckdb_path", str(tmp_path / "t.duckdb"))
    grid = SheetGrid("Pay", [["grade", "step", "salary"]] + [[f"GS-{g}", 5, 80000 + g] for g in range(10, 14)])
    cls = SheetClassification("Pay", "clean", 0, ["text", "number", "number"], "clean table")
    con = connect_tabular(read_only=False)
    table, _ = load_sheet_to_duckdb(con, "doc1", "Pay", cls, grid)
    con.close()
    reg = SchemaRegistry()
    reg.register(schema_from_sheet("doc1", "Pay", cls, grid, acl_groups=["ALL"]))
    return reg, table


@pytest.mark.asyncio
async def test_analytical_runs_sql_against_duckdb(tmp_path, monkeypatch):
    reg, table = _make_pay_duckdb(tmp_path, monkeypatch)
    monkeypatch.setattr(analytical, "generate",
                        lambda **kw: f'SELECT grade, salary FROM "{table}" WHERE step = 5 ORDER BY salary')

    state = {"question": "engineer pay", "user_groups": ["ALL"], "retrieval_attempts": 0}
    result = await retrieve_analytical(state, vector_store=MagicMock(), schema_registry=reg)

    assert result["sql_results"][0] == {"grade": "GS-10", "salary": 80010.0}
    assert len(result["sql_results"]) == 4


@pytest.mark.asyncio
async def test_analytical_falls_back_when_no_schemas(monkeypatch):
    import src.agent.strategies.map_reduce as mr
    async def fake_mr(state, vector_store):
        return {"retrieved_chunks": [], "fellback": True}
    monkeypatch.setattr(mr, "retrieve_map_reduce", fake_mr)

    state = {"question": "x", "user_groups": ["ALL"], "retrieval_attempts": 0}
    result = await retrieve_analytical(state, vector_store=MagicMock(), schema_registry=SchemaRegistry())
    assert result.get("fellback") is True


@pytest.mark.asyncio
async def test_analytical_falls_back_when_sql_references_disallowed_table(tmp_path, monkeypatch):
    reg, table = _make_pay_duckdb(tmp_path, monkeypatch)
    # LLM emits SQL referencing a table the user is NOT allowed -> allowlist rejects -> fallback
    monkeypatch.setattr(analytical, "generate", lambda **kw: 'SELECT * FROM "doc_other_secret"')
    import src.agent.strategies.map_reduce as mr
    async def fake_mr(state, vector_store):
        return {"retrieved_chunks": [], "fellback": True}
    monkeypatch.setattr(mr, "retrieve_map_reduce", fake_mr)

    state = {"question": "x", "user_groups": ["ALL"], "retrieval_attempts": 0}
    result = await retrieve_analytical(state, vector_store=MagicMock(), schema_registry=reg)
    assert result.get("fellback") is True
```

- [ ] **Step 2: Run test to verify it fails**

Run: `docker run --rm -v "$(pwd)":/app -w /app sauron-api python -m pytest tests/test_agent/test_strategies/test_analytical.py -q`
Expected: FAIL — `test_analytical_runs_sql_against_duckdb` fails because `retrieve_analytical` currently executes against sqlite (`execute_sql`), where the `doc_doc1_pay` table does not exist, so it falls back to map-reduce and returns no `sql_results`.

- [ ] **Step 3: Rewrite `retrieve_analytical` to query DuckDB with the allowlist**

Replace the entire contents of `src/agent/strategies/analytical.py` with:

```python
import asyncio

from src.agent.state import AgentState
from src.db.schema_registry import SchemaRegistry
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


async def retrieve_analytical(state: AgentState, vector_store, schema_registry: SchemaRegistry) -> dict:
    question = state["question"]
    user_groups = state["user_groups"]

    schema_prompt = schema_registry.schemas_to_prompt(user_groups)
    if schema_prompt == "No database schemas available.":
        from src.agent.strategies.map_reduce import retrieve_map_reduce
        return await retrieve_map_reduce(state, vector_store=vector_store)

    sql = generate(
        system_prompt=TEXT_TO_SQL_PROMPT.format(schema=schema_prompt),
        user_prompt=f"Question: {question}",
        temperature=0.0,
        max_tokens=2048,
    )
    sql = sql.strip().strip("`").removeprefix("sql\n").removeprefix("sql").strip()

    allowed_tables = {s.table for s in schema_registry.list_for_user(user_groups)}

    def _run_query():
        from src.ingestion.tabular_store import connect_tabular, execute_duckdb_sql
        con = connect_tabular(read_only=True)
        try:
            return execute_duckdb_sql(con, sql, allowed_tables=allowed_tables)
        finally:
            con.close()

    try:
        rows = await asyncio.to_thread(_run_query)
    except Exception:
        # Bad/blocked/failed SQL -> fall back to comprehensive retrieval.
        from src.agent.strategies.map_reduce import retrieve_map_reduce
        return await retrieve_map_reduce(state, vector_store=vector_store)

    return {
        "retrieved_chunks": [],
        "sql_results": rows,
        "retrieval_attempts": state.get("retrieval_attempts", 0) + 1,
    }
```

(This drops the unused `from src.db.sql_executor import execute_sql` and `from src.ingestion.embedder import embed_query` imports — they are no longer used.)

- [ ] **Step 4: Run test to verify it passes**

Run: `docker run --rm -v "$(pwd)":/app -w /app sauron-api python -m pytest tests/test_agent/test_strategies/test_analytical.py -q`
Expected: PASS (3 passed).

- [ ] **Step 5: Commit**

```bash
git add src/agent/strategies/analytical.py tests/test_agent/test_strategies/test_analytical.py
git commit -m "feat: route analytical queries to DuckDB with table allowlist

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Final verification

- [ ] Run the affected suites:

```bash
docker run --rm -v "$(pwd)":/app -w /app sauron-api python -m pytest \
  tests/test_agent/test_classifier.py \
  tests/test_agent/test_strategies/test_analytical.py \
  tests/test_agent/test_strategies/test_map_reduce.py -q
```
Expected: all PASS (classifier: 5; analytical: 3; map_reduce: unchanged).

- [ ] Confirm only the intended files changed:

```bash
git diff --stat <first-commit-of-this-plan>^..HEAD
```
Expected: `src/agent/classifier.py`, `src/agent/graph.py`, `src/agent/strategies/analytical.py`, and the two new test files.

- [ ] **Deploy (changes live behavior):**

```bash
docker compose build && docker compose up -d
```
Confirm health, then sanity-check end to end: ingest a GS pay spreadsheet (so a table + schema register), then ask a pay question and confirm the trace classifies `ANALYTICAL` and returns `sql_results` (not a map-reduce note).

## Notes for the implementer

- **The classifier only ever sees the user's own tables** (via `list_for_user`), so the LLM cannot be prompted toward another user's table name — and `retrieve_analytical` passes that same ACL-filtered set as `allowed_tables` to `execute_duckdb_sql`, enforcing the boundary at execution too.
- **`retrieve_analytical` now executes against DuckDB only.** The old sqlite `execute_sql` path served hypothetical external databases that were never wired; all registered schemas are DuckDB spreadsheet tables (`database="spreadsheets"`). If external SQL databases are ever added, that becomes a separate router branch — out of scope here.
- **DuckDB work runs in `asyncio.to_thread`** (connect + query + close) so the blocking driver doesn't stall the event loop; the connection is read-only and closed in `finally`.
- **Fall-back preserved:** no schemas, blocked SQL (allowlist/forbidden token), or any execution error returns the existing map-reduce result — so a misfire degrades to comprehensive retrieval rather than an error. Combined with Phase 0, that fallback is now bounded (no timeout storms).
- **The SQL prompt asks for bare table names** (no `database.` prefix) so generated SQL matches the DuckDB table names and the lexical allowlist; a schema-qualified name would fail and fall back.
- **End-to-end payoff:** with a GS pay table ingested (Plan 2c) and this routing, "what's the GS-12 step 5 base pay in the Rest of US locality" classifies ANALYTICAL → generates `SELECT ... FROM doc_..._pay WHERE grade='GS-12' AND step=5` → exact row, fast — instead of the 2600s map-reduce that started this whole effort.
```
