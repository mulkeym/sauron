# Structured Lookup Playground Step — Design

**Date:** 2026-05-27
**Status:** Approved (brainstorming) — pending spec review before implementation planning

## Goal

Add a dedicated, collapsible **"Structured Lookup"** step to the admin playground trace
that — whenever the structured/SQL retrieval path was *considered* — shows the decision,
the relevance-gate scores, the generated SQL, and the result (rows, or the skip/error/
0-row reason). Today the SQL string, gate decision, and row count are produced during
retrieval but dropped before the UI (the SQL only appears in server logs), so an operator
running a query in the playground has no visibility into whether/why a DB lookup happened
or what it returned.

## Background

- **Playground stack** (`src/admin/templates/playground.html`, `src/admin/routes.py`):
  Jinja template + vanilla-JS polling (500ms) for steps, SSE for the final answer. Steps
  are listed in a JS `STEPS` array; the backend pre-renders each step's detail HTML via
  `_format_live_step()` (live polling) and `format_step_detail()` (final assembled trace).
  Adding a step is non-destructive — the render loop iterates `STEPS` and fills details as
  they arrive; `markStepCompleted(stepId, time, detailHtml)` renders one step.
- **Today's steps:** cache_check → classify → retrieve → enrich → synthesize, each carrying
  `{step, time, detail-HTML}`.
- **The gap:** `structured_sql_rows` (`src/agent/strategies/structured.py`) generates SQL,
  logs it, executes it, and returns only the rows. The SQL string, the gate decision
  (`tables_relevant_to`), and the row count never reach the trace. Structured retrieval runs
  *inside* the `retrieve` graph node (alongside sweep/map-reduce), not as its own node.
- **Where structured runs:** `retrieve_analytical` (runs SQL unconditionally, no gate),
  `retrieve_structured` (sweep branch — gated by `tables_relevant_to` ≥ 0.30), and
  `retrieve_cross_reference` (uses the schema registry). The step appears for these; it is
  absent for pure `lookup`/`temporal` queries that never touch structured retrieval.
- **Security posture:** the playground is admin-only and retrieval already applies the
  selected play-user's ACL; SQL + sample rows are already in the answer context, and
  surfacing SQL is consistent with the already-accepted SQL-logging tradeoff
  (`sql-classifier-logging-accepted-pii-tradeoff` memory).

## Component 1 — `StructuredLookupTrace` (data model)

New dataclass in `src/agent/strategies/structured.py`, populated by the structured
strategies and carried to the UI:

```python
@dataclass
class StructuredLookupTrace:
    query_type: str                      # "analytical" | "sweep" | "cross_reference"
    gate: list[tuple[str, float, bool]] | None  # (table, score, passed); None when no gate (analytical)
    sql: str = ""                        # generated SQL (captured even when execution errors)
    status: str = "ran"                  # "ran" | "skipped" | "error"
    skip_reason: str = ""                # e.g. "no table >= 0.30 relevance"
    error: str = ""                      # SQL/exec error message; "" if none
    row_count: int = 0
    sample_rows: list[dict] = field(default_factory=list)  # first 5 rows
    fell_back: bool = False              # did the strategy fall back to map-reduce?

    def to_dict(self) -> dict: ...        # plain-dict form for the AgentState / JSON response
```

**Capturing SQL on the error path.** `structured_sql_rows` currently generates the SQL
inside the function and raises from execution, so the SQL is lost when execution fails.
Refactor into two seams so the caller holds the SQL independently of execution:

- `generate_sql(question, schemas, con) -> str` — the prompt build + LLM call + `_extract_sql`.
- `run_sql(con, sql, allowed_tables) -> list[dict]` — `execute_duckdb_sql`.

`structured_sql_rows` remains as a thin wrapper (`run_sql(con, generate_sql(...), ...)`) so
existing callers/tests are unaffected. The strategies that build the trace call the two
seams directly so they can record `sql` before `run_sql`, and on a `run_sql` exception set
`status="error"`, `error=str(e)`, and (for sweep/cross_reference) `fell_back=True`.

## Component 2 — Strategies populate the trace

- `retrieve_analytical`: `gate=None` (runs unconditionally). On success → `status="ran"`,
  `row_count`, `sample_rows=rows[:5]`. On exception → `status="error"`, `error`,
  `fell_back=True` (it already falls back to map-reduce). Returns `structured_trace` in its
  dict.
- `retrieve_structured` (sweep): build `gate` from `tables_relevant_to` scores (each table
  with its cosine score and whether it passed the 0.30 threshold). If no table passes →
  `status="skipped"`, `skip_reason="no table >= 0.30 relevance"`, no SQL. Else run via the
  two seams and fill `sql`/`row_count`/`sample_rows` or the error fields. Returns
  `structured_trace` alongside the existing `sql_results`/`retrieved_chunks`.
- `retrieve_cross_reference`: same pattern as analytical/sweep as applicable; returns
  `structured_trace`.

## Component 3 — Thread to the UI

- `graph.py` retrieve node: when any branch returns `structured_trace`, copy it onto the
  node result and into `AgentState` (new field `structured_trace: dict`, serialized as a
  plain dict via `to_dict()`).
- `src/admin/routes.py`: new `_format_structured_lookup(trace: dict) -> str` used by **both**
  `_format_live_step()` and `format_step_detail()` so the live and final renders are
  identical. The retrieve-node handler emits the `structured_lookup` step (with its detail
  HTML) into `completed_steps` when `structured_trace` is present.
- `src/admin/templates/playground.html`: add a `structured_lookup` row to the `STEPS` array
  positioned **after `classify`, before `retrieve`**. It renders only when the backend
  emitted that step (no `structured_trace` → row absent), reusing `markStepCompleted`.

Note on timing vs display order: structured retrieval runs *inside* the retrieve node, so
the `structured_lookup` step's data is derived from that node's output and is emitted to
`completed_steps` when the retrieve node completes — even though the step is *displayed*
above Retrieve. This is fine: the `STEPS` array fixes display order, while polling fills
each step's detail in as it becomes available regardless of completion order (the existing
retrieve/enrich steps already complete in parallel and fill in this way).

## Render (the detail HTML)

Using the existing `<details>`/`<pre>` trace styling:

- **Decision line:** `Decision: <query_type> → gate ran` / `→ no gate (analytical)`.
- **Gate scores** (when present): one line per table, `table  0.71 ✓` / `0.18 ✗`, compared
  to the 0.30 threshold.
- **SQL:** the generated statement in a `<pre>` (escaped). Omitted when `status="skipped"`.
- **Result line:** `15 rows` / `skipped: <reason>` / `error: <message> (fell back to map-reduce)`
  / `0 rows (filter matched nothing)`.
- **Sample rows:** first 5 rows behind a `[view sample ▸]` `<details>` (`<pre>` JSON), shown
  only when `row_count > 0`.

## Data flow

`retrieve_{analytical,structured,cross_reference}` build `StructuredLookupTrace` →
returned under `structured_trace` → `graph.py` retrieve node copies to `AgentState`
(`to_dict()`) → playground `run_query` emits a `structured_lookup` step with detail HTML
from `_format_structured_lookup` → JS renders the new step row. No change to the answer
synthesis path; `sql_results` continues to flow as today.

## Error handling

Fully additive and fail-open. Building the trace never affects retrieval results or the
answer (a trace-build failure is caught and degrades to `status="error"` with the message,
or an absent step). `_format_structured_lookup` is wrapped so a formatter exception yields a
minimal fallback string rather than breaking the trace render. No new external calls.

## Testing (TDD, pytest)

- **Strategy/trace population** (`tests/test_agent/test_strategies/`):
  - analytical success → `structured_trace` has `gate=None`, `status="ran"`, correct
    `row_count`, `sample_rows` = first 5.
  - sweep gate skip → `status="skipped"`, `skip_reason` set, no SQL, `gate` lists the
    sub-threshold tables with `passed=False`.
  - error path → SQL captured in `trace.sql`, `status="error"`, `error` set, `fell_back=True`.
  - 0 rows → `status="ran"`, `row_count=0`, empty `sample_rows`.
- **`generate_sql`/`run_sql` split:** existing `structured_sql_rows` tests stay green
  (wrapper preserved); add a test that `generate_sql` returns extracted SQL and `run_sql`
  executes it.
- **Formatter** (`tests/test_admin/` or alongside routes tests): `_format_structured_lookup`
  output contains the SQL, gate scores, and result line for each status (ran/skipped/error/
  0-rows), and includes the sample-rows `<details>` only when rows > 0.
- Frontend JS has no test harness today; it is exercised indirectly via the formatter HTML
  tests, consistent with the existing playground code.

## Out of scope / deferred

- No JS unit-test harness is introduced (none exists today).
- No change to which strategies run or to answer synthesis.
- Multi-table/year disambiguation (3 `all_gs` year tables + `allleo`) is a separate retrieval
  concern, not part of this visualization.
- Streaming the structured step via SSE — it renders through the existing polling path like
  the other pre-synthesis steps.
