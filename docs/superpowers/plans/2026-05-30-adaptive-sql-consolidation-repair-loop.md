# Adaptive SQL Consolidation + Bounded Repair Loop — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Stop broad questions against wide structured tables from overflowing the model context by steering text-to-SQL away from `SELECT *` on big tables and giving the model a bounded loop to repair unsatisfactory (too-large / empty / degenerate / errored) results.

**Architecture:** A single shared helper `_generate_run_fit` in `src/agent/strategies/structured.py` runs a pre-flight wide-table gate, then a bounded generate→run→classify loop (max 3 generations). It is called by both `run_structured_lookup` and `structured_sql_rows`. The synthesizer's existing row-cap remains the final fallback.

**Tech Stack:** Python 3.11, DuckDB, pydantic-settings, pytest. LLM via `src.generation.llm_client.generate`.

**Spec:** `docs/superpowers/specs/2026-05-30-adaptive-sql-consolidation-repair-loop-design.md`

---

## File Structure

- `src/config.py` — add four config knobs (modify).
- `src/agent/strategies/structured.py` — all new logic: classification, feedback, gate, relevance judge, the `_generate_run_fit` loop; wire it into `run_structured_lookup` and `structured_sql_rows`; soften the `SELECT *` instruction (modify).
- `tests/test_agent/test_strategies/test_sql_repair.py` — new unit tests for the helpers and the loop (create).
- `tests/test_agent/test_synthesizer.py` — add the realistic wide-table regression test (modify).

Note: `_classify_sql_result`, `_repair_feedback`, `_wide_table_steering`, `_relevance_judge`, and `_generate_run_fit` are all module-level functions in `structured.py` so they can be unit-tested with stubs (no live LLM, in-memory DuckDB).

---

## Task 1: Config knobs

**Files:**
- Modify: `src/config.py:62-65` (insert after `map_doc_char_budget`)

- [ ] **Step 1: Add the knobs**

In `src/config.py`, immediately after the line `map_doc_char_budget: int = 80000  # ...` (line 65), add:

```python
    # Structured/SQL consolidation + repair loop
    sql_result_budget_chars: int = 130000  # serialized SQL-result size (chars) that counts as "too large"; ~65% of llm_max_context, kept under the synthesizer cap. Effective budget is min(this, 0.65*llm_max_context).
    sql_wide_table_cell_threshold: int = 5000  # rows*cols above which the pre-flight gate steers text-to-SQL away from SELECT *
    sql_repair_max_retries: int = 2  # retries after the first generation (so max 3 generations total)
    sql_relevance_judge_enabled: bool = True  # on a flagged result, ask the LLM why it's unhelpful and feed that into the retry
```

- [ ] **Step 2: Verify it imports**

Run: `python3 -c "from src.config import settings; print(settings.sql_result_budget_chars, settings.sql_wide_table_cell_threshold, settings.sql_repair_max_retries, settings.sql_relevance_judge_enabled)"`
Expected: `130000 5000 2 True`

- [ ] **Step 3: Commit**

```bash
git add src/config.py
git commit -m "feat: config knobs for SQL consolidation + repair loop"
```

---

## Task 2: Result classification

Pure function that labels a result list. No LLM. Error results are handled by exception in the loop, not here.

**Files:**
- Modify: `src/agent/strategies/structured.py` (add after `_extract_sql`, around line 73)
- Test: `tests/test_agent/test_strategies/test_sql_repair.py` (create)

- [ ] **Step 1: Write the failing test**

Create `tests/test_agent/test_strategies/test_sql_repair.py`:

```python
import json
from src.agent.strategies import structured as S


def test_classify_empty():
    assert S._classify_sql_result([]) == "empty"


def test_classify_degenerate_all_null():
    rows = [{"a": None, "b": None}, {"a": None, "b": None}]
    assert S._classify_sql_result(rows) == "degenerate"


def test_classify_satisfactory():
    rows = [{"locality": "Tampa", "salary": 86415}]
    assert S._classify_sql_result(rows) == "satisfactory"


def test_classify_too_large(monkeypatch):
    from src.config import settings
    monkeypatch.setattr(settings, "sql_result_budget_chars", 50)
    monkeypatch.setattr(settings, "llm_max_context", 200000)
    rows = [{"locality": f"loc{i}", "salary": i} for i in range(100)]
    assert S._classify_sql_result(rows) == "too_large"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_agent/test_strategies/test_sql_repair.py -v`
Expected: FAIL with `AttributeError: module ... has no attribute '_classify_sql_result'`

- [ ] **Step 3: Implement `_classify_sql_result`**

In `src/agent/strategies/structured.py`, add `import json` to the imports at the top (it is not currently imported), then add after `_extract_sql` (after line 73):

```python
def _effective_sql_budget() -> int:
    """Char budget for a SQL result before it counts as too large. Capped at
    65% of the model context so it always leaves room for the rest of the
    synthesis context even if the knob is set high."""
    from src.config import settings
    return min(settings.sql_result_budget_chars, int(settings.llm_max_context * 0.65))


def _classify_sql_result(rows: list[dict]) -> str:
    """Label a SQL result for the repair loop. One of:
    'empty' | 'degenerate' | 'too_large' | 'satisfactory'.
    Errors are handled separately (the query raised), not here."""
    if not rows:
        return "empty"
    if all(v is None for r in rows for v in r.values()):
        return "degenerate"
    if len(json.dumps(rows, default=str)) > _effective_sql_budget():
        return "too_large"
    return "satisfactory"
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/test_agent/test_strategies/test_sql_repair.py -v`
Expected: 4 passed

- [ ] **Step 5: Commit**

```bash
git add src/agent/strategies/structured.py tests/test_agent/test_strategies/test_sql_repair.py
git commit -m "feat: classify SQL results for repair loop (empty/degenerate/too_large)"
```

---

## Task 3: Retry feedback builder

Pure function mapping a verdict to the text appended to the next text-to-SQL attempt.

**Files:**
- Modify: `src/agent/strategies/structured.py` (add after `_classify_sql_result`)
- Test: `tests/test_agent/test_strategies/test_sql_repair.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_agent/test_strategies/test_sql_repair.py`:

```python
def test_feedback_too_large_mentions_aggregate():
    rows = [{"a": i} for i in range(885)]
    fb = S._repair_feedback("too_large", rows=rows, sql="SELECT * FROM t", question="q")
    assert "885" in fb
    assert "aggregate" in fb.lower()
    assert "SELECT * FROM t" in fb


def test_feedback_empty_mentions_filter():
    fb = S._repair_feedback("empty", rows=[], sql="SELECT * FROM t WHERE x='z'", question="q")
    assert "no rows" in fb.lower()
    assert "filter" in fb.lower()


def test_feedback_error_includes_error():
    fb = S._repair_feedback("error", rows=[], sql="SELEKT", question="q", error="syntax error")
    assert "syntax error" in fb


def test_feedback_includes_judge_reason():
    rows = [{"a": i} for i in range(885)]
    fb = S._repair_feedback("too_large", rows=rows, sql="SELECT * FROM t",
                            question="q", judge_reason="wrong column")
    assert "wrong column" in fb
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_agent/test_strategies/test_sql_repair.py -k feedback -v`
Expected: FAIL with `AttributeError: ... '_repair_feedback'`

- [ ] **Step 3: Implement `_repair_feedback`**

In `src/agent/strategies/structured.py`, add after `_classify_sql_result`:

```python
def _repair_feedback(verdict: str, *, rows: list[dict], sql: str, question: str,
                     error: str = "", judge_reason: str = "") -> str:
    """Failure-specific guidance appended to the next text-to-SQL attempt."""
    if verdict == "too_large":
        chars = len(json.dumps(rows, default=str))
        base = (f"Your previous query returned {len(rows)} rows (~{chars} chars), too large "
                f"to use. Rewrite it to summarize/aggregate (MIN/MAX/AVG with GROUP BY on "
                f"low-cardinality columns) or scope it more tightly with WHERE/LIMIT, while "
                f"still answering the question.")
    elif verdict == "empty":
        base = ("Your previous query returned no rows. Your filter or a column name may be "
                "wrong — loosen the filter, check the column names, or pick the closest "
                "available value.")
    elif verdict == "degenerate":
        base = ("Your previous query returned only NULLs. The selected columns are probably "
                "wrong for this question — choose different columns.")
    elif verdict == "error":
        base = f"Your previous query failed to run: {error}. Fix the SQL."
    else:
        base = "Improve the previous query to better answer the question."
    parts = [base, f"Previous SQL: {sql}"]
    if judge_reason:
        parts.append(f"It also does not answer the question well because: {judge_reason}.")
    parts.append(f"Question: {question}")
    return "\n".join(parts)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/test_agent/test_strategies/test_sql_repair.py -k feedback -v`
Expected: 4 passed

- [ ] **Step 5: Commit**

```bash
git add src/agent/strategies/structured.py tests/test_agent/test_strategies/test_sql_repair.py
git commit -m "feat: failure-specific retry feedback for SQL repair loop"
```

---

## Task 4: Pre-flight wide-table gate

Compute `rows × cols` per candidate table; return a steering block (or `""`) to append to the first text-to-SQL attempt.

**Files:**
- Modify: `src/agent/strategies/structured.py` (add after `_repair_feedback`)
- Test: `tests/test_agent/test_strategies/test_sql_repair.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_agent/test_strategies/test_sql_repair.py`:

```python
import duckdb
from src.db.schema_registry import TableSchema, ColumnSchema


def _schema(table, ncols):
    return TableSchema(database="tab", table=table,
                       columns=[ColumnSchema(name=f"c{i}", dtype="DOUBLE") for i in range(ncols)])


def test_wide_table_gate_fires_for_big_table(monkeypatch):
    from src.config import settings
    monkeypatch.setattr(settings, "sql_wide_table_cell_threshold", 100)
    con = duckdb.connect(":memory:")
    con.execute("CREATE TABLE big AS SELECT range AS c0, range AS c1 FROM range(60)")  # 60 rows
    schema = _schema("big", 2)  # 60*2 = 120 > 100
    block = S._wide_table_steering(con, [schema])
    assert "big" in block
    assert "aggregat" in block.lower()


def test_wide_table_gate_silent_for_small_table(monkeypatch):
    from src.config import settings
    monkeypatch.setattr(settings, "sql_wide_table_cell_threshold", 100000)
    con = duckdb.connect(":memory:")
    con.execute("CREATE TABLE small AS SELECT range AS c0 FROM range(3)")
    assert S._wide_table_steering(con, [_schema("small", 1)]) == ""
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_agent/test_strategies/test_sql_repair.py -k wide_table -v`
Expected: FAIL with `AttributeError: ... '_wide_table_steering'`

- [ ] **Step 3: Implement `_wide_table_steering`**

In `src/agent/strategies/structured.py`, add after `_repair_feedback`:

```python
def _wide_table_steering(con, schemas) -> str:
    """Pre-flight: for each candidate table, estimate rows*cols. If any exceeds
    the configured cell threshold, return a steering block telling the model to
    aggregate/scope rather than SELECT *. Returns '' when no table is wide.
    Never raises — a missing/unreadable table is simply skipped."""
    from src.config import settings
    wide = []
    for s in schemas:
        try:
            nrows = con.execute(f'SELECT COUNT(*) FROM "{s.table}"').fetchone()[0]
        except Exception:
            continue
        ncols = len(s.columns)
        if nrows * ncols > settings.sql_wide_table_cell_threshold:
            wide.append(f'{s.table} (~{nrows} rows x {ncols} cols)')
    if not wide:
        return ""
    return ("\nNOTE: " + "; ".join(wide) + " — returning every row is unhelpful and will be "
            "truncated. Prefer aggregation (MIN/MAX/AVG with GROUP BY on low-cardinality "
            "columns such as locality/grade) or scope with WHERE/LIMIT to directly answer "
            "the question.")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/test_agent/test_strategies/test_sql_repair.py -k wide_table -v`
Expected: 2 passed

- [ ] **Step 5: Commit**

```bash
git add src/agent/strategies/structured.py tests/test_agent/test_strategies/test_sql_repair.py
git commit -m "feat: pre-flight wide-table gate steers text-to-SQL away from SELECT *"
```

---

## Task 5: Soften the global `SELECT *` instruction

**Files:**
- Modify: `src/agent/strategies/structured.py:45`
- Test: `tests/test_agent/test_strategies/test_sql_repair.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_agent/test_strategies/test_sql_repair.py`:

```python
def test_prompt_no_longer_prefers_select_star():
    assert "prefer `SELECT *`" not in S.TEXT_TO_SQL_PROMPT
    assert "narrowest set of columns" in S.TEXT_TO_SQL_PROMPT
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_agent/test_strategies/test_sql_repair.py -k select_star -v`
Expected: FAIL (the prompt still says "prefer `SELECT *`")

- [ ] **Step 3: Edit the prompt**

In `src/agent/strategies/structured.py`, change line 45 from:

```python
- When in doubt, prefer `SELECT *` over a hand-picked subset of columns.
```

to:

```python
- When in doubt, prefer the narrowest set of columns that answers the question over returning everything; never select more columns than the answer needs.
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/test_agent/test_strategies/test_sql_repair.py -k select_star -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/agent/strategies/structured.py tests/test_agent/test_strategies/test_sql_repair.py
git commit -m "feat: soften text-to-SQL prompt away from SELECT * default"
```

---

## Task 6: Optional relevance judge

One LLM call (only used inside the loop, on already-flagged results) returning whether the rows answer the question and why not.

**Files:**
- Modify: `src/agent/strategies/structured.py` (add after `_wide_table_steering`)
- Test: `tests/test_agent/test_strategies/test_sql_repair.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_agent/test_strategies/test_sql_repair.py`:

```python
def test_relevance_judge_parses_unhelpful():
    def fake_gen(system_prompt, user_prompt, **kw):
        return '{"helpful": false, "reason": "rows are about grades not localities"}'
    helpful, reason = S._relevance_judge(fake_gen, "what are pay rates by locality?",
                                         [{"grade": "GS-12"}])
    assert helpful is False
    assert "localities" in reason


def test_relevance_judge_defaults_helpful_on_bad_json():
    def fake_gen(system_prompt, user_prompt, **kw):
        return "not json at all"
    helpful, reason = S._relevance_judge(fake_gen, "q", [{"a": 1}])
    assert helpful is True
    assert reason == ""
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_agent/test_strategies/test_sql_repair.py -k relevance_judge -v`
Expected: FAIL with `AttributeError: ... '_relevance_judge'`

- [ ] **Step 3: Implement `_relevance_judge`**

In `src/agent/strategies/structured.py`, add after `_wide_table_steering`:

```python
_JUDGE_PROMPT = """You check whether SQL result rows answer a user's question.
Respond with ONLY a JSON object: {"helpful": true|false, "reason": "<short reason if not helpful>"}.
Mark helpful=false only when the rows clearly do not address the question (wrong entity,
wrong columns, off-topic). If they plausibly answer it, mark helpful=true."""


def _relevance_judge(gen, question: str, rows: list[dict]) -> tuple[bool, str]:
    """Ask the LLM whether a sample of rows answers the question. Fail-open:
    any parse/LLM problem returns (True, '') so we never block on the judge."""
    sample = json.dumps(rows[:5], default=str)
    try:
        raw = gen(system_prompt=_JUDGE_PROMPT,
                  user_prompt=f"Question: {question}\nResult rows (sample): {sample}",
                  temperature=0.0, max_tokens=256)
        m = re.search(r"\{.*\}", raw, re.DOTALL)
        data = json.loads(m.group(0)) if m else {}
        return bool(data.get("helpful", True)), str(data.get("reason", "") or "")
    except Exception:
        return True, ""
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/test_agent/test_strategies/test_sql_repair.py -k relevance_judge -v`
Expected: 2 passed

- [ ] **Step 5: Commit**

```bash
git add src/agent/strategies/structured.py tests/test_agent/test_strategies/test_sql_repair.py
git commit -m "feat: optional LLM relevance judge for SQL repair loop"
```

---

## Task 7: Extend `generate_sql` to accept extra context + temperature

Single text-to-SQL entry point used by the loop for both the steered first attempt and the feedback retries.

**Files:**
- Modify: `src/agent/strategies/structured.py:107-119`
- Test: `tests/test_agent/test_strategies/test_sql_repair.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_agent/test_strategies/test_sql_repair.py`:

```python
def test_generate_sql_passes_extra_context_and_temp():
    seen = {}
    def fake_gen(system_prompt, user_prompt, temperature=0.1, max_tokens=2048):
        seen["user"] = user_prompt
        seen["temp"] = temperature
        return "SELECT 1"
    sql = S.generate_sql("SCHEMA", "my question", generate_fn=fake_gen,
                         extra_user_context="\nNOTE: aggregate please", temperature=0.3)
    assert sql == "SELECT 1"
    assert "my question" in seen["user"]
    assert "aggregate please" in seen["user"]
    assert seen["temp"] == 0.3
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_agent/test_strategies/test_sql_repair.py -k generate_sql_passes -v`
Expected: FAIL with `TypeError: generate_sql() got an unexpected keyword argument 'extra_user_context'`

- [ ] **Step 3: Modify `generate_sql`**

Replace `generate_sql` (lines 107-119) with:

```python
def generate_sql(schema_prompt: str, question: str, generate_fn=None,
                 *, extra_user_context: str = "", temperature: float = 0.0) -> str:
    """LLM text-to-SQL for one question + rendered schema prompt; returns the
    extracted SQL string (robust to prose/code-fence wrapping). ``extra_user_context``
    carries pre-flight steering or retry feedback; ``temperature`` is raised on
    retries so the model does not deterministically regenerate the same query."""
    gen = generate_fn or generate
    raw = gen(
        system_prompt=TEXT_TO_SQL_PROMPT.format(schema=schema_prompt),
        user_prompt=f"Question: {question}{extra_user_context}",
        temperature=temperature,
        max_tokens=2048,
    )
    sql = _extract_sql(raw)
    logger.info("Text-to-SQL for %r -> %s", question, sql)
    return sql
```

Note: the two new parameters are keyword-only (after `*`), so the existing call signature is unchanged; all current callers pass `generate_fn=` by keyword and keep working. The previous hardcoded `temperature=0.0` becomes the parameter default, so behavior is identical when the loop does not override it.

- [ ] **Step 4: Run the test plus the existing structured tests**

Run: `python3 -m pytest tests/test_agent/test_strategies/test_sql_repair.py -k generate_sql_passes tests/test_agent/ -k "structured or sql" -v`
Expected: PASS (new test passes; no regressions)

- [ ] **Step 5: Commit**

```bash
git add src/agent/strategies/structured.py tests/test_agent/test_strategies/test_sql_repair.py
git commit -m "feat: generate_sql accepts steering/feedback context + temperature"
```

---

## Task 8: The `_generate_run_fit` loop

Orchestrates gate → generate → run → classify → (judge) → feedback → retry, bounded. Returns the best valid result.

**Files:**
- Modify: `src/agent/strategies/structured.py` (add after `generate_sql`/`run_sql`, before `run_structured_lookup`)
- Test: `tests/test_agent/test_strategies/test_sql_repair.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_agent/test_strategies/test_sql_repair.py`:

```python
def _make_con_with_pay():
    con = duckdb.connect(":memory:")
    con.execute("CREATE TABLE pay AS SELECT * FROM (VALUES "
                "('Tampa','GS-12',86415),('Boston','GS-12',92000),('Denver','GS-13',99000)) "
                "AS t(locality, grade, salary)")
    return con


def test_fit_clean_result_one_generation():
    calls = []
    def fake_gen(system_prompt, user_prompt, temperature=0.0, max_tokens=2048):
        calls.append(user_prompt)
        return "SELECT * FROM pay"
    con = _make_con_with_pay()
    res = S._generate_run_fit(con, "what are the pay rates?", [_schema("pay", 3)],
                              generate_fn=fake_gen)
    assert res.verdict == "satisfactory"
    assert res.attempts == 1
    assert len(res.rows) == 3
    assert len(calls) == 1  # no retry, no judge


def test_fit_retries_when_first_is_empty(monkeypatch):
    monkeypatch.setattr("src.config.settings.sql_relevance_judge_enabled", False)
    seq = iter(["SELECT * FROM pay WHERE locality='nowhere'",  # empty
                "SELECT * FROM pay"])                          # good
    def fake_gen(system_prompt, user_prompt, temperature=0.0, max_tokens=2048):
        return next(seq)
    con = _make_con_with_pay()
    res = S._generate_run_fit(con, "pay rates?", [_schema("pay", 3)], generate_fn=fake_gen)
    assert res.verdict == "satisfactory"
    assert res.attempts == 2
    assert len(res.rows) == 3


def test_fit_exhausts_and_returns_best_valid(monkeypatch):
    monkeypatch.setattr("src.config.settings.sql_relevance_judge_enabled", False)
    monkeypatch.setattr("src.config.settings.sql_result_budget_chars", 10)  # force too_large
    monkeypatch.setattr("src.config.settings.sql_repair_max_retries", 2)
    def fake_gen(system_prompt, user_prompt, temperature=0.0, max_tokens=2048):
        return "SELECT * FROM pay"  # always too_large under budget=10
    con = _make_con_with_pay()
    res = S._generate_run_fit(con, "pay rates?", [_schema("pay", 3)], generate_fn=fake_gen)
    assert res.verdict == "too_large"
    assert res.attempts == 3  # orig + 2 retries
    assert len(res.rows) == 3  # best valid result kept, not discarded


def test_fit_raises_when_all_attempts_error(monkeypatch):
    monkeypatch.setattr("src.config.settings.sql_relevance_judge_enabled", False)
    def fake_gen(system_prompt, user_prompt, temperature=0.0, max_tokens=2048):
        return "SELEKT bad sql"  # raises in run_sql
    con = _make_con_with_pay()
    import pytest
    with pytest.raises(Exception):
        S._generate_run_fit(con, "q", [_schema("pay", 3)], generate_fn=fake_gen)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_agent/test_strategies/test_sql_repair.py -k fit -v`
Expected: FAIL with `AttributeError: ... '_generate_run_fit'`

- [ ] **Step 3: Implement `_generate_run_fit` and `SqlFitResult`**

In `src/agent/strategies/structured.py`, add after `run_sql` (after line 128) and before `run_structured_lookup`:

```python
@dataclass
class SqlFitResult:
    """Outcome of the gate + bounded repair loop."""
    sql: str
    rows: list = field(default_factory=list)
    attempts: int = 0
    verdict: str = "satisfactory"


def _generate_run_fit(con, question: str, schemas, *, hints=None, generate_fn=None) -> SqlFitResult:
    """Pre-flight wide-table gate, then a bounded generate->run->classify loop.
    Returns the first satisfactory result, or the best valid (non-error) result
    after exhausting retries. Raises only if EVERY attempt errored (no valid
    result was ever produced) so callers' existing error paths still engage.

    Uses one read-only connection supplied by the caller. Synchronous."""
    from src.config import settings
    from src.ingestion.tabular_store import schema_prompt_with_values
    gen = generate_fn or generate
    allowed = {s.table for s in schemas}
    base_schema_prompt = schema_prompt_with_values(schemas, con, hints=hints)
    steering = _wide_table_steering(con, schemas)

    best = None            # (sql, rows) best valid result seen
    best_verdict = "error"
    extra = steering       # first attempt carries the gate steering
    last_error = ""
    max_attempts = settings.sql_repair_max_retries + 1

    for attempt in range(max_attempts):
        temperature = 0.0 if attempt == 0 else 0.3
        sql = generate_sql(base_schema_prompt, question, generate_fn=gen,
                           extra_user_context=extra, temperature=temperature)
        try:
            rows = run_sql(con, sql, allowed)
        except Exception as e:
            last_error = str(e)
            verdict = "error"
            extra = "\n" + _repair_feedback("error", rows=[], sql=sql,
                                            question=question, error=last_error)
            continue

        verdict = _classify_sql_result(rows)
        if verdict == "satisfactory":
            return SqlFitResult(sql=sql, rows=rows, attempts=attempt + 1, verdict=verdict)

        best = (sql, rows)          # valid but unsatisfactory — keep as fallback
        best_verdict = verdict
        if attempt < max_attempts - 1:
            reason = ""
            if settings.sql_relevance_judge_enabled:
                helpful, reason = _relevance_judge(gen, question, rows)
                if helpful:
                    reason = ""
            extra = "\n" + _repair_feedback(verdict, rows=rows, sql=sql,
                                            question=question, judge_reason=reason)

    if best is not None:
        return SqlFitResult(sql=best[0], rows=best[1], attempts=max_attempts, verdict=best_verdict)
    raise RuntimeError(f"text-to-SQL produced no valid query: {last_error}")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/test_agent/test_strategies/test_sql_repair.py -k fit -v`
Expected: 4 passed

- [ ] **Step 5: Commit**

```bash
git add src/agent/strategies/structured.py tests/test_agent/test_strategies/test_sql_repair.py
git commit -m "feat: bounded SQL repair loop (_generate_run_fit) with gate + judge"
```

---

## Task 9: Wire the loop into `run_structured_lookup`

**Files:**
- Modify: `src/agent/strategies/structured.py:131-158`
- Test: `tests/test_agent/test_strategies/test_sql_repair.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_agent/test_strategies/test_sql_repair.py`:

```python
def test_run_structured_lookup_uses_loop(monkeypatch):
    monkeypatch.setattr("src.config.settings.sql_relevance_judge_enabled", False)
    # connect_tabular returns our in-memory con with a pay table
    con = _make_con_with_pay()
    monkeypatch.setattr("src.ingestion.tabular_store.connect_tabular",
                        lambda read_only=False: con)
    seq = iter(["SELECT * FROM pay WHERE locality='nope'", "SELECT * FROM pay"])
    def fake_gen(system_prompt, user_prompt, temperature=0.0, max_tokens=2048):
        return next(seq)
    trace = S.run_structured_lookup("pay rates?", [_schema("pay", 3)], "analytical",
                                    generate_fn=fake_gen)
    assert trace.status == "ran"
    assert trace.row_count == 3      # recovered after the empty first attempt
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_agent/test_strategies/test_sql_repair.py -k run_structured_lookup_uses_loop -v`
Expected: FAIL (current code runs only one generation; the empty result is returned with row_count 0)

- [ ] **Step 3: Rewrite `run_structured_lookup` to use the loop**

Replace the body of `run_structured_lookup` (lines 131-158) with:

```python
def run_structured_lookup(question: str, schemas, query_type: str,
                          gate: list | None = None, generate_fn=None, hints=None) -> StructuredLookupTrace:
    """Generate + run SQL (with the bounded repair loop) and capture a full trace.
    Never raises: a failure is recorded as status='error' (with the SQL, if
    generated) and fell_back=True so the caller can fall back. Sync (run via
    asyncio.to_thread from async callers)."""
    from src.ingestion.tabular_store import (
        connect_tabular, schema_context_for_synthesis)
    trace = StructuredLookupTrace(query_type=query_type, gate=gate)
    con = connect_tabular(read_only=True)
    try:
        fit = _generate_run_fit(con, question, schemas, hints=hints, generate_fn=generate_fn)
        trace.sql = fit.sql
        # Carry the meaning of the queried table(s) forward to the synthesizer.
        referenced = [s for s in schemas if s.table in fit.sql] or list(schemas)
        trace.schema_context = schema_context_for_synthesis(referenced, hints=hints)
        trace.status = "ran"
        trace.rows = fit.rows
        trace.row_count = len(fit.rows)
        trace.sample_rows = fit.rows[:5]
        trace.fell_back = fit.attempts > 1  # signal the loop had to retry
    except Exception as e:
        trace.status = "error"
        trace.error = str(e)
        trace.fell_back = True
    finally:
        con.close()
    return trace
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/test_agent/test_strategies/test_sql_repair.py -v`
Expected: all pass

- [ ] **Step 5: Commit**

```bash
git add src/agent/strategies/structured.py tests/test_agent/test_strategies/test_sql_repair.py
git commit -m "feat: run_structured_lookup uses bounded SQL repair loop"
```

---

## Task 10: Wire the loop into `structured_sql_rows`

**Files:**
- Modify: `src/agent/strategies/structured.py:161-176`
- Test: `tests/test_agent/test_strategies/test_sql_repair.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_agent/test_strategies/test_sql_repair.py`:

```python
def test_structured_sql_rows_uses_loop(monkeypatch):
    monkeypatch.setattr("src.config.settings.sql_relevance_judge_enabled", False)
    con = _make_con_with_pay()
    monkeypatch.setattr("src.ingestion.tabular_store.connect_tabular",
                        lambda read_only=False: con)
    seq = iter(["SELECT * FROM pay WHERE grade='none'", "SELECT * FROM pay"])
    def fake_gen(system_prompt, user_prompt, temperature=0.0, max_tokens=2048):
        return next(seq)
    rows = S.structured_sql_rows("pay rates?", [_schema("pay", 3)], generate_fn=fake_gen)
    assert len(rows) == 3  # recovered after empty first attempt
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_agent/test_strategies/test_sql_repair.py -k structured_sql_rows_uses_loop -v`
Expected: FAIL (current code returns the 0-row first result)

- [ ] **Step 3: Rewrite `structured_sql_rows` to use the loop**

Replace `structured_sql_rows` (lines 161-176) with:

```python
def structured_sql_rows(question: str, schemas, generate_fn=None, hints=None) -> list[dict]:
    """Generate SQL (with the bounded repair loop) and run it against the tabular
    DuckDB, restricted to ``schemas`` as the allowlist. One read-only connection;
    raises only if no valid query could be produced. Synchronous (run via
    ``asyncio.to_thread`` from async callers)."""
    from src.ingestion.tabular_store import connect_tabular
    con = connect_tabular(read_only=True)
    try:
        fit = _generate_run_fit(con, question, schemas, hints=hints, generate_fn=generate_fn)
        return fit.rows
    finally:
        con.close()
```

- [ ] **Step 4: Run the full new suite plus existing structured/analytical tests**

Run: `python3 -m pytest tests/test_agent/test_strategies/test_sql_repair.py tests/test_agent/ -k "structured or analytical or sql" -v`
Expected: all pass (no regressions)

- [ ] **Step 5: Commit**

```bash
git add src/agent/strategies/structured.py tests/test_agent/test_strategies/test_sql_repair.py
git commit -m "feat: structured_sql_rows uses bounded SQL repair loop"
```

---

## Task 11: Realistic wide-table synthesizer regression test

The missing regression test for the row-cap fix, using a realistic 885×32 shape.

**Files:**
- Modify: `tests/test_agent/test_synthesizer.py`

- [ ] **Step 1: Write the test**

Append to `tests/test_agent/test_synthesizer.py`:

```python
def test_synthesis_context_caps_wide_sql_result():
    from src.agent.synthesizer import build_synthesis_context
    from src.config import settings
    # Realistic OPM GS shape: 885 rows x 32 cols of real-looking values.
    big = [{f"col_{c}": f"value_{r}_{c}_xxxxxxxxxx" for c in range(32)} for r in range(885)]
    state = {
        "question": "what are the pay rates?",
        "retrieved_chunks": [],
        "sql_results": big,
        "structured_trace": {"sql": "SELECT * FROM gs_pay", "schema_context": "gs_pay(...)"},
    }
    ctx = build_synthesis_context(state)
    assert len(ctx) <= settings.llm_max_context          # never overflows
    assert "showing" in ctx and "of 885" in ctx          # truncation is disclosed
```

- [ ] **Step 2: Run test to verify it passes**

Run: `python3 -m pytest tests/test_agent/test_synthesizer.py::test_synthesis_context_caps_wide_sql_result -v`
Expected: PASS (the cap already shipped; this locks it in)

- [ ] **Step 3: Commit**

```bash
git add tests/test_agent/test_synthesizer.py
git commit -m "test: regression for synthesizer cap on realistic wide SQL result"
```

---

## Task 12: Full suite + manual smoke

**Files:** none (verification only)

- [ ] **Step 1: Run the relevant suites**

Run: `python3 -m pytest tests/test_agent/ -v`
Expected: all pass except the pre-existing unrelated `test_cross_reference_combines_doc_and_sql` failure (stale `embed_query` mock — out of scope for this work). Confirm no NEW failures.

- [ ] **Step 2: Manual smoke against the running container (optional, requires deploy)**

After deploying, replay the original failing query and confirm no overflow:

```bash
docker exec sauron-api-1 python3 -c "
import asyncio
from src.agent.strategies.structured import structured_sql_rows
# (smoke) confirm the helper imports and the loop wiring is intact
print('import + wiring OK')
"
```

Then, via the API/playground, ask "what are the pay rates?" and confirm a summarized answer (ranges by grade/locality) with no `VLLMValidationError` in the logs:

```bash
docker logs sauron-api-1 --since 2m 2>&1 | grep -i "VLLMValidationError" || echo "no overflow"
```

- [ ] **Step 3: Final commit (if any cleanup)**

```bash
git status   # should be clean; commit any stragglers
```

---

## Self-Review Notes (completed by plan author)

- **Spec coverage:** pre-flight gate (Task 4), soften SELECT * (Task 5), budget signal (Task 2 `_effective_sql_budget`), satisfaction check incl. empty/degenerate/too_large (Task 2), repair loop max 3 gens + temp 0.0→0.3 (Task 8), failure-specific feedback (Task 3), relevance judge on suspect only (Tasks 6+8), best-valid-kept + raise-only-on-all-errors (Task 8), both call sites (Tasks 9+10), row-cap fallback unchanged (relied upon, regression-tested Task 11), all four config knobs (Task 1). No gaps.
- **Type consistency:** `_generate_run_fit` returns `SqlFitResult(sql, rows, attempts, verdict)`; consumers in Tasks 9/10 read exactly those fields. `generate_sql(..., extra_user_context=, temperature=)` defined in Task 7 and used in Task 8. `_classify_sql_result`/`_repair_feedback`/`_wide_table_steering`/`_relevance_judge` signatures match their call sites.
- **Placeholder scan:** none — every code step is complete.
