# Reasoning-Assisted SQL Generation — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** When the wide-table gate fires, run the SQL-generation call(s) with the model's reasoning enabled plus a strengthened aggregation suggestion, so broad questions against wide tables produce an aggregated query instead of a lazy `LIMIT`.

**Architecture:** Thread a keyword-only `thinking` flag from `_generate_run_fit` (set to `bool(steering) and settings.sql_thinking_on_wide_table`) through `generate_sql → generate → _call_llm`, which adds the reasoning toggle + a larger token budget to the request. Thinking stays off for all other LLM calls. The wide-table steering text is reworded to lead with aggregation and a worked example.

**Tech Stack:** Python 3.11, DuckDB, pydantic-settings, requests (OpenAI-compatible vLLM), pytest.

**Spec:** `docs/superpowers/specs/2026-05-30-reasoning-assisted-sql-generation-design.md`

---

## File Structure

- `src/config.py` — add two knobs (modify).
- `src/generation/llm_client.py` — `_call_llm` and `generate` gain `thinking` (modify).
- `src/agent/strategies/structured.py` — `generate_sql` gains `thinking`; `_generate_run_fit` computes + threads it; `_wide_table_steering` text reworded (modify).
- `tests/test_generation/test_llm_client.py` — payload/forwarding tests (modify).
- `tests/test_agent/test_strategies/test_sql_repair.py` — generate_sql + loop + steering tests (modify).

Backward-compat note: `generate_sql` passes `thinking` to its `gen` callable **only when `thinking` is True** (conditional kwarg). This keeps every existing non-thinking test stub (which doesn't accept a `thinking` kwarg) working unchanged.

---

## Task 1: Config knobs

**Files:**
- Modify: `src/config.py` (after line 71, the `sql_relevance_judge_enabled` knob)

- [ ] **Step 1: Add the knobs**

In `src/config.py`, immediately after the line `sql_relevance_judge_enabled: bool = True  # ...` (line 71), add:

```python
    sql_thinking_on_wide_table: bool = True  # enable model reasoning for SQL generation when the wide-table gate fires (off elsewhere for speed)
    sql_thinking_max_tokens: int = 4096  # max_tokens for a thinking SQL-generation call (reasoning + SQL needs more than the default 2048)
```

- [ ] **Step 2: Verify import**

Run: `python3 -c "from src.config import settings; print(settings.sql_thinking_on_wide_table, settings.sql_thinking_max_tokens)"`
Expected: `True 4096`

- [ ] **Step 3: Commit**

```bash
git add src/config.py
git commit -m "feat: config knobs for reasoning-assisted SQL generation"
```

---

## Task 2: `_call_llm` + `generate` thinking parameter

**Files:**
- Modify: `src/generation/llm_client.py:25-35` (`_call_llm` signature + payload) and `:161-171` (`generate`)
- Test: `tests/test_generation/test_llm_client.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_generation/test_llm_client.py`:

```python
def test_call_llm_thinking_adds_reasoning_toggle(monkeypatch):
    from src.config import settings
    monkeypatch.setattr(settings, "sql_thinking_max_tokens", 4096)
    captured = {}

    class FakeResp:
        def raise_for_status(self):
            pass

        def json(self):
            return {"choices": [{"message": {"content": "SELECT 1"}}]}

    def fake_post(url, json, headers, timeout, verify):
        captured["payload"] = json
        return FakeResp()

    monkeypatch.setattr("src.generation.llm_client.requests.post", fake_post)
    from src.generation.llm_client import _call_llm
    out = _call_llm([{"role": "user", "content": "x"}], model="m",
                    temperature=0.0, max_tokens=2048, thinking=True)
    assert out == "SELECT 1"
    assert captured["payload"]["chat_template_kwargs"] == {"enable_thinking": True}
    assert captured["payload"]["max_tokens"] == 4096


def test_call_llm_no_thinking_by_default(monkeypatch):
    captured = {}

    class FakeResp:
        def raise_for_status(self):
            pass

        def json(self):
            return {"choices": [{"message": {"content": "ok"}}]}

    def fake_post(url, json, headers, timeout, verify):
        captured["payload"] = json
        return FakeResp()

    monkeypatch.setattr("src.generation.llm_client.requests.post", fake_post)
    from src.generation.llm_client import _call_llm
    _call_llm([{"role": "user", "content": "x"}], model="m", temperature=0.0, max_tokens=2048)
    assert "chat_template_kwargs" not in captured["payload"]
    assert captured["payload"]["max_tokens"] == 2048


def test_generate_forwards_thinking(monkeypatch):
    import src.generation.llm_client as llm

    seen = {}

    def fake_call(messages, model, temperature, max_tokens, thinking=False):
        seen["thinking"] = thinking
        return "SELECT 1"

    monkeypatch.setattr(llm, "_call_llm", fake_call)
    out = llm.generate("sys", "usr", thinking=True)
    assert out == "SELECT 1"
    assert seen["thinking"] is True
```

- [ ] **Step 2: Run to verify failure**

Run: `python3 -m pytest tests/test_generation/test_llm_client.py -k "thinking or forwards" -v`
Expected: FAIL with `TypeError: _call_llm() got an unexpected keyword argument 'thinking'`.

- [ ] **Step 3: Modify `_call_llm`**

In `src/generation/llm_client.py`, change the `_call_llm` signature and payload block (lines 25-35) from:

```python
def _call_llm(messages: list, model: str, temperature: float, max_tokens: int) -> str:
    """Call LLM via requests to an OpenAI-compatible endpoint."""
    logger.info(f"LLM call: model={model}, temperature={temperature}, max_tokens={max_tokens}")

    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "seed": settings.llm_seed,
    }
```

to:

```python
def _call_llm(messages: list, model: str, temperature: float, max_tokens: int,
              *, thinking: bool = False) -> str:
    """Call LLM via requests to an OpenAI-compatible endpoint. When ``thinking``
    is set, enable the model's reasoning (chat-template toggle) and raise the
    token budget. If the served template ignores the toggle, generation simply
    proceeds non-thinking — never an error."""
    if thinking:
        max_tokens = settings.sql_thinking_max_tokens
    logger.info(f"LLM call: model={model}, temperature={temperature}, max_tokens={max_tokens}, thinking={thinking}")

    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "seed": settings.llm_seed,
    }
    if thinking:
        payload["chat_template_kwargs"] = {"enable_thinking": True}
```

- [ ] **Step 4: Modify `generate`**

In `src/generation/llm_client.py`, change `generate` (lines 161-171) from:

```python
def generate(system_prompt, user_prompt, temperature=0.1, max_tokens=2048):
    """Generate text using the LLM."""
    original_content = _call_llm(
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        model=settings.vllm_model_name,
        temperature=temperature,
        max_tokens=max_tokens,
    )
```

to:

```python
def generate(system_prompt, user_prompt, temperature=0.1, max_tokens=2048, *, thinking=False):
    """Generate text using the LLM. ``thinking`` enables model reasoning for this call."""
    original_content = _call_llm(
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        model=settings.vllm_model_name,
        temperature=temperature,
        max_tokens=max_tokens,
        thinking=thinking,
    )
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `python3 -m pytest tests/test_generation/test_llm_client.py -v`
Expected: all pass (existing + 3 new).

- [ ] **Step 6: Commit**

```bash
git add src/generation/llm_client.py tests/test_generation/test_llm_client.py
git commit -m "feat: optional thinking flag in llm_client (_call_llm + generate)"
```

---

## Task 3: `generate_sql` thinking parameter

**Files:**
- Modify: `src/agent/strategies/structured.py` (`generate_sql`)
- Test: `tests/test_agent/test_strategies/test_sql_repair.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_agent/test_strategies/test_sql_repair.py`:

```python
def test_generate_sql_forwards_thinking_when_true():
    seen = {}
    def fake_gen(system_prompt, user_prompt, temperature=0.0, max_tokens=2048, **kwargs):
        seen.update(kwargs)
        return "SELECT 1"
    assert S.generate_sql("SCHEMA", "q", generate_fn=fake_gen, thinking=True) == "SELECT 1"
    assert seen.get("thinking") is True


def test_generate_sql_omits_thinking_when_false():
    # Stub WITHOUT **kwargs: if generate_sql passed thinking when False, this raises.
    def fake_gen(system_prompt, user_prompt, temperature=0.0, max_tokens=2048):
        return "SELECT 1"
    assert S.generate_sql("SCHEMA", "q", generate_fn=fake_gen) == "SELECT 1"
```

- [ ] **Step 2: Run to verify failure**

Run: `python3 -m pytest tests/test_agent/test_strategies/test_sql_repair.py -k "generate_sql_forwards or generate_sql_omits" -v`
Expected: `test_generate_sql_forwards_thinking_when_true` FAILS with `TypeError: generate_sql() got an unexpected keyword argument 'thinking'`.

- [ ] **Step 3: Modify `generate_sql`**

In `src/agent/strategies/structured.py`, replace the current `generate_sql` with:

```python
def generate_sql(schema_prompt: str, question: str, generate_fn=None,
                 *, extra_user_context: str = "", temperature: float = 0.0,
                 thinking: bool = False) -> str:
    """LLM text-to-SQL for one question + rendered schema prompt; returns the
    extracted SQL string (robust to prose/code-fence wrapping). ``extra_user_context``
    carries pre-flight steering or retry feedback; ``temperature`` is raised on
    retries; ``thinking`` enables model reasoning for this generation. ``thinking``
    is forwarded to ``gen`` only when True so non-thinking stubs stay compatible."""
    gen = generate_fn or generate
    extra_kwargs = {"thinking": True} if thinking else {}
    raw = gen(
        system_prompt=TEXT_TO_SQL_PROMPT.format(schema=schema_prompt),
        user_prompt=f"Question: {question}{extra_user_context}",
        temperature=temperature,
        max_tokens=2048,
        **extra_kwargs,
    )
    sql = _extract_sql(raw)
    logger.info("Text-to-SQL for %r -> %s", question, sql)
    return sql
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/test_agent/test_strategies/test_sql_repair.py -k "generate_sql" -v`
Expected: all `generate_sql` tests pass (existing + 2 new).

- [ ] **Step 5: Commit**

```bash
git add src/agent/strategies/structured.py tests/test_agent/test_strategies/test_sql_repair.py
git commit -m "feat: generate_sql forwards thinking flag (conditional, backward-compatible)"
```

---

## Task 4: `_generate_run_fit` enables thinking on the wide-table gate

**Files:**
- Modify: `src/agent/strategies/structured.py` (`_generate_run_fit`)
- Test: `tests/test_agent/test_strategies/test_sql_repair.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_agent/test_strategies/test_sql_repair.py`:

```python
def test_fit_enables_thinking_on_wide_table(monkeypatch):
    monkeypatch.setattr("src.config.settings.sql_relevance_judge_enabled", False)
    monkeypatch.setattr("src.config.settings.sql_wide_table_cell_threshold", 1)  # pay (9 cells) counts as wide
    monkeypatch.setattr("src.config.settings.sql_thinking_on_wide_table", True)
    seen = {}
    def fake_gen(system_prompt, user_prompt, temperature=0.0, max_tokens=2048, **kwargs):
        seen["thinking"] = kwargs.get("thinking", False)
        return "SELECT * FROM pay"
    con = _make_con_with_pay()
    S._generate_run_fit(con, "pay rates?", [_schema("pay", 3)], generate_fn=fake_gen)
    assert seen["thinking"] is True


def test_fit_no_thinking_on_small_table(monkeypatch):
    monkeypatch.setattr("src.config.settings.sql_relevance_judge_enabled", False)
    monkeypatch.setattr("src.config.settings.sql_wide_table_cell_threshold", 100000)  # not wide
    monkeypatch.setattr("src.config.settings.sql_thinking_on_wide_table", True)
    seen = {}
    def fake_gen(system_prompt, user_prompt, temperature=0.0, max_tokens=2048, **kwargs):
        seen["thinking"] = kwargs.get("thinking", False)
        return "SELECT * FROM pay"
    con = _make_con_with_pay()
    S._generate_run_fit(con, "pay rates?", [_schema("pay", 3)], generate_fn=fake_gen)
    assert seen["thinking"] is False


def test_fit_thinking_disabled_by_config(monkeypatch):
    monkeypatch.setattr("src.config.settings.sql_relevance_judge_enabled", False)
    monkeypatch.setattr("src.config.settings.sql_wide_table_cell_threshold", 1)  # wide
    monkeypatch.setattr("src.config.settings.sql_thinking_on_wide_table", False)  # but master switch off
    seen = {}
    def fake_gen(system_prompt, user_prompt, temperature=0.0, max_tokens=2048, **kwargs):
        seen["thinking"] = kwargs.get("thinking", False)
        return "SELECT * FROM pay"
    con = _make_con_with_pay()
    S._generate_run_fit(con, "pay rates?", [_schema("pay", 3)], generate_fn=fake_gen)
    assert seen["thinking"] is False


def test_fit_judge_call_never_thinks(monkeypatch):
    monkeypatch.setattr("src.config.settings.sql_relevance_judge_enabled", True)
    monkeypatch.setattr("src.config.settings.sql_wide_table_cell_threshold", 1)   # wide -> sql gen thinks
    monkeypatch.setattr("src.config.settings.sql_thinking_on_wide_table", True)
    monkeypatch.setattr("src.config.settings.sql_result_budget_chars", 10)        # force too_large -> judge runs
    monkeypatch.setattr("src.config.settings.sql_repair_max_retries", 1)
    judge_thinking = []
    def fake_gen(system_prompt, user_prompt, temperature=0.0, max_tokens=2048, **kwargs):
        if system_prompt == S._JUDGE_PROMPT:
            judge_thinking.append(kwargs.get("thinking", False))
            return '{"helpful": true}'
        return "SELECT * FROM pay"
    con = _make_con_with_pay()
    S._generate_run_fit(con, "pay rates?", [_schema("pay", 3)], generate_fn=fake_gen)
    assert judge_thinking and all(t is False for t in judge_thinking)
```

- [ ] **Step 2: Run to verify failure**

Run: `python3 -m pytest tests/test_agent/test_strategies/test_sql_repair.py -k "fit_enables_thinking or fit_no_thinking or fit_thinking_disabled or fit_judge_call" -v`
Expected: `test_fit_enables_thinking_on_wide_table` FAILS (currently `seen["thinking"]` is False — the loop never requests thinking).

- [ ] **Step 3: Modify `_generate_run_fit`**

In `src/agent/strategies/structured.py`, inside `_generate_run_fit`, find the line that computes steering:

```python
    steering = _wide_table_steering(con, schemas)
```

Immediately after it, add:

```python
    thinking = bool(steering) and settings.sql_thinking_on_wide_table
```

Then find the `generate_sql` call inside the loop:

```python
            sql = generate_sql(base_schema_prompt, question, generate_fn=gen,
                               extra_user_context=extra, temperature=temperature)
```

and change it to:

```python
            sql = generate_sql(base_schema_prompt, question, generate_fn=gen,
                               extra_user_context=extra, temperature=temperature,
                               thinking=thinking)
```

(The `_relevance_judge` call is unchanged — it calls `gen` without a `thinking` kwarg, so the judge never thinks.)

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/test_agent/test_strategies/test_sql_repair.py -v`
Expected: all pass (the 4 new + all existing).

- [ ] **Step 5: Commit**

```bash
git add src/agent/strategies/structured.py tests/test_agent/test_strategies/test_sql_repair.py
git commit -m "feat: enable thinking for SQL generation when the wide-table gate fires"
```

---

## Task 5: Reword the wide-table steering (lead with aggregation + example)

**Files:**
- Modify: `src/agent/strategies/structured.py` (`_wide_table_steering` return text)
- Test: `tests/test_agent/test_strategies/test_sql_repair.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_agent/test_strategies/test_sql_repair.py`:

```python
def test_wide_table_steering_leads_with_aggregation_example(monkeypatch):
    from src.config import settings
    monkeypatch.setattr(settings, "sql_wide_table_cell_threshold", 100)
    con = duckdb.connect(":memory:")
    con.execute("CREATE TABLE big AS SELECT range AS c0, range AS c1 FROM range(60)")
    block = S._wide_table_steering(con, [_schema("big", 2)])
    assert "GROUP BY" in block            # worked aggregation example present
    assert "aggregation" in block.lower()
    assert "LIMIT" not in block            # no longer offered as an equal option
```

- [ ] **Step 2: Run to verify failure**

Run: `python3 -m pytest tests/test_agent/test_strategies/test_sql_repair.py -k "steering_leads_with_aggregation" -v`
Expected: FAIL (current text contains "LIMIT" and no "GROUP BY" example).

- [ ] **Step 3: Reword the steering return**

In `src/agent/strategies/structured.py`, in `_wide_table_steering`, replace the final `return` block:

```python
    return ("\nNOTE: " + "; ".join(wide) + " — returning every row is unhelpful and will be "
            "truncated. Prefer aggregation (MIN/MAX/AVG with GROUP BY on low-cardinality "
            "columns such as locality/grade) or scope with WHERE/LIMIT to directly answer "
            "the question.")
```

with:

```python
    first = wide[0].split(" ")[0]  # a representative wide table name for the example
    return ("\nNOTE: " + "; ".join(wide) + " — returning every row is rarely what's wanted. "
            "The most useful answer is usually an aggregation — for example "
            f"`SELECT grade, MIN(annual1), MAX(annual10) FROM {first} GROUP BY grade`. "
            "Aggregate the measure columns over the low-cardinality identifying columns "
            "(e.g. grade, locality). Only return raw rows if the question genuinely asks "
            "for specific records.")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/test_agent/test_strategies/test_sql_repair.py -k "steering or wide_table" -v`
Expected: all pass (the new test + the existing `test_wide_table_gate_fires_for_big_table`, which only asserts the table name and `"aggregat"` substring — both still present).

- [ ] **Step 5: Commit**

```bash
git add src/agent/strategies/structured.py tests/test_agent/test_strategies/test_sql_repair.py
git commit -m "feat: wide-table steering leads with aggregation example, drops LIMIT option"
```

---

## Task 6: Full suite + live verification

**Files:** none (verification only)

- [ ] **Step 1: Run the affected suites**

Run: `python3 -m pytest tests/test_generation/test_llm_client.py tests/test_agent/test_strategies/test_sql_repair.py -q`
Expected: all pass.

Run: `python3 -m pytest tests/test_agent/ -q`
Expected: pass except the known unrelated pre-existing failures only if any remain (there should be none after the earlier test-fix work — confirm no NEW failures).

- [ ] **Step 2: Confirm the reasoning toggle param against the LIVE vLLM (REQUIRED before trusting it)**

After deploying (rebuild + recreate `sauron-api-1`), verify the `chat_template_kwargs={"enable_thinking": True}` toggle actually engages reasoning for this `gemma-4` build:

```bash
docker exec sauron-api-1 python3 -c "
from src.generation import llm_client as llm
# Same prompt, thinking off vs on; a thinking call should take longer / emit reasoning.
import time
for flag in (False, True):
    t0 = time.monotonic()
    out = llm._call_llm([{'role':'user','content':'Reply with only the word OK.'}],
                        model=__import__('src.config', fromlist=['settings']).settings.vllm_model_name,
                        temperature=0.0, max_tokens=2048, thinking=flag)
    print(f'thinking={flag}: {time.monotonic()-t0:.1f}s -> {out[:60]!r}')
"
```

Expected: the `thinking=True` call is meaningfully slower (reasoning ran). If `thinking=True` behaves identically to `thinking=False` (param ignored), the toggle name is wrong for this build — try alternatives (`extra_body={"chat_template_kwargs": {"enable_thinking": True}}`, or a vendor-specific key) and update `_call_llm` accordingly, re-running this check. If no toggle works, the behavior degrades safely to non-thinking; report this and stop (the feature is then a no-op pending a model/template that supports it).

- [ ] **Step 3: End-to-end acceptance — replay the original query**

```bash
docker exec sauron-api-1 python3 -c "
import asyncio, json
from src.api.routes_ingest import get_metadata_store
from src.agent.strategies.structured import _generate_run_fit
from src.ingestion.tabular_store import connect_tabular
store = get_metadata_store()
schemas = asyncio.run(store.load_all_schemas())
gs = max(schemas, key=lambda s: len(s.columns))
con = connect_tabular(read_only=True)
fit = _generate_run_fit(con, 'what are the pay rates?', [gs])
print('attempts:', fit.attempts, 'verdict:', fit.verdict, 'rows:', len(fit.rows))
print('SQL:', fit.sql)
con.close()
"
```

Expected: the SQL now contains `GROUP BY` and an aggregate (`MIN`/`MAX`/`AVG`) instead of a bare `LIMIT 100`. Record the actual SQL in the completion notes. (Reasoning quality is probabilistic — if it still emits a bare LIMIT, note it; the deferred follow-ups are stronger wording or an opt-in lazy-LIMIT retry.)

- [ ] **Step 4: Final commit (if any cleanup)**

```bash
git status   # should be clean
```

---

## Self-Review Notes (completed by plan author)

- **Spec coverage:** thinking tied to gate (Task 4: `thinking = bool(steering) and settings.sql_thinking_on_wide_table`); `generate`/`_call_llm` plumbing + max_tokens + toggle param (Task 2); `generate_sql` flag (Task 3); steering rewrite leading with aggregation, dropping LIMIT (Task 5); judge stays non-thinking (Task 4 test `test_fit_judge_call_never_thinks`); both config knobs (Task 1); fail-safe degrade-to-non-thinking (Task 2 docstring + Task 6 Step 2 verification); manual smoke acceptance (Task 6 Step 3). No gaps.
- **Type/signature consistency:** `thinking: bool = False` is keyword-only in `_call_llm`, `generate`, and `generate_sql`; `_generate_run_fit` passes `thinking=thinking` (a bool) into `generate_sql`. The conditional `extra_kwargs = {"thinking": True} if thinking else {}` is the only place the flag is conditionally forwarded, justified for stub backward-compat and covered by `test_generate_sql_omits_thinking_when_false`.
- **Placeholder scan:** none — every code step shows complete code; the only "try alternatives" is the live toggle-param verification, which is inherent to the empirical step and bounded with a defined fallback.
