# Phase 0: Map-Reduce Timeout & Retry Fixes — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Stop map-reduce from burning ~25 minutes re-running deterministic timeouts, and cap per-document payloads so no single MAP call can run to the request timeout.

**Architecture:** Introduce typed LLM exceptions so a timeout is distinguishable from a transient connection error. Map-reduce then retries *only* transient failures, at full concurrency, and caps each document's content to a tight char budget. Permanent failures (timeouts) are reported via the existing incomplete-note mechanism instead of being retried.

**Tech Stack:** Python 3.11, asyncio, pydantic-settings, requests, pytest/pytest-asyncio. Tests run inside the app image (host lacks `lancedb`).

**Why:** A confirmed production incident ran 2590s; the retry pass alone wasted 1554s re-running size-driven timeouts at half concurrency, producing zero usable results. See `docs/superpowers/specs/2026-05-25-tabular-spreadsheet-ingestion-design.md` (Phase 0 section).

---

## File Structure

- `src/generation/llm_client.py` — **modify**: add `LLMError`/`LLMTimeoutError`/`LLMConnectionError`; raise them from `_call_llm`.
- `src/config.py` — **modify**: add `map_doc_char_budget` setting.
- `src/agent/strategies/map_reduce.py` — **modify**: add `_classify_failure` + `_cap_content` helpers; wire `map_document` to use them; make `_map_documents` retry only transient failures; change the call site to full-concurrency retry.
- `tests/test_generation/test_llm_client.py` — **create**: typed-exception tests.
- `tests/test_agent/test_strategies/test_map_reduce.py` — **modify**: helper tests + updated retry-semantics tests.

## How to run tests

Host Python has no `lancedb`, so `map_reduce` tests must run in the app image with the working tree mounted (this does **not** touch the live `sauron-api-1` service):

```bash
docker run --rm -v "$(pwd)":/app -w /app sauron-api python -m pytest <path> -q
```

`test_llm_client.py` imports only `config` and can run on host, but use the same command for consistency.

---

### Task 1: Typed LLM exceptions

**Files:**
- Modify: `src/generation/llm_client.py` (add classes after line 9; update `except` blocks at lines 37-42)
- Test: `tests/test_generation/test_llm_client.py` (create)

- [ ] **Step 1: Write the failing test**

Create `tests/test_generation/test_llm_client.py`:

```python
"""Typed LLM exceptions let callers distinguish a deterministic timeout
(not worth retrying) from a transient connection error (worth retrying)."""
import pytest
import requests

from src.generation import llm_client
from src.generation.llm_client import (
    LLMError, LLMTimeoutError, LLMConnectionError,
)


def test_timeout_raises_typed_timeout_error(monkeypatch):
    def raise_timeout(*args, **kwargs):
        raise requests.Timeout()
    monkeypatch.setattr(llm_client.requests, "post", raise_timeout)
    with pytest.raises(LLMTimeoutError):
        llm_client._call_llm([{"role": "user", "content": "hi"}], "m", 0.0, 8)


def test_connection_error_raises_typed_connection_error(monkeypatch):
    def raise_conn(*args, **kwargs):
        raise requests.ConnectionError()
    monkeypatch.setattr(llm_client.requests, "post", raise_conn)
    with pytest.raises(LLMConnectionError):
        llm_client._call_llm([{"role": "user", "content": "hi"}], "m", 0.0, 8)


def test_typed_errors_are_runtimeerror_subclasses():
    # Existing `except RuntimeError` / `except Exception` handlers must still catch them.
    assert issubclass(LLMTimeoutError, LLMError)
    assert issubclass(LLMConnectionError, LLMError)
    assert issubclass(LLMError, RuntimeError)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `docker run --rm -v "$(pwd)":/app -w /app sauron-api python -m pytest tests/test_generation/test_llm_client.py -q`
Expected: FAIL — `ImportError: cannot import name 'LLMError'`.

- [ ] **Step 3: Add the exception classes**

In `src/generation/llm_client.py`, immediately after `logger = logging.getLogger(__name__)` (line 9), add:

```python


class LLMError(RuntimeError):
    """Base class for LLM call failures."""


class LLMTimeoutError(LLMError):
    """The LLM request exceeded vllm_request_timeout. Deterministic for a given
    payload size — re-running it only wastes another full timeout."""


class LLMConnectionError(LLMError):
    """Could not reach the LLM endpoint. Transient — worth retrying."""
```

- [ ] **Step 4: Raise the typed exceptions in `_call_llm`**

In `src/generation/llm_client.py`, replace the `except` blocks (currently lines 37-42):

```python
    except requests.ConnectionError as e:
        raise RuntimeError(f"LLM connection failed: {e}")
    except requests.Timeout:
        raise RuntimeError(f"LLM request timed out after {settings.vllm_request_timeout}s")
    except requests.HTTPError as e:
        raise RuntimeError(f"LLM HTTP error: {e}")
```

with:

```python
    except requests.ConnectionError as e:
        raise LLMConnectionError(f"LLM connection failed: {e}")
    except requests.Timeout:
        raise LLMTimeoutError(f"LLM request timed out after {settings.vllm_request_timeout}s")
    except requests.HTTPError as e:
        raise LLMError(f"LLM HTTP error: {e}")
```

- [ ] **Step 5: Run test to verify it passes**

Run: `docker run --rm -v "$(pwd)":/app -w /app sauron-api python -m pytest tests/test_generation/test_llm_client.py -q`
Expected: PASS (3 passed).

- [ ] **Step 6: Commit**

```bash
git add src/generation/llm_client.py tests/test_generation/test_llm_client.py
git commit -m "feat: add typed LLM exceptions (timeout vs connection)

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task 2: Failure classification + payload-cap helpers

**Files:**
- Modify: `src/config.py` (add setting after line 60)
- Modify: `src/agent/strategies/map_reduce.py` (import at line 14; helpers after `_normalize_relevance` ~line 116; `map_document` content cap ~lines 391-393 and `except` block ~lines 407-411)
- Test: `tests/test_agent/test_strategies/test_map_reduce.py` (add tests)

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_agent/test_strategies/test_map_reduce.py`. Extend the existing import from `src.agent.strategies.map_reduce` to also import `_classify_failure` and `_cap_content`, and add a new import line:

```python
from src.generation.llm_client import LLMTimeoutError, LLMConnectionError
```

Then add this test section:

```python
# --- Failure classification & payload cap ---------------------------------

def test_timeout_failure_is_permanent():
    assert _classify_failure(LLMTimeoutError("x")) == "permanent"


def test_connection_failure_is_transient():
    assert _classify_failure(LLMConnectionError("x")) == "transient"


def test_unknown_error_is_permanent():
    # Unknown errors default to permanent — never retried, to avoid wasted timeouts.
    assert _classify_failure(ValueError("x")) == "permanent"


def test_cap_content_truncates_over_budget():
    capped = _cap_content("a" * 100, 10)
    assert capped.startswith("a" * 10)
    assert "[truncated]" in capped
    assert len(capped) < 100


def test_cap_content_leaves_small_content_untouched():
    assert _cap_content("short", 100) == "short"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `docker run --rm -v "$(pwd)":/app -w /app sauron-api python -m pytest tests/test_agent/test_strategies/test_map_reduce.py -q`
Expected: FAIL — `ImportError: cannot import name '_classify_failure'`.

- [ ] **Step 3: Add the `map_doc_char_budget` setting**

In `src/config.py`, after line 60 (`llm_max_output_tokens: int = 32768 ...`), add:

```python
    map_doc_char_budget: int = 80000  # max chars per MAP extraction call (~20K tokens); tighter than llm_max_context so one oversized doc can't run to the request timeout
```

- [ ] **Step 4: Add the helpers and update the import**

In `src/agent/strategies/map_reduce.py`, change the import (line 14) from:

```python
from src.generation.llm_client import generate
```

to:

```python
from src.generation.llm_client import generate, LLMConnectionError
```

Then, immediately after the `_normalize_relevance` function (before `async def _prefilter_by_summary`, ~line 116), add:

```python


def _classify_failure(exc: Exception) -> str:
    """Whether a failed MAP call is worth retrying.

    Connection errors are transient (the endpoint may recover); timeouts and
    everything else are permanent — a timeout on an oversized doc is
    deterministic and re-running it only wastes another full timeout.
    """
    return "transient" if isinstance(exc, LLMConnectionError) else "permanent"


def _cap_content(content: str, budget: int) -> str:
    """Truncate per-document MAP content to a char budget so no single call can
    run to the request timeout."""
    if len(content) > budget:
        return content[:budget] + "\n... [truncated]"
    return content
```

- [ ] **Step 5: Wire `map_document` to use the helpers**

In `src/agent/strategies/map_reduce.py`, inside `map_document`, replace the content-truncation block (currently ~lines 391-393):

```python
        max_content = _settings.llm_max_context
        if len(content) > max_content:
            content = content[:max_content] + "\n... [truncated]"
```

with:

```python
        content = _cap_content(content, _settings.map_doc_char_budget)
```

Then replace the `except` block (currently ~lines 407-411):

```python
        except Exception as e:
            # A timeout/error is NOT "no data" — flag it so it can be retried
            # and, if it still fails, reported instead of silently dropped.
            logger.warning(f"Map failed for {filename}: {e}")
            return {"doc_id": doc_id, "filename": filename, "extraction": "", "status": "failed"}
```

with:

```python
        except Exception as e:
            # A timeout/error is NOT "no data" — flag its kind so transient
            # failures are retried and permanent ones (timeouts) are reported
            # instead of being re-run into another timeout.
            kind = _classify_failure(e)
            logger.warning(f"Map {kind} failure for {filename}: {e}")
            return {"doc_id": doc_id, "filename": filename, "extraction": "", "status": "failed", "failure_kind": kind}
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `docker run --rm -v "$(pwd)":/app -w /app sauron-api python -m pytest tests/test_agent/test_strategies/test_map_reduce.py -q`
Expected: the 5 new tests PASS, and all pre-existing tests still PASS (the retry semantics don't change until Task 3, which rewrites `_map_documents`).

- [ ] **Step 7: Commit**

```bash
git add src/config.py src/agent/strategies/map_reduce.py tests/test_agent/test_strategies/test_map_reduce.py
git commit -m "feat: classify MAP failures and cap per-doc payload size

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task 3: Retry only transient failures, at full concurrency

**Files:**
- Modify: `src/agent/strategies/map_reduce.py` (`_map_documents` ~lines 137-167; call site ~lines 419-425)
- Test: `tests/test_agent/test_strategies/test_map_reduce.py` (update existing retry tests)

- [ ] **Step 1: Update the existing retry tests to the new semantics**

In `tests/test_agent/test_strategies/test_map_reduce.py`:

(a) In `test_failed_map_is_retried_and_can_succeed`, the first-attempt failure must be tagged transient to be retried. Change the failing return to:

```python
        if doc_id == "d2" and attempts[doc_id] == 1:
            return {"doc_id": doc_id, "filename": "f2", "extraction": "", "status": "failed", "failure_kind": "transient"}
```

(b) Replace `test_persistent_failures_are_reported_not_dropped` entirely with these two tests:

```python
@pytest.mark.asyncio
async def test_permanent_failure_is_reported_not_retried():
    # A timeout-style (permanent) failure is reported, never retried.
    attempts = {}

    async def map_one(doc_id):
        attempts[doc_id] = attempts.get(doc_id, 0) + 1
        if doc_id == "bad":
            return {"doc_id": doc_id, "filename": "bad", "extraction": "", "status": "failed", "failure_kind": "permanent"}
        return {"doc_id": doc_id, "filename": doc_id, "extraction": "x", "status": "ok"}

    results, failed = await _map_documents(
        ["good", "bad"], map_one, concurrency=2, retry_concurrency=2, max_retries=2,
    )

    assert failed == ["bad"]
    assert attempts["bad"] == 1  # permanent failure was NOT retried
    assert any(r["doc_id"] == "good" and r["status"] == "ok" for r in results)


@pytest.mark.asyncio
async def test_unrecovered_transient_failure_is_reported():
    # A transient failure that never recovers is still reported, not dropped.
    async def map_one(doc_id):
        return {"doc_id": doc_id, "filename": doc_id, "extraction": "", "status": "failed", "failure_kind": "transient"}

    results, failed = await _map_documents(
        ["d1"], map_one, concurrency=1, retry_concurrency=1, max_retries=1,
    )
    assert failed == ["d1"]
```

- [ ] **Step 2: Run tests to verify the retry tests fail**

Run: `docker run --rm -v "$(pwd)":/app -w /app sauron-api python -m pytest tests/test_agent/test_strategies/test_map_reduce.py -q`
Expected: FAIL — `test_permanent_failure_is_reported_not_retried` fails because the current `_map_documents` retries any `status == "failed"` doc regardless of `failure_kind` (so `attempts["bad"] == 3`, not 1).

- [ ] **Step 3: Rewrite `_map_documents` to retry only transient failures**

In `src/agent/strategies/map_reduce.py`, replace the body of `_map_documents` (currently ~lines 137-167) with:

```python
async def _map_documents(doc_ids, map_one, concurrency: int, retry_concurrency: int, max_retries: int = 1):
    """Run ``map_one`` over ``doc_ids``, re-attempting only *transient* failures
    before returning.

    ``map_one(doc_id)`` must return a dict with a ``status`` of ``"ok"``,
    ``"empty"`` (read fine, no relevant content), or ``"failed"``. A failed
    result also carries ``failure_kind``: ``"transient"`` (worth retrying) or
    ``"permanent"`` (timeout/size-driven — never retried, since re-running it
    only wastes another full timeout). Returns ``(results, still_failed_ids)`` —
    every surviving failure is reported, never silently counted as "no data".
    """
    async def run(ids, conc):
        sem = asyncio.Semaphore(max(1, conc))

        async def bounded(did):
            async with sem:
                return await map_one(did)
        return await asyncio.gather(*[bounded(d) for d in ids])

    def _transient(rs):
        return [r["doc_id"] for r in rs if r["status"] == "failed" and r.get("failure_kind") == "transient"]

    results = await run(doc_ids, concurrency)
    by_id = {r["doc_id"]: r for r in results}

    retryable = _transient(results)
    retries = 0
    while retryable and retries < max_retries:
        retries += 1
        logger.info(f"Map-reduce: retrying {len(retryable)} transient-failure docs (attempt {retries}) at concurrency {retry_concurrency}")
        retry_results = await run(retryable, retry_concurrency)
        for r in retry_results:
            by_id[r["doc_id"]] = r
        retryable = _transient(retry_results)

    still_failed = [doc_id for doc_id, r in by_id.items() if r["status"] == "failed"]
    return list(by_id.values()), still_failed
```

- [ ] **Step 4: Change the call site to retry at full concurrency**

In `src/agent/strategies/map_reduce.py`, in `retrieve_map_reduce` (~lines 419-425), replace:

```python
    map_results, still_failed = await _map_documents(
        relevant_doc_ids,
        map_document,
        concurrency=_settings.llm_concurrency,
        retry_concurrency=max(1, _settings.llm_concurrency // 2),
        max_retries=1,
    )
```

with:

```python
    map_results, still_failed = await _map_documents(
        relevant_doc_ids,
        map_document,
        concurrency=_settings.llm_concurrency,
        retry_concurrency=_settings.llm_concurrency,  # full concurrency; only transient failures are retried
        max_retries=1,
    )
```

- [ ] **Step 5: Run the full map_reduce test file to verify all pass**

Run: `docker run --rm -v "$(pwd)":/app -w /app sauron-api python -m pytest tests/test_agent/test_strategies/test_map_reduce.py -q`
Expected: PASS (all tests, including `test_failed_map_is_retried_and_can_succeed`, the two new failure tests, and the unchanged `test_genuine_no_data_is_not_treated_as_failure` and integration test).

- [ ] **Step 6: Commit**

```bash
git add src/agent/strategies/map_reduce.py tests/test_agent/test_strategies/test_map_reduce.py
git commit -m "fix: retry only transient MAP failures at full concurrency

Timeouts are deterministic and size-driven; re-running them (previously at
half concurrency) wasted ~25min per query for zero results. They are now
reported via the incomplete-note instead of retried.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Final verification

- [ ] Run the full affected test suites:

```bash
docker run --rm -v "$(pwd)":/app -w /app sauron-api python -m pytest \
  tests/test_generation/test_llm_client.py \
  tests/test_agent/test_strategies/test_map_reduce.py -q
```
Expected: all PASS.

- [ ] Confirm no other caller relied on the removed `RuntimeError` messages or the half-concurrency retry:

```bash
git grep -n "llm_max_context\|retry_concurrency\|LLM request timed out" src/
```
Expected: `llm_max_context` still used for synthesis/reduce (untouched); `retry_concurrency` only in `_map_documents` + the one call site; no code matches on the literal timeout message string.

## Notes for the implementer

- **Subclassing matters:** `LLMError(RuntimeError)` keeps every existing `except RuntimeError`/`except Exception` handler working — e.g. the pre-MAP gate's fail-open `except Exception` (in `_prefilter_by_summary`) still catches timeouts and keeps the doc. Do not change those handlers.
- **Permanent-by-default is intentional:** an unrecognized error is classified permanent so we never re-run into a timeout. Connection errors are the only thing we retry.
- **This plan does not make spreadsheets answerable** — it only bounds latency and stops wasted retries. The GS pay tables will still appear in the incomplete-note until the tabular-ingestion plan lands; that is expected and correct for Phase 0.
