# Shared Cache Decision (API/Playground Parity) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the public query API and the admin playground reach the **identical** query-cache decision by routing both through one shared `judged_cache_lookup` helper (embed → lookup → LLM applicability judge → decide). Today the API trusts a vector-similarity hit and can serve a stale `[Cached result from ...]` answer; the playground additionally runs `cache_judge`. After this, both call the same function and cannot drift.

**Architecture:** New `CacheDecision` dataclass + `async judged_cache_lookup(question, user_groups, *, skip_cache=False)` in `src/retrieval/query_cache.py`. `agent_query` (API) and the playground `run_query` both call it. API returns a **clean** cached answer plus a new `cached`/`cached_query` flag (no prefix). Playground builds its existing (byte-identical) trace HTML from the returned `CacheDecision`. Synthesis is unchanged.

**Spec:** `docs/superpowers/specs/2026-05-30-shared-cache-decision-design.md`.

**Tech Stack:** Python, FastAPI, pytest (`pytest-asyncio`), LanceDB query cache, LLM judge.

**Reference facts (verified in current code):**
- `cache_lookup(query_vector, user_groups, similarity_threshold=0.92)` (`src/retrieval/query_cache.py:116`) — returns dict `{answer, citations, query_type, source_doc_ids, cached_at, cached_query}` or `None`; already does ACL match + `_has_new_related_docs` freshness skip.
- `async cache_judge(original_query, new_query, cached_answer)` (`query_cache.py:164`) — returns `{applicable: bool, confidence: float, reason: str}`; **already fail-open** (`{"applicable": True, ...}` on exception).
- `cache_store(query_text, query_vector, answer, citations, user_groups, source_doc_ids, query_type="")` (`query_cache.py:207`).
- `embed_query` is `from src.ingestion.embedder import embed_query` (sync; both callers wrap in `asyncio.to_thread`).
- API path: `routes_query.py:11 query()` → `agent_query(...)` (`src/generation/rag_chain.py:61`). `agent_query` currently: embed (to_thread) → `cache_lookup` → on hit returns `RAGResponse(answer=f"[Cached result from: \"{cached['cached_query']}\"]\n\n{cached['answer']}", ...)` (line 80); else `run_agent` (line 86) then `cache_store` (line 97).
- `RAGResponse` is a `@dataclass` (`rag_chain.py:23`) with `answer: str`, `citations: list[Citation]`.
- `QueryResponse(BaseModel)` (`src/api/models.py:40`) — `answer: str`, `citations: list[CitationResponse]`. `routes_query.py:17` constructs it.
- Playground cache block: `src/admin/routes.py` lines **786–807** (embed/lookup/judge), `cache_accepted` at **808**, judge-reject fall-through at **810–813**, `if cache_accepted:` trace HTML at **815–890**, miss path at **892+**, `cache_store(...)` on miss at **~1195** (reuses `query_vector`). Playground reads `cache_time`, `judge_time`, `judgment["confidence"/"reason"]`, `cached`, `cache_accepted`.
- No existing cache tests. `tests/test_generation/test_rag_chain.py` and `tests/test_retrieval/` exist.

**Running tests during implementation:** host Python is too old; the baked image lacks new files. Mount host `src` + `tests`:
`docker compose run --rm --no-deps -e PYTHONPATH=/app -w /app -v "$PWD/src:/app/src" -v "$PWD/tests:/app/tests" api python -m pytest <args>`

**Deploy:** `docker compose build api && docker compose up -d api` (host ports remapped to 8880/8890/8891 per current docker-compose.yml).

---

## File Structure

- Modify `src/retrieval/query_cache.py` — add `CacheDecision` dataclass + `judged_cache_lookup`.
- Modify `src/generation/rag_chain.py` — `agent_query` uses the helper; clean answer + `cached` fields on `RAGResponse`.
- Modify `src/api/models.py` — `cached`/`cached_query` on `QueryResponse`.
- Modify `src/api/routes_query.py` — map the new fields.
- Modify `src/admin/routes.py` — `run_query` uses the helper; trace HTML byte-identical.
- Tests: `tests/test_retrieval/test_query_cache.py` (new), additions to `tests/test_generation/test_rag_chain.py`.

---

## Task 1: `CacheDecision` + `judged_cache_lookup` helper

**Files:**
- Modify: `src/retrieval/query_cache.py`
- Test: `tests/test_retrieval/test_query_cache.py` (new)

- [ ] **Step 1: Write the failing tests**

Create `tests/test_retrieval/test_query_cache.py`:

```python
"""Shared cache-decision helper."""
import pytest

from src.retrieval import query_cache as qc


@pytest.mark.asyncio
async def test_accepted_hit(monkeypatch):
    monkeypatch.setattr(qc, "embed_query", lambda q: [0.1, 0.2])
    monkeypatch.setattr(qc, "cache_lookup",
        lambda v, g, **k: {"answer": "A", "cached_query": "old q", "citations": [], "cached_at": 0})
    async def _judge(original_query, new_query, cached_answer):
        return {"applicable": True, "confidence": 0.9, "reason": "same"}
    monkeypatch.setattr(qc, "cache_judge", _judge)

    d = await qc.judged_cache_lookup("new q", ["ALL"])
    assert d.hit is True and d.accepted is True
    assert d.cached["answer"] == "A"
    assert d.judgment["applicable"] is True
    assert d.query_vector == [0.1, 0.2]


@pytest.mark.asyncio
async def test_judge_rejects(monkeypatch):
    monkeypatch.setattr(qc, "embed_query", lambda q: [0.0])
    monkeypatch.setattr(qc, "cache_lookup",
        lambda v, g, **k: {"answer": "A", "cached_query": "old", "citations": [], "cached_at": 0})
    async def _judge(**k):
        return {"applicable": False, "confidence": 0.1, "reason": "different"}
    monkeypatch.setattr(qc, "cache_judge", _judge)

    d = await qc.judged_cache_lookup("q", ["ALL"])
    assert d.hit is True and d.accepted is False


@pytest.mark.asyncio
async def test_miss_does_not_judge(monkeypatch):
    monkeypatch.setattr(qc, "embed_query", lambda q: [0.0])
    monkeypatch.setattr(qc, "cache_lookup", lambda v, g, **k: None)
    called = {"judge": False}
    async def _judge(**k):
        called["judge"] = True
        return {"applicable": True}
    monkeypatch.setattr(qc, "cache_judge", _judge)

    d = await qc.judged_cache_lookup("q", ["ALL"])
    assert d.hit is False and d.accepted is False
    assert d.query_vector == [0.0]
    assert called["judge"] is False


@pytest.mark.asyncio
async def test_skip_cache_skips_lookup(monkeypatch):
    monkeypatch.setattr(qc, "embed_query", lambda q: [0.5])
    called = {"lookup": False}
    def _lookup(v, g, **k):
        called["lookup"] = True
        return {"answer": "A", "cached_query": "x", "citations": [], "cached_at": 0}
    monkeypatch.setattr(qc, "cache_lookup", _lookup)

    d = await qc.judged_cache_lookup("q", ["ALL"], skip_cache=True)
    assert d.hit is False and d.accepted is False
    assert d.query_vector == [0.5]      # still embedded for a later cache_store
    assert called["lookup"] is False


@pytest.mark.asyncio
async def test_embed_failure_is_fail_open(monkeypatch):
    def _boom(q):
        raise RuntimeError("embed down")
    monkeypatch.setattr(qc, "embed_query", _boom)
    d = await qc.judged_cache_lookup("q", ["ALL"])
    assert d.query_vector is None and d.hit is False and d.accepted is False


@pytest.mark.asyncio
async def test_judge_failure_fails_open_to_accept(monkeypatch):
    # cache_judge already returns applicable=True on its own internal error;
    # the helper must honor that (serve the cache when the judge is down).
    monkeypatch.setattr(qc, "embed_query", lambda q: [0.0])
    monkeypatch.setattr(qc, "cache_lookup",
        lambda v, g, **k: {"answer": "A", "cached_query": "old", "citations": [], "cached_at": 0})
    async def _judge(**k):
        return {"applicable": True, "confidence": 0.5, "reason": "Judge unavailable, using cache"}
    monkeypatch.setattr(qc, "cache_judge", _judge)

    d = await qc.judged_cache_lookup("q", ["ALL"])
    assert d.accepted is True
```

- [ ] **Step 2: Run to verify they fail**

`docker compose run --rm --no-deps -e PYTHONPATH=/app -w /app -v "$PWD/src:/app/src" -v "$PWD/tests:/app/tests" api python -m pytest tests/test_retrieval/test_query_cache.py -v`
Expected: FAIL — `AttributeError: module ... has no attribute 'judged_cache_lookup'` (and `embed_query` not importable from `qc` until added).

- [ ] **Step 3: Implement the helper**

In `src/retrieval/query_cache.py`:

Add a module-level import so tests can monkeypatch `qc.embed_query` and the helper resolves it from the module:
```python
from src.ingestion.embedder import embed_query
```
(near the top imports, after `from src.config import settings`).

Add `from dataclasses import dataclass` and `import asyncio` to the imports.

Append after `cache_judge`:

```python
@dataclass
class CacheDecision:
    """Outcome of the shared cache lookup+judge sequence.
    Both the API (agent_query) and the admin playground consume this so the
    cache decision lives in exactly one place."""
    query_vector: list | None = None   # reuse for cache_store; None if embed failed
    hit: bool = False                  # cache_lookup found a vector+ACL+freshness match
    accepted: bool = False             # hit AND judge applicable -> serve the cache
    cached: dict | None = None         # the cache_lookup result
    judgment: dict | None = None       # {applicable, confidence, reason}; None if no hit
    cache_time: float = 0.0            # seconds: embed + lookup
    judge_time: float = 0.0            # seconds: judge (0 if no hit)


async def judged_cache_lookup(question: str, user_groups: list,
                              *, skip_cache: bool = False) -> CacheDecision:
    """Embed the question, look up the cache, and (on a hit) run the LLM
    applicability judge. Single source of truth for "is there a usable cache
    hit". Fail-open throughout: embed failure -> no hit (and no vector to store);
    cache_judge already returns applicable=True on its own error."""
    d = CacheDecision()
    t0 = time.time()
    try:
        d.query_vector = await asyncio.to_thread(embed_query, question)
    except Exception as e:
        logger.warning(f"Cache embed failed: {e}")
        d.cache_time = round(time.time() - t0, 2)
        return d

    if skip_cache:
        d.cache_time = round(time.time() - t0, 2)
        return d

    d.cached = cache_lookup(d.query_vector, user_groups)
    d.cache_time = round(time.time() - t0, 2)
    if not d.cached:
        return d

    d.hit = True
    tj = time.time()
    d.judgment = await cache_judge(
        original_query=d.cached.get("cached_query", ""),
        new_query=question,
        cached_answer=d.cached.get("answer", ""),
    )
    d.judge_time = round(time.time() - tj, 2)
    d.accepted = bool(d.judgment.get("applicable", False))
    return d
```

- [ ] **Step 4: Run to verify pass**

`docker compose run --rm --no-deps -e PYTHONPATH=/app -w /app -v "$PWD/src:/app/src" -v "$PWD/tests:/app/tests" api python -m pytest tests/test_retrieval/test_query_cache.py -v`
Expected: PASS (6 tests).

- [ ] **Step 5: Commit**

```bash
git add src/retrieval/query_cache.py tests/test_retrieval/test_query_cache.py
git commit -m "feat: shared judged_cache_lookup helper (CacheDecision)"
```

---

## Task 2: API uses the helper + `cached` response fields

**Files:**
- Modify: `src/generation/rag_chain.py`, `src/api/models.py`, `src/api/routes_query.py`
- Test: `tests/test_generation/test_rag_chain.py`

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_generation/test_rag_chain.py`:

```python
import pytest
from src.generation import rag_chain as rc
from src.retrieval.query_cache import CacheDecision


@pytest.mark.asyncio
async def test_agent_query_accepted_cache_is_clean(monkeypatch):
    async def _decision(q, g, **k):
        return CacheDecision(query_vector=[0.1], hit=True, accepted=True,
            cached={"answer": "cached A", "cached_query": "old q", "citations": []},
            judgment={"applicable": True})
    monkeypatch.setattr(rc, "judged_cache_lookup", _decision)
    called = {"run_agent": False}
    async def _run_agent(**k):
        called["run_agent"] = True
    monkeypatch.setattr(rc, "run_agent", _run_agent, raising=False)

    resp = await rc.agent_query("new q", ["ALL"], vector_store=None, schema_registry=None)
    assert resp.answer == "cached A"            # no "[Cached result from ...]" prefix
    assert resp.cached is True
    assert resp.cached_query == "old q"
    assert called["run_agent"] is False


@pytest.mark.asyncio
async def test_agent_query_rejected_cache_runs_agent(monkeypatch):
    async def _decision(q, g, **k):
        return CacheDecision(query_vector=[0.1], hit=True, accepted=False,
            cached={"answer": "stale", "cached_query": "old", "citations": []},
            judgment={"applicable": False})
    monkeypatch.setattr(rc, "judged_cache_lookup", _decision)
    from src.generation.rag_chain import RAGResponse
    from src.retrieval.models import Citation
    async def _run_agent(**k):
        return RAGResponse(answer="fresh", citations=[])
    monkeypatch.setattr(rc, "run_agent", _run_agent, raising=False)
    stored = {"n": 0}
    monkeypatch.setattr(rc, "cache_store", lambda **k: stored.__setitem__("n", stored["n"] + 1))

    resp = await rc.agent_query("q", ["ALL"], vector_store=None, schema_registry=None)
    assert resp.answer == "fresh" and resp.cached is False
    assert stored["n"] == 1
```

- [ ] **Step 2: Run to verify they fail**

`docker compose run --rm --no-deps -e PYTHONPATH=/app -w /app -v "$PWD/src:/app/src" -v "$PWD/tests:/app/tests" api python -m pytest tests/test_generation/test_rag_chain.py -k "cache" -v`
Expected: FAIL — `judged_cache_lookup` not imported in `rag_chain`; `RAGResponse` has no `cached` field.

- [ ] **Step 3: Implement**

In `src/generation/rag_chain.py`:

Add `cached`/`cached_query` to `RAGResponse`:
```python
@dataclass
class RAGResponse:
    answer: str
    citations: list[Citation]
    cached: bool = False
    cached_query: str | None = None
```

Rewrite the cache portion of `agent_query` (replace the embed + `cache_lookup` block and the cached-return at line ~80) to use the helper. Import at module scope so tests can patch `rc.judged_cache_lookup` / `rc.run_agent` / `rc.cache_store`:
```python
from src.retrieval.query_cache import judged_cache_lookup, cache_store
from src.agent.graph import run_agent
```
Body:
```python
async def agent_query(question, user_groups, vector_store, schema_registry, metadata_store=None) -> RAGResponse:
    decision = await judged_cache_lookup(question, user_groups)
    if decision.accepted:
        cached = decision.cached
        citations = [Citation(
            doc_id=c.get("doc_id", ""), filename=c.get("filename", ""),
            doc_type=c.get("doc_type", ""), chunk_index=c.get("chunk_index", 0),
            page=c.get("page"), snippet=c.get("snippet", ""), relevance=c.get("relevance", 0.0),
        ) for c in cached.get("citations", [])]
        return RAGResponse(answer=cached["answer"], citations=citations,
                           cached=True, cached_query=cached.get("cached_query"))

    result = await run_agent(question=question, user_groups=user_groups,
                             vector_store=vector_store, schema_registry=schema_registry,
                             metadata_store=metadata_store)

    if decision.query_vector is not None:
        try:
            citation_dicts = [{"doc_id": c.doc_id, "filename": c.filename, "doc_type": c.doc_type,
                "chunk_index": c.chunk_index, "page": c.page, "snippet": c.snippet,
                "relevance": c.relevance} for c in result.citations]
            cache_store(query_text=question, query_vector=decision.query_vector,
                answer=result.answer, citations=citation_dicts, user_groups=user_groups,
                source_doc_ids=list({c.doc_id for c in result.citations}))
        except Exception:
            pass
    return result
```
(Keep the existing module-top `embed_query`/`Citation` imports as needed; remove the now-unused inline imports of `cache_lookup`/`cache_store`/`embed_query` inside the function.)

In `src/api/models.py`, extend `QueryResponse`:
```python
class QueryResponse(BaseModel):
    answer: str
    citations: list[CitationResponse]
    cached: bool = False
    cached_query: str | None = None
```

In `src/api/routes_query.py`, map the fields in the returned `QueryResponse`:
```python
    return QueryResponse(
        answer=result.answer,
        citations=[...],
        cached=result.cached,
        cached_query=result.cached_query,
    )
```

- [ ] **Step 4: Run to verify pass**

`docker compose run --rm --no-deps -e PYTHONPATH=/app -w /app -v "$PWD/src:/app/src" -v "$PWD/tests:/app/tests" api python -m pytest tests/test_generation/test_rag_chain.py tests/test_retrieval/test_query_cache.py -v`
Expected: PASS (new + existing rag_chain tests).

- [ ] **Step 5: Commit**

```bash
git add src/generation/rag_chain.py src/api/models.py src/api/routes_query.py tests/test_generation/test_rag_chain.py
git commit -m "feat: API cache decision via judged_cache_lookup; clean answer + cached flag"
```

---

## Task 3: Playground uses the helper (trace HTML byte-identical)

**Files:**
- Modify: `src/admin/routes.py`
- Test: manual + targeted import/behavior check

- [ ] **Step 1: Replace the inline cache block**

In `src/admin/routes.py` `run_query` (lines ~786–813), replace the inline embed/`cache_lookup`/`cache_judge`/`cache_accepted` logic with one helper call, preserving the exact variable names the downstream trace HTML uses (`cache_time`, `judge_time`, `judgment`, `cached`, `cache_accepted`, `query_vector`):

```python
            # Check query cache first (unless skip_cache is set) — shared decision
            _skip_cache = skip_cache == "true"
            _playground_jobs[query_id]["step"] = "cache_check"
            from src.retrieval.query_cache import judged_cache_lookup, cache_store
            _decision = await judged_cache_lookup(question, user_groups, skip_cache=_skip_cache)
            query_vector = _decision.query_vector
            cached = _decision.cached
            cache_time = _decision.cache_time
            judge_time = _decision.judge_time
            judgment = _decision.judgment or {}
            cache_accepted = _decision.accepted
            if cached and not cache_accepted:
                cached = None            # judge rejected -> fall through to full pipeline
                _playground_jobs[query_id]["step"] = "classify"
```

Leave everything from `if cache_accepted:` onward (trace HTML, lines ~815–890), the miss path, and the later `cache_store(query_vector=query_vector, ...)` unchanged.

- [ ] **Step 2: Verify the module imports and the helper is wired**

`docker compose run --rm --no-deps -e PYTHONPATH=/app -w /app -v "$PWD/src:/app/src" -v "$PWD/tests:/app/tests" api python -c "import ast,sys; ast.parse(open('src/admin/routes.py').read()); print('parse OK')"`
And confirm the old inline judge call is gone and the helper is referenced:
`grep -n "judged_cache_lookup\|cache_judge(" src/admin/routes.py`
Expected: `judged_cache_lookup` present in `run_query`; no direct `cache_judge(` call remaining in `run_query` (it now lives in the helper).

- [ ] **Step 3: Run the admin test suite (no regressions)**

`docker compose run --rm --no-deps -e PYTHONPATH=/app -w /app -v "$PWD/src:/app/src" -v "$PWD/tests:/app/tests" api python -m pytest tests/test_admin -q`
Expected: no NEW failures vs. baseline (pre-existing/environmental failures unchanged).

- [ ] **Step 4: Commit**

```bash
git add src/admin/routes.py
git commit -m "refactor: playground cache decision via shared judged_cache_lookup"
```

---

## Task 4: Deploy + end-to-end parity check

- [ ] **Step 1: Full affected-suite run**

`docker compose run --rm --no-deps -e PYTHONPATH=/app -w /app -v "$PWD/src:/app/src" -v "$PWD/tests:/app/tests" api python -m pytest tests/test_retrieval/test_query_cache.py tests/test_generation tests/test_admin tests/test_api -q`
Expected: new tests pass; no new failures.

- [ ] **Step 2: Deploy**

```bash
docker compose build api && docker compose up -d api
```
Wait for healthy.

- [ ] **Step 3: E2E parity (deployed)**

```bash
docker exec -i sauron-api-1 env PYTHONPATH=/app python - <<'PY'
import json, urllib.request
from src.auth.jwt import create_token
tok = create_token("verify", ["executives"])
def ask(q):
    req = urllib.request.Request("http://localhost:8080/api/v1/query",
        data=json.dumps({"question": q}).encode(),
        headers={"Content-Type":"application/json","X-API-Key":"dev-key-1","Authorization":f"Bearer {tok}"},
        method="POST")
    return json.load(urllib.request.urlopen(req, timeout=300))
# 1) seed an answer
a1 = ask("How many PDFs do we have?")
print("seed cached flag:", a1.get("cached"), "| answer:", a1["answer"][:80])
# 2) exact repeat -> should be an accepted cache hit, CLEAN answer (no prefix), cached True
a2 = ask("How many PDFs do we have?")
print("repeat cached flag:", a2.get("cached"), "| starts-with-prefix:", a2["answer"].startswith("[Cached result"))
PY
docker logs sauron-api-1 2>&1 | grep -iE "Cache hit|Cache judge|Cached result" | tail -6
```
Expected: the repeat returns `cached: true`, the answer does **NOT** start with `[Cached result from ...]`, and (if a vector-similar but semantically different question is asked) the judge can reject it so the API runs fresh — matching the playground.

- [ ] **Step 4: Report results.**

---

## Notes for the implementer

- YAGNI: only the cache *decision* is shared. Do NOT touch synthesis, streaming, or the graph.
- Fail-open is a hard requirement: embed failure → run fresh (no vector → skip store); judge outage → serve cache (via `cache_judge`'s own default).
- The playground's rendered trace HTML must stay byte-identical — only the variables feeding it now come from `CacheDecision`.
- Do NOT modify the dead `POST /admin/api/playground/query` endpoint; it is out of scope (flagged in the spec).
- The host API port is 8880 (compose remap); use `localhost:8080` *inside* the container as shown.
