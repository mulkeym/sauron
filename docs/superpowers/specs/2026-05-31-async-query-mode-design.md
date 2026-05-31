# Async Query Mode — Design

**Date:** 2026-05-31
**Status:** Approved (design); pending implementation plan

## Problem

The external HTTP API exposes only **blocking** query endpoints (`POST /api/v1/query`,
`POST /v1/chat/completions`). Some questions take a long time to answer, so a caller's
HTTP request can hang for minutes with no feedback and a real risk of client/proxy timeouts.

We want an **advanced async mode**: a caller submits a question, immediately receives a
token, then polls with that token to see the current status (including live progress) and,
once finished, the final answer.

## Goals

- Submit a question and get back a token immediately (no long-held request).
- Poll by token to get coarse status **and** human-readable live step progress.
- Retrieve the final answer + citations from the same poll once complete.
- Reuse existing patterns and preserve API/playground cache parity.

## Non-goals

- Durability across an API restart (in-memory only — see Decisions).
- Server-push / SSE / websockets to external callers (polling only).
- Changing the existing blocking endpoints' behavior.
- Auto-resume of in-flight jobs after restart.

## Decisions (from brainstorming)

| Decision | Choice |
|---|---|
| Durability | **In-memory + TTL** (mirror `src/mcp/jobs.py`); a restart loses jobs, caller resubmits |
| Status detail | **Rich step progress** (coarse state + current pipeline step) |
| Ownership | **Owner-scoped** — poll requires same API key + JWT; other users get 404 |
| Concurrency | **Bounded pool** (default 3 workers; extras wait in `queued`) |

## Architecture

The agent pipeline is a LangGraph: `classify → (retrieve ∥ enrich) → merge → synthesize`.
The admin playground already streams it node-by-node via
`graph.astream(stream_mode="updates")` to surface live steps. The async worker reuses that
same streaming approach so progress comes essentially for free.

To avoid a third copy of the cache-lookup + `cache_store` logic (it currently lives in
`agent_query` and inline in the playground), we extract a single shared streaming helper
and have `agent_query` delegate to it.

### Components

**1. `src/api/query_jobs.py`** (new) — mirrors `src/mcp/jobs.py` + the ingest-queue worker:
- `QueryStatus(StrEnum)`: `QUEUED / PROCESSING / COMPLETE / FAILED`
- `@dataclass QueryJob`: `token, question, username, groups, status, step,
  answer, citations, cached, cached_query, error, created_at, completed_at`
- `class QueryJobQueue`: in-memory `_jobs: dict[str, QueryJob]` + `asyncio.Queue`,
  bounded pool of N workers (`settings.max_parallel_async_query`, default 3),
  lazy TTL eviction (`settings.async_query_ttl_seconds`, default 3600).
  Methods: `enqueue(question, username, groups) -> token`, `get_job`, `update_step`,
  `complete`, `fail`, `start_worker(...)`, `_worker_loop(...)`.
  Singleton `query_queue = QueryJobQueue()`.
- Worker loop calls `agent_query_streamed(...)` with
  `step_callback=lambda node: self.update_step(token, node)`, then `complete`/`fail`.

**2. `src/generation/rag_chain.py`** — new `agent_query_streamed(question, user_groups,
vector_store, schema_registry, metadata_store, step_callback=None) -> RAGResponse`:
- Same cache-parity path as today (`judged_cache_lookup` → on hit, return cached).
- On miss, stream the compiled graph; after each node emit `step_callback(node_name)`;
  build `RAGResponse` from the final state; `cache_store(...)` exactly as `agent_query` does.
- `agent_query` becomes a thin wrapper calling `agent_query_streamed(step_callback=None)`
  (no behavior change for the existing blocking endpoints).

**3. `src/api/routes_query.py`** — two new endpoints, both behind `require_auth`
(API key + JWT), owner-scoped:
- `POST /api/v1/query/async` — body `QueryRequest` (`{question}`); lazily starts the
  worker pool; returns `AsyncQuerySubmitResponse{token, status:"queued"}`.
- `GET /api/v1/query/async/{token}` — returns `AsyncQueryStatusResponse` with status +
  current step; includes the full answer/citations when `complete`, or `error` when
  failed. **404** if the token is unknown *or* owned by a different `username`
  (no cross-user leakage, no existence oracle).

**4. `src/api/models.py`** — new response models:
- `AsyncQuerySubmitResponse{token: str, status: str}`
- `AsyncQueryStatusResponse{token: str, status: str, step: str, answer: str | None,
  citations: list[CitationResponse], cached: bool, cached_query: str | None,
  error: str | None, created_at: float, completed_at: float | None}`

**5. Step labels** — map graph node names to human-readable text:
`queued`, `checking cache`, `classify`→"classifying", `retrieve`→"retrieving documents",
`enrich`→"searching knowledge graph", `synthesize`→"synthesizing answer", `complete`.

**6. `src/config.py`** — `max_parallel_async_query: int = 3`,
`async_query_ttl_seconds: int = 3600`.

## Data flow

```
POST /api/v1/query/async  {question}
  → require_auth (api key + jwt) → UserContext{username, groups}
  → query_queue.start_worker() (idempotent)
  → token = query_queue.enqueue(question, username, groups)   # status=QUEUED
  ← {token, status:"queued"}

worker slot picks up token:
  status=PROCESSING
  agent_query_streamed(..., step_callback=update_step)
     step → checking cache → classify → retrieve/enrich → synthesize
  → complete(token, answer, citations, cached, cached_query)   # or fail(token, error)

GET /api/v1/query/async/{token}
  → require_auth → owner check (job.username == user.username) else 404
  ← {token, status, step, answer?, citations?, cached, cached_query, error?, ...}
```

## Error handling

- Worker wraps processing in try/except → `fail_job(token, error)`; status=`FAILED`,
  `error` surfaced on poll.
- Bounded pool caps concurrent vLLM load; extra submissions wait in `queued`.
- TTL eviction is lazy (checked on `enqueue`/`get_job`); expired tokens return 404.
- Unknown / other-owner / expired token → 404 (uniform, no information leak).

## Testing (TDD)

- **Queue unit tests:** token issued on enqueue; QUEUED→PROCESSING→COMPLETE/FAILED
  transitions; owner scoping (get_job by another user denied at route layer); TTL expiry
  removes job; bounded pool runs ≤ N concurrently while extras stay queued.
- **Endpoint tests:** submit returns `{token, status:"queued"}`; poll transitions to
  `complete` with answer + citations; 404 for unknown token, other-user token, expired
  token; 403 without API key / JWT.
- **Helper test:** `step_callback` fires once per graph node; `agent_query` output
  unchanged (regression — delegates with no callback).
