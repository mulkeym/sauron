# Async query mode (submit → token → poll)

## Summary

Adds an **advanced async query mode** to the external API for long-running questions.
Instead of holding an HTTP request open for minutes, a caller submits a question,
gets a token immediately, then polls for live progress and the final answer.

- `POST /api/v1/query/async` `{question}` → `{token, status:"queued"}`
- `GET /api/v1/query/async/{token}` → status + current step, and the answer +
  citations once `complete` (or `error` once `failed`)

Both endpoints sit behind the existing auth (`X-API-Key` + JWT). Results are
**owner-scoped**: another user's token returns an identical `404` (no existence oracle).

## Design

- **In-memory + TTL** job store (`src/api/query_jobs.py`), mirroring the existing
  `src/ingestion/queue.py` and `src/mcp/jobs.py` patterns. A restart drops jobs;
  the caller resubmits.
- **Rich step progress** — the agent LangGraph (`classify → retrieve ∥ enrich →
  merge → synthesize`) is streamed via a new single-pass `run_agent_streamed`,
  reporting a human-readable step per node (`checking cache`, `classifying
  question`, `retrieving documents`, `searching knowledge graph`, `synthesizing
  answer`).
- **Cache parity preserved** — `agent_query_streamed` keeps the exact
  `judged_cache_lookup` / `cache_store` path; `agent_query` is now a thin delegate,
  so the existing blocking endpoint is behavior-identical.
- **Bounded worker pool** (`max_parallel_async_query`, default 3); extras wait in
  `queued`.
- **Resource bounds**: `max_async_query_jobs` cap (enqueue past it → HTTP 503) and a
  generous per-job `async_query_timeout_seconds` ceiling (default 600s) so a wedged
  query can't hold a worker slot forever.
- **No raw error leak**: callers get a generic "Query processing failed"; full
  detail is logged server-side only (matching the sync path's opaque 500).

## Config added (`src/config.py`)

| setting | default | purpose |
|---|---|---|
| `max_parallel_async_query` | 3 | concurrent worker slots |
| `async_query_ttl_seconds` | 3600 | retention of finished jobs |
| `max_async_query_jobs` | 100 | cap on tracked jobs (else 503) |
| `async_query_timeout_seconds` | 600 | per-job ceiling |

## Tests

- `tests/test_generation/test_agent_query_streamed.py` (3) — callback per node, cache
  hit short-circuit, `agent_query` delegates with no callback.
- `tests/test_api/test_query_jobs.py` (12) — state transitions, TTL eviction, bounded
  concurrency, queue-full rejection, per-job timeout.
- `tests/test_api/test_routes_query_async.py` (7) — submit, poll-to-complete, owner-scoped
  404, auth required, 503 when full.

All feature tests pass. Full-suite shows the same 20 pre-existing failures as the
master baseline (Qdrant→migration rot, unrelated) — **zero new failures**.

## Not in scope

- Durability across restart (in-memory by design).
- Streaming/SSE to external callers (polling only).
- Async parity for the OpenAI-compatible `/v1/chat/completions` endpoint.

🤖 Generated with [Claude Code](https://claude.com/claude-code)
