# Shared Cache Decision: API/Playground Parity

**Date:** 2026-05-30
**Status:** Approved (design)

## Problem

The public query API (`POST /api/v1/query` → `agent_query`) and the admin
playground (`POST /admin/api/playground/start`) answer questions through largely
the same agent graph, but their **query-cache decision logic diverges**:

- **API** (`src/generation/rag_chain.py:agent_query`): on a `cache_lookup` hit it
  returns the cached answer immediately, trusting vector similarity (≥0.92) + ACL
  match + freshness. It does **not** run the LLM applicability judge.
- **Playground** (`src/admin/routes.py`, the SSE `run_query` flow): after a
  `cache_lookup` hit it additionally calls `cache_judge` (an LLM check of whether
  the cached answer actually applies to the new question) and only serves the
  cache when the judge says `applicable`; otherwise it falls through to the full
  pipeline.

Consequence: the API can return a stale/inapplicable `[Cached result from: "..."]`
answer for a question the playground would (correctly) treat as a miss. This was
observed during this session's metadata work — reworded questions returned
pre-fix cached answers from the API.

Goal: make both paths reach the **identical cache decision** through one shared
function, so the behavior cannot drift again.

## Non-Goals (explicitly out of scope)

- **Answer synthesis.** Both paths already share the synthesis prompt, context
  builder (`build_synthesis_context`), and citation construction
  (`build_citations`). The API synthesizes in-graph (sync `generate`); the
  playground synthesizes out-of-graph via `generate_stream` (SSE) with a sync
  fallback. This streaming-vs-sync difference is intentional (the playground
  streams tokens to the browser) and is **not** changed here.
- **The dead non-SSE endpoint** `POST /admin/api/playground/query`
  (`src/admin/routes.py:~1380`), which calls `run_agent_with_trace` and runs the
  graph twice (`astream` then `ainvoke`). The playground UI does not call it.
  Flagged as a known issue; not touched by this work.
- The `query_type`-staleness in the cache and the `tables_relevant_to` gate —
  pre-existing, unrelated.

## Architecture

Introduce a single shared cache-decision function in `src/retrieval/query_cache.py`
that performs the whole "embed → lookup → judge → decide" sequence and returns a
rich result both callers can consume. Both `agent_query` (API) and the playground
`run_query` call it. Neither caller re-implements the sequence.

```
                      ┌───────────────────────────────────────┐
   API agent_query ──▶│  judged_cache_lookup(question,         │
                      │      user_groups, skip_cache=False)    │
 Playground run_query▶│    embed_query → cache_lookup          │
                      │      → (on hit) cache_judge            │
                      │    returns CacheDecision               │
                      └───────────────────────────────────────┘
                                      │
              accepted? ──────────────┴───────────────
              yes: serve cached answer        no: run full agent pipeline,
              (clean text + cached flag)          then cache_store(query_vector)
```

The decision lives in one place; the two callers differ only in how they *render*
the outcome (API → `RAGResponse`/`QueryResponse`; playground → trace HTML).

## Components

### `CacheDecision` (new dataclass, `query_cache.py`)

Carries everything either caller needs:

| Field          | Type                 | Meaning |
|----------------|----------------------|---------|
| `query_vector` | `list[float] | None` | Embedding of the question; reused for `cache_store`. `None` if embedding failed. |
| `hit`          | `bool`               | `cache_lookup` returned an entry (vector+ACL+freshness passed). |
| `accepted`     | `bool`               | `hit` **and** judge `applicable` → serve the cache. |
| `cached`       | `dict | None`        | The `cache_lookup` result dict (answer, citations, cached_query, cached_at, …). |
| `judgment`     | `dict | None`        | `{applicable, confidence, reason}`; `None` when there was no hit to judge. |
| `cache_time`   | `float`              | Seconds for embed + lookup (playground trace). |
| `judge_time`   | `float`              | Seconds for the judge call; `0.0` when no hit. |

### `judged_cache_lookup(question, user_groups, *, skip_cache=False) -> CacheDecision` (new, async, `query_cache.py`)

- `skip_cache=True` → return `CacheDecision(query_vector=<embedded or None>, hit=False)`
  immediately (no lookup), so the playground's "skip cache" toggle still works and
  the vector is still available for a later `cache_store`.
- Embed the question (`asyncio.to_thread(embed_query, …)`); on failure set
  `query_vector=None` and return a no-hit decision (fail-open — same as today's
  `try/except: pass` around the API embed).
- `cache_lookup(query_vector, user_groups)`. No hit → return `hit=False`.
- On hit → `cache_judge(cached["cached_query"], question, cached["answer"])`,
  timing it. Set `accepted = judgment["applicable"]`. (Judge failure already
  returns `applicable: True` inside `cache_judge`, preserving today's fail-open
  "use the cache if the judge is down" behavior.)
- Return the fully-populated `CacheDecision`.

### API: `src/generation/rag_chain.py:agent_query`

- Replace the inline embed + `cache_lookup` block with
  `decision = await judged_cache_lookup(question, user_groups)`.
- If `decision.accepted`: return `RAGResponse(answer=decision.cached["answer"],
  citations=…, cached=True, cached_query=decision.cached["cached_query"])`.
  **No `[Cached result from: "..."]` prefix** — the answer text is clean and the
  cache signal moves to the `cached` flag.
- Else: run `run_agent(...)` as today, then `cache_store(...)` using
  `decision.query_vector` (skip the store if `query_vector is None`).

### Response models: `RAGResponse` and `QueryResponse`

- `RAGResponse` (`rag_chain.py`): add `cached: bool = False` and
  `cached_query: str | None = None`.
- `QueryResponse` (`src/api/models.py`): add the same two fields.
- `routes_query.py`: map `result.cached` / `result.cached_query` into the
  response. Defaults keep this backward-compatible for existing clients.

### Playground: `src/admin/routes.py:run_query`

- Replace the inline embed/`cache_lookup`/`cache_judge` block (~lines 789–812)
  with one `judged_cache_lookup(question, user_groups, skip_cache=_skip_cache)`
  call.
- Build the existing trace HTML from `decision`: `cache_time`, `judge_time`,
  `judgment["confidence"]`, `judgment["reason"]`, `decision.cached`,
  `decision.accepted`. The rendered HTML (cache-hit panel, "skipped" step rows,
  result card) stays **byte-identical** to today.
- The later `cache_store` on a miss reuses `decision.query_vector`.
- Synthesis/streaming path is unchanged.

## Data Flow

1. Request arrives (API route or playground start).
2. `judged_cache_lookup` embeds the question and looks up the cache.
3. On hit, the LLM judge rules on applicability.
4. **Accepted** → caller serves the cached answer (API: clean text + `cached=True`;
   playground: cache-hit trace HTML).
5. **Not accepted** (miss or judge-rejected) → caller runs the full agent
   pipeline, then `cache_store` with the decision's `query_vector`.

## Error Handling (fail-open preserved)

- **Embed failure** → `query_vector=None`, `hit=False`: caller runs the pipeline;
  `cache_store` skipped (no vector). Matches today's API behavior.
- **`cache_lookup` failure** → already caught inside `cache_lookup` (returns
  `None`); surfaces as `hit=False`.
- **`cache_judge` failure** → `cache_judge` already returns
  `{"applicable": True, ...}`, so a judge outage means the cache is still served
  (unchanged).

## Testing

- **`judged_cache_lookup` (unit, `tests/test_retrieval/`):**
  - accepted hit (lookup returns entry, judge `applicable=True`) → `accepted=True`,
    `cached`/`judgment` populated, `query_vector` set.
  - judge rejects (`applicable=False`) → `hit=True`, `accepted=False`.
  - miss (lookup returns `None`) → `hit=False`, `accepted=False`, judge not called.
  - `skip_cache=True` → lookup not called, `query_vector` still set.
  - embed raises → `query_vector=None`, `hit=False` (no exception).
  - judge raises → fail-open `accepted=True` (via `cache_judge` default).
  (Inject/patch `embed_query`, `cache_lookup`, `cache_judge`.)
- **API `agent_query` (`tests/`):**
  - accepted hit → clean answer (no prefix), `cached=True`,
    `cached_query` set, `run_agent` not called.
  - judge-rejected hit → `run_agent` runs; result cached with `query_vector`.
  - miss → `run_agent` runs + `cache_store` called.
- **Playground regression:** a rejected judge falls through to the full pipeline
  (assert the decision path, not the HTML bytes).

## Verification

- Run affected suites in the mounted-`src` container.
- Deploy (`docker compose build api && up -d api`, ports per current
  docker-compose).
- E2E: ask a question, confirm it caches; ask a clearly-different question that
  vector-matches it, confirm the API now treats it as a miss (judge rejects)
  rather than returning `[Cached result from ...]`. Confirm the playground's
  cache-hit trace still renders for a true repeat.
