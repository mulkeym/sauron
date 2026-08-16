# Dashboard Recent Queries

**Date:** 2026-08-16
**Status:** Design approved, pending spec review

## Problem

The admin dashboard is five stat cards (documents, categories, proposals,
vector chunks, graph entities). An operator cannot see who just queried
Sauron, through which surface, with which groups, or how long the answer
took.

Activity is not stored in a usable form today:

- `src/audit/logger.py` (`AuditEntry` / JSONL) is unused in production and
  has no groups, duration, or status.
- `query_metrics` is Playground-only quality telemetry (MAP precision, docs
  cited). It has no username, surface, or tool name. Settings → Maintenance
  already renders it as a research table.

## Goals

- Show the last 10 finished calls under the dashboard stat cards.
- Cover every answer-producing HTTP surface **and every MCP tool**
  (including metadata tools such as `list_documents` / `list_sources`).
- Each row: when, type (surface + tool), strategy, username, groups,
  truncated question, duration, status (`ok` / `error` / `cache`).
- A logging failure must never fail the user request.
- A failed activity read must not take down the rest of the dashboard.

## Non-goals

- Live polling / auto-refresh (page reload is enough).
- Pagination, a “see all” page, or changing `/admin/audit`.
- Changing Settings → query-metrics or the `query_metrics` schema.
- Logging MCP resources (`document://`, `category://`, `schema://`).
- Logging MCP `get_result` (job poll, not a user question).
- Showing the generated answer on the dashboard.
- Retention / TTL / purge of old activity rows.
- Writing the unused JSONL audit logger.

## Decisions (from brainstorming)

| Decision | Choice |
|---|---|
| What counts as a call | Every answer-producing HTTP surface **plus all MCP tools** except `get_result` |
| Type column | Surface + tool (`MCP · ask`), strategy as its own column |
| Question | Truncated in the table (80 chars) |
| Failures | Included, with status |
| Cache hits | Included; status displays as `cache` |
| Storage | New `query_activity` table (not `query_metrics`, not JSONL) |
| Instrumentation | Outermost surface only (no log inside `ask()` / `agent_query`) |

## Architecture

```
 Playground / REST /query[/async] / OpenAI chat / MCP tools
        │
        ▼
  query_activity_span (times, catches, always writes)
        │
        ▼
  query_activity  (SQLite, newest first)
        │
        ▼
  GET /admin/  → last 10 rows under the stat cards
```

Log at the **outermost surface only**. A `query_database` fallback that
calls `ask()` is one `MCP · query_database` row, not a second `ask` row.

### Components

**1. `QueryActivity` (`src/db/models.py`)**

New table, created by existing `MetadataStore.init()` → `create_all`.
Import the model from `src/db/metadata.py` so the metadata registry
includes it. No ALTER — this is a new table.

| Column | Type | Notes |
|---|---|---|
| `id` | int PK | autoincrement |
| `created_at` | timezone-aware datetime | UTC, same default as other models |
| `source` | str | `mcp` / `rest` / `openai` / `playground` |
| `tool` | str | see tool catalog below |
| `username` | str | JWT / OpenWebUI identity / Playground persona; empty if unknown |
| `user_groups` | JSON list | groups actually used for the call |
| `query_text` | str | question (or tool argument), stored cap **500** chars |
| `strategy` | str | classifier `query_type`, `structured`, `cache`, or empty |
| `duration_seconds` | float | processing wall time, not time-in-queue |
| `status` | str | `ok` or `error` |
| `cache_hit` | bool | default false |
| `error` | str | failure only; cap **200** chars; no traceback |

**2. Helper (`src/audit/activity.py`)** — do not extend `logger.py`.

- `record_query_activity(...)` — async insert via `MetadataStore` (or an
  injected store in tests). Best-effort: log and drop on DB failure.
- `list_recent_query_activity(limit=10)` — newest `created_at` first.
- `query_activity_span(...)` — **async** context manager that:
  - starts a timer on enter
  - yields a span object with writable `strategy`, `cache_hit`, `status`,
    `error`
  - on an uncaught exception: `status=error`, `error=str(exc)[:200]`
  - always records on exit
  - does **not** swallow the exception (`__aexit__` returns false)

Sync MCP handlers in `src/mcp/server.py` become `async def` so they can
`async with` the span; they still call the existing sync implementations.

**3. Strategy plumbing (`RAGResponse`)**

`RAGResponse` gains optional `query_type: str = ""`.

- Cache accepted in `_agent_query_streamed_bound` → `query_type="cache"`,
  `cached=True`.
- Graph finish (`run_agent_streamed` / `run_agent_with_trace`) →
  `query_type` from classify (`lookup`, `sweep`, `analytical`,
  `cross_reference`, `temporal`, `metadata`).
- Error before classify → leave strategy empty.

Playground already sees classify output; it does not need `RAGResponse`
for strategy but should use the same string values.

High-level MCP helpers that call `agent_query` (`ask`, `summarize_topic`,
`compare`) include `query_type` and `cached` in their return dict so the
`server.py` wrapper can set the span. `query_database` forwards those
fields when it falls back to `ask`.

**4. Dashboard**

`GET /admin/` loads `list_recent_query_activity(10)` and passes `activity`
into `dashboard.html`. A new “Recent queries” block sits under
`.stats-grid`. Existing table CSS; no new endpoint; no HTMX poll.

### Tool catalog

| Surface | `source` | `tool` | Type shown | `query_text` | `strategy` |
|---|---|---|---|---|---|
| Playground | `playground` | `playground` | `Playground · playground` | question | classify result; `cache` on cache hit |
| `POST /api/v1/query` | `rest` | `query` | `REST · query` | question | from `RAGResponse.query_type` |
| REST async complete/fail | `rest` | `query_async` | `REST · query_async` | question | `job.classification.query_type` when present; else from result |
| `POST /v1/chat/completions` | `openai` | `chat.completions` | `OpenAI · chat.completions` | last user message | from `RAGResponse.query_type` |
| MCP `tool_ask` | `mcp` | `ask` | `MCP · ask` | question | from `RAGResponse.query_type` |
| MCP `tool_summarize_topic` | `mcp` | `summarize_topic` | `MCP · summarize_topic` | topic | from `RAGResponse.query_type` |
| MCP `tool_compare` | `mcp` | `compare` | `MCP · compare` | `{item_a} vs {item_b}` | from `RAGResponse.query_type` |
| MCP `tool_query_database` | `mcp` | `query_database` | `MCP · query_database` | question | `structured` on SQL path; agent `query_type` if it falls back to `ask` |
| MCP `tool_summarize_documents` | `mcp` | `summarize_documents` | `MCP · summarize_documents` | category, or empty for all visible | empty |
| MCP `tool_search_documents` | `mcp` | `search_documents` | `MCP · search_documents` | search query | empty |
| MCP `tool_lookup_document` | `mcp` | `lookup_document` | `MCP · lookup_document` | doc id / filename | empty |
| MCP `tool_list_sources` | `mcp` | `list_sources` | `MCP · list_sources` | empty | empty |
| MCP `tool_list_documents` | `mcp` | `list_documents` | `MCP · list_documents` | category filter, or empty | empty |
| MCP `tool_search_meetings` | `mcp` | `search_meetings` | `MCP · search_meetings` | topic / speaker / type (joined, skip blanks) | empty |
| MCP `tool_search_knowledge_graph` | `mcp` | `search_knowledge_graph` | `MCP · search_knowledge_graph` | graph query | empty |

**Not logged:** `tool_get_result`, MCP resources.

Playground is recorded **once** when the job reaches a terminal state
(`complete` or `error`) inside `playground_start`'s background task, and
again independently on the legacy `POST /api/playground/query` handler.
Do not record on status polls or the SSE stream endpoint.

### Identity

| Surface | Username | Groups |
|---|---|---|
| REST `/query` and async | JWT `user.username` | `user.groups` |
| OpenAI-compat | JWT username when present, else empty | groups actually used (`["ALL"]` when API-key-only) |
| MCP | `MCPContext.username` | `MCPContext.groups` |
| Playground | selected `play_user` persona | resolved persona groups |

### Async duration

Start the timer when the worker begins processing, not at enqueue.
`duration_seconds` is generation time, not queue wait.

### MCP error dicts

Some tools return `{error: ...}` instead of raising (`query_database`,
`lookup_document` on miss, `get_result` — the last is not logged).
If the returned value is a dict with a non-empty `error` key, set
`status=error` and store that message (200-char cap).

## UI

Block title: **Recent queries**.

| Column | Render |
|---|---|
| When | `MM-DD HH:MM` (same as Settings metrics) |
| Type | `{Surface} · {tool}` using the “Type shown” column above |
| Strategy | stored strategy, or — |
| User | username, or — if empty |
| Groups | comma-separated; more than 3 → first three names + `+N` (5 groups → `finance, executives, engineering, +2`) |
| Question | first 80 chars + ellipsis; — if blank |
| Time | `{duration:.1f}s` |
| Status | `ok` / `error` / `cache` (`ok` + `cache_hit` → `cache`) |

Empty state: *No queries recorded yet.*

Read failure: stat cards still render; the section shows
*Unable to load recent queries.*

In-flight work is not listed.

## Error handling

- Insert failures: log at warning, drop the row, never raise to the caller.
- Uncaught exceptions in a span: row is still written (`status=error`).
- No stack traces in `error`.
- Dashboard read is isolated from the five stat-card queries so an activity
  failure cannot blank the counts.

## Testing

- Helper: several inserts; `list_recent_query_activity(10)` is newest-first
  and capped at 10.
- Helper: exception inside the span records `status=error` and a positive
  duration; the exception still propagates.
- Helper: `ok` + `cache_hit` is what the template will show as `cache`.
- Dashboard: no rows → HTTP 200 and empty-state copy.
- Dashboard: one fixture row → page includes `MCP · ask`, username,
  truncated question, and status.
- Dashboard: list helper raises → stat cards still present, error copy shown.
- One route-level test: `POST /api/v1/query` with `agent_query` mocked
  writes an activity row (username, groups, duration, `source=rest`,
  `tool=query`). Other surfaces use the same helper; do not run a full
  agent per surface.

## Out of scope (explicit)

- Wiring `AuditLogger` / changing `/admin/audit`.
- Extending `QueryMetrics` or the Settings metrics dashboard.
- Auto-refresh, pagination, answer preview.
- MCP `get_result` and resources.
