# LLM Session Headers for Switchyard

**Date:** 2026-08-15
**Status:** Implemented (2026-08-16)

## Problem

Sauron answers one user question with many LLM HTTP calls (cache judge,
classify, table routing, MAP extracts, synthesizer, LightRAG query enrich).
None of those calls send a session header. Switchyard therefore treats each
call as its own session: live view, session stats, overflow memory, and
(if enabled) session affinity cannot group them.

OpenWebUI and other apps already emit a session/chat id on API and MCP
requests. Sauron drops it before talking to Switchyard.

## Goals

- Reuse an inbound session id when the caller sent one (API or MCP).
- Otherwise mint one UUID that covers every LLM call for that question.
- Attach a stable user identity when Sauron already has one
  (`x-switchyard-agent-id`).
- Cover playground, API (`/query`, async, OpenAI-compat), and MCP ask paths,
  plus LightRAG query enrich on those paths.
- Leave ingest / KG extract / figure-vision unbound.

## Non-goals

- Changing Switchyard routing or turning session affinity on/off.
- Playground UI changes.
- Forwarding email/name as extra headers.
- Persisting sessions across process restarts.
- Grouping ingest LLM traffic.

## Decisions

| Decision | Choice |
|---|---|
| Session scope | Inbound header if present, else one id per question |
| Call sites | Answer pipeline + LightRAG query enrich |
| Mechanism | `ContextVar` + bind at answer entry (approach A) |
| User detail | `x-switchyard-agent-id` from existing auth / persona |
| Affinity | Documented caveat; leave Switchyard `session_affinity` off for Sauron if each step should be judged independently |

## Header contract

### Inbound session (first non-empty wins)

1. `x-switchyard-session-id`
2. `x-session-id`
3. `session-id`
4. `x-openwebui-chat-id`

Missing/blank → mint a UUID for this question.

### Inbound / resolved agent id (first non-empty wins)

1. `x-switchyard-agent-id`
2. `x-openwebui-user-id`
3. Explicit caller identity: OpenWebUI JWT `sub` (MCP `agent_id`), else API
   username, else playground `play_user`
4. Omit if nothing is available

Do not forward email or display name.

### Outbound (every bound LLM HTTP call)

| Header | Value |
|---|---|
| `x-switchyard-session-id` | resolved / minted session id |
| `x-switchyard-request-id` | new UUID per HTTP call |
| `x-switchyard-agent-id` | resolved agent id, omitted when unknown |

Unbound calls (ingest, tests, one-off scripts) send none of these.

## Architecture

```
 playground Ask / API /query / MCP tool_ask
        │
        ▼
  resolve session + agent from request headers
  + already-authenticated identity
        │
        ▼
  llm_session(...) ContextVar bind
        │
        ├─ judged_cache_lookup  → generate()          ─┐
        ├─ classify / SQL / MAP → generate()           ├─ same session
        ├─ synthesizer          → generate[_stream]()  │
        └─ LightRAG aquery      → extra_headers        ─┘
```

`asyncio.to_thread` on Python 3.11 copies context, so threaded `generate()`
keeps the bind.

### Components

**`src/generation/llm_client.py`**

- `ContextVar`s for session id and agent id
- `resolve_llm_identity(headers=..., agent_id=..., session_id=...)`
- `llm_session(...)` context manager (`reset()` is fail-open if the token
  was created in another `copy_context()`)
- `outbound_llm_headers()` used by `_call_llm` and `generate_vision`
- `generate_stream(..., session_id=, agent_id=)` attaches Switchyard headers
  on that POST without a ContextVar bind (playground SSE; see below)

**`src/generation/rag_chain.py`**

- `agent_query` / `agent_query_streamed` accept optional `session_headers`,
  `agent_id`, `session_id` and bind `llm_session` around cache + graph.

**Entry points**

| Entry | Bind | Session | Agent |
|---|---|---|---|
| Playground `run_query` | own `llm_session` | inbound headers or mint | `play_user` |
| `POST /api/v1/query` | via `agent_query` | request headers or mint | JWT username |
| `POST /api/v1/query/async` | resolve at enqueue; worker passes stored ids | request headers or mint | JWT username |
| OpenAI-compat `/v1/chat/completions` | via `agent_query` | request headers or mint | JWT username when present |
| MCP `ask` / `summarize_topic` / `compare` | via `agent_query` | MCP HTTP headers or mint | JWT `sub` or username |

Async jobs re-bind the ids stored at submit time because the worker does not
inherit the submitter's context.

**`src/knowledge/graph_rag.py`**

- `_llm_func` forwards `outbound_llm_headers()` as OpenAI `extra_headers`.
- Ingest extract uses the same function but never binds, so no session
  headers.

## Error handling

- Header attach is fail-open: unset context → today's no-session behavior.
- Binding never changes retry/timeout semantics.
- Unknown inbound headers are ignored.

## Testing

- Inbound precedence (switchyard header beats `x-session-id`, etc.)
- Mint when no header
- Bound `_call_llm` / `generate_stream` / `generate_vision` send session +
  unique request-id + agent-id
- Unbound calls send none
- `to_thread(generate)` sees the bound session
- Two concurrent binds do not leak
- LightRAG `_llm_func` includes `extra_headers` when bound
- API / async enqueue / MCP helpers pass headers and identity through

## Playground SSE (implementation note)

Starlette iterates a sync `StreamingResponse` generator with one `next()` per
worker thread, each in a fresh `copy_context()`. A `with llm_session()` around
that generator set the ContextVar in the first copy and `reset()` in a later
copy, which raised `Token was created in a different Context` after the last
token (the UI replaced the answer with that error).

Playground streaming therefore passes `session_id` / `agent_id` into
`generate_stream()` as arguments. The answer-pipeline `llm_session` bind on
the async `run_query` task is unchanged.

`generate_stream` also skips usage chunks with `choices: []` (Switchyard /
OpenAI `include_usage` frames). Indexing `[0]` on those chunks used to raise
`list index out of range` at the end of a successful stream.

## Affinity note

If the Switchyard route has `session_affinity = true`, the first easy call
(classify) can pin later hard calls to the cheap model. Leave affinity **off**
on the Sauron route to judge each step independently. The session id still
groups live view, stats, and overflow memory.
