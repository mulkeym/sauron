# Playground Answer Streaming — Design

**Date:** 2026-05-27
**Status:** Approved (pending spec review)

## Problem

In the admin playground, the synthesized answer ("Generate Answer" step) appears
all at once after a multi-second pause, rather than streaming token-by-token.

The streaming infrastructure already exists but is **disconnected**:

- An SSE endpoint `GET /admin/api/playground/stream/{query_id}`
  (`src/admin/routes.py`) already calls `generate_stream()` and yields tokens.
- A frontend `EventSource` consumer (`src/admin/templates/playground.html`)
  already renders tokens live into `#stream-output` with a cursor and
  incremental markdown, and merges the streamed answer with trace + citations on
  completion.

It never activates because:

1. The frontend only opens the `EventSource` when `status.step === 'streaming'`,
   and **the backend never sets `step = "streaming"`**.
2. The background runner builds the full agent graph **including** the
   non-streaming `synthesize` node (`graph.py` wires `merge -> synthesize ->
   END`), so the complete answer is produced in-graph and dropped in at once when
   `step` becomes `"complete"`. The comment at `routes.py` ("Build graph WITHOUT
   synthesize — we'll stream that separately") describes an intent the code does
   not implement.

Additionally, the dormant streaming path builds its **own** synthesis context
(`routes.py`, the `stream_context` assembly) which does NOT include the
schema-reference + executed-SQL context block added on 2026-05-27 for structured
answers. Wiring streaming naively would regress structured answers on exactly
that fix.

## Goal

Make the playground stream the synthesized answer token-by-token into the
"Generate Answer" section, with the streamed answer **identical** to what the
non-streaming path produces (so it inherits the SQL/schema-reference context
fix), and the final result card still showing the full trace and citations.

Scope: **playground only**. The production/MCP answer path (`run_agent`,
`run_agent_with_trace`) is unchanged.

## Approach

Reuse the synthesizer's context/citation logic from both the non-streaming path
and the playground stream (the "clean" approach), rather than maintaining a
divergent second context builder.

## Components

### 1. Synthesizer — extract two pure helpers (`src/agent/synthesizer.py`)

Refactor `synthesize_answer` into reusable pieces (no behavior change for the
production path):

- `build_synthesis_context(state) -> str` — assembles the context string:
  filtered/sorted chunks (existing map-reduce / KG / table_row logic) plus the
  `[Database query results]` block (Table & column reference + Executed SQL +
  Result rows) introduced today. This is the **single source of truth** for
  synthesis context.
- `build_citations(state) -> list[Citation]` — dedup chunk citations by document
  plus SQL-source-document citations. Independent of the answer text.
- `synthesize_answer(state)` becomes: empty-context check (no chunks and no
  `sql_results` -> the "could not find any relevant information" answer) ->
  `build_synthesis_context` -> `generate(...)` -> `_strip_reasoning_artifacts`
  -> `build_citations` -> `{"answer", "citations"}`.

`SYSTEM_PROMPT` and `USER_PROMPT_TEMPLATE` remain the public constants the SSE
endpoint already imports.

### 2. Graph — opt-out flag (`src/agent/graph.py`)

`create_agent_graph(..., include_synthesize: bool = True)`. When `False`, the
graph omits the `synthesize` node and finishes at `merge` (`merge -> END`).
Default `True` keeps `run_agent` and `run_agent_with_trace` byte-identical.

### 3. Playground runner — actually stream (`src/admin/routes.py`, `run_query`)

- Build the graph with `include_synthesize=False`.
- After the graph reaches `merge` (retrieve + enrich + merge done):
  - Compute context via `build_synthesis_context(final_state)`.
  - **Empty context** (no chunks and no sql_results): set the "no info" answer
    directly, skip streaming, proceed to completion.
  - Otherwise: set `step = "streaming"`, `stream_ready = True`, and
    `stream_context = {"context": <built context>, "question": question}`,
    replacing the hand-built context block.
- **Wait** for the SSE endpoint to populate `streamed_answer` (poll
  `_playground_jobs[query_id]` with a timeout). On timeout or stream error, fall
  back to a non-streamed `generate(...)` using the same context so the job always
  completes.
- Build citations via `build_citations(final_state)`, assemble `result_html`
  with the (streamed or fallback) answer + trace + citations, cache the result,
  log metrics, set `step = "complete"`.

### 4. SSE endpoint (`/stream`) — unchanged mechanism

Already waits for `stream_ready`, calls `generate_stream(SYSTEM_PROMPT,
USER_PROMPT_TEMPLATE.format(context, question))`, yields `{token}` events, then
`{done, answer}`, and stores `streamed_answer`. It now simply receives the
shared-builder context. Keep its existing error event.

### 5. Frontend (`src/admin/templates/playground.html`) — expected no change

Already opens `EventSource` on `step==='streaming'`, renders tokens live, and on
`step==='complete'` merges the streamed answer with the trace + citations card.
Verify in the browser; adjust only if a gap surfaces.

## Data Flow

```
POST /start
  -> background run_query:
       graph.astream (classify -> retrieve||enrich -> merge)   [no answer yet]
       -> step = "streaming", stream_context = build_synthesis_context(state)
  -> frontend status poll sees step=="streaming"
  -> frontend opens EventSource GET /stream
       -> generate_stream tokens -> #stream-output (live)
       -> on done: store streamed_answer
  -> run_query wait-loop sees streamed_answer
       -> build_citations(state), result_html, cache, metrics
       -> step = "complete"
  -> frontend status poll sees step=="complete"
       -> swap in final card (trace + streamed answer + citations)
```

## Concurrency / Failure Handling

- `run_query` and the SSE handler coordinate via `_playground_jobs[query_id]`.
  Ordering: `run_query` sets `stream_ready`/`step=streaming` before the SSE
  handler (triggered by the frontend) generates and sets `streamed_answer`;
  `run_query` then waits for `streamed_answer`.
- `run_query`'s wait-loop has a timeout and a non-streamed `generate` fallback,
  so a closed browser tab / never-opened EventSource / stream error cannot hang
  the job.
- Empty-context queries never enter streaming (handled before `step=streaming`).

## Testing

- Unit (`tests/test_agent/test_synthesizer.py`):
  - `build_synthesis_context` includes the Table & column reference + Executed
    SQL block when a `structured_trace` is present (extends today's test).
  - `build_citations` returns the same citations the current `synthesize_answer`
    produces (chunk dedup + SQL-source citation cases).
  - `synthesize_answer` behavior unchanged — existing tests stay green.
- Unit (`tests/test_agent/test_graph.py`):
  - `create_agent_graph(include_synthesize=False)` produces a graph that finishes
    at `merge` and yields no `answer` (no `synthesize` node executed).
  - Default graph still includes `synthesize` (regression guard).
- Manual / browser (Playwright or by hand) on the deployed stack:
  - Re-run "What are the GS salary rates in Tampa?"; confirm tokens type out live
    in the Generate Answer section and the final card shows trace + citations.

## Non-Goals

- Streaming in the production REST/MCP answer path.
- Changing the answer content, prompts, or retrieval behavior.
- Frontend redesign (reuse the existing stream-output rendering).
