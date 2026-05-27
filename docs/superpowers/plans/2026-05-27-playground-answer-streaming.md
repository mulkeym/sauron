# Playground Answer Streaming Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the admin playground stream the synthesized answer token-by-token into the "Generate Answer" section, with the streamed answer identical to the non-streaming path.

**Architecture:** Extract the synthesizer's context/citation building into two reusable, `state`-only helpers. Add an `include_synthesize=False` flag to the agent graph so the playground runs retrieval→merge without generating the answer in-graph. The playground runner sets `step="streaming"` and a shared-builder context; the existing SSE endpoint + frontend `EventSource` (already built) then stream the answer; the runner waits for the streamed answer (with a non-streamed fallback) before finalizing citations and the result card.

**Tech Stack:** Python, FastAPI (SSE `StreamingResponse`), LangGraph, vanilla JS `EventSource`, pytest.

---

## File Structure

- `src/agent/synthesizer.py` — add `build_synthesis_context(state)` and `build_citations(state)`; `synthesize_answer` composes them. Single source of truth for synthesis context + citations.
- `src/agent/graph.py` — `create_agent_graph(..., include_synthesize=True)`; when `False`, finish at `merge`.
- `src/admin/routes.py` — `run_query` (the `/api/playground/start` background task): build graph with `include_synthesize=False`, set `step="streaming"` + shared context after merge, wait for `streamed_answer` (fallback to non-streamed `generate`), finalize citations via `build_citations`.
- `src/admin/templates/playground.html` — expected no change; verify in browser.
- Tests: `tests/test_agent/test_synthesizer.py`, `tests/test_agent/test_graph.py`.

---

## Task 1: Extract `build_synthesis_context` and `build_citations` from `synthesize_answer`

**Files:**
- Modify: `src/agent/synthesizer.py` (current `synthesize_answer` spans lines 84-255)
- Test: `tests/test_agent/test_synthesizer.py`

This is a behavior-preserving refactor. The existing context logic (current lines 95-158) becomes `build_synthesis_context`; the existing citation logic (current lines 169-254) becomes `build_citations`. `synthesize_answer` then calls both.

- [ ] **Step 1: Write/extend failing tests**

Add to `tests/test_agent/test_synthesizer.py`:

```python
def test_build_synthesis_context_includes_sql_block():
    from src.agent.synthesizer import build_synthesis_context
    state = AgentState(
        question="GS rates in Tampa", user_groups=["finance"],
        query_type=QueryType.ANALYTICAL, retrieved_chunks=[],
        sql_results=[{"annual1": 23440.0}],
        structured_trace={
            "status": "ran", "row_count": 15,
            "sql": "SELECT * FROM all_gs WHERE locname = 'RUS'",
            "schema_context": "Table all_gs — GS pay\nValue meanings:\n  - locname: RUS = Rest of U.S.",
        },
    )
    ctx = build_synthesis_context(state)
    assert "WHERE locname = 'RUS'" in ctx
    assert "RUS = Rest of U.S." in ctx
    assert "23440" in ctx


def test_build_citations_dedupes_chunks_by_document():
    from src.agent.synthesizer import build_citations
    state = AgentState(
        question="policy?", user_groups=["finance"], query_type=QueryType.LOOKUP,
        retrieved_chunks=[_make_chunk("a", doc_id="d1", score=0.8),
                          _make_chunk("b", doc_id="d1", score=0.95)],
        sql_results=[],
    )
    cits = build_citations(state)
    assert len(cits) == 1                      # one per document
    assert cits[0].doc_id == "d1"
    assert cits[0].relevance == 0.95           # best score kept
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_agent/test_synthesizer.py::test_build_synthesis_context_includes_sql_block tests/test_agent/test_synthesizer.py::test_build_citations_dedupes_chunks_by_document -q`
Expected: FAIL with `ImportError: cannot import name 'build_synthesis_context'` (and `build_citations`).

- [ ] **Step 3: Add `build_synthesis_context`**

Insert this function directly above `def synthesize_answer` in `src/agent/synthesizer.py`. Its body is the current context-building logic (lines 95-158) verbatim, returning `context`:

```python
def build_synthesis_context(state: AgentState) -> str:
    """Assemble the LLM synthesis context from retrieved chunks + structured SQL
    results. Single source of truth shared by synthesize_answer (non-streaming)
    and the playground streaming path. Returns "" when there is nothing to say."""
    chunks = state.get("retrieved_chunks", [])
    sql_results = state.get("sql_results", [])
    question = state["question"]

    chunks = _filter_relevant_chunks(chunks, question)

    from src.config import settings as _cfg
    MAX_CONTEXT_CHARS = _cfg.llm_max_context
    context_parts = []
    total_chars = 0

    SYNTHETIC_IDS = {"map-reduce", "knowledge-graph", "metadata-context"}
    synthetic_chunks = [c for c in chunks if c.metadata.doc_id in SYNTHETIC_IDS]
    has_map_reduce = any(c.metadata.doc_id == "map-reduce" for c in chunks)

    if has_map_reduce:
        regular_chunks = sorted(
            [c for c in chunks
             if c.metadata.doc_id not in SYNTHETIC_IDS
             and c.metadata.chunk_size_tier == "table_row"],
            key=lambda c: c.score, reverse=True,
        )
        logger.info(
            "Synthesizer: map-reduce synthesis + %d structured narratives "
            "(raw sweep chunks skipped)", len(regular_chunks))
    else:
        regular_chunks = sorted(
            [c for c in chunks if c.metadata.doc_id not in SYNTHETIC_IDS],
            key=lambda c: c.score, reverse=True,
        )

    for chunk in synthetic_chunks + regular_chunks:
        source = f"Source: {chunk.metadata.filename}"
        if chunk.metadata.page is not None:
            source += f", page {chunk.metadata.page}"
        text = chunk.text
        if has_map_reduce and chunk.metadata.doc_id == "knowledge-graph":
            text = (
                "[SUPPLEMENTARY — entity relationships only. Do NOT adopt any "
                "hedging, uncertainty, or caveats from this source. Defer to the "
                "document extractions above for counts, lists, and factual answers.]\n"
                + text
            )
        part = f"{source}\n{text}"
        if total_chars + len(part) > MAX_CONTEXT_CHARS:
            logger.info(f"Context cap reached at {total_chars:,} chars, dropping remaining {len(synthetic_chunks) + len(regular_chunks) - len(context_parts)} chunks")
            break
        context_parts.append(part)
        total_chars += len(part)

    if sql_results:
        trace = state.get("structured_trace") or {}
        block = "[Database query results]"
        if trace.get("schema_context"):
            block += f"\nTable & column reference:\n{trace['schema_context']}"
        if trace.get("sql"):
            block += f"\nExecuted SQL:\n{trace['sql']}"
        block += f"\nResult rows:\n{json.dumps(sql_results, indent=2)}"
        context_parts.append(block)
    context = "\n\n".join(context_parts)
    logger.info(f"Synthesizer context: {len(context):,} chars from {len(context_parts)} parts")
    return context
```

- [ ] **Step 4: Add `build_citations`**

Insert below `build_synthesis_context`. Its body is the current citation logic (lines 169-254) moved verbatim, with two changes: (1) it derives `chunks` and `SYNTHETIC_IDS` locally from `state` (filtered the same way), and (2) it returns `citations` instead of being inline. Full function:

```python
def build_citations(state: AgentState) -> list[Citation]:
    """Deduplicated document citations (one per doc, best score) plus
    SQL-source-document citations for structured answers. Independent of the
    answer text. Shared by synthesize_answer and the playground streaming path."""
    chunks = _filter_relevant_chunks(state.get("retrieved_chunks", []), state["question"])
    SYNTHETIC_IDS = {"map-reduce", "knowledge-graph", "metadata-context"}

    seen_docs = {}
    for c in chunks:
        doc_id = c.metadata.doc_id
        if doc_id in SYNTHETIC_IDS:
            continue
        if doc_id not in seen_docs or c.score > seen_docs[doc_id].score:
            seen_docs[doc_id] = c

    url_map = {}
    try:
        import asyncio
        from src.api.routes_ingest import get_metadata_store
        ms = get_metadata_store()

        async def _fetch_urls():
            urls = {}
            for doc_id in seen_docs:
                doc_rec = await ms.get_document(doc_id)
                if doc_rec and getattr(doc_rec, 'source_url', ''):
                    urls[doc_id] = doc_rec.source_url
            return urls

        try:
            url_map = asyncio.run(_fetch_urls())
        except RuntimeError:
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as pool:
                url_map = pool.submit(asyncio.run, _fetch_urls()).result()
    except Exception as e:
        logger.debug(f"Source URL lookup skipped: {e}")

    citations = [
        Citation(
            doc_id=c.metadata.doc_id,
            filename=c.metadata.filename,
            doc_type=c.metadata.doc_type,
            chunk_index=c.metadata.chunk_index,
            page=c.metadata.page,
            snippet=c.text[:200],
            relevance=c.score,
            source_url=url_map.get(c.metadata.doc_id, ""),
        )
        for c in seen_docs.values()
    ]

    trace = state.get("structured_trace") or {}
    if trace.get("status") == "ran" and trace.get("row_count", 0) > 0 and trace.get("sql"):
        try:
            import asyncio
            from src.api.routes_ingest import get_metadata_store
            from src.ingestion.tabular_store import referenced_source_docs
            _ms = get_metadata_store()

            async def _fetch_sql_docs():
                docs = await _ms.list_documents()
                src_ids = referenced_source_docs(trace["sql"], [d.doc_id for d in docs])
                by_id = {d.doc_id: d for d in docs}
                return [by_id[i] for i in src_ids if i in by_id]

            try:
                sql_docs = asyncio.run(_fetch_sql_docs())
            except RuntimeError:
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as pool:
                    sql_docs = pool.submit(asyncio.run, _fetch_sql_docs()).result()

            existing = {c.doc_id for c in citations}
            for rec in sql_docs:
                if rec.doc_id in existing:
                    continue
                citations.append(Citation(
                    doc_id=rec.doc_id,
                    filename=rec.filename,
                    doc_type=getattr(rec, "doc_type", "") or "",
                    chunk_index=0,
                    page=None,
                    snippet=f"Structured query returned {trace['row_count']} rows from this table.",
                    relevance=1.0,
                    source_url=getattr(rec, "source_url", "") or "",
                ))
                existing.add(rec.doc_id)
        except Exception as e:
            logger.debug(f"SQL-source citations skipped: {e}")

    return citations
```

- [ ] **Step 5: Rewrite `synthesize_answer` to compose the helpers**

Replace the entire current `synthesize_answer` body (lines 84-255) with:

```python
def synthesize_answer(state: AgentState) -> dict:
    if not state.get("retrieved_chunks") and not state.get("sql_results"):
        return {
            "answer": "I could not find any relevant information in the documents you have access to.",
            "citations": [],
        }
    from src.config import settings as _cfg
    context = build_synthesis_context(state)
    answer = generate(
        system_prompt=SYSTEM_PROMPT,
        user_prompt=USER_PROMPT_TEMPLATE.format(context=context, question=state["question"]),
        max_tokens=_cfg.llm_max_output_tokens,
    )
    answer = _strip_reasoning_artifacts(answer)
    citations = build_citations(state)
    return {"answer": answer, "citations": citations}
```

- [ ] **Step 6: Run the full synthesizer suite**

Run: `python -m pytest tests/test_agent/test_synthesizer.py -q`
Expected: PASS (the two new tests + all pre-existing tests, proving the refactor preserved behavior).

- [ ] **Step 7: Commit**

```bash
git add src/agent/synthesizer.py tests/test_agent/test_synthesizer.py
git commit -m "refactor: extract build_synthesis_context + build_citations from synthesize_answer"
```

---

## Task 2: Add `include_synthesize` flag to the agent graph

**Files:**
- Modify: `src/agent/graph.py` (function `create_agent_graph`, signature at line 20; node/edge wiring at lines 222-234)
- Test: `tests/test_agent/test_graph.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_agent/test_graph.py`:

```python
def test_create_agent_graph_without_synthesize_ends_at_merge():
    from unittest.mock import MagicMock
    from src.agent.graph import create_agent_graph
    g = create_agent_graph(vector_store=MagicMock(), schema_registry=MagicMock(),
                           metadata_store=MagicMock(), include_synthesize=False)
    nodes = set(g.get_graph().nodes.keys())
    assert "merge" in nodes
    assert "synthesize" not in nodes


def test_create_agent_graph_includes_synthesize_by_default():
    from unittest.mock import MagicMock
    from src.agent.graph import create_agent_graph
    g = create_agent_graph(vector_store=MagicMock(), schema_registry=MagicMock(),
                           metadata_store=MagicMock())
    assert "synthesize" in set(g.get_graph().nodes.keys())
```

- [ ] **Step 2: Run tests to verify failure**

Run: `python -m pytest tests/test_agent/test_graph.py::test_create_agent_graph_without_synthesize_ends_at_merge -q`
Expected: FAIL with `TypeError: create_agent_graph() got an unexpected keyword argument 'include_synthesize'`.

- [ ] **Step 3: Add the parameter and conditional wiring**

Change the signature (line 20) from:

```python
def create_agent_graph(vector_store: VectorStore, schema_registry: SchemaRegistry, metadata_store: MetadataStore | None = None):
```

to:

```python
def create_agent_graph(vector_store: VectorStore, schema_registry: SchemaRegistry, metadata_store: MetadataStore | None = None, include_synthesize: bool = True):
```

Then replace the tail wiring (current lines 222-234) — the block that adds `merge`/`synthesize` nodes and their edges through `return graph.compile()` — with:

```python
    graph.add_node("merge", merge_results)

    graph.set_entry_point("classify")

    graph.add_edge("classify", "retrieve")
    graph.add_edge("classify", "enrich")
    graph.add_edge("retrieve", "merge")
    graph.add_edge("enrich", "merge")
    if include_synthesize:
        graph.add_node("synthesize", synthesize_answer)
        graph.add_edge("merge", "synthesize")
        graph.add_edge("synthesize", END)
    else:
        graph.add_edge("merge", END)

    return graph.compile()
```

(Note: this removes the unconditional `graph.add_node("synthesize", ...)` at line 223 — it now lives inside the `if`.)

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_agent/test_graph.py -q`
Expected: PASS (both new tests + existing graph tests unaffected).

- [ ] **Step 5: Commit**

```bash
git add src/agent/graph.py tests/test_agent/test_graph.py
git commit -m "feat: add include_synthesize flag to create_agent_graph (omit answer node)"
```

---

## Task 3: Wire live streaming into the playground runner

**Files:**
- Modify: `src/admin/routes.py` — `run_query` background task inside `playground_start` (graph build at line 931; in-loop synthesize block at lines 1008-1027; answer determination at lines 1029-1036; citation source at line 1137).

No unit test (background task + SSE concurrency); verified via Task 4 (browser). Each edit below is mechanical and localized.

- [ ] **Step 1: Build the graph without synthesize**

At line 931, change:

```python
            graph = create_agent_graph(vector_store=vs, schema_registry=sr, metadata_store=ms)
```

to:

```python
            graph = create_agent_graph(vector_store=vs, schema_registry=sr, metadata_store=ms, include_synthesize=False)
```

- [ ] **Step 2: Replace the dead in-loop synthesize block with nothing**

Delete the in-loop block at lines 1008-1027 (the `if node_name == "synthesize" and not _playground_jobs[query_id].get("stream_ready"):` block). It keyed on a node that no longer runs; streaming is now set up after the loop (Step 3). Leave the rest of the loop intact.

- [ ] **Step 3: After the astream loop, set up streaming + wait for the answer**

The astream loop ends at the line `final_state.update(...)`. Immediately AFTER the loop (before the existing `total_time = ...` / `answer = final_state.get("answer", "No answer")` at lines 1029-1036), insert:

```python
            # Answer is produced by streaming (graph was built without synthesize).
            has_context = bool(final_state.get("retrieved_chunks")) or bool(final_state.get("sql_results"))
            answer = "I could not find any relevant information in the documents you have access to."
            synth_start = time.time()
            if has_context:
                from src.agent.synthesizer import build_synthesis_context
                _playground_jobs[query_id]["stream_context"] = {
                    "context": build_synthesis_context(final_state),
                    "question": question,
                }
                _playground_jobs[query_id]["stream_ready"] = True
                _playground_jobs[query_id]["step"] = "streaming"

                # The SSE endpoint (opened by the frontend) streams the answer and
                # stores it back as streamed_answer. Wait for it; fall back to a
                # non-streamed generate so a closed tab can never hang the job.
                for _ in range(1500):  # ~5 min at 0.2s
                    if _playground_jobs[query_id].get("streamed_answer") is not None:
                        break
                    await asyncio.sleep(0.2)
                streamed = _playground_jobs[query_id].get("streamed_answer")
                if streamed is not None:
                    answer = streamed
                else:
                    from src.agent.synthesizer import (
                        SYSTEM_PROMPT, USER_PROMPT_TEMPLATE, _strip_reasoning_artifacts)
                    from src.generation.llm_client import generate as _gen
                    ctx = _playground_jobs[query_id]["stream_context"]["context"]
                    answer = _strip_reasoning_artifacts(_gen(
                        system_prompt=SYSTEM_PROMPT,
                        user_prompt=USER_PROMPT_TEMPLATE.format(context=ctx, question=question),
                        max_tokens=4096))

            # Synthesize wasn't a graph node; build citations + a trace step here.
            from src.agent.synthesizer import build_citations
            final_state["citations"] = build_citations(final_state) if has_context else []
            synth_elapsed = round(time.time() - synth_start, 2)
            steps_data.append({"step": "synthesize", "time": synth_elapsed,
                               "output": {"answer": answer, "citations": final_state["citations"]}})
            _playground_jobs[query_id]["completed_steps"].append(
                {"step": "synthesize", "time": synth_elapsed,
                 "detail": f"<strong>Answer length:</strong> {len(answer)} chars<br><strong>Citations:</strong> {len(final_state['citations'])}"})
```

Then DELETE the now-redundant lines 1029-1036:

```python
            total_time = sum(s["time"] for s in steps_data)
            answer = final_state.get("answer", "No answer")
            chunks = final_state.get("retrieved_chunks", [])
            query_type = str(final_state.get("query_type", "lookup"))

            # Use streamed answer if available (from SSE endpoint)
            if _playground_jobs[query_id].get("streamed_answer"):
                answer = _playground_jobs[query_id]["streamed_answer"]
```

and replace with (keep `total_time`, `chunks`, `query_type`; drop the `answer` reassignments since `answer` is now set above):

```python
            total_time = sum(s["time"] for s in steps_data)
            chunks = final_state.get("retrieved_chunks", [])
            query_type = str(final_state.get("query_type", "lookup"))
```

`asyncio` is already imported in `run_query` (used by the existing loop); `time` is already imported at module level (used by `step_start = time.time()`). The downstream `result_html`, `cache_store`, metrics, and `final_state.get("citations", [])` (line 1137) all consume `answer` / `final_state["citations"]` unchanged.

- [ ] **Step 4: Deploy and smoke-test the endpoint (no hang)**

Rebuild + restart the api container (per the project's docker deploy), then confirm a query reaches completion via the API (no browser yet):

```bash
docker compose build api && docker compose up -d api
# wait for healthy, then drive the job lifecycle:
QID=$(curl -s -X POST localhost:8080/admin/api/playground/start \
  --data-urlencode "question=What are the GS salary rates in Tampa?" \
  --data-urlencode "play_user=finance" --data-urlencode "mode=full" | python -c "import sys,json;print(json.load(sys.stdin)['query_id'])")
# open the stream (consumes tokens) in the background, then poll status to 'complete':
curl -sN localhost:8080/admin/api/playground/stream/$QID >/dev/null &
for i in $(seq 1 60); do s=$(curl -s localhost:8080/admin/api/playground/status/$QID | python -c "import sys,json;print(json.load(sys.stdin)['step'])"); echo $s; [ "$s" = complete ] && break; sleep 2; done
```

Expected: prints `streaming` then `complete` (proves the runner sets `step=streaming`, the SSE stream feeds `streamed_answer`, and the runner finalizes without hanging).

- [ ] **Step 5: Commit**

```bash
git add src/admin/routes.py
git commit -m "feat: stream the playground answer via SSE (run graph to merge, stream synthesize)"
```

---

## Task 4: Verify live streaming in the browser

**Files:** none (verification). Possible touch: `src/admin/templates/playground.html` only if a gap surfaces.

- [ ] **Step 1: Drive the playground with Playwright**

Navigate to `http://localhost:8080` admin playground, submit "What are the GS salary rates in Tampa?", and observe the "Generate Answer"/result area.

Expected:
- The `#stream-output` area appears with a blinking cursor and text that grows incrementally (token streaming), not all-at-once.
- On completion, the final card shows the trace (including a "Generate Answer" step) and the citations.

- [ ] **Step 2: If (and only if) the answer still appears all-at-once**

Inspect `playground.html` around the status poll: confirm the `if (status.step === 'streaming' ...)` branch fires and opens `EventSource('/admin/api/playground/stream/${queryId}')`. Fix any mismatch (e.g. the runner must set `step` to exactly the string `"streaming"`). Re-deploy and re-verify. Commit any template change:

```bash
git add src/admin/templates/playground.html
git commit -m "fix: trigger playground answer streaming in the frontend"
```

- [ ] **Step 3: Confirm no regression in the non-streaming path**

Run: `python -m pytest tests/test_agent/ -q`
Expected: the pre-existing environmental failures only (test_graph run-agent, lookup, sweep, cross_reference, as noted on master) — no NEW failures in `test_synthesizer.py` or the streaming-related areas.

---

## Self-Review

**Spec coverage:**
- Synthesizer two helpers → Task 1. ✓
- Graph `include_synthesize` flag → Task 2. ✓
- Runner: graph-without-synthesize, step=streaming, shared context, wait + fallback, citations, complete → Task 3. ✓
- SSE endpoint unchanged (receives shared context) → no task needed; it already imports `SYSTEM_PROMPT`/`USER_PROMPT_TEMPLATE` and reads `stream_context["context"]`. ✓
- Frontend expected no change → Task 4 verifies; Step 2 covers the contingency. ✓
- Concurrency/failure (runner sets stream_ready first; timeout + non-streamed fallback) → Task 3 Step 3. ✓
- Empty-context handled before streaming → Task 3 Step 3 (`has_context`). ✓
- Testing (unit context/citations, graph flag, browser) → Tasks 1, 2, 4. ✓

**Placeholder scan:** No TBD/TODO; all code blocks are complete; the citation/context bodies are reproduced in full.

**Type consistency:** `build_synthesis_context(state) -> str` and `build_citations(state) -> list[Citation]` are used with those exact names/signatures in Task 1 (synthesize_answer) and Task 3 (runner). `include_synthesize` keyword matches between Task 2 (definition) and Task 3 Step 1 (call). `_playground_jobs[query_id]` keys (`stream_context`, `stream_ready`, `streamed_answer`, `step`, `completed_steps`) match the existing SSE endpoint and frontend.
