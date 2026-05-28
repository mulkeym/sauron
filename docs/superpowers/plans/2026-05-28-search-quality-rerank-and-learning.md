# Search Quality: Reranking + Dormant-Learning Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Activate three retrieval-quality capabilities the codebase has but doesn't fully use — final-stage CrossEncoder reranking (F3), relevance-feedback boosts across all strategies (F2), and Strategy Memory read-back into routing (F1).

**Architecture:** All changes are fail-open and flag-guarded. F3 mutates `chunk.score` in place inside the graph's `merge_results` node (consumers already sort by score, and the `retrieved_chunks` reducer blocks reordering via node return). F2 fetches existing `get_feedback_boosts()` per strategy and applies a shared helper before each strategy's score cutoff. F1 fixes `get_best_strategy()`'s metric and adds a confidence-gated soft override in an async classify node.

**Tech Stack:** Python, LangGraph, LanceDB, `sentence_transformers.CrossEncoder`, pytest, SQLAlchemy async.

**Spec:** `docs/superpowers/specs/2026-05-28-search-quality-rerank-and-learning-design.md`

**Build order:** Phase A (F3) → Phase B (F2) → Phase C (F1). Each phase is independently shippable.

**Baseline caveat:** `tests/test_agent/` carries ~8 pre-existing failures and `tests/test_ingestion/` has environmental failures (numpy/sklearn, OpenAI 401). Success = the same pre-existing failure set, not zero failures. Run new tests in isolation to confirm they pass.

---

## Phase A — F3: Final-N CrossEncoder Rerank

### Task 1: Add rerank settings

**Files:**
- Modify: `src/config.py:80-81` (after the Strategy memory block)

- [ ] **Step 1: Add the settings**

In `src/config.py`, after line 81 (`strategy_memory_enabled: bool = True`), add:

```python

    # Final-N reranking
    rerank_final_enabled: bool = True
    rerank_final_top_n: int = 50  # cap on chunks the final CrossEncoder pass scores/leads
    rerank_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
```

- [ ] **Step 2: Verify it imports**

Run: `python -c "from src.config import settings; print(settings.rerank_final_enabled, settings.rerank_final_top_n, settings.rerank_model)"`
Expected: `True 50 cross-encoder/ms-marco-MiniLM-L-6-v2`

- [ ] **Step 3: Commit**

```bash
git add src/config.py
git commit -m "feat: add final-N rerank settings"
```

---

### Task 2: Add `VectorStore.rerank_chunks`

**Files:**
- Modify: `src/retrieval/vector_store.py` (add a model accessor near `_get_cross_encoder` at line 37, and a `rerank_chunks` method after `hybrid_search_reranked` at line 204)
- Test: `tests/test_retrieval/test_vector_store.py`

This reranks the highest-scoring **non-synthetic** chunks so they lead, ordered by CrossEncoder relevance, with the F2 feedback boost re-added. It mutates `chunk.score` in place (consumers sort by score) and is fail-open at every external call. Synthetic chunks (`map-reduce`, `knowledge-graph`, `metadata-context`) are never reranked or demoted.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_retrieval/test_vector_store.py`:

```python
from unittest.mock import patch
from src.retrieval.models import RetrievedChunk, ChunkMetadata


def _chunk(doc_id, idx, score, text):
    return RetrievedChunk(
        text=text, score=score,
        metadata=ChunkMetadata(
            doc_id=doc_id, filename=f"{doc_id}.txt", doc_type="text",
            chunk_index=idx, start_char=0, acl_groups=["ALL"],
        ),
    )


class _FakeCE:
    """Fake CrossEncoder: score = 1.0 if 'match' in text else 0.0."""
    def predict(self, pairs):
        return [1.0 if "match" in text else 0.0 for _q, text in pairs]


def test_rerank_chunks_reorders_by_crossencoder(monkeypatch):
    from src.retrieval.vector_store import VectorStore
    vs = VectorStore.__new__(VectorStore)  # no DB init needed
    chunks = [
        _chunk("d1", 0, 0.9, "irrelevant text"),
        _chunk("d2", 1, 0.1, "this is a match"),
    ]
    with patch.object(VectorStore, "_get_cross_encoder_model", return_value=_FakeCE()):
        out = vs.rerank_chunks(chunks, "find the match", top_n=50, boosts=None)
    # d2 (CE match) must now outscore d1 despite lower original score
    by_id = {c.metadata.doc_id: c.score for c in out}
    assert by_id["d2"] > by_id["d1"]


def test_rerank_chunks_applies_feedback_boost(monkeypatch):
    from src.retrieval.vector_store import VectorStore
    vs = VectorStore.__new__(VectorStore)
    chunks = [_chunk("d1", 0, 0.5, "a match"), _chunk("d2", 1, 0.5, "another match")]
    with patch.object(VectorStore, "_get_cross_encoder_model", return_value=_FakeCE()):
        out = vs.rerank_chunks(chunks, "match", top_n=50, boosts={"d2": 0.5})
    by_id = {c.metadata.doc_id: c.score for c in out}
    assert by_id["d2"] > by_id["d1"]  # equal CE score, d2 wins on boost


def test_rerank_chunks_skips_synthetic(monkeypatch):
    from src.retrieval.vector_store import VectorStore
    vs = VectorStore.__new__(VectorStore)
    synth = _chunk("map-reduce", 0, 0.42, "extracted data")
    chunks = [synth, _chunk("d1", 0, 0.1, "a match")]
    with patch.object(VectorStore, "_get_cross_encoder_model", return_value=_FakeCE()):
        vs.rerank_chunks(chunks, "match", top_n=50, boosts=None)
    assert synth.score == 0.42  # synthetic chunk score untouched


def test_rerank_chunks_failopen_on_model_error(monkeypatch):
    from src.retrieval.vector_store import VectorStore
    vs = VectorStore.__new__(VectorStore)
    chunks = [_chunk("d1", 0, 0.9, "x"), _chunk("d2", 1, 0.1, "y")]
    with patch.object(VectorStore, "_get_cross_encoder_model", side_effect=RuntimeError("boom")):
        out = vs.rerank_chunks(chunks, "q", top_n=50, boosts=None)
    assert [c.score for c in out] == [0.9, 0.1]  # unchanged
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `pytest tests/test_retrieval/test_vector_store.py -k rerank_chunks -v`
Expected: FAIL — `AttributeError: ... has no attribute 'rerank_chunks'`

- [ ] **Step 3: Implement the model accessor**

In `src/retrieval/vector_store.py`, after the existing `_get_cross_encoder` classmethod (ends at line 43), add a sibling classmethod:

```python
    _cross_encoder_model = None  # class-level cache for the raw CrossEncoder

    @classmethod
    def _get_cross_encoder_model(cls):
        """Lazy-load and cache a raw sentence_transformers CrossEncoder for
        scoring explicit (query, text) pairs (the lancedb reranker only drives
        Lance query pipelines)."""
        if cls._cross_encoder_model is None:
            from sentence_transformers import CrossEncoder
            from src.config import settings
            cls._cross_encoder_model = CrossEncoder(settings.rerank_model)
        return cls._cross_encoder_model
```

Note: place the `_cross_encoder_model = None` class attribute next to the existing `_cross_encoder = None` at line 29 if you prefer; either location works since both are class-level.

- [ ] **Step 4: Implement `rerank_chunks`**

In `src/retrieval/vector_store.py`, after `hybrid_search_reranked` (after line 204), add:

```python
    def rerank_chunks(self, chunks, text_query, top_n, boosts=None):
        """Rerank the top_n highest-scoring non-synthetic chunks with a
        CrossEncoder so they lead (ordered by relevance), re-adding the
        doc-level feedback boost. Mutates chunk.score in place and returns
        the same list. Fail-open: any error leaves scores unchanged."""
        boosts = boosts or {}
        SYNTHETIC = {"map-reduce", "knowledge-graph", "metadata-context"}
        if not chunks or len(chunks) < 2:
            return chunks
        regular = [c for c in chunks if c.metadata.doc_id not in SYNTHETIC]
        if len(regular) < 2:
            return chunks
        try:
            model = self._get_cross_encoder_model()
        except Exception as e:
            logger.warning(f"Rerank model load failed, skipping rerank: {e}")
            return chunks

        regular.sort(key=lambda c: c.score, reverse=True)
        head = regular[:top_n]
        tail = regular[top_n:]
        try:
            raw = list(model.predict([(text_query, c.text) for c in head]))
        except Exception as e:
            logger.warning(f"Rerank predict failed, skipping rerank: {e}")
            return chunks

        lo, hi = min(raw), max(raw)
        span = (hi - lo) or 1.0
        tail_max = max((c.score for c in tail), default=0.0)
        base = tail_max + 0.01  # guarantee reranked head leads the tail
        for c, r in zip(head, raw):
            norm = (r - lo) / span  # [0, 1]
            c.score = base + norm + boosts.get(c.metadata.doc_id, 0.0)
        return chunks
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `pytest tests/test_retrieval/test_vector_store.py -k rerank_chunks -v`
Expected: 4 PASS

- [ ] **Step 6: Commit**

```bash
git add src/retrieval/vector_store.py tests/test_retrieval/test_vector_store.py
git commit -m "feat: VectorStore.rerank_chunks — final-N CrossEncoder rerank, fail-open"
```

---

### Task 3: Wire rerank into `merge_results` + add `feedback_boosts` state field

**Files:**
- Modify: `src/agent/state.py:43` (add field)
- Modify: `src/agent/graph.py:218-220` (`merge_results`)
- Test: `tests/test_agent/test_graph.py`

`merge_results` is currently a no-op. It runs in BOTH the API path (`include_synthesize=True`) and the playground path (`include_synthesize=False`), so it's the one chokepoint that covers every strategy. It reads `state["feedback_boosts"]` (default `{}`; populated in Phase B — until then it's simply empty and rerank runs boost-free).

- [ ] **Step 1: Add the state field**

In `src/agent/state.py`, after line 43 (`structured_trace: dict`), add:

```python
    feedback_boosts: dict  # {doc_id: boost} from relevance feedback, passed to final rerank
    strategy_memory: dict  # routing decision from Strategy Memory (F1 observability)
```

(Both fields use the default last-write-wins reducer — no `Annotated` needed.)

- [ ] **Step 2: Write the failing test**

Add to `tests/test_agent/test_graph.py`:

```python
def test_merge_results_reranks_when_enabled(monkeypatch):
    from src.agent.graph import create_agent_graph
    from src.retrieval.vector_store import VectorStore
    from src.config import settings

    calls = {}

    def fake_rerank(chunks, text_query, top_n, boosts=None):
        calls["args"] = (text_query, top_n, boosts)
        return chunks

    vs = VectorStore.__new__(VectorStore)
    monkeypatch.setattr(vs, "rerank_chunks", fake_rerank)
    monkeypatch.setattr(settings, "rerank_final_enabled", True)
    monkeypatch.setattr(settings, "rerank_final_top_n", 7)

    # Reach merge_results via the compiled graph's node function is awkward;
    # instead test the extracted helper directly (see Step 4 — _rerank_merge).
    from src.agent.graph import _rerank_merge
    state = {"question": "hello", "retrieved_chunks": [1, 2], "feedback_boosts": {"d1": 0.3}}
    out = _rerank_merge(state, vs)
    assert out == {}
    assert calls["args"] == ("hello", 7, {"d1": 0.3})


def test_merge_results_noop_when_disabled(monkeypatch):
    from src.agent.graph import _rerank_merge
    from src.retrieval.vector_store import VectorStore
    from src.config import settings

    vs = VectorStore.__new__(VectorStore)
    def boom(*a, **k):
        raise AssertionError("rerank should not run when disabled")
    monkeypatch.setattr(vs, "rerank_chunks", boom)
    monkeypatch.setattr(settings, "rerank_final_enabled", False)
    out = _rerank_merge({"question": "x", "retrieved_chunks": [1]}, vs)
    assert out == {}
```

- [ ] **Step 3: Run the test to verify it fails**

Run: `pytest tests/test_agent/test_graph.py -k merge_results -v`
Expected: FAIL — `ImportError: cannot import name '_rerank_merge'`

- [ ] **Step 4: Implement — extract a testable helper + call it from `merge_results`**

In `src/agent/graph.py`, add a module-level helper (near the top, after imports) so it's unit-testable without compiling the graph:

```python
def _rerank_merge(state, vector_store) -> dict:
    """Final-N rerank of the consolidated chunk set (mutates chunk.score in
    place; consumers sort by score). Fail-open + flag-guarded."""
    from src.config import settings
    if not settings.rerank_final_enabled:
        return {}
    chunks = state.get("retrieved_chunks", [])
    if not chunks:
        return {}
    boosts = state.get("feedback_boosts", {}) or {}
    try:
        vector_store.rerank_chunks(
            chunks, state.get("question", ""), settings.rerank_final_top_n, boosts=boosts,
        )
    except Exception as e:
        import logging
        logging.getLogger("retrieval").warning(f"Final rerank skipped: {e}")
    return {}
```

Then replace the `merge_results` body at lines 218-220:

```python
    def merge_results(state: AgentState) -> dict:
        """Final-N rerank over the chunks both branches produced (mutates
        scores in place; the additive reducer means we return {})."""
        return _rerank_merge(state, vector_store)
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `pytest tests/test_agent/test_graph.py -k merge_results -v`
Expected: 2 PASS

- [ ] **Step 6: Run the broader graph/agent suite to confirm no new breakage**

Run: `pytest tests/test_agent/test_graph.py -v`
Expected: the 2 new tests PASS; any failures match the known pre-existing set (compare against `git stash && pytest tests/test_agent/test_graph.py -v` on master if unsure).

- [ ] **Step 7: Commit**

```bash
git add src/agent/state.py src/agent/graph.py tests/test_agent/test_graph.py
git commit -m "feat: apply final-N rerank in merge_results; add feedback_boosts/strategy_memory state fields"
```

---

## Phase B — F2: Relevance-Feedback Boosts Across Strategies

### Task 4: Shared boost helpers in `feedback.py`

**Files:**
- Modify: `src/retrieval/feedback.py` (add two functions at end of file)
- Test: `tests/test_retrieval/test_feedback.py` (create if absent)

- [ ] **Step 1: Write the failing test**

Create/append `tests/test_retrieval/test_feedback.py`:

```python
from src.retrieval.models import RetrievedChunk, ChunkMetadata


def _chunk(doc_id, idx, score):
    return RetrievedChunk(
        text="t", score=score,
        metadata=ChunkMetadata(doc_id=doc_id, filename="f", doc_type="text",
                               chunk_index=idx, start_char=0, acl_groups=["ALL"]),
    )


def test_apply_boosts_adds_and_resorts():
    from src.retrieval.feedback import apply_feedback_boosts_to_chunks
    chunks = [_chunk("d1", 0, 0.5), _chunk("d2", 1, 0.4)]
    out = apply_feedback_boosts_to_chunks(chunks, {"d2": 0.3})
    assert out[0].metadata.doc_id == "d2"  # 0.4 + 0.3 = 0.7 > 0.5
    assert abs(out[0].score - 0.7) < 1e-9


def test_apply_boosts_empty_is_noop():
    from src.retrieval.feedback import apply_feedback_boosts_to_chunks
    chunks = [_chunk("d1", 0, 0.5), _chunk("d2", 1, 0.9)]
    out = apply_feedback_boosts_to_chunks(chunks, {})
    assert [c.metadata.doc_id for c in out] == ["d1", "d2"]  # order untouched


def test_get_feedback_boosts_sync_failopen(monkeypatch):
    from src.retrieval import feedback
    async def boom(*a, **k):
        raise RuntimeError("db down")
    monkeypatch.setattr(feedback, "get_feedback_boosts", boom)
    # Sync wrapper must swallow the error and return {}
    assert feedback.get_feedback_boosts_sync([0.1, 0.2], ["ALL"]) == {}
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `pytest tests/test_retrieval/test_feedback.py -v`
Expected: FAIL — `ImportError: cannot import name 'apply_feedback_boosts_to_chunks'`

- [ ] **Step 3: Implement both helpers**

Append to `src/retrieval/feedback.py`:

```python
def apply_feedback_boosts_to_chunks(chunks, boosts):
    """Add each chunk's owning-doc boost to chunk.score, re-sort desc, return.
    No-op when boosts is empty. Mutates chunk.score in place."""
    if not boosts:
        return chunks
    for c in chunks:
        b = boosts.get(c.metadata.doc_id, 0.0)
        if b:
            c.score += b
    chunks.sort(key=lambda c: c.score, reverse=True)
    return chunks


def get_feedback_boosts_sync(query_vector, user_groups):
    """Synchronous wrapper for get_feedback_boosts, for sync strategy callers
    (e.g. retrieve_lookup runs in a worker thread). Fail-open -> {}."""
    import asyncio
    try:
        return asyncio.run(get_feedback_boosts(query_vector, user_groups))
    except Exception as e:
        logger.warning(f"Sync feedback boost fetch failed: {e}")
        return {}
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `pytest tests/test_retrieval/test_feedback.py -v`
Expected: 3 PASS

- [ ] **Step 5: Commit**

```bash
git add src/retrieval/feedback.py tests/test_retrieval/test_feedback.py
git commit -m "feat: shared feedback-boost helpers (chunk application + sync wrapper)"
```

---

### Task 5: Apply boosts in `sweep` (doc-level)

**Files:**
- Modify: `src/agent/strategies/sweep.py:42-62`
- Test: `tests/test_agent/test_strategies/test_sweep.py`

Sweep currently cuts docs below 30% of top score (lines 44-50). Add boosts to each doc's representative score **before** that cutoff and drop negative-boost docs — mirroring `map_reduce.py:326-345`.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_agent/test_strategies/test_sweep.py`:

```python
import pytest


@pytest.mark.asyncio
async def test_sweep_returns_feedback_boosts(monkeypatch):
    """retrieve_sweep should fetch and surface feedback_boosts in its result."""
    from src.agent.strategies import sweep as sweep_mod
    from src.retrieval.models import RetrievedChunk, ChunkMetadata

    async def fake_boosts(qv, ug):
        return {"docA": 0.5}
    monkeypatch.setattr(sweep_mod, "get_feedback_boosts", fake_boosts, raising=False)
    monkeypatch.setattr(sweep_mod, "embed_query", lambda q: [0.0, 0.1])

    class FakeVS:
        def search(self, **k):
            return [RetrievedChunk(text="t", score=0.8,
                    metadata=ChunkMetadata(doc_id="docA", filename="f", doc_type="text",
                                           chunk_index=0, start_char=0, acl_groups=["ALL"]))]
        def hybrid_search(self, **k):
            return []
        def get_chunks_by_doc(self, *a, **k):
            return []

    # Disable PRF + date filter side-paths for a focused test
    monkeypatch.setattr(sweep_mod, "_extract_date_filter", lambda *a, **k: [])
    state = {"question": "q", "user_groups": ["ALL"]}
    result = await sweep_mod.retrieve_sweep(state, vector_store=FakeVS())
    assert result.get("feedback_boosts") == {"docA": 0.5}
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `pytest tests/test_agent/test_strategies/test_sweep.py -k feedback -v`
Expected: FAIL — `feedback_boosts` not in result (KeyError/None)

- [ ] **Step 3: Implement**

In `src/agent/strategies/sweep.py`, immediately after `query_vector = await asyncio.to_thread(embed_query, question)` (line 22), fetch boosts fail-open:

```python
    feedback_boosts = {}
    try:
        from src.retrieval.feedback import get_feedback_boosts
        feedback_boosts = await get_feedback_boosts(query_vector, user_groups)
    except Exception:
        pass
```

Then, in the score-cutoff block (lines 44-50), apply boosts to each chunk's score and drop negative-boost docs before computing the threshold. Replace lines 44-50 with:

```python
    if initial_results:
        if feedback_boosts:
            initial_results = [
                c for c in initial_results if feedback_boosts.get(c.metadata.doc_id, 0) >= 0
            ]
            for c in initial_results:
                c.score += feedback_boosts.get(c.metadata.doc_id, 0.0)
        top_score = max((c.score for c in initial_results), default=0)
        score_threshold = top_score * 0.3 if top_score > 0 else 0
        before_count = len(initial_results)
        initial_results = [c for c in initial_results if c.score >= score_threshold]
        if len(initial_results) < before_count:
            logger.info(f"Sweep: score cutoff ({score_threshold:.3f}) reduced {before_count} → {len(initial_results)} results")
```

Finally, surface boosts in the return dict. Find the `return` at the end of `retrieve_sweep` (the dict with `"retrieved_chunks"`) and add `"feedback_boosts": feedback_boosts,` to it.

- [ ] **Step 4: Run the test to verify it passes**

Run: `pytest tests/test_agent/test_strategies/test_sweep.py -k feedback -v`
Expected: PASS

- [ ] **Step 5: Run the full sweep test file**

Run: `pytest tests/test_agent/test_strategies/test_sweep.py -v`
Expected: new test PASS; others match the pre-existing baseline.

- [ ] **Step 6: Commit**

```bash
git add src/agent/strategies/sweep.py tests/test_agent/test_strategies/test_sweep.py
git commit -m "feat: apply relevance-feedback boosts in sweep doc selection"
```

---

### Task 6: Apply boosts in `lookup` (chunk-level, sync)

**Files:**
- Modify: `src/agent/strategies/lookup.py:23-39`
- Test: `tests/test_agent/test_strategies/test_lookup.py`

`retrieve_lookup` is synchronous (run via `to_thread`), so it uses `get_feedback_boosts_sync`. Apply boosts before the 30% cutoff (line 27-33).

- [ ] **Step 1: Write the failing test**

Add to `tests/test_agent/test_strategies/test_lookup.py`:

```python
def test_lookup_applies_and_returns_feedback_boosts(monkeypatch):
    from src.agent.strategies import lookup as lk
    from src.retrieval.models import RetrievedChunk, ChunkMetadata

    monkeypatch.setattr(lk, "embed_query", lambda q: [0.0, 0.1])
    monkeypatch.setattr(lk, "get_feedback_boosts_sync", lambda qv, ug: {"docB": 0.4})
    # neutralize the date-filter import side path
    import src.agent.strategies.sweep as sweep_mod
    monkeypatch.setattr(sweep_mod, "_extract_date_filter", lambda *a, **k: [])

    def _c(doc_id, idx, score):
        return RetrievedChunk(text="t", score=score,
            metadata=ChunkMetadata(doc_id=doc_id, filename="f", doc_type="text",
                                   chunk_index=idx, start_char=0, acl_groups=["ALL"]))

    class FakeVS:
        def hybrid_search_reranked(self, **k):
            return [_c("docA", 0, 0.9), _c("docB", 1, 0.7)]
        def expand_window(self, chunks, window=2):
            return chunks

    state = {"question": "q", "user_groups": ["ALL"]}
    result = lk.retrieve_lookup(state, vector_store=FakeVS())
    assert result["feedback_boosts"] == {"docB": 0.4}
    # docB boosted 0.7+0.4=1.1 should now lead docA (0.9)
    assert result["retrieved_chunks"][0].metadata.doc_id == "docB"
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `pytest tests/test_agent/test_strategies/test_lookup.py -k feedback -v`
Expected: FAIL — `KeyError: 'feedback_boosts'`

- [ ] **Step 3: Implement**

In `src/agent/strategies/lookup.py`, add the import + boost application. After line 23 (the `hybrid_search_reranked` call) and BEFORE the cutoff block at line 27, insert:

```python
    from src.retrieval.feedback import get_feedback_boosts_sync, apply_feedback_boosts_to_chunks
    feedback_boosts = get_feedback_boosts_sync(query_vector, user_groups)
    chunks = apply_feedback_boosts_to_chunks(chunks, feedback_boosts)
```

Then change the return dict (lines 36-39) to include the boosts:

```python
    return {
        "retrieved_chunks": chunks,
        "retrieval_attempts": state.get("retrieval_attempts", 0) + 1,
        "feedback_boosts": feedback_boosts,
    }
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `pytest tests/test_agent/test_strategies/test_lookup.py -k feedback -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/agent/strategies/lookup.py tests/test_agent/test_strategies/test_lookup.py
git commit -m "feat: apply relevance-feedback boosts in lookup (sync)"
```

---

### Task 7: Apply boosts in `cross_reference` (chunk-level, async)

**Files:**
- Modify: `src/agent/strategies/cross_reference.py:21,51-59`
- Test: `tests/test_agent/test_strategies/test_cross_reference.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_agent/test_strategies/test_cross_reference.py`:

```python
import pytest


@pytest.mark.asyncio
async def test_cross_reference_returns_feedback_boosts(monkeypatch):
    from src.agent.strategies import cross_reference as xr
    from src.retrieval.models import RetrievedChunk, ChunkMetadata

    async def fake_embed(texts, kind):
        return [[0.0, 0.1] for _ in texts]
    monkeypatch.setattr("src.ingestion.embedder.embed_texts", fake_embed)

    async def fake_boosts(qv, ug):
        return {"docB": 0.6}
    monkeypatch.setattr(xr, "get_feedback_boosts", fake_boosts, raising=False)

    def _c(doc_id, idx, score):
        return RetrievedChunk(text="t", score=score,
            metadata=ChunkMetadata(doc_id=doc_id, filename="f", doc_type="text",
                                   chunk_index=idx, start_char=0, acl_groups=["ALL"]))

    class FakeVS:
        def hybrid_search_reranked(self, **k):
            return [_c("docA", 0, 0.9), _c("docB", 1, 0.5)]
        def expand_window(self, chunks, window=2):
            return chunks

    class FakeRegistry:
        def list_for_user(self, ug):
            return []

    state = {"question": "q", "user_groups": ["ALL"], "sub_tasks": ["q"]}
    result = await xr.retrieve_cross_reference(state, vector_store=FakeVS(), schema_registry=FakeRegistry())
    assert result["feedback_boosts"] == {"docB": 0.6}
    assert result["retrieved_chunks"][0].metadata.doc_id == "docB"  # 0.5+0.6 > 0.9
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `pytest tests/test_agent/test_strategies/test_cross_reference.py -k feedback -v`
Expected: FAIL — `KeyError: 'feedback_boosts'`

- [ ] **Step 3: Implement**

In `src/agent/strategies/cross_reference.py`, after the dedup loop builds `unique_chunks` (after line 41) and before `sql_results = []` (line 43), add a fail-open fetch + apply:

```python
    feedback_boosts = {}
    try:
        from src.retrieval.feedback import get_feedback_boosts, apply_feedback_boosts_to_chunks
        # Reuse the first sub-task vector as the representative query vector
        feedback_boosts = await get_feedback_boosts(task_vectors[0], user_groups)
        unique_chunks = apply_feedback_boosts_to_chunks(unique_chunks, feedback_boosts)
    except Exception:
        pass
```

Then add `"feedback_boosts": feedback_boosts,` to the `result` dict (line 52-56).

- [ ] **Step 4: Run the test to verify it passes**

Run: `pytest tests/test_agent/test_strategies/test_cross_reference.py -k feedback -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/agent/strategies/cross_reference.py tests/test_agent/test_strategies/test_cross_reference.py
git commit -m "feat: apply relevance-feedback boosts in cross_reference"
```

---

### Task 8: Apply boosts in `structured` narrative search

**Files:**
- Modify: `src/agent/strategies/structured.py:266-276`
- Test: `tests/test_agent/test_strategies/test_structured.py`

`retrieve_structured` searches `tier="table_row"` narratives at line 269. Apply boosts to those chunks before returning.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_agent/test_strategies/test_structured.py`:

```python
import pytest


@pytest.mark.asyncio
async def test_structured_applies_feedback_boosts_to_narratives(monkeypatch):
    from src.agent.strategies import structured as st
    from src.retrieval.models import RetrievedChunk, ChunkMetadata

    # Make the gate pass with one relevant table
    class S:
        table = "all_gs"
    monkeypatch.setattr(st, "tables_relevant_scored", lambda q, schemas: [(S(), 0.9, True)])

    class FakeTrace:
        rows = [{"x": 1}]
        def to_dict(self):
            return {"status": "ran"}
    monkeypatch.setattr(st, "run_structured_lookup", lambda *a, **k: FakeTrace())
    monkeypatch.setattr(st, "embed_query", lambda q: [0.0, 0.1])

    async def fake_boosts(qv, ug):
        return {"docB": 0.5}
    monkeypatch.setattr(st, "get_feedback_boosts", fake_boosts, raising=False)

    def _c(doc_id, idx, score):
        return RetrievedChunk(text="t", score=score,
            metadata=ChunkMetadata(doc_id=doc_id, filename="f", doc_type="text",
                                   chunk_index=idx, start_char=0, acl_groups=["ALL"]))

    class FakeVS:
        def search(self, **k):
            return [_c("docA", 0, 0.8), _c("docB", 1, 0.4)]

    class FakeRegistry:
        def list_for_user(self, ug):
            return [S()]

    state = {"question": "q", "user_groups": ["ALL"]}
    result = await st.retrieve_structured(state, vector_store=FakeVS(), schema_registry=FakeRegistry())
    chunks = result["retrieved_chunks"]
    assert chunks[0].metadata.doc_id == "docB"  # 0.4+0.5 > 0.8
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `pytest tests/test_agent/test_strategies/test_structured.py -k feedback -v`
Expected: FAIL — docA still leads (boosts not applied)

- [ ] **Step 3: Implement**

In `src/agent/strategies/structured.py`, in the narrative-search block (lines 266-273), after `chunks` is populated and before the `return` at line 275, add a fail-open boost application:

```python
    try:
        from src.retrieval.feedback import get_feedback_boosts, apply_feedback_boosts_to_chunks
        _boosts = await get_feedback_boosts(qv, user_groups)
        chunks = apply_feedback_boosts_to_chunks(chunks, _boosts)
    except Exception:
        pass
```

Note: `qv` is the query vector embedded at line 268; it is in scope. If the embed step failed (`chunks = []`), `qv` may be unbound — guard by wrapping the whole block in the `try` (already done) so a `NameError` falls through fail-open.

- [ ] **Step 4: Run the test to verify it passes**

Run: `pytest tests/test_agent/test_strategies/test_structured.py -k feedback -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/agent/strategies/structured.py tests/test_agent/test_strategies/test_structured.py
git commit -m "feat: apply relevance-feedback boosts to structured narrative chunks"
```

---

### Task 9: Surface `feedback_boosts` from `map_reduce` and propagate through the SWEEP retrieve node

**Files:**
- Modify: `src/agent/strategies/map_reduce.py` (around line 260-266 fetch + final return dict)
- Modify: `src/agent/graph.py:73-77` (SWEEP branch result dict)
- Test: `tests/test_agent/test_strategies/test_map_reduce.py`

map_reduce already fetches `feedback_boosts` (line 261-266) but doesn't return them. Surface them, then have the SWEEP retrieve branch copy them into the node result so `merge_results` (Task 3) can re-apply them in the final rerank.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_agent/test_strategies/test_map_reduce.py`:

```python
import pytest


@pytest.mark.asyncio
async def test_map_reduce_returns_feedback_boosts(monkeypatch):
    from src.agent.strategies import map_reduce as mr
    async def fake_boosts(qv, ug):
        return {"docZ": 0.2}
    monkeypatch.setattr(mr, "get_feedback_boosts", fake_boosts, raising=False)
    # Find the result dict returned by retrieve_map_reduce and assert the key.
    # (Use the file's existing fixtures/mocks for vector_store + metadata_store;
    # this assertion only checks the new key is present.)
    # See existing test_map_reduce.py setup for the FakeVS/monkeypatch pattern.
```

Note for the implementer: reuse this file's existing `retrieve_map_reduce` invocation fixture (it already mocks the vector store and metadata store). Add the single assertion `assert result.get("feedback_boosts") == {"docZ": 0.2}` to an existing passing happy-path test, or build a minimal invocation mirroring the existing ones. Do not invent new mocks if a working harness already exists in the file.

- [ ] **Step 2: Run the test to verify it fails**

Run: `pytest tests/test_agent/test_strategies/test_map_reduce.py -k feedback -v`
Expected: FAIL — `feedback_boosts` absent from result.

- [ ] **Step 3: Implement in `map_reduce.py`**

Locate the final `return` dict of `retrieve_map_reduce` (the one containing `"retrieved_chunks"` and `"doc_relevance"`). Add:

```python
        "feedback_boosts": feedback_boosts,
```

`feedback_boosts` is already defined at line 261. If the function has multiple return points, add the key to the primary success-path return.

- [ ] **Step 4: Propagate in the SWEEP branch of `graph.py`**

In `src/agent/graph.py`, in the SWEEP branch after building `result` (after line 73), add:

```python
            if mr_result.get("feedback_boosts"):
                result["feedback_boosts"] = mr_result["feedback_boosts"]
```

For the non-SWEEP branches (LOOKUP/CROSS_REFERENCE/ANALYTICAL/TEMPORAL), `result` is the strategy's own dict, which already carries `feedback_boosts` from Tasks 6-8 — it flows up automatically.

- [ ] **Step 5: Run the test to verify it passes**

Run: `pytest tests/test_agent/test_strategies/test_map_reduce.py -k feedback -v`
Expected: PASS

- [ ] **Step 6: Run the full strategies suite**

Run: `pytest tests/test_agent/test_strategies/ -v`
Expected: all new feedback tests PASS; remainder matches the pre-existing baseline.

- [ ] **Step 7: Commit**

```bash
git add src/agent/strategies/map_reduce.py src/agent/graph.py tests/test_agent/test_strategies/test_map_reduce.py
git commit -m "feat: surface feedback_boosts from map_reduce and propagate to final rerank"
```

---

## Phase C — F1: Strategy Memory → Routing

### Task 10: Fix the `get_best_strategy` selection metric

**Files:**
- Modify: `src/retrieval/strategy_memory.py:123-147`
- Test: `tests/test_retrieval/test_strategy_memory.py` (create if absent)

Replace precision-based ranking (which favors narrow strategies) with a composite: primary `avg_docs_cited`, tiebreak `avg_relevant`. Return `count` (winner run-count) and `margin` (winner composite − runner-up composite, normalized to the winner's composite; `1.0` when only one strategy has records).

- [ ] **Step 1: Write the failing test**

Create `tests/test_retrieval/test_strategy_memory.py`:

```python
import pytest


class _Rec:
    def __init__(self, qt, discovered, relevant, cited, t=1.0):
        self.query_type = qt
        self.strategy_used = qt
        self.docs_discovered = discovered
        self.docs_relevant = relevant
        self.docs_cited = cited
        self.total_time_seconds = t


@pytest.mark.asyncio
async def test_best_strategy_prefers_cited_over_precision(monkeypatch):
    from src.retrieval import strategy_memory as sm

    # lookup: precision 1.0 but only 1 cited; sweep: precision 0.8 but 8 cited.
    records = [
        _Rec("lookup", discovered=1, relevant=1, cited=1),
        _Rec("lookup", discovered=1, relevant=1, cited=1),
        _Rec("lookup", discovered=1, relevant=1, cited=1),
        _Rec("sweep", discovered=10, relevant=8, cited=8),
        _Rec("sweep", discovered=10, relevant=8, cited=8),
        _Rec("sweep", discovered=10, relevant=8, cited=8),
    ]

    class FakeSession:
        async def __aenter__(self): return self
        async def __aexit__(self, *a): return False
        async def execute(self, q):
            class R:
                def scalars(self_):
                    class S:
                        def all(self__): return records
                    return S()
            return R()

    class FakeStore:
        def session_factory(self): return FakeSession()

    monkeypatch.setattr(sm.settings, "strategy_memory_enabled", True)
    monkeypatch.setattr("src.api.routes_ingest.get_metadata_store", lambda: FakeStore())

    best = await sm.get_best_strategy("how many contracts did the army award")
    assert best["strategy"] == "sweep"        # cited-weighted winner
    assert best["count"] == 3
    assert best["margin"] > 0
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `pytest tests/test_retrieval/test_strategy_memory.py -v`
Expected: FAIL — current code returns `lookup` (precision 1.0) and has no `margin` key.

- [ ] **Step 3: Implement the metric**

In `src/retrieval/strategy_memory.py`, replace the selection block (lines 123-147) with:

```python
    # Composite ranking: primary = avg docs cited (truest success proxy),
    # tiebreak = avg relevant. Precision is reported but no longer the key
    # (it structurally favored narrow strategies and punished recall).
    def composite(stats):
        avg_cited = sum(stats["cited"]) / len(stats["cited"])
        avg_relevant = sum(stats["relevant"]) / len(stats["relevant"])
        return avg_cited + 0.001 * avg_relevant  # relevant breaks cited ties

    ranked = sorted(
        strategy_stats.items(), key=lambda kv: composite(kv[1]), reverse=True
    )
    top_strategy, top_stats = ranked[0]
    top_comp = composite(top_stats)
    runner_comp = composite(ranked[1][1]) if len(ranked) > 1 else 0.0
    margin = (top_comp - runner_comp) / top_comp if top_comp > 0 else (1.0 if len(ranked) == 1 else 0.0)

    avg_discovered = sum(top_stats["discovered"]) / len(top_stats["discovered"])
    avg_relevant = sum(top_stats["relevant"]) / len(top_stats["relevant"])
    avg_cited = sum(top_stats["cited"]) / len(top_stats["cited"])
    precision = avg_relevant / avg_discovered if avg_discovered > 0 else 0
    best = {
        "strategy": top_strategy,
        "avg_relevant": round(avg_relevant, 1),
        "avg_discovered": round(avg_discovered, 1),
        "avg_cited": round(avg_cited, 1),
        "avg_time": round(sum(top_stats["times"]) / len(top_stats["times"]), 1),
        "precision": round(precision, 3),
        "count": len(top_stats["times"]),
        "margin": round(margin, 3),
    }

    if best:
        logger.info(f"Strategy memory: pattern='{pattern}' -> best={best['strategy']} "
                   f"(cited={best['avg_cited']}, margin={best['margin']:.0%}, n={best['count']})")

    return best
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `pytest tests/test_retrieval/test_strategy_memory.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/retrieval/strategy_memory.py tests/test_retrieval/test_strategy_memory.py
git commit -m "feat: rank strategy memory by docs-cited composite with margin/count"
```

---

### Task 11: Add the soft-override gate settings

**Files:**
- Modify: `src/config.py:81` (after `strategy_memory_enabled`)

- [ ] **Step 1: Add the settings**

In `src/config.py`, immediately after line 81 (`strategy_memory_enabled: bool = True`), add:

```python
    strategy_memory_min_runs: int = 3   # min recorded runs before memory may override routing
    strategy_memory_margin: float = 0.15  # min normalized composite margin to override
```

- [ ] **Step 2: Verify it imports**

Run: `python -c "from src.config import settings; print(settings.strategy_memory_min_runs, settings.strategy_memory_margin)"`
Expected: `3 0.15`

- [ ] **Step 3: Commit**

```bash
git add src/config.py
git commit -m "feat: add strategy-memory soft-override gate settings"
```

---

### Task 12: Async classify node with confidence-gated soft override + observability

**Files:**
- Modify: `src/agent/classifier.py:64-73` (`_classify_node_factory`)
- Modify: `src/agent/graph.py:291-292` (capture `strategy_memory` into the trace)
- Test: `tests/test_agent/test_classifier.py`

The classify node becomes `async` so it can `await get_best_strategy`. It overrides the LLM `query_type` only when all gates pass, and always emits a `strategy_memory` decision dict for observability.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_agent/test_classifier.py`:

```python
import pytest
from src.agent.state import QueryType


@pytest.mark.asyncio
async def test_classify_node_soft_override_applies(monkeypatch):
    import src.agent.classifier as clf

    # LLM says LOOKUP; memory confidently says SWEEP -> override.
    monkeypatch.setattr(clf, "classify_query",
                        lambda state, available_tables="": {"query_type": QueryType.LOOKUP, "sub_tasks": ["q"]})
    async def fake_best(q):
        return {"strategy": "sweep", "count": 5, "margin": 0.5, "avg_cited": 8.0}
    monkeypatch.setattr(clf, "get_best_strategy", fake_best, raising=False)
    monkeypatch.setattr(clf.settings, "strategy_memory_enabled", True)
    monkeypatch.setattr(clf.settings, "strategy_memory_min_runs", 3)
    monkeypatch.setattr(clf.settings, "strategy_memory_margin", 0.15)

    node = clf._classify_node_factory(schema_registry=None)
    out = await node({"question": "q", "user_groups": ["ALL"]})
    assert out["query_type"] == QueryType.SWEEP
    assert out["strategy_memory"]["overrode"] is True


@pytest.mark.asyncio
async def test_classify_node_respects_min_runs(monkeypatch):
    import src.agent.classifier as clf

    monkeypatch.setattr(clf, "classify_query",
                        lambda state, available_tables="": {"query_type": QueryType.LOOKUP, "sub_tasks": ["q"]})
    async def fake_best(q):
        return {"strategy": "sweep", "count": 2, "margin": 0.9, "avg_cited": 8.0}  # too few runs
    monkeypatch.setattr(clf, "get_best_strategy", fake_best, raising=False)
    monkeypatch.setattr(clf.settings, "strategy_memory_enabled", True)
    monkeypatch.setattr(clf.settings, "strategy_memory_min_runs", 3)
    monkeypatch.setattr(clf.settings, "strategy_memory_margin", 0.15)

    node = clf._classify_node_factory(schema_registry=None)
    out = await node({"question": "q", "user_groups": ["ALL"]})
    assert out["query_type"] == QueryType.LOOKUP        # not overridden
    assert out["strategy_memory"]["overrode"] is False


@pytest.mark.asyncio
async def test_classify_node_failopen(monkeypatch):
    import src.agent.classifier as clf
    monkeypatch.setattr(clf, "classify_query",
                        lambda state, available_tables="": {"query_type": QueryType.LOOKUP, "sub_tasks": ["q"]})
    async def boom(q):
        raise RuntimeError("db down")
    monkeypatch.setattr(clf, "get_best_strategy", boom, raising=False)
    monkeypatch.setattr(clf.settings, "strategy_memory_enabled", True)

    node = clf._classify_node_factory(schema_registry=None)
    out = await node({"question": "q", "user_groups": ["ALL"]})
    assert out["query_type"] == QueryType.LOOKUP        # fail-open keeps LLM pick
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `pytest tests/test_agent/test_classifier.py -k "soft_override or min_runs or failopen" -v`
Expected: FAIL — node is sync / `strategy_memory` key absent.

- [ ] **Step 3: Implement the async node**

In `src/agent/classifier.py`, add imports at the top (the file already imports `settings`? confirm — `from src.config import settings`; if absent, add it). Then replace `_classify_node_factory` (lines 64-73) with:

```python
def _classify_node_factory(schema_registry):
    """Build an async LangGraph 'classify' node: LLM classification, then a
    confidence-gated soft override from Strategy Memory."""
    async def classify_node(state: AgentState) -> dict:
        import asyncio
        available = ""
        if schema_registry is not None:
            schemas = schema_registry.list_for_user(state.get("user_groups", ["ALL"]))
            available = format_available_tables(schemas)
        # classify_query makes a blocking LLM call — run it off the event loop
        # (the old sync node was run by LangGraph in a threadpool).
        result = await asyncio.to_thread(classify_query, state, available)
        llm_pick = result["query_type"]

        memory_decision = {"llm_pick": str(llm_pick), "overrode": False, "reason": "disabled"}
        if settings.strategy_memory_enabled:
            try:
                best = await get_best_strategy(state["question"])
                memory_decision["reason"] = "no record"
                if best:
                    memory_decision.update({
                        "memory_best": best["strategy"], "count": best["count"],
                        "margin": best["margin"], "reason": "below gate",
                    })
                    try:
                        mem_type = QueryType(best["strategy"])
                    except ValueError:
                        mem_type = None
                    if (mem_type is not None
                            and mem_type != llm_pick
                            and best["count"] >= settings.strategy_memory_min_runs
                            and best["margin"] >= settings.strategy_memory_margin):
                        result["query_type"] = mem_type
                        memory_decision["overrode"] = True
                        memory_decision["reason"] = "override"
                        logger.info("Strategy memory override: %s -> %s (n=%d, margin=%.0f%%)",
                                    llm_pick, mem_type, best["count"], best["margin"] * 100)
            except Exception as e:
                logger.warning("Strategy memory lookup failed, keeping LLM pick: %s", e)
                memory_decision["reason"] = "error"

        result["strategy_memory"] = memory_decision
        return result
    return classify_node
```

Add the import for `get_best_strategy` at the top of `classifier.py`:

```python
from src.retrieval.strategy_memory import get_best_strategy
```

(Confirm `from src.config import settings` is present at the top; add it if not.)

- [ ] **Step 4: Capture the decision in the trace**

In `src/agent/graph.py`, in `run_agent_with_trace`, extend the classify capture at lines 291-292:

```python
            if node_name == "classify":
                trace.query_type = str(node_output.get("query_type", ""))
                trace.strategy_memory = node_output.get("strategy_memory")
```

And add the field to the `AgentTrace` dataclass (after line 263):

```python
    strategy_memory: dict | None = None
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `pytest tests/test_agent/test_classifier.py -k "soft_override or min_runs or failopen" -v`
Expected: 3 PASS

- [ ] **Step 6: Confirm the graph still compiles with an async classify node**

Run: `python -c "from src.agent.graph import create_agent_graph; print('ok')"`
Expected: prints `ok` (LangGraph supports async nodes; classify already had async peers).

- [ ] **Step 7: Surface the decision in the playground (modest, no new step row)**

In `src/admin/routes.py`, find where the `classify` node step detail is formatted (search for `node_name == "classify"` or the classify step in `_format_live_step`/`format_step_detail`). Add a single line to the classify step detail when `strategy_memory.overrode` is true, e.g.:

```python
# inside the classify-step detail builder, after the query_type line:
sm = (node_output or {}).get("strategy_memory") or {}
if sm.get("overrode"):
    detail += f"<br><strong>Strategy memory override:</strong> {sm.get('llm_pick')} → {sm.get('memory_best')} (n={sm.get('count')}, margin={sm.get('margin')})"
```

If the exact formatting seam is unclear, keep this change minimal and additive — it must not alter existing classify rendering when `strategy_memory` is absent or `overrode` is false.

- [ ] **Step 8: Run the admin + agent suites for regressions**

Run: `pytest tests/test_agent/ tests/test_admin/ -q`
Expected: new tests PASS; failures match the known pre-existing baseline (compare to master if uncertain).

- [ ] **Step 9: Commit**

```bash
git add src/agent/classifier.py src/agent/graph.py src/admin/routes.py tests/test_agent/test_classifier.py
git commit -m "feat: wire Strategy Memory into routing (confidence-gated soft override + trace)"
```

---

## Final verification

- [ ] **Step 1: Full affected-suite run**

Run: `pytest tests/test_retrieval/ tests/test_agent/ tests/test_admin/ -q`
Expected: all newly added tests PASS; the only failures are the documented pre-existing set.

- [ ] **Step 2: Regression check — all flags off reproduces current behavior**

Manually confirm: with `rerank_final_enabled=False`, `feedback_enabled=False`, `strategy_memory_enabled=False`, `merge_results` returns `{}` without reranking, strategies skip boost fetches (the function self-guards on `feedback_enabled`), and the classify node keeps the LLM pick with `strategy_memory.reason == "disabled"`.

- [ ] **Step 3: Smoke test (optional, needs the app)**

Run a representative SWEEP question ("GS salary rates in Tampa") through the playground and confirm: the answer still resolves, the classify step shows any strategy-memory decision, and reranked chunks lead the citation list.

---

## Self-Review Notes (for the implementer)

- **Spec coverage:** F1 = Tasks 10-12; F2 = Tasks 4-9; F3 = Tasks 1-3. New settings = Tasks 1, 11. Observability = Task 12. All spec sections map to tasks.
- **Sync/async:** only `retrieve_lookup` is sync — it uses `get_feedback_boosts_sync` (Task 6). All other strategies are async and use `get_feedback_boosts` directly.
- **Reducer interaction:** F3 mutates `chunk.score` in place rather than returning a reordered list, because `retrieved_chunks` has the additive `_merge_chunks` reducer that would discard a reordered return. Verified consumers (`synthesizer.py:127-139`, `build_citations`, playground) all sort by `c.score`.
- **Naming consistency:** `rerank_chunks`, `apply_feedback_boosts_to_chunks`, `get_feedback_boosts_sync`, `_get_cross_encoder_model`, `_rerank_merge`, state fields `feedback_boosts`/`strategy_memory`, settings `rerank_final_enabled`/`rerank_final_top_n`/`rerank_model`/`strategy_memory_min_runs`/`strategy_memory_margin` — used identically across all tasks.
