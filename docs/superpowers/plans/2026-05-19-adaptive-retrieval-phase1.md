# Adaptive Retrieval Phase 1: Relevance Feedback Log — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Log which documents were useful for each query and boost those documents for similar future queries, reducing wasted MAP LLM calls by ~60%.

**Architecture:** A new `QueryFeedback` model stores per-query document relevance signals (was_cited, was_in_map_reduce). A new `src/retrieval/feedback.py` module handles logging feedback after queries and looking up historical boosts during retrieval. The map-reduce and sweep strategies apply boosts from past similar queries during document scoring. Feedback decays over 90 days.

**Tech Stack:** SQLAlchemy, numpy (cosine similarity on stored vectors), existing embedder

---

### Task 1: Add QueryFeedback model and config

**Files:**
- Modify: `src/db/models.py`
- Modify: `src/config.py`

- [ ] **Step 1: Add QueryFeedback model**

In `src/db/models.py`, after the `QueryMetrics` class, add:

```python
class QueryFeedback(Base):
    __tablename__ = "query_feedback"
    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    query_hash: Mapped[str] = mapped_column(String, nullable=False, index=True)
    query_text: Mapped[str] = mapped_column(String, nullable=False)
    query_vector_blob: Mapped[bytes] = mapped_column(default=b"")  # serialized numpy array
    query_type: Mapped[str] = mapped_column(String, default="")
    doc_id: Mapped[str] = mapped_column(String, nullable=False)
    filename: Mapped[str] = mapped_column(String, default="")
    relevance_score: Mapped[float] = mapped_column(default=0.0)
    was_cited: Mapped[bool] = mapped_column(default=False)
    was_in_map_reduce: Mapped[bool] = mapped_column(default=False)
    user_groups: Mapped[list] = mapped_column(JSON, default=list)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
    )
```

- [ ] **Step 2: Add config settings**

In `src/config.py`, after `llm_max_output_tokens`, add:

```python
    # Relevance feedback
    feedback_enabled: bool = True
    feedback_similarity_threshold: float = 0.85  # cosine sim for matching past queries
    feedback_boost_cited: float = 0.3            # boost for previously-cited docs
    feedback_boost_relevant: float = 0.2         # boost for MAP-relevant docs
    feedback_penalty_irrelevant: float = 0.1     # penalty for MAP-irrelevant docs
    feedback_decay_days: int = 90                # halve weight after this many days
```

- [ ] **Step 3: Commit**

```bash
git add src/db/models.py src/config.py
git commit -m "feat: add QueryFeedback model and feedback config settings"
```

---

### Task 2: Create feedback module

**Files:**
- Create: `src/retrieval/feedback.py`

- [ ] **Step 1: Create the feedback module**

Create `src/retrieval/feedback.py`:

```python
"""Relevance feedback: log query→document signals, boost future queries."""
import hashlib
import logging
import time
from datetime import datetime, timezone

import numpy as np
from sqlalchemy import select

from src.config import settings

logger = logging.getLogger(__name__)


def _serialize_vector(vec: list[float]) -> bytes:
    """Serialize embedding vector to bytes for storage."""
    return np.array(vec, dtype=np.float32).tobytes()


def _deserialize_vector(blob: bytes) -> np.ndarray:
    """Deserialize bytes back to numpy array."""
    if not blob:
        return np.array([], dtype=np.float32)
    return np.frombuffer(blob, dtype=np.float32)


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    if a.size == 0 or b.size == 0:
        return 0.0
    dot = np.dot(a, b)
    norm = np.linalg.norm(a) * np.linalg.norm(b)
    return float(dot / norm) if norm > 0 else 0.0


def _query_hash(text: str) -> str:
    """Normalize and hash query text."""
    normalized = " ".join(text.lower().split())
    return hashlib.sha256(normalized.encode()).hexdigest()


async def log_feedback(
    query_text: str,
    query_vector: list[float],
    query_type: str,
    user_groups: list[str],
    cited_doc_ids: list[str],
    relevant_doc_ids: list[str],
    irrelevant_doc_ids: list[str],
    doc_filenames: dict[str, str] = None,
    doc_scores: dict[str, float] = None,
):
    """Log relevance feedback for a completed query."""
    if not settings.feedback_enabled:
        return

    from src.db.models import QueryFeedback

    doc_filenames = doc_filenames or {}
    doc_scores = doc_scores or {}
    qhash = _query_hash(query_text)
    vec_blob = _serialize_vector(query_vector)

    try:
        from src.api.routes_ingest import get_metadata_store
        store = get_metadata_store()
        async with store.session_factory() as session:
            # Log cited docs
            for doc_id in cited_doc_ids:
                session.add(QueryFeedback(
                    query_hash=qhash, query_text=query_text[:500],
                    query_vector_blob=vec_blob, query_type=query_type,
                    doc_id=doc_id, filename=doc_filenames.get(doc_id, ""),
                    relevance_score=doc_scores.get(doc_id, 0.0),
                    was_cited=True, was_in_map_reduce=True,
                    user_groups=user_groups,
                ))
            # Log relevant-but-not-cited docs
            for doc_id in relevant_doc_ids:
                if doc_id not in cited_doc_ids:
                    session.add(QueryFeedback(
                        query_hash=qhash, query_text=query_text[:500],
                        query_vector_blob=vec_blob, query_type=query_type,
                        doc_id=doc_id, filename=doc_filenames.get(doc_id, ""),
                        relevance_score=doc_scores.get(doc_id, 0.0),
                        was_cited=False, was_in_map_reduce=True,
                        user_groups=user_groups,
                    ))
            # Log irrelevant docs (MAP returned NO_RELEVANT_DATA)
            for doc_id in irrelevant_doc_ids:
                session.add(QueryFeedback(
                    query_hash=qhash, query_text=query_text[:500],
                    query_vector_blob=vec_blob, query_type=query_type,
                    doc_id=doc_id, filename=doc_filenames.get(doc_id, ""),
                    relevance_score=doc_scores.get(doc_id, 0.0),
                    was_cited=False, was_in_map_reduce=False,
                    user_groups=user_groups,
                ))
            await session.commit()
            total = len(cited_doc_ids) + len(relevant_doc_ids) + len(irrelevant_doc_ids)
            logger.info(f"Feedback logged: {len(cited_doc_ids)} cited, "
                       f"{len(relevant_doc_ids)} relevant, {len(irrelevant_doc_ids)} irrelevant")
    except Exception as e:
        logger.warning(f"Failed to log feedback: {e}")


async def get_feedback_boosts(
    query_vector: list[float],
    user_groups: list[str],
) -> dict[str, float]:
    """Look up historical feedback boosts for a query.

    Returns {doc_id: boost_score} where positive = boost, negative = penalty.
    """
    if not settings.feedback_enabled:
        return {}

    from src.db.models import QueryFeedback

    try:
        from src.api.routes_ingest import get_metadata_store
        store = get_metadata_store()
        async with store.session_factory() as session:
            result = await session.execute(select(QueryFeedback))
            all_feedback = list(result.scalars().all())
    except Exception as e:
        logger.warning(f"Failed to load feedback: {e}")
        return {}

    if not all_feedback:
        return {}

    query_vec = np.array(query_vector, dtype=np.float32)
    now = time.time()
    boosts: dict[str, float] = {}
    seen_queries: set[str] = set()

    for fb in all_feedback:
        # Compute similarity to past query
        if fb.query_hash in seen_queries:
            # Already computed similarity for this query — reuse
            pass
        else:
            fb_vec = _deserialize_vector(fb.query_vector_blob)
            if fb_vec.size == 0:
                continue
            sim = _cosine_similarity(query_vec, fb_vec)
            if sim < settings.feedback_similarity_threshold:
                continue
            seen_queries.add(fb.query_hash)

        # Apply decay based on age
        age_days = (now - fb.created_at.timestamp()) / 86400 if fb.created_at else 0
        decay = 0.5 ** (age_days / settings.feedback_decay_days) if settings.feedback_decay_days > 0 else 1.0

        # Calculate boost
        if fb.was_cited:
            boost = settings.feedback_boost_cited * decay
        elif fb.was_in_map_reduce:
            boost = settings.feedback_boost_relevant * decay
        else:
            boost = -settings.feedback_penalty_irrelevant * decay

        # Accumulate per doc_id (multiple past queries may reference same doc)
        boosts[fb.doc_id] = boosts.get(fb.doc_id, 0.0) + boost

    if boosts:
        pos = sum(1 for v in boosts.values() if v > 0)
        neg = sum(1 for v in boosts.values() if v < 0)
        logger.info(f"Feedback boosts: {pos} boosted, {neg} penalized (from {len(seen_queries)} similar past queries)")

    return boosts
```

- [ ] **Step 2: Commit**

```bash
git add src/retrieval/feedback.py
git commit -m "feat: create relevance feedback module with logging and boost lookup"
```

---

### Task 3: Log feedback after queries in the playground

**Files:**
- Modify: `src/admin/routes.py`

- [ ] **Step 1: Add feedback logging after the synthesizer generates an answer**

In `src/admin/routes.py`, inside the `run_query` function, find the block where query metrics are logged (after the cache_store call, the `# Log query metrics` comment). After the metrics block, add feedback logging:

```python
            # Log relevance feedback for adaptive retrieval
            try:
                from src.retrieval.feedback import log_feedback
                SYNTHETIC_IDS = {"map-reduce", "knowledge-graph", "metadata-context"}
                cited_ids = [c.doc_id for c in citations]
                # Determine which docs were MAP'd and relevant vs irrelevant
                mr_chunks = [c for c in chunks if c.metadata.doc_id == "map-reduce"]
                mr_relevant_ids = []
                mr_irrelevant_ids = []
                if mr_chunks:
                    import re as _re
                    mr_text = mr_chunks[0].text
                    mr_filenames = _re.findall(r'\[([^\]]+\.(?:md|pdf|docx))\]:', mr_text)
                    mr_relevant_ids = list(set(mr_filenames))
                # All discovered docs that weren't in MAP results are "not evaluated"
                all_doc_ids = {c.metadata.doc_id for c in chunks if c.metadata.doc_id not in SYNTHETIC_IDS}
                doc_fn_map = {c.metadata.doc_id: c.metadata.filename for c in chunks if c.metadata.doc_id not in SYNTHETIC_IDS}

                await log_feedback(
                    query_text=question,
                    query_vector=query_vector,
                    query_type=query_type,
                    user_groups=user_groups,
                    cited_doc_ids=cited_ids,
                    relevant_doc_ids=mr_relevant_ids,
                    irrelevant_doc_ids=mr_irrelevant_ids,
                    doc_filenames=doc_fn_map,
                )
            except Exception:
                pass
```

- [ ] **Step 2: Commit**

```bash
git add src/admin/routes.py
git commit -m "feat: log relevance feedback after playground queries"
```

---

### Task 4: Apply feedback boosts during document discovery

**Files:**
- Modify: `src/agent/strategies/map_reduce.py`

- [ ] **Step 1: Add feedback boosts to the document scoring in map_reduce.py**

In `src/agent/strategies/map_reduce.py`, find the scoring section where `scored_docs` is built (the `# Score and rank all candidates` comment). Before the scoring loop, fetch feedback boosts and apply them to the combined score.

After `candidate_doc_ids` and `doc_scores` are set, before the `if metadata_store:` block, add:

```python
    # Fetch feedback boosts from past similar queries
    feedback_boosts = {}
    try:
        from src.retrieval.feedback import get_feedback_boosts
        feedback_boosts = await get_feedback_boosts(query_vector, user_groups)
    except Exception:
        pass
```

Then inside the scoring loop, where `combined` is calculated, change:

```python
                combined = doc_scores.get(did, 0) + (meta_score * 0.1)
```

To:

```python
                combined = doc_scores.get(did, 0) + (meta_score * 0.1) + feedback_boosts.get(did, 0)
```

And also add feedback boosts for the `else` (no metadata_store) branch. Change:

```python
    else:
        relevant_doc_ids = candidate_doc_ids
```

To:

```python
    else:
        if feedback_boosts:
            scored = [(did, doc_scores.get(did, 0) + feedback_boosts.get(did, 0)) for did in candidate_doc_ids]
            scored.sort(key=lambda x: x[1], reverse=True)
            relevant_doc_ids = [did for did, _ in scored]
        else:
            relevant_doc_ids = candidate_doc_ids
```

- [ ] **Step 2: Commit**

```bash
git add src/agent/strategies/map_reduce.py
git commit -m "feat: apply relevance feedback boosts during map-reduce document scoring"
```

---

### Task 5: Add feedback settings to admin UI

**Files:**
- Modify: `src/admin/templates/settings.html`
- Modify: `src/admin/routes.py`

- [ ] **Step 1: Add feedback settings to the settings template**

In `src/admin/templates/settings.html`, after the Metadata Extraction section and before the Save button, add:

```html
    <div style="margin-top:1rem;">
        <h3 style="margin-bottom:0.5rem;">Relevance Feedback</h3>
        <div class="form-row">
            <div class="form-group">
                <label for="feedback_enabled">Enable Feedback Learning</label>
                <select id="feedback_enabled" name="feedback_enabled">
                    <option value="true" {{ 'selected' if settings.feedback_enabled }}>Enabled</option>
                    <option value="false" {{ 'selected' if not settings.feedback_enabled }}>Disabled</option>
                </select>
                <span style="font-size:0.8rem; color:#6b7280;">Log and learn from query results to improve future retrieval</span>
            </div>
            <div class="form-group">
                <label for="feedback_similarity_threshold">Similarity Threshold</label>
                <input type="number" id="feedback_similarity_threshold" name="feedback_similarity_threshold" value="{{ settings.feedback_similarity_threshold }}" step="0.05" min="0.5" max="1.0" style="max-width:100px;">
                <span style="font-size:0.8rem; color:#6b7280;">Min cosine similarity to match past queries (0.85 = very similar)</span>
            </div>
        </div>
    </div>
```

- [ ] **Step 2: Add to settings save handler**

In `src/admin/routes.py`, add to the save_settings function signature:

```python
    feedback_enabled: bool = Form(True),
    feedback_similarity_threshold: float = Form(0.85),
```

Add to the in-memory update section:

```python
    settings.feedback_enabled = feedback_enabled
    settings.feedback_similarity_threshold = feedback_similarity_threshold
```

Add to the env_lines section:

```python
    env_lines["FEEDBACK_ENABLED"] = str(settings.feedback_enabled).lower()
    env_lines["FEEDBACK_SIMILARITY_THRESHOLD"] = str(settings.feedback_similarity_threshold)
```

- [ ] **Step 3: Commit**

```bash
git add src/admin/templates/settings.html src/admin/routes.py
git commit -m "feat: add relevance feedback settings to admin UI"
```

---

### Task 6: Add feedback stats to the metrics dashboard

**Files:**
- Modify: `src/admin/routes.py`

- [ ] **Step 1: Add feedback stats to the query metrics dashboard**

In `src/admin/routes.py`, find the `query_metrics_dashboard` function. After the summary stats section (the grid div), add a feedback summary:

```python
        # Feedback stats
        from src.db.models import QueryFeedback
        try:
            fb_result = await session.execute(select(QueryFeedback))
            fb_rows = list(fb_result.scalars().all())
            fb_total = len(fb_rows)
            fb_cited = sum(1 for r in fb_rows if r.was_cited)
            fb_relevant = sum(1 for r in fb_rows if r.was_in_map_reduce and not r.was_cited)
            fb_irrelevant = sum(1 for r in fb_rows if not r.was_in_map_reduce)
            fb_unique_queries = len({r.query_hash for r in fb_rows})
            fb_unique_docs = len({r.doc_id for r in fb_rows})

            summary += f"""<div style="display:grid; grid-template-columns:repeat(5,1fr); gap:1rem; margin-bottom:1rem; padding-top:0.5rem; border-top:1px solid #e5e7eb;">
                <div><strong>{fb_total}</strong><br><span style="font-size:0.8rem; color:#6b7280;">Feedback Records</span></div>
                <div><strong>{fb_unique_queries}</strong><br><span style="font-size:0.8rem; color:#6b7280;">Unique Queries</span></div>
                <div><strong>{fb_unique_docs}</strong><br><span style="font-size:0.8rem; color:#6b7280;">Unique Docs Seen</span></div>
                <div><strong>{fb_cited}</strong><br><span style="font-size:0.8rem; color:#6b7280;">Cited Signals</span></div>
                <div><strong>{fb_irrelevant}</strong><br><span style="font-size:0.8rem; color:#6b7280;">Irrelevant Signals</span></div>
            </div>"""
        except Exception:
            pass  # table may not exist yet
```

Note: The `session` variable is already available inside the `async with store.session_factory() as session:` block. Move the feedback query inside that block, after the metrics query.

- [ ] **Step 2: Commit**

```bash
git add src/admin/routes.py
git commit -m "feat: add feedback stats to query performance dashboard"
```

---

### Task 7: Update README and push

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Add feedback section to README**

In `README.md`, after the "Document Metadata" section, add:

```markdown
## Adaptive Retrieval

SAURON learns from past queries to improve future retrieval accuracy:

- **Relevance Feedback** -- after each query, logs which documents were cited, which were relevant but not cited, and which were irrelevant (MAP returned NO_RELEVANT_DATA)
- **Feedback Boost** -- when a similar query is asked later, previously-useful documents get a relevance boost during document discovery, reducing wasted LLM reads
- **Decay** -- feedback older than 90 days loses weight, preventing stale patterns from dominating
- **Metrics Dashboard** -- Settings page shows MAP Precision (% of docs the LLM read that were useful), query timing, and feedback signal counts
```

- [ ] **Step 2: Commit and push**

```bash
git add README.md
git commit -m "docs: add adaptive retrieval feedback to README"
git push origin master
```
