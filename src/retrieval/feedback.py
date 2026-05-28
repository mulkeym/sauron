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
    return np.array(vec, dtype=np.float32).tobytes()


def _deserialize_vector(blob: bytes) -> np.ndarray:
    if not blob:
        return np.array([], dtype=np.float32)
    return np.frombuffer(blob, dtype=np.float32)


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    if a.size == 0 or b.size == 0:
        return 0.0
    dot = np.dot(a, b)
    norm = np.linalg.norm(a) * np.linalg.norm(b)
    return float(dot / norm) if norm > 0 else 0.0


def _query_hash(text: str) -> str:
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
            for doc_id in cited_doc_ids:
                session.add(QueryFeedback(
                    query_hash=qhash, query_text=query_text[:500],
                    query_vector_blob=vec_blob, query_type=query_type,
                    doc_id=doc_id, filename=doc_filenames.get(doc_id, ""),
                    relevance_score=doc_scores.get(doc_id, 0.0),
                    was_cited=True, was_in_map_reduce=True,
                    user_groups=user_groups,
                ))
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
    query_similarities: dict[str, float] = {}

    for fb in all_feedback:
        # Compute similarity per unique query (cache by hash)
        if fb.query_hash not in query_similarities:
            fb_vec = _deserialize_vector(fb.query_vector_blob)
            if fb_vec.size == 0:
                query_similarities[fb.query_hash] = 0.0
                continue
            query_similarities[fb.query_hash] = _cosine_similarity(query_vec, fb_vec)

        sim = query_similarities[fb.query_hash]
        if sim < settings.feedback_similarity_threshold:
            continue

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

        boosts[fb.doc_id] = boosts.get(fb.doc_id, 0.0) + boost

    if boosts:
        pos = sum(1 for v in boosts.values() if v > 0)
        neg = sum(1 for v in boosts.values() if v < 0)
        similar_queries = sum(1 for s in query_similarities.values() if s >= settings.feedback_similarity_threshold)
        logger.info(f"Feedback boosts: {pos} boosted, {neg} penalized (from {similar_queries} similar past queries)")

    return boosts


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
