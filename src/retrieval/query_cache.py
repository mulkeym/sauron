from __future__ import annotations
"""Opt-in answer reuse, bounded by source/access/configuration revisions and expiry.

Exact mode matches the original question; semantic mode additionally requires
an affirmative applicability judgment. All uncertain checks fall back to retrieval.
"""
import asyncio
import json
import logging
import time
import uuid
from dataclasses import dataclass

import pyarrow as pa

from src.config import settings
from src.ingestion.embedder import embed_query

logger = logging.getLogger(__name__)

_cache_table = None


def _get_cache_table():
    """Get or create the query cache LanceDB table."""
    global _cache_table
    if _cache_table is not None:
        return _cache_table

    import lancedb
    db = lancedb.connect(settings.lancedb_path)

    try:
        _cache_table = db.open_table("query_cache")
    except Exception:
        from src.retrieval.vector_store import _detect_vector_size
        dim = _detect_vector_size()
        schema = pa.schema([
            pa.field("id", pa.string()),
            pa.field("query_text", pa.string()),
            pa.field("vector", pa.list_(pa.float32(), dim)),
            pa.field("answer", pa.string()),
            pa.field("citations_json", pa.string()),
            pa.field("acl_groups_json", pa.string()),  # sorted JSON array of groups
            pa.field("source_doc_ids_json", pa.string()),
            pa.field("doc_count", pa.int32()),
            pa.field("query_type", pa.string()),
            pa.field("created_at", pa.float64()),
        ])
        _cache_table = db.create_table("query_cache", schema=schema)
        logger.info("Created query_cache table")

    # Old rows have no revision and are intentionally never reused.
    if "scope_revision" not in _cache_table.schema.names:
        _cache_table.add_columns({"scope_revision": "CAST(NULL AS STRING)"})
    return _cache_table


def _acl_key(user_groups: list[str]) -> str:
    """Normalize ACL groups to a comparable string."""
    return json.dumps(sorted(set(user_groups)))


def cache_lookup(query_vector: list[float], user_groups: list[str],
                 similarity_threshold: float = 0.92, *, scope_revision: str = "",
                 query_text: str = "") -> dict | None:
    """Search cache for a semantically similar query with matching ACL.

    Returns cached result dict or None if no hit.
    """
    if settings.query_cache_mode == "off" or not scope_revision or not user_groups:
        return None
    table = _get_cache_table()
    if table.count_rows() == 0:
        return None

    acl_key = _acl_key(user_groups)

    try:
        safe_revision = scope_revision.replace("'", "''")
        search = table.search(query_vector).where(f"scope_revision = '{safe_revision}'", prefilter=True)
        if settings.query_cache_mode == "exact":
            exact = query_text.strip().replace("'", "''")
            search = search.where(f"scope_revision = '{safe_revision}' AND query_text = '{exact}'", prefilter=True)
        results = search.limit(5).to_list()

        for row in results:
            # Check similarity
            score = 1.0 / (1.0 + row.get("_distance", 999))
            if score < similarity_threshold:
                continue

            # Check ACL match
            if row.get("acl_groups_json", "") != acl_key:
                continue

            cached_at = row.get("created_at", 0)
            if not 0 <= time.time() - cached_at <= settings.query_cache_ttl_seconds:
                continue
            if row.get("scope_revision") != scope_revision:
                continue
            source_doc_ids = json.loads(row.get("source_doc_ids_json", "[]"))
            if not source_doc_ids:
                continue

            logger.info(f"Cache hit: \"{row['query_text'][:60]}\" (similarity: {score:.3f})")
            return {
                "answer": row["answer"],
                "citations": json.loads(row.get("citations_json", "[]")),
                "query_type": row.get("query_type", ""),
                "source_doc_ids": source_doc_ids,
                "cached_at": cached_at,
                "cached_query": row["query_text"],
            }

    except Exception as e:
        logger.warning(f"Cache lookup failed: {e}")

    return None


async def cache_judge(original_query: str, new_query: str, cached_answer: str) -> dict:
    """Ask LLM to judge if a cached result is applicable to the new query.

    Returns {"applicable": bool, "confidence": float, "reason": str}
    """
    import asyncio
    from src.generation.llm_client import generate, parse_json_response

    prompt = f"""You are judging whether a cached answer is applicable to a new question.

Cached question: "{original_query}"
New question: "{new_query}"

First 500 chars of cached answer:
{cached_answer[:500]}

Is the cached answer applicable to the new question? Consider:
- Do they ask about the same topic/entities?
- Would the cached answer satisfy the new question?
- Are there important differences that make the cache invalid?

Respond with ONLY JSON:
{{"applicable": true/false, "confidence": 0.0-1.0, "reason": "brief explanation"}}"""

    try:
        response = await asyncio.to_thread(
            generate,
            system_prompt="You judge cache applicability. Return ONLY JSON.",
            user_prompt=prompt,
            temperature=0.0,
            max_tokens=1024,
        )
        parsed = parse_json_response(response)
        return {
            "applicable": parsed.get("applicable", False),
            "confidence": parsed.get("confidence", 0.0),
            "reason": parsed.get("reason", ""),
        }
    except Exception as e:
        logger.warning(f"Cache judge failed: {e}")
        return {"applicable": False, "confidence": 0.0, "reason": "Judge unavailable; retrieve fresh evidence"}


@dataclass
class CacheDecision:
    """Outcome of the shared cache lookup+judge sequence.
    Both the API (agent_query) and the admin playground consume this so the
    cache decision lives in exactly one place."""
    scope_revision: str = ""
    query_vector: list | None = None   # reuse for cache_store; None if embed failed
    hit: bool = False                  # cache_lookup found a vector+ACL+freshness match
    accepted: bool = False             # hit AND judge applicable -> serve the cache
    cached: dict | None = None         # the cache_lookup result
    judgment: dict | None = None       # {applicable, confidence, reason}; None if no hit
    cache_time: float = 0.0            # seconds: embed + lookup
    judge_time: float = 0.0            # seconds: judge (0 if no hit)


async def judged_cache_lookup(question: str, user_groups: list,
                              *, skip_cache: bool = False, metadata_store=None,
                              dataset_id=0, allowed_doc_ids=None, mode="full", answer_profile=None) -> CacheDecision:
    """Return fresh retrieval whenever cache scope or applicability is uncertain."""
    d = CacheDecision()
    t0 = time.time()
    if settings.query_cache_mode == "off" or not user_groups:
        return d
    try:
        from src.retrieval.query_scope import resolve_query_scope
        scope = await resolve_query_scope(user_groups, metadata_store,
            dataset_id=dataset_id, allowed_doc_ids=allowed_doc_ids, mode=mode, answer_profile=answer_profile)
        d.scope_revision = scope.revision
        if not scope.doc_ids:
            return d
        d.query_vector = await asyncio.to_thread(embed_query, question)
    except Exception as e:
        logger.warning(f"Cache embed failed: {e}")
        d.cache_time = round(time.time() - t0, 2)
        return d

    if skip_cache:
        d.cache_time = round(time.time() - t0, 2)
        return d

    try:
        d.cached = await asyncio.to_thread(cache_lookup, d.query_vector, user_groups,
            scope_revision=d.scope_revision, query_text=question)
    except Exception:
        logger.warning("Cache unavailable; retrieving fresh evidence")
        return d
    d.cache_time = round(time.time() - t0, 2)
    if not d.cached:
        return d

    d.hit = True
    if settings.query_cache_mode == "exact":
        d.accepted = True
        d.judgment = {"applicable": True, "confidence": 1.0, "reason": "Exact question and scope revision match"}
        return d
    tj = time.time()
    d.judgment = await cache_judge(
        original_query=d.cached.get("cached_query", ""),
        new_query=question,
        cached_answer=d.cached.get("answer", ""),
    )
    d.judge_time = round(time.time() - tj, 2)
    confidence = d.judgment.get("confidence", 0)
    d.accepted = (d.judgment.get("applicable") is True
                  and isinstance(confidence, (int, float))
                  and settings.query_cache_min_confidence <= confidence <= 1.0)
    return d


def cache_store(query_text: str, query_vector: list[float], answer: str,
                citations: list[dict], user_groups: list[str],
                source_doc_ids: list[str], query_type: str = "", *, scope_revision: str = ""):
    """Store only answers with evidence and a checked catalog revision."""
    if settings.query_cache_mode == "off" or not scope_revision or not source_doc_ids or not citations:
        return
    if any(c.get("source_kind") != "document" for c in citations):
        return  # Derived graph/SQL summaries need their own freshness tracking.
    table = _get_cache_table()

    record = {
        "id": str(uuid.uuid4()),
        "query_text": query_text.strip(),
        "scope_revision": scope_revision,
        "vector": query_vector,
        "answer": answer,
        "citations_json": json.dumps(citations),
        "acl_groups_json": _acl_key(user_groups),
        "source_doc_ids_json": json.dumps(source_doc_ids),
        "doc_count": len(source_doc_ids),
        "query_type": query_type,
        "created_at": time.time(),
    }

    try:
        table.add([record])
        logger.info(f"Cached result for: \"{query_text[:60]}\"")
    except Exception as e:
        logger.warning(f"Cache store failed: {e}")


def cache_purge() -> int:
    """Purge all cached query results. Returns count of entries deleted."""
    global _cache_table
    table = _get_cache_table()
    count = table.count_rows()
    if count > 0:
        import lancedb
        db = lancedb.connect(settings.lancedb_path)
        db.drop_table("query_cache")
        _cache_table = None
        logger.info(f"Purged {count} cached query results")
    return count


def cache_stats() -> dict:
    """Get cache statistics."""
    table = _get_cache_table()
    count = table.count_rows()
    return {"entries": count}
