from __future__ import annotations
"""Query result cache — stores previous answers for semantic reuse.

Cached results are matched by embedding similarity (not exact query match),
so "army contracts" and "What did the army award?" can share a cache entry.

ACL-aware: cached results are only returned if the user's groups match
the groups that were used when the cache entry was created.
"""
import asyncio
import json
import logging
import time
import uuid
from dataclasses import dataclass

import numpy as np
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

    return _cache_table


def _has_new_related_docs(source_doc_ids: list[str], cached_at: float) -> bool:
    """Check if new documents were added in the same categories since the cache was built."""
    import asyncio
    from datetime import datetime, timezone

    if not source_doc_ids:
        return False

    try:
        from src.db.metadata import MetadataStore

        async def _check():
            store = MetadataStore()
            await store.init()

            # Get categories of the cached source documents
            source_categories = set()
            for doc_id in source_doc_ids:
                doc = await store.get_document(doc_id)
                if doc and doc.category:
                    source_categories.add(doc.category)

            if not source_categories:
                return False

            # Check if any documents in those categories were created after the cache
            cached_time = datetime.fromtimestamp(cached_at, tz=timezone.utc)
            all_docs = await store.list_documents(None)
            for doc in all_docs:
                if doc.category in source_categories and doc.doc_id not in source_doc_ids:
                    doc_time = doc.created_at
                    if doc_time is None:
                        continue
                    # Normalize: make naive datetimes UTC-aware for comparison
                    if doc_time.tzinfo is None:
                        doc_time = doc_time.replace(tzinfo=timezone.utc)
                    if doc_time > cached_time:
                        return True
            return False

        # Run async check — handle both async and sync contexts
        try:
            loop = asyncio.get_running_loop()
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as pool:
                return pool.submit(asyncio.run, _check()).result()
        except RuntimeError:
            return asyncio.run(_check())

    except Exception as e:
        logger.warning(f"New doc check failed: {e}")
        return False  # on error, use cache (conservative)


def _acl_key(user_groups: list[str]) -> str:
    """Normalize ACL groups to a comparable string."""
    return json.dumps(sorted(set(user_groups)))


def cache_lookup(query_vector: list[float], user_groups: list[str],
                 similarity_threshold: float = 0.92) -> dict | None:
    """Search cache for a semantically similar query with matching ACL.

    Returns cached result dict or None if no hit.
    """
    table = _get_cache_table()
    if table.count_rows() == 0:
        return None

    acl_key = _acl_key(user_groups)

    try:
        results = table.search(query_vector).limit(5).to_list()

        for row in results:
            # Check similarity
            score = 1.0 / (1.0 + row.get("_distance", 999))
            if score < similarity_threshold:
                continue

            # Check ACL match
            if row.get("acl_groups_json", "") != acl_key:
                continue

            # Check if new related documents have been added since cache was built
            cached_at = row.get("created_at", 0)
            source_doc_ids = json.loads(row.get("source_doc_ids_json", "[]"))
            if _has_new_related_docs(source_doc_ids, cached_at):
                logger.info(f"Cache stale: new related documents added since {time.strftime('%Y-%m-%d %H:%M', time.localtime(cached_at))}")
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
        return {"applicable": True, "confidence": 0.5, "reason": "Judge unavailable, using cache"}


@dataclass
class CacheDecision:
    """Outcome of the shared cache lookup+judge sequence.
    Both the API (agent_query) and the admin playground consume this so the
    cache decision lives in exactly one place."""
    query_vector: list | None = None   # reuse for cache_store; None if embed failed
    hit: bool = False                  # cache_lookup found a vector+ACL+freshness match
    accepted: bool = False             # hit AND judge applicable -> serve the cache
    cached: dict | None = None         # the cache_lookup result
    judgment: dict | None = None       # {applicable, confidence, reason}; None if no hit
    cache_time: float = 0.0            # seconds: embed + lookup
    judge_time: float = 0.0            # seconds: judge (0 if no hit)


async def judged_cache_lookup(question: str, user_groups: list,
                              *, skip_cache: bool = False) -> CacheDecision:
    """Embed the question, look up the cache, and (on a hit) run the LLM
    applicability judge. Single source of truth for "is there a usable cache
    hit". Fail-open throughout: embed failure -> no hit (and no vector to store);
    cache_judge already returns applicable=True on its own error."""
    d = CacheDecision()
    t0 = time.time()
    try:
        d.query_vector = await asyncio.to_thread(embed_query, question)
    except Exception as e:
        logger.warning(f"Cache embed failed: {e}")
        d.cache_time = round(time.time() - t0, 2)
        return d

    if skip_cache:
        d.cache_time = round(time.time() - t0, 2)
        return d

    d.cached = cache_lookup(d.query_vector, user_groups)
    d.cache_time = round(time.time() - t0, 2)
    if not d.cached:
        return d

    d.hit = True
    tj = time.time()
    d.judgment = await cache_judge(
        original_query=d.cached.get("cached_query", ""),
        new_query=question,
        cached_answer=d.cached.get("answer", ""),
    )
    d.judge_time = round(time.time() - tj, 2)
    d.accepted = bool(d.judgment.get("applicable", False))
    return d


def cache_store(query_text: str, query_vector: list[float], answer: str,
                citations: list[dict], user_groups: list[str],
                source_doc_ids: list[str], query_type: str = ""):
    """Store a query result in the cache."""
    table = _get_cache_table()

    record = {
        "id": str(uuid.uuid4()),
        "query_text": query_text,
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
