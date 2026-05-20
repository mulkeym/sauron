"""Pseudo-Relevance Feedback: expand queries using terms from top results."""
import logging
from collections import Counter

from src.config import settings

logger = logging.getLogger(__name__)


async def expand_query_with_prf(
    question: str,
    query_vector: list[float],
    user_groups: list[str],
    vector_store,
    doc_ids: list[str] | None = None,
) -> tuple[str, list[float]]:
    """Expand a query using terms from top initial results.

    Returns (expanded_query_text, expanded_query_vector).
    If PRF is disabled or fails, returns the original query and vector.
    """
    if not settings.prf_enabled:
        return question, query_vector

    try:
        # Step 1: Fast initial search for top results
        top_results = vector_store.search(
            vector=query_vector, user_groups=user_groups,
            top_k=settings.prf_top_k, tier="summary", doc_ids=doc_ids,
        )
        if not top_results:
            top_results = vector_store.search(
                vector=query_vector, user_groups=user_groups,
                top_k=settings.prf_top_k, tier="xlarge", doc_ids=doc_ids,
            )

        if not top_results:
            return question, query_vector

        # Step 2: Extract key terms from top results' metadata
        top_doc_ids = list({c.metadata.doc_id for c in top_results})
        terms = await _extract_terms_from_docs(top_doc_ids)

        if not terms:
            return question, query_vector

        # Step 3: Build expanded query
        q_lower = question.lower()
        new_terms = [t for t in terms if t.lower() not in q_lower][:settings.prf_max_terms]

        if not new_terms:
            return question, query_vector

        expanded = f"{question} {' '.join(new_terms)}"
        logger.info(f"PRF expanded query with {len(new_terms)} terms: {', '.join(new_terms)}")

        # Step 4: Re-embed the expanded query
        from src.ingestion.embedder import embed_query
        import asyncio
        expanded_vector = await asyncio.to_thread(embed_query, expanded)

        return expanded, expanded_vector

    except Exception as e:
        logger.warning(f"PRF failed, using original query: {e}")
        return question, query_vector


async def _extract_terms_from_docs(doc_ids: list[str]) -> list[str]:
    """Extract the most common metadata terms across a set of documents."""
    try:
        from src.api.routes_ingest import get_metadata_store
        store = get_metadata_store()
    except Exception:
        return []

    term_counter = Counter()
    fields_to_extract = ["entities", "organizations", "topics", "identifiers", "people", "locations"]

    for doc_id in doc_ids:
        doc = await store.get_document(doc_id)
        if not doc:
            continue
        meta = getattr(doc, 'metadata_tags', {}) or {}
        for field in fields_to_extract:
            for val in meta.get(field, []):
                if val and len(val) > 2:
                    term_counter[val] += 1

    return [term for term, count in term_counter.most_common(settings.prf_max_terms * 2)]
