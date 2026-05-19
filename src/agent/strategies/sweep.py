from __future__ import annotations
import asyncio
import logging
import re
from src.agent.state import AgentState
from src.ingestion.embedder import embed_query
from src.retrieval.models import RetrievedChunk
from src.retrieval.vector_store import VectorStore

logger = logging.getLogger(__name__)


async def retrieve_sweep(state: AgentState, vector_store: VectorStore, top_k: int = 500) -> dict:
    """Exhaustive sweep: find relevant documents, then retrieve all their chunks.

    For date-specific queries, also filters by filename to narrow down results.
    """
    question = state["question"]
    user_groups = state["user_groups"]
    doc_ids = state.get("allowed_doc_ids")

    query_vector = await asyncio.to_thread(embed_query, question)

    # Step 1: Check if question references a specific date — use filename filter
    date_filter_docs = _extract_date_filter(question, vector_store)

    if date_filter_docs:
        # Date-specific: only retrieve chunks from date-matched documents
        relevant_doc_ids = date_filter_docs
        logger.info(f"Sweep: date filter matched {len(relevant_doc_ids)} documents")
    else:
        # Search summary embeddings first, fall back to xlarge
        initial_results = vector_store.search(
            vector=query_vector, user_groups=user_groups,
            top_k=top_k, tier="summary", doc_ids=doc_ids,
        )
        if not initial_results:
            initial_results = vector_store.hybrid_search(
                vector=query_vector, text_query=question,
                user_groups=user_groups, top_k=top_k, tier="xlarge", doc_ids=doc_ids,
            )
            logger.info("Sweep: no summary embeddings, falling back to xlarge")
        relevant_doc_ids = list({chunk.metadata.doc_id for chunk in initial_results})
        logger.info(f"Sweep: found {len(relevant_doc_ids)} relevant documents from summary search")

    # Step 2: Retrieve large-tier chunks from relevant documents in parallel
    async def get_doc(doc_id):
        chunks = await asyncio.to_thread(vector_store.get_chunks_by_doc, doc_id, 200, "large")
        if not chunks:
            chunks = await asyncio.to_thread(vector_store.get_chunks_by_doc, doc_id)
        return chunks

    doc_results = await asyncio.gather(*[get_doc(did) for did in relevant_doc_ids])

    all_chunks: list[RetrievedChunk] = []
    for doc_chunks in doc_results:
        all_chunks.extend(doc_chunks)

    logger.info(f"Sweep: retrieved {len(all_chunks)} large chunks from {len(relevant_doc_ids)} documents")

    all_chunks.sort(key=lambda c: (c.metadata.doc_id, c.metadata.chunk_index))

    return {
        "retrieved_chunks": all_chunks,
        "retrieval_attempts": state.get("retrieval_attempts", 0) + 1,
    }


def _extract_date_filter(question: str, vector_store: VectorStore) -> list[str] | None:
    """If the question mentions a specific date, find documents with that date in metadata or filename."""
    month_names = {
        'jan': '01', 'january': '01', 'feb': '02', 'february': '02',
        'mar': '03', 'march': '03', 'apr': '04', 'april': '04',
        'may': '05', 'jun': '06', 'june': '06', 'jul': '07', 'july': '07',
        'aug': '08', 'august': '08', 'sep': '09', 'september': '09',
        'oct': '10', 'october': '10', 'nov': '11', 'november': '11',
        'dec': '12', 'december': '12',
    }
    month_words = {v: k for k, v in month_names.items()}  # "01" -> "jan"

    q_lower = question.lower()

    m = re.search(r'(jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:tember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)\w*\.?\s+(\d{1,2})', q_lower)
    if m:
        month = month_names.get(m.group(1)[:3], '')
        day = m.group(2).zfill(2)
        if month and day:
            # Build date patterns to search for in metadata and filenames
            # e.g. "01-30", "Jan 30", "January 30", "1/30", "Jan. 30"
            month_word = m.group(1)[:3].capitalize()
            date_patterns = [
                f"-{month}-{day}",        # 2026-01-30 in filenames
                f"{month}-{day}",          # 01-30
                f"{month}/{day}",          # 01/30
                f"{month_word} {int(day)}", # Jan 30
                f"{month_word}. {int(day)}", # Jan. 30
            ]

            try:
                # Search metadata_tags.dates field across all documents
                from src.api.routes_ingest import get_metadata_store
                import asyncio
                store = get_metadata_store()

                # Run async in sync context
                async def _find_by_metadata():
                    docs = await store.list_documents()
                    matched = set()
                    for doc in docs:
                        # Check metadata dates
                        meta = getattr(doc, 'metadata_tags', {}) or {}
                        doc_dates = meta.get('dates', [])
                        for d in doc_dates:
                            d_lower = d.lower()
                            if any(p.lower() in d_lower for p in date_patterns):
                                matched.add(doc.doc_id)
                                break

                        # Also check filename as fallback
                        if doc.doc_id not in matched:
                            if any(p in doc.filename for p in date_patterns):
                                matched.add(doc.doc_id)
                    return list(matched)

                try:
                    loop = asyncio.get_running_loop()
                    import concurrent.futures
                    with concurrent.futures.ThreadPoolExecutor() as pool:
                        matched_ids = pool.submit(asyncio.run, _find_by_metadata()).result()
                except RuntimeError:
                    matched_ids = asyncio.run(_find_by_metadata())

                if matched_ids:
                    logger.info(f"Date filter: '{month_word} {int(day)}' matched {len(matched_ids)} documents via metadata+filename")
                    return matched_ids
            except Exception:
                pass

    return None
