import asyncio
import logging
import re
from src.agent.state import AgentState
from src.ingestion.embedder import embed_query
from src.retrieval.models import RetrievedChunk
from src.retrieval.vector_store import VectorStore

logger = logging.getLogger(__name__)


async def retrieve_sweep(state: AgentState, vector_store: VectorStore, top_k: int = 50) -> dict:
    """Exhaustive sweep: find relevant documents, then retrieve all their chunks.

    For date-specific queries, also filters by filename to narrow down results.
    """
    question = state["question"]
    user_groups = state["user_groups"]

    query_vector = await asyncio.to_thread(embed_query, question)

    # Step 1: Check if question references a specific date — use filename filter
    date_filter_docs = _extract_date_filter(question, vector_store)

    if date_filter_docs:
        # Date-specific: only retrieve chunks from date-matched documents
        relevant_doc_ids = date_filter_docs
        logger.info(f"Sweep: date filter matched {len(relevant_doc_ids)} documents")
    else:
        # General sweep: find relevant documents via search
        initial_results = vector_store.hybrid_search(
            vector=query_vector, text_query=question,
            user_groups=user_groups, top_k=top_k, tier="xlarge",
        )
        relevant_doc_ids = list({chunk.metadata.doc_id for chunk in initial_results})
        logger.info(f"Sweep: found {len(relevant_doc_ids)} relevant documents from xlarge search")

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
    """If the question mentions a specific date, find documents matching that date by filename."""
    # Match patterns like "Jan 2, 2026", "January 2, 2026", "Jan. 2", "01-02", "1/2/2026"
    month_names = {
        'jan': '01', 'january': '01', 'feb': '02', 'february': '02',
        'mar': '03', 'march': '03', 'apr': '04', 'april': '04',
        'may': '05', 'jun': '06', 'june': '06', 'jul': '07', 'july': '07',
        'aug': '08', 'august': '08', 'sep': '09', 'september': '09',
        'oct': '10', 'october': '10', 'nov': '11', 'november': '11',
        'dec': '12', 'december': '12',
    }

    q_lower = question.lower()

    # Try "Jan 2, 2026" or "January 2, 2026" format
    m = re.search(r'(jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:tember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)\w*\.?\s+(\d{1,2})', q_lower)
    if m:
        month = month_names.get(m.group(1)[:3], '')
        day = m.group(2).zfill(2)
        if month and day:
            date_str = f"-{month}-{day}"  # matches "2026-01-02" in filename
            try:
                results = vector_store.table.search().where(f"chunk_size_tier = 'large'").limit(500).to_list()
                matched_ids = set()
                for r in results:
                    if date_str in r.get('filename', ''):
                        matched_ids.add(r['doc_id'])
                if matched_ids:
                    return list(matched_ids)
            except Exception:
                pass

    return None
