import logging

from src.agent.state import AgentState
from src.ingestion.embedder import embed_query
from src.retrieval.feedback import get_feedback_boosts_sync, apply_feedback_boosts_to_chunks
from src.retrieval.vector_store import VectorStore

logger = logging.getLogger(__name__)

def retrieve_lookup(state: AgentState, vector_store: VectorStore, top_k: int = 30) -> dict:
    """Lookup uses medium chunks — balanced precision and context."""
    question = state["question"]
    user_groups = state["user_groups"]
    doc_ids = state.get("allowed_doc_ids")
    query_vector = embed_query(question)

    # Check for date-specific query — restrict to date-matched docs if found
    from src.agent.strategies.sweep import _extract_date_filter
    date_doc_ids = _extract_date_filter(question, vector_store, user_groups)
    if date_doc_ids:
        # Override doc_ids to only search date-matched documents
        doc_ids = date_doc_ids

    chunks = vector_store.hybrid_search_reranked(vector=query_vector, text_query=question, user_groups=user_groups, top_k=top_k, tier="medium", doc_ids=doc_ids)

    feedback_boosts = get_feedback_boosts_sync(query_vector, user_groups)
    chunks = apply_feedback_boosts_to_chunks(chunks, feedback_boosts)

    # Score-based filter: drop chunks below 30% of top score after reranking
    # to avoid pulling in loosely-matching documents for targeted lookups
    if chunks:
        top_score = max(c.score for c in chunks)
        score_threshold = top_score * 0.3 if top_score > 0 else 0
        before_count = len(chunks)
        chunks = [c for c in chunks if c.score >= score_threshold]
        if len(chunks) < before_count:
            logger.info(f"Lookup: score cutoff ({score_threshold:.3f}) reduced {before_count} → {len(chunks)} chunks")

    chunks = vector_store.expand_window(chunks, window=2)
    return {
        "retrieved_chunks": chunks,
        "retrieval_attempts": state.get("retrieval_attempts", 0) + 1,
        "feedback_boosts": feedback_boosts,
    }
