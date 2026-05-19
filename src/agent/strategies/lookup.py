from src.agent.state import AgentState
from src.ingestion.embedder import embed_query
from src.retrieval.vector_store import VectorStore

def retrieve_lookup(state: AgentState, vector_store: VectorStore, top_k: int = 30) -> dict:
    """Lookup uses medium chunks — balanced precision and context."""
    question = state["question"]
    user_groups = state["user_groups"]
    doc_ids = state.get("allowed_doc_ids")
    query_vector = embed_query(question)

    # Check for date-specific query — restrict to date-matched docs if found
    from src.agent.strategies.sweep import _extract_date_filter
    date_doc_ids = _extract_date_filter(question, vector_store)
    if date_doc_ids:
        # Override doc_ids to only search date-matched documents
        doc_ids = date_doc_ids

    chunks = vector_store.hybrid_search_reranked(vector=query_vector, text_query=question, user_groups=user_groups, top_k=top_k, tier="medium", doc_ids=doc_ids)
    chunks = vector_store.expand_window(chunks, window=2)
    return {
        "retrieved_chunks": chunks,
        "retrieval_attempts": state.get("retrieval_attempts", 0) + 1,
    }
