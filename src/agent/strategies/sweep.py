import logging
from src.agent.state import AgentState
from src.ingestion.embedder import embed_query
from src.retrieval.models import RetrievedChunk
from src.retrieval.vector_store import VectorStore

logger = logging.getLogger(__name__)


def retrieve_sweep(state: AgentState, vector_store: VectorStore, top_k: int = 50) -> dict:
    """Exhaustive sweep uses xlarge chunks to capture full sections,
    then finds relevant documents and retrieves all their large chunks."""
    question = state["question"]
    user_groups = state["user_groups"]

    query_vector = embed_query(question)

    # Step 1: Search xlarge chunks to find relevant documents quickly
    initial_results = vector_store.hybrid_search(
        vector=query_vector, text_query=question,
        user_groups=user_groups, top_k=top_k, tier="xlarge",
    )

    # Identify unique relevant doc_ids
    relevant_doc_ids = set()
    for chunk in initial_results:
        relevant_doc_ids.add(chunk.metadata.doc_id)

    logger.info(f"Sweep: found {len(relevant_doc_ids)} relevant documents from xlarge search")

    # Step 2: Retrieve all large chunks from each relevant document
    all_chunks: list[RetrievedChunk] = []
    for doc_id in relevant_doc_ids:
        doc_chunks = vector_store.get_chunks_by_doc(doc_id)
        # Filter to large tier only — avoids duplicate content from smaller tiers
        large_chunks = [c for c in doc_chunks if c.metadata.chunk_size_tier == "large"]
        if large_chunks:
            all_chunks.extend(large_chunks)
        else:
            # Fallback if no large chunks (old data without tiers)
            all_chunks.extend(doc_chunks)

    logger.info(f"Sweep: retrieved {len(all_chunks)} large chunks from {len(relevant_doc_ids)} documents")

    all_chunks.sort(key=lambda c: (c.metadata.doc_id, c.metadata.chunk_index))

    return {
        "retrieved_chunks": all_chunks,
        "retrieval_attempts": state.get("retrieval_attempts", 0) + 1,
    }
