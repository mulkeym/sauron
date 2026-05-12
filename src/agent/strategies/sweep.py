import logging
from src.agent.state import AgentState
from src.ingestion.embedder import embed_query
from src.retrieval.models import RetrievedChunk
from src.retrieval.vector_store import VectorStore

logger = logging.getLogger(__name__)


def retrieve_sweep(state: AgentState, vector_store: VectorStore, top_k: int = 50) -> dict:
    """Exhaustive sweep: find relevant documents, then retrieve ALL chunks from them."""
    question = state["question"]
    user_groups = state["user_groups"]

    query_vector = embed_query(question)

    # Step 1: Use hybrid search to identify which documents are relevant
    initial_results = vector_store.hybrid_search(
        vector=query_vector, text_query=question,
        user_groups=user_groups, top_k=top_k,
    )

    # Identify unique relevant doc_ids
    relevant_doc_ids = set()
    for chunk in initial_results:
        relevant_doc_ids.add(chunk.metadata.doc_id)

    logger.info(f"Sweep: found {len(relevant_doc_ids)} relevant documents")

    # Step 2: Retrieve ALL chunks from each relevant document
    all_chunks: list[RetrievedChunk] = []
    for doc_id in relevant_doc_ids:
        doc_chunks = vector_store.get_chunks_by_doc(doc_id)
        all_chunks.extend(doc_chunks)

    logger.info(f"Sweep: retrieved {len(all_chunks)} total chunks from {len(relevant_doc_ids)} documents")

    # Sort by document order
    all_chunks.sort(key=lambda c: (c.metadata.doc_id, c.metadata.chunk_index))

    return {
        "retrieved_chunks": all_chunks,
        "retrieval_attempts": state.get("retrieval_attempts", 0) + 1,
    }
