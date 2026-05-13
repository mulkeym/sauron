import asyncio
import logging
from src.agent.state import AgentState
from src.ingestion.embedder import embed_query
from src.retrieval.models import RetrievedChunk
from src.retrieval.vector_store import VectorStore

logger = logging.getLogger(__name__)


async def retrieve_sweep(state: AgentState, vector_store: VectorStore, top_k: int = 50) -> dict:
    """Exhaustive sweep uses xlarge chunks to capture full sections,
    then finds relevant documents and retrieves all their large chunks in parallel."""
    question = state["question"]
    user_groups = state["user_groups"]

    query_vector = await asyncio.to_thread(embed_query, question)

    # Step 1: Search xlarge chunks to find relevant documents quickly
    initial_results = vector_store.hybrid_search(
        vector=query_vector, text_query=question,
        user_groups=user_groups, top_k=top_k, tier="xlarge",
    )

    # Identify unique relevant doc_ids
    relevant_doc_ids = list({chunk.metadata.doc_id for chunk in initial_results})
    logger.info(f"Sweep: found {len(relevant_doc_ids)} relevant documents from xlarge search")

    # Step 2: Retrieve large-tier chunks from ALL relevant documents in parallel
    async def get_doc(doc_id):
        chunks = await asyncio.to_thread(vector_store.get_chunks_by_doc, doc_id, 200, "large")
        if not chunks:
            chunks = await asyncio.to_thread(vector_store.get_chunks_by_doc, doc_id)
        return chunks

    doc_results = await asyncio.gather(*[get_doc(did) for did in relevant_doc_ids])

    all_chunks: list[RetrievedChunk] = []
    for doc_chunks in doc_results:
        all_chunks.extend(doc_chunks)

    logger.info(f"Sweep: retrieved {len(all_chunks)} large chunks from {len(relevant_doc_ids)} documents (parallel)")

    all_chunks.sort(key=lambda c: (c.metadata.doc_id, c.metadata.chunk_index))

    return {
        "retrieved_chunks": all_chunks,
        "retrieval_attempts": state.get("retrieval_attempts", 0) + 1,
    }
