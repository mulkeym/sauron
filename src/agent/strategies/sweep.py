import logging
from src.agent.state import AgentState
from src.ingestion.embedder import embed_query
from src.retrieval.models import RetrievedChunk
from src.retrieval.vector_store import VectorStore

logger = logging.getLogger(__name__)


def retrieve_sweep(state: AgentState, vector_store: VectorStore, top_k: int = 50) -> dict:
    """Exhaustive sweep: find relevant documents, then retrieve ALL chunks from them.

    Unlike lookup (which finds the best chunks), sweep finds which DOCUMENTS
    are relevant and then retrieves all their content. This ensures we don't
    miss contractors listed 20 chunks after the "ARMY" header.
    """
    question = state["question"]
    user_groups = state["user_groups"]

    query_vector = embed_query(question)

    # Step 1: Use hybrid search to identify which documents are relevant
    initial_results = vector_store.hybrid_search(
        vector=query_vector, text_query=question,
        user_groups=user_groups, top_k=top_k,
    )

    # Identify unique relevant doc_ids (any doc that had a matching chunk)
    relevant_doc_ids = set()
    for chunk in initial_results:
        relevant_doc_ids.add(chunk.metadata.doc_id)

    logger.info(f"Sweep: found {len(relevant_doc_ids)} relevant documents from initial search")

    # Step 2: Retrieve ALL chunks from each relevant document
    from qdrant_client.models import Filter, FieldCondition, MatchValue
    all_chunks: list[RetrievedChunk] = []
    for doc_id in relevant_doc_ids:
        try:
            results = vector_store.client.scroll(
                collection_name=vector_store.collection,
                scroll_filter=Filter(must=[
                    FieldCondition(key="doc_id", match=MatchValue(value=doc_id)),
                ]),
                limit=200,
                with_payload=True,
            )
            for point in results[0]:
                payload = dict(point.payload)
                text = payload.pop("text", "")
                all_chunks.append(RetrievedChunk(
                    text=text,
                    score=0.5,  # Uniform score since we're getting all chunks
                    metadata=vector_store._points_to_chunks.__func__  # Can't call, build manually
                ))
        except Exception:
            pass

    # Simpler approach: use scroll results directly
    all_chunks = []
    from src.retrieval.models import ChunkMetadata
    for doc_id in relevant_doc_ids:
        try:
            results = vector_store.client.scroll(
                collection_name=vector_store.collection,
                scroll_filter=Filter(must=[
                    FieldCondition(key="doc_id", match=MatchValue(value=doc_id)),
                ]),
                limit=200,
                with_payload=True,
            )
            for point in results[0]:
                payload = dict(point.payload)
                text = payload.pop("text", "")
                all_chunks.append(RetrievedChunk(
                    text=text, score=0.5,
                    metadata=ChunkMetadata(**payload),
                ))
        except Exception as e:
            logger.warning(f"Failed to retrieve chunks for doc {doc_id}: {e}")

    logger.info(f"Sweep: retrieved {len(all_chunks)} total chunks from {len(relevant_doc_ids)} documents")

    # Sort by document order for coherent reading
    all_chunks.sort(key=lambda c: (c.metadata.doc_id, c.metadata.chunk_index))

    return {
        "retrieved_chunks": all_chunks,
        "retrieval_attempts": state.get("retrieval_attempts", 0) + 1,
    }
