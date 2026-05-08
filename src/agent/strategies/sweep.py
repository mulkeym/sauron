from src.agent.state import AgentState
from src.ingestion.embedder import embed_query
from src.retrieval.models import RetrievedChunk
from src.retrieval.vector_store import VectorStore

def retrieve_sweep(state: AgentState, vector_store: VectorStore, top_k: int = 50) -> dict:
    question = state["question"]
    user_groups = state["user_groups"]
    all_chunks: list[RetrievedChunk] = []

    query_vector = embed_query(question)
    semantic_results = vector_store.search(vector=query_vector, user_groups=user_groups, top_k=top_k)
    all_chunks.extend(semantic_results)

    metadata_results = vector_store.search(vector=query_vector, user_groups=user_groups, top_k=top_k * 2)
    all_chunks.extend(metadata_results)

    seen = set()
    unique_chunks = []
    for chunk in all_chunks:
        key = (chunk.metadata.doc_id, chunk.metadata.chunk_index)
        if key not in seen:
            seen.add(key)
            unique_chunks.append(chunk)
    unique_chunks.sort(key=lambda c: c.score, reverse=True)

    return {
        "retrieved_chunks": unique_chunks,
        "retrieval_attempts": state.get("retrieval_attempts", 0) + 1,
    }
