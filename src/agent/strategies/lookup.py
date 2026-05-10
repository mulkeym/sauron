from src.agent.state import AgentState
from src.ingestion.embedder import embed_query
from src.retrieval.vector_store import VectorStore

def retrieve_lookup(state: AgentState, vector_store: VectorStore, top_k: int = 8) -> dict:
    question = state["question"]
    user_groups = state["user_groups"]
    query_vector = embed_query(question)
    chunks = vector_store.search(vector=query_vector, user_groups=user_groups, top_k=top_k)
    return {
        "retrieved_chunks": chunks,
        "retrieval_attempts": state.get("retrieval_attempts", 0) + 1,
    }
