from src.ingestion.embedder import embed_query
from src.retrieval.models import RetrievedChunk
from src.retrieval.vector_store import VectorStore


def retrieve(query, user_groups, vector_store, top_k=8, min_score=0.0):
    query_vector = embed_query(query)
    results = vector_store.search(vector=query_vector, user_groups=user_groups, top_k=top_k)
    if min_score > 0:
        results = [r for r in results if r.score >= min_score]
    return results
