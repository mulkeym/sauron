from src.agent.state import AgentState
from src.agent.strategies.analytical import retrieve_analytical
from src.db.schema_registry import SchemaRegistry
from src.ingestion.embedder import embed_query
from src.retrieval.models import RetrievedChunk
from src.retrieval.vector_store import VectorStore


async def retrieve_cross_reference(
    state: AgentState,
    vector_store: VectorStore,
    schema_registry: SchemaRegistry,
) -> dict:
    question = state["question"]
    user_groups = state["user_groups"]
    sub_tasks = state.get("sub_tasks", [question])
    all_chunks: list[RetrievedChunk] = []

    for task in sub_tasks:
        query_vector = embed_query(task)
        chunks = vector_store.hybrid_search(vector=query_vector, text_query=task, user_groups=user_groups, top_k=30)
        all_chunks.extend(chunks)

    seen = set()
    unique_chunks = []
    for chunk in all_chunks:
        key = (chunk.metadata.doc_id, chunk.metadata.chunk_index)
        if key not in seen:
            seen.add(key)
            unique_chunks.append(chunk)

    sql_results = []
    has_schemas = len(schema_registry.list_for_user(user_groups)) > 0
    if has_schemas:
        analytical_result = await retrieve_analytical(state, vector_store=vector_store, schema_registry=schema_registry)
        sql_results = analytical_result.get("sql_results", [])

    return {
        "retrieved_chunks": unique_chunks,
        "sql_results": sql_results,
        "retrieval_attempts": state.get("retrieval_attempts", 0) + 1,
    }
