import asyncio
from src.agent.state import AgentState
from src.agent.strategies.analytical import retrieve_analytical
from src.db.schema_registry import SchemaRegistry
from src.retrieval.feedback import get_feedback_boosts, apply_feedback_boosts_to_chunks
from src.retrieval.models import RetrievedChunk
from src.retrieval.vector_store import VectorStore


async def retrieve_cross_reference(
    state: AgentState,
    vector_store: VectorStore,
    schema_registry: SchemaRegistry,
) -> dict:
    question = state["question"]
    user_groups = state["user_groups"]
    doc_ids = state.get("allowed_doc_ids")
    sub_tasks = state.get("sub_tasks", [question])

    # Embed all sub-tasks in one batch (model isn't thread-safe for concurrent calls)
    from src.ingestion.embedder import embed_texts
    task_vectors = await asyncio.to_thread(embed_texts, sub_tasks, "query")

    # Search in parallel (LanceDB handles concurrent reads)
    async def search_task(task, vector):
        return vector_store.hybrid_search_reranked(
            vector=vector, text_query=task,
            user_groups=user_groups, top_k=30, tier="medium", doc_ids=doc_ids,
        )

    all_results = await asyncio.gather(
        *[search_task(t, v) for t, v in zip(sub_tasks, task_vectors)]
    )

    seen = set()
    unique_chunks = []
    for task_chunks in all_results:
        for chunk in task_chunks:
            key = (chunk.metadata.doc_id, chunk.metadata.chunk_index)
            if key not in seen:
                seen.add(key)
                unique_chunks.append(chunk)

    feedback_boosts = {}
    try:
        feedback_boosts = await get_feedback_boosts(task_vectors[0], user_groups)
        unique_chunks = apply_feedback_boosts_to_chunks(unique_chunks, feedback_boosts)
    except Exception:
        pass

    sql_results = []
    structured_trace = None
    has_schemas = len(schema_registry.list_for_user(user_groups)) > 0
    if has_schemas:
        analytical_result = await retrieve_analytical(state, vector_store=vector_store, schema_registry=schema_registry)
        sql_results = analytical_result.get("sql_results", [])
        structured_trace = analytical_result.get("structured_trace")

    unique_chunks = vector_store.expand_window(unique_chunks, window=2)
    result = {
        "retrieved_chunks": unique_chunks,
        "sql_results": sql_results,
        "retrieval_attempts": state.get("retrieval_attempts", 0) + 1,
        "feedback_boosts": feedback_boosts,
    }
    if structured_trace:
        result["structured_trace"] = structured_trace
    return result
