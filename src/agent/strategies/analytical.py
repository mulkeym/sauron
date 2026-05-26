import asyncio

from src.agent.state import AgentState
from src.db.schema_registry import SchemaRegistry
from src.agent.strategies.structured import structured_sql_rows


async def retrieve_analytical(state: AgentState, vector_store, schema_registry: SchemaRegistry) -> dict:
    question = state["question"]
    user_groups = state["user_groups"]

    schemas = schema_registry.list_for_user(user_groups)
    if not schemas:
        from src.agent.strategies.map_reduce import retrieve_map_reduce
        return await retrieve_map_reduce(state, vector_store=vector_store)

    try:
        rows = await asyncio.to_thread(structured_sql_rows, question, schemas)
    except Exception:
        # No usable SQL (LLM error), blocked SQL, or execution error -> comprehensive retrieval.
        from src.agent.strategies.map_reduce import retrieve_map_reduce
        return await retrieve_map_reduce(state, vector_store=vector_store)

    return {
        "retrieved_chunks": [],
        "sql_results": rows,
        "retrieval_attempts": state.get("retrieval_attempts", 0) + 1,
    }
