import asyncio

from src.agent.state import AgentState
from src.db.schema_registry import SchemaRegistry


async def retrieve_analytical(state: AgentState, vector_store, schema_registry: SchemaRegistry) -> dict:
    question = state["question"]
    user_groups = state["user_groups"]

    from src.agent.strategies.structured import run_structured_lookup, StructuredLookupTrace

    schemas = schema_registry.list_for_user(user_groups)
    if not schemas:
        from src.agent.strategies.map_reduce import retrieve_map_reduce
        result = await retrieve_map_reduce(state, vector_store=vector_store)
        result["structured_trace"] = StructuredLookupTrace(
            query_type="analytical", gate=None, status="skipped",
            skip_reason="no registered tables").to_dict()
        return result

    trace = await asyncio.to_thread(run_structured_lookup, question, schemas, "analytical")
    if trace.status == "error":
        from src.agent.strategies.map_reduce import retrieve_map_reduce
        result = await retrieve_map_reduce(state, vector_store=vector_store)
        result["structured_trace"] = trace.to_dict()
        return result

    return {
        "retrieved_chunks": [],
        "sql_results": trace.rows,
        "structured_trace": trace.to_dict(),
        "retrieval_attempts": state.get("retrieval_attempts", 0) + 1,
    }
