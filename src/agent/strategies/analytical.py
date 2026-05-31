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

    # Route the (possibly large) ACL-visible corpus down to the relevant tables
    # before SQL generation, so the value-dumping schema prompt stays within the
    # model context. Without this, a broad question over a large corpus sends
    # every table's schema and overflows.
    from src.agent.strategies.structured import resolve_hints_for_schemas, select_relevant_tables
    from src.api.routes_ingest import get_hint_store, get_metadata_store
    schemas = await asyncio.to_thread(select_relevant_tables, question, schemas)
    try:
        hints = await resolve_hints_for_schemas(schemas, get_hint_store(), get_metadata_store())
    except Exception:
        hints = None
    trace = await asyncio.to_thread(run_structured_lookup, question, schemas, "analytical", None, None, hints)

    if trace.status == "error":
        from src.agent.strategies.map_reduce import retrieve_map_reduce
        result = await retrieve_map_reduce(state, vector_store=vector_store)
        result["structured_trace"] = trace.to_dict()
        return result

    # Runnable-but-empty SQL (e.g. WHERE col_0='officer' when values are O-1..O-10)
    # is a miss, not an answer. Fall back to the structured row-narrative path so the
    # glossary-annotated table_row chunks reach the synthesizer; then map-reduce.
    if trace.status == "ran" and trace.row_count == 0:
        trace.fell_back = True
        from src.agent.strategies.structured import retrieve_structured
        structured = await retrieve_structured(state, vector_store, schema_registry)
        if structured.get("retrieved_chunks") or structured.get("sql_results"):
            structured["structured_trace"] = trace.to_dict()
            structured["retrieval_attempts"] = state.get("retrieval_attempts", 0) + 1
            return structured
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
