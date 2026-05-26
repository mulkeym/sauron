import asyncio

from src.agent.state import AgentState
from src.db.schema_registry import SchemaRegistry
from src.generation.llm_client import generate

TEXT_TO_SQL_PROMPT = """You are a SQL query generator. Given a natural language question and database schema, generate a single SELECT query.

Rules:
- Output ONLY the SQL query, no explanation
- Only use tables and columns from the provided schema
- Use the table name exactly as given, with no database/schema prefix
- Always use SELECT (never INSERT, UPDATE, DELETE, DROP, etc.)
- Keep queries simple and correct

Schema:
{schema}"""


async def retrieve_analytical(state: AgentState, vector_store, schema_registry: SchemaRegistry) -> dict:
    question = state["question"]
    user_groups = state["user_groups"]

    schema_prompt = schema_registry.schemas_to_prompt(user_groups)
    if schema_prompt == "No database schemas available.":
        from src.agent.strategies.map_reduce import retrieve_map_reduce
        return await retrieve_map_reduce(state, vector_store=vector_store)

    sql = generate(
        system_prompt=TEXT_TO_SQL_PROMPT.format(schema=schema_prompt),
        user_prompt=f"Question: {question}",
        temperature=0.0,
        max_tokens=2048,
    )
    sql = sql.strip().strip("`").removeprefix("sql\n").removeprefix("sql").strip()

    allowed_tables = {s.table for s in schema_registry.list_for_user(user_groups)}

    def _run_query():
        from src.ingestion.tabular_store import connect_tabular, execute_duckdb_sql
        con = connect_tabular(read_only=True)
        try:
            return execute_duckdb_sql(con, sql, allowed_tables=allowed_tables)
        finally:
            con.close()

    try:
        rows = await asyncio.to_thread(_run_query)
    except Exception:
        # Bad/blocked/failed SQL -> fall back to comprehensive retrieval.
        from src.agent.strategies.map_reduce import retrieve_map_reduce
        return await retrieve_map_reduce(state, vector_store=vector_store)

    return {
        "retrieved_chunks": [],
        "sql_results": rows,
        "retrieval_attempts": state.get("retrieval_attempts", 0) + 1,
    }
