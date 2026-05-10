from src.agent.state import AgentState
from src.db.schema_registry import SchemaRegistry
from src.db.sql_executor import execute_sql
from src.generation.llm_client import generate

TEXT_TO_SQL_PROMPT = """You are a SQL query generator. Given a natural language question and database schema, generate a single SELECT query.

Rules:
- Output ONLY the SQL query, no explanation
- Only use tables and columns from the provided schema
- Always use SELECT (never INSERT, UPDATE, DELETE, DROP, etc.)
- Keep queries simple and correct

Schema:
{schema}"""


async def retrieve_analytical(state: AgentState, vector_store, schema_registry: SchemaRegistry) -> dict:
    from src.ingestion.embedder import embed_query

    question = state["question"]
    user_groups = state["user_groups"]
    schema_prompt = schema_registry.schemas_to_prompt(user_groups)
    if schema_prompt == "No database schemas available.":
        # Fall back to vector search when no database schemas are available
        query_vector = embed_query(question)
        chunks = vector_store.search(vector=query_vector, user_groups=user_groups, top_k=30)
        return {
            "retrieved_chunks": chunks,
            "sql_results": [],
            "retrieval_attempts": state.get("retrieval_attempts", 0) + 1,
        }

    sql = generate(
        system_prompt=TEXT_TO_SQL_PROMPT.format(schema=schema_prompt),
        user_prompt=f"Question: {question}",
        temperature=0.0,
        max_tokens=2048,
    )
    sql = sql.strip().strip("`").removeprefix("sql\n").removeprefix("sql").strip()

    from src.config import settings
    database_url = settings.database_url

    try:
        rows = await execute_sql(database_url=database_url, sql=sql)
    except (ValueError, Exception) as e:
        # Fall back to vector search if SQL fails
        query_vector = embed_query(question)
        chunks = vector_store.search(vector=query_vector, user_groups=user_groups, top_k=30)
        return {
            "retrieved_chunks": chunks,
            "sql_results": [],
            "retrieval_attempts": state.get("retrieval_attempts", 0) + 1,
        }
    return {
        "retrieved_chunks": [],
        "sql_results": rows,
        "retrieval_attempts": state.get("retrieval_attempts", 0) + 1,
    }
