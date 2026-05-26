"""Structured retrieval shared across strategies: text-to-SQL against the
tabular DuckDB store plus retrievable row narratives.

`structured_sql_rows` is the SQL core used by both `retrieve_analytical` and
the SWEEP branch. `tables_relevant_to` is a cheap (no-LLM) gate. `retrieve_structured`
combines them, fail-open, for the SWEEP strategy.
"""
import asyncio
import math

from src.generation.llm_client import generate
from src.ingestion.embedder import embed_query

TEXT_TO_SQL_PROMPT = """You are a SQL query generator. Given a natural language question and database schema, generate a single SELECT query.

Rules:
- Output ONLY the SQL query, no explanation
- Only use tables and columns from the provided schema
- Use the table name exactly as given, with no database/schema prefix
- Always use SELECT (never INSERT, UPDATE, DELETE, DROP, etc.)
- Keep queries simple and correct

Schema:
{schema}"""


def structured_sql_rows(question: str, schemas, generate_fn=None) -> list[dict]:
    """Generate SQL from the (value-enriched) schema prompt and run it against
    the tabular DuckDB, restricted to ``schemas`` as the allowlist.

    One read-only connection; raises on any failure (LLM, blocked/empty SQL,
    execution) — callers decide the fallback. Synchronous (run via
    ``asyncio.to_thread`` from async callers).
    """
    from src.ingestion.tabular_store import (
        connect_tabular, execute_duckdb_sql, schema_prompt_with_values,
    )
    gen = generate_fn or generate
    allowed_tables = {s.table for s in schemas}
    con = connect_tabular(read_only=True)
    try:
        schema_prompt = schema_prompt_with_values(schemas, con)
        sql = gen(
            system_prompt=TEXT_TO_SQL_PROMPT.format(schema=schema_prompt),
            user_prompt=f"Question: {question}",
            temperature=0.0,
            max_tokens=2048,
        )
        sql = sql.strip().strip("`").removeprefix("sql\n").removeprefix("sql").strip()
        return execute_duckdb_sql(con, sql, allowed_tables=allowed_tables)
    finally:
        con.close()


RELEVANCE_THRESHOLD = 0.30  # permissive: bias toward attempting SQL (a false positive just costs one try)


def _cosine(a, b) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


def _table_text(schema) -> str:
    cols = ", ".join(c.name for c in schema.columns)
    return f"{schema.table}. {schema.description}. Columns: {cols}"


def tables_relevant_to(question: str, schemas, threshold: float = RELEVANCE_THRESHOLD,
                       embed_query_fn=None, embed_texts_fn=None) -> list:
    """Cheap, no-LLM gate: keep tables whose text is embedding-similar to the
    question above ``threshold``. Operates only on the ACL-filtered schema list
    passed in. ``embed_*_fn`` are injectable for tests.
    """
    if not schemas:
        return []
    from src.ingestion.embedder import embed_query, embed_texts
    eq = embed_query_fn or embed_query
    et = embed_texts_fn or embed_texts
    qv = eq(question)
    tvs = et([_table_text(s) for s in schemas])
    return [s for s, tv in zip(schemas, tvs) if _cosine(qv, tv) >= threshold]


async def retrieve_structured(state, vector_store, schema_registry) -> dict:
    """Gated structured retrieval for the SWEEP branch.

    If a registered (ACL-visible) table is relevant to the question, return its
    exact SQL rows AND top-k row-narrative chunks. Fail-open: any failure yields
    whatever succeeded; an irrelevant question returns {} (sweep proceeds RAG-only).
    """
    question = state["question"]
    user_groups = state["user_groups"]

    schemas = schema_registry.list_for_user(user_groups)
    relevant = tables_relevant_to(question, schemas)
    if not relevant:
        return {}

    sql_results: list = []
    try:
        sql_results = await asyncio.to_thread(structured_sql_rows, question, relevant)
    except Exception:
        sql_results = []

    chunks: list = []
    try:
        qv = await asyncio.to_thread(embed_query, question)
        chunks = vector_store.search(
            vector=qv, user_groups=user_groups, top_k=20, tier="table_row",
        )
    except Exception:
        chunks = []

    return {"sql_results": sql_results, "retrieved_chunks": chunks}
