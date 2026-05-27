"""Structured retrieval shared across strategies: text-to-SQL against the
tabular DuckDB store plus retrievable row narratives.

`structured_sql_rows` is the SQL core used by both `retrieve_analytical` and
the SWEEP branch. `tables_relevant_to` is a cheap (no-LLM) gate. `retrieve_structured`
combines them, fail-open, for the SWEEP strategy.
"""
import asyncio
import logging
import math
import re
from dataclasses import dataclass, field

from src.generation.llm_client import generate
from src.ingestion.embedder import embed_query, embed_texts

logger = logging.getLogger(__name__)

TEXT_TO_SQL_PROMPT = """You are a SQL query generator. Given a natural language question and database schema, generate a single SELECT query.

Rules:
- Output ONLY the SQL query, no explanation
- Only use tables and columns from the provided schema
- Use the table name exactly as given, with no database/schema prefix
- Always use SELECT (never INSERT, UPDATE, DELETE, DROP, etc.)
- Keep queries simple and correct

Schema:
{schema}"""


_SQL_START = re.compile(r"(?is)\b(?:WITH|SELECT)\b")


def _extract_sql(response: str) -> str:
    """Pull the SQL statement out of an LLM response that may wrap it in prose
    and/or markdown code fences. Prefers the LAST fenced code block (models often
    emit their final answer last); within the chosen text, starts at the first
    WITH/SELECT keyword so leading prose is dropped. Falls back to the stripped
    response. Defends against the observed failure where the model hedges in prose
    and then emits a ```sql block — the old strip-only logic passed that whole blob
    to DuckDB and the query errored."""
    text = response.strip()
    fences = re.findall(r"```(?:sql)?\s*(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
    candidates = [f.strip() for f in fences if f.strip()]
    block = candidates[-1] if candidates else text
    m = _SQL_START.search(block)
    if m:
        block = block[m.start():]
    return block.strip().strip("`").removeprefix("sql\n").removeprefix("sql").strip()


@dataclass
class StructuredLookupTrace:
    """Per-query record of the structured/SQL retrieval attempt, for the
    playground 'Structured Lookup' step. ``rows`` is the transient full result
    set (used to populate sql_results); it is excluded from ``to_dict``."""
    query_type: str
    gate: list | None = None            # list of [table, score, passed]; None when no gate (analytical)
    sql: str = ""
    status: str = "ran"                 # "ran" | "skipped" | "error"
    skip_reason: str = ""
    error: str = ""
    row_count: int = 0
    sample_rows: list = field(default_factory=list)
    fell_back: bool = False
    rows: list = field(default_factory=list, repr=False)

    def to_dict(self) -> dict:
        return {
            "query_type": self.query_type,
            "gate": self.gate,
            "sql": self.sql,
            "status": self.status,
            "skip_reason": self.skip_reason,
            "error": self.error,
            "row_count": self.row_count,
            "sample_rows": self.sample_rows,
            "fell_back": self.fell_back,
        }


def generate_sql(schema_prompt: str, question: str, generate_fn=None) -> str:
    """LLM text-to-SQL for one question + rendered schema prompt; returns the
    extracted SQL string (robust to prose/code-fence wrapping)."""
    gen = generate_fn or generate
    raw = gen(
        system_prompt=TEXT_TO_SQL_PROMPT.format(schema=schema_prompt),
        user_prompt=f"Question: {question}",
        temperature=0.0,
        max_tokens=2048,
    )
    sql = _extract_sql(raw)
    logger.info("Text-to-SQL for %r -> %s", question, sql)
    return sql


def run_sql(con, sql: str, allowed_tables: set) -> list[dict]:
    """Execute SELECT-only SQL against the tabular DuckDB, restricted to
    ``allowed_tables``. Raises on blocked/invalid SQL or execution error."""
    from src.ingestion.tabular_store import execute_duckdb_sql
    rows = execute_duckdb_sql(con, sql, allowed_tables=allowed_tables)
    logger.info("Text-to-SQL returned %d row(s)", len(rows))
    return rows


def run_structured_lookup(question: str, schemas, query_type: str,
                          gate: list | None = None, generate_fn=None) -> StructuredLookupTrace:
    """Generate + run SQL and capture a full trace. Never raises: a failure is
    recorded as status='error' (with the SQL, if generated) and fell_back=True so
    the caller can fall back. Sync (run via asyncio.to_thread from async callers)."""
    from src.ingestion.tabular_store import connect_tabular, schema_prompt_with_values
    trace = StructuredLookupTrace(query_type=query_type, gate=gate)
    con = connect_tabular(read_only=True)
    try:
        trace.sql = generate_sql(schema_prompt_with_values(schemas, con), question,
                                 generate_fn=generate_fn)
        rows = run_sql(con, trace.sql, {s.table for s in schemas})
        trace.status = "ran"
        trace.rows = rows
        trace.row_count = len(rows)
        trace.sample_rows = rows[:5]
    except Exception as e:
        trace.status = "error"
        trace.error = str(e)
        trace.fell_back = True
    finally:
        con.close()
    return trace


def structured_sql_rows(question: str, schemas, generate_fn=None) -> list[dict]:
    """Generate SQL from the (value-enriched) schema prompt and run it against
    the tabular DuckDB, restricted to ``schemas`` as the allowlist.

    One read-only connection; raises on any failure (LLM, blocked/empty SQL,
    execution) — callers decide the fallback. Synchronous (run via
    ``asyncio.to_thread`` from async callers).
    """
    from src.ingestion.tabular_store import connect_tabular, schema_prompt_with_values
    con = connect_tabular(read_only=True)
    try:
        sql = generate_sql(schema_prompt_with_values(schemas, con), question,
                           generate_fn=generate_fn)
        return run_sql(con, sql, {s.table for s in schemas})
    finally:
        con.close()


RELEVANCE_THRESHOLD = 0.30  # permissive: bias toward attempting SQL (a false positive just costs one try)


def _cosine(a, b) -> float:
    if len(a) != len(b):
        return 0.0  # dimension mismatch -> treat as unrelated
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


def _table_text(schema) -> str:
    cols = ", ".join(c.name for c in schema.columns)
    return f"{schema.table}. {schema.description}. Columns: {cols}"


def tables_relevant_scored(question: str, schemas, threshold: float = RELEVANCE_THRESHOLD,
                           embed_query_fn=None, embed_texts_fn=None) -> list:
    """Score every ACL-filtered table against the question. Returns
    ``[(schema, score, passed), ...]`` so callers can show all scores (the gate)
    and pick the passers. No-LLM; ``embed_*_fn`` injectable for tests."""
    if not schemas:
        return []
    eq = embed_query_fn or embed_query
    et = embed_texts_fn or embed_texts
    qv = eq(question)
    tvs = et([_table_text(s) for s in schemas])
    return [(s, _cosine(qv, tv), _cosine(qv, tv) >= threshold) for s, tv in zip(schemas, tvs)]


def tables_relevant_to(question: str, schemas, threshold: float = RELEVANCE_THRESHOLD,
                       embed_query_fn=None, embed_texts_fn=None) -> list:
    """Cheap, no-LLM gate: tables whose text is embedding-similar above
    ``threshold``. Thin wrapper over ``tables_relevant_scored``."""
    return [s for s, _score, passed in tables_relevant_scored(
        question, schemas, threshold, embed_query_fn, embed_texts_fn) if passed]


async def retrieve_structured(state, vector_store, schema_registry) -> dict:
    """Gated structured retrieval for the SWEEP branch.

    If a registered (ACL-visible) table is relevant to the question, return its
    exact SQL rows AND top-k row-narrative chunks. Fail-open: any failure yields
    whatever succeeded; an irrelevant question returns {} (sweep proceeds RAG-only).
    """
    question = state["question"]
    user_groups = state["user_groups"]

    try:
        schemas = schema_registry.list_for_user(user_groups)
        relevant = tables_relevant_to(question, schemas)
    except Exception:
        return {}   # gate/registry error -> RAG-only sweep (fail-open)
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
