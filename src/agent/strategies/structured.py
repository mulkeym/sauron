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
from src.agent.strategies.hint_resolver import resolve_hints

logger = logging.getLogger(__name__)

TEXT_TO_SQL_PROMPT = """You are a SQL query generator. Given a natural language question and database schema, generate a single SELECT query.

Rules:
- Output ONLY the SQL query, no explanation
- Only use tables and columns from the provided schema
- Use the table name exactly as given, with no database/schema prefix
- Always use SELECT (never INSERT, UPDATE, DELETE, DROP, etc.)
- Keep queries simple and correct

Using column values and codes:
- A column may list its allowed values. Codes are often annotated as `CODE (meaning)`
  (e.g. `TU (Tucson-Nogales, AZ)`). When the question names a place, category, or
  entity, match it to the corresponding `(meaning)` and filter on the CODE itself,
  not the human-readable name.
- Apply any `Notes:` guidance shown for a table or column when choosing values.
- If the exact value the user named is not in the list, do NOT refuse — choose the
  closest applicable code, or a catch-all / "rest" / total category if the values
  or notes indicate one.

Choosing which columns to return:
- Return enough columns that each row is self-explanatory. ALWAYS include the
  identifying / label columns that say WHICH entity a row describes (e.g. name,
  code, grade, category, locality, year/date) — not only the numeric measure
  columns. A reader who sees only the measures cannot tell the rows apart.
- When in doubt, prefer `SELECT *` over a hand-picked subset of columns.

Always return your single best-effort SELECT query. Never refuse, never apologize,
and never output prose — output SQL only, even when you are unsure.

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
    schema_context: str = ""            # column meanings + value glossary for the queried table(s); fed to the synthesizer so it can interpret the raw rows
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
            "schema_context": self.schema_context,
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
                          gate: list | None = None, generate_fn=None, hints=None) -> StructuredLookupTrace:
    """Generate + run SQL and capture a full trace. Never raises: a failure is
    recorded as status='error' (with the SQL, if generated) and fell_back=True so
    the caller can fall back. Sync (run via asyncio.to_thread from async callers)."""
    from src.ingestion.tabular_store import (
        connect_tabular, schema_prompt_with_values, schema_context_for_synthesis)
    trace = StructuredLookupTrace(query_type=query_type, gate=gate)
    con = connect_tabular(read_only=True)
    try:
        trace.sql = generate_sql(schema_prompt_with_values(schemas, con, hints=hints), question,
                                 generate_fn=generate_fn)
        # Carry the meaning of the queried table(s) forward to the synthesizer.
        # Scope to the tables the SQL actually referenced so the context stays small.
        referenced = [s for s in schemas if s.table in trace.sql] or list(schemas)
        trace.schema_context = schema_context_for_synthesis(referenced, hints=hints)
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


def structured_sql_rows(question: str, schemas, generate_fn=None, hints=None) -> list[dict]:
    """Generate SQL from the (value-enriched) schema prompt and run it against
    the tabular DuckDB, restricted to ``schemas`` as the allowlist.

    One read-only connection; raises on any failure (LLM, blocked/empty SQL,
    execution) — callers decide the fallback. Synchronous (run via
    ``asyncio.to_thread`` from async callers).
    """
    from src.ingestion.tabular_store import connect_tabular, schema_prompt_with_values
    con = connect_tabular(read_only=True)
    try:
        sql = generate_sql(schema_prompt_with_values(schemas, con, hints=hints), question,
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
    scored = [(s, _cosine(qv, tv)) for s, tv in zip(schemas, tvs)]
    return [(s, c, c >= threshold) for s, c in scored]


def tables_relevant_to(question: str, schemas, threshold: float = RELEVANCE_THRESHOLD,
                       embed_query_fn=None, embed_texts_fn=None) -> list:
    """Cheap, no-LLM gate: tables whose text is embedding-similar above
    ``threshold``. Thin wrapper over ``tables_relevant_scored``."""
    return [s for s, _score, passed in tables_relevant_scored(
        question, schemas, threshold, embed_query_fn, embed_texts_fn) if passed]


async def resolve_hints_for_schemas(schemas, hint_store, metadata_store) -> dict:
    """Map each schema's table name -> ResolvedHints, by finding its owning
    document (table name prefix == duckdb_table_name(doc_id, "")) and resolving
    that document's category/dataset-scoped hints. Fail-open: returns {} on any
    error; tables with no owning doc or no hints are omitted."""
    from src.ingestion.tabular_store import duckdb_table_name
    try:
        docs = await metadata_store.list_documents()
    except Exception:
        return {}
    out = {}
    for s in schemas:
        owner = next((d for d in docs if s.table.startswith(duckdb_table_name(d.doc_id, ""))), None)
        if owner is None:
            continue
        rh = resolve_hints(s, owner, hint_store)
        if rh.column_glossaries or rh.column_notes or rh.table_notes:
            out[s.table] = rh
    return out


async def retrieve_structured(state, vector_store, schema_registry) -> dict:
    """Gated structured retrieval for the SWEEP branch. Returns exact SQL rows +
    top-k row-narrative chunks when a registered table is relevant, plus a
    ``structured_trace`` describing the decision/SQL/result. Fail-open: gate or
    registry errors yield {} (RAG-only sweep)."""
    question = state["question"]
    user_groups = state["user_groups"]

    try:
        schemas = schema_registry.list_for_user(user_groups)
        scored = tables_relevant_scored(question, schemas)
    except Exception:
        return {}   # gate/registry error -> RAG-only sweep (fail-open)

    gate = [[s.table, round(score, 3), passed] for s, score, passed in scored]
    relevant = [s for s, _score, passed in scored if passed]

    if not relevant:
        if gate:  # tables existed but none cleared the threshold -> a visible "skipped" decision
            trace = StructuredLookupTrace(
                query_type="sweep", gate=gate, status="skipped",
                skip_reason=f"no table >= {RELEVANCE_THRESHOLD} relevance")
            return {"structured_trace": trace.to_dict()}
        return {}

    trace = await asyncio.to_thread(run_structured_lookup, question, relevant, "sweep", gate)

    chunks: list = []
    try:
        qv = await asyncio.to_thread(embed_query, question)
        chunks = vector_store.search(
            vector=qv, user_groups=user_groups, top_k=20, tier="table_row",
        )
    except Exception:
        chunks = []

    return {"sql_results": trace.rows, "retrieved_chunks": chunks,
            "structured_trace": trace.to_dict()}
