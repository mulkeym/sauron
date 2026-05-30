"""Structured retrieval shared across strategies: text-to-SQL against the
tabular DuckDB store plus retrievable row narratives.

`structured_sql_rows` is the SQL core used by both `retrieve_analytical` and
the SWEEP branch. `tables_relevant_to` is a cheap (no-LLM) gate. `retrieve_structured`
combines them, fail-open, for the SWEEP strategy.
"""
import asyncio
import json
import logging
import math
import re
from dataclasses import dataclass, field

from src.generation.llm_client import generate
from src.ingestion.embedder import embed_query, embed_texts
from src.agent.strategies.hint_resolver import resolve_hints
from src.retrieval.feedback import get_feedback_boosts, apply_feedback_boosts_to_chunks

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
- When in doubt, prefer the narrowest set of columns that answers the question over returning everything; never select more columns than the answer needs.

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


def _effective_sql_budget() -> int:
    """Char budget for a SQL result before it counts as too large. Capped at
    65% of the model context so it always leaves room for the rest of the
    synthesis context even if the knob is set high."""
    from src.config import settings
    return min(settings.sql_result_budget_chars, int(settings.llm_max_context * 0.65))


def _classify_sql_result(rows: list[dict]) -> str:
    """Label a SQL result for the repair loop. One of:
    'empty' | 'degenerate' | 'too_large' | 'satisfactory'.
    Errors are handled separately (the query raised), not here."""
    if not rows:
        return "empty"
    if all(v is None for r in rows for v in r.values()):
        return "degenerate"
    if len(json.dumps(rows, default=str)) > _effective_sql_budget():
        return "too_large"
    return "satisfactory"


def _repair_feedback(verdict: str, *, rows: list[dict], sql: str, question: str,
                     error: str = "", judge_reason: str = "") -> str:
    """Failure-specific guidance appended to the next text-to-SQL attempt."""
    if verdict == "too_large":
        chars = len(json.dumps(rows, default=str))
        base = (f"Your previous query returned {len(rows)} rows (~{chars} chars), too large "
                f"to use. Rewrite it to summarize/aggregate (MIN/MAX/AVG with GROUP BY on "
                f"low-cardinality columns) or scope it more tightly with WHERE/LIMIT, while "
                f"still answering the question.")
    elif verdict == "empty":
        base = ("Your previous query returned no rows. Your filter or a column name may be "
                "wrong — loosen the filter, check the column names, or pick the closest "
                "available value.")
    elif verdict == "degenerate":
        base = ("Your previous query returned only NULLs. The selected columns are probably "
                "wrong for this question — choose different columns.")
    elif verdict == "error":
        base = f"Your previous query failed to run: {error}. Fix the SQL."
    else:
        base = "Improve the previous query to better answer the question."
    parts = [base, f"Previous SQL: {sql}"]
    if judge_reason:
        parts.append(f"It also does not answer the question well because: {judge_reason}.")
    parts.append(f"Question: {question}")
    return "\n".join(parts)


def _wide_table_steering(con, schemas) -> str:
    """Pre-flight: for each candidate table, estimate rows*cols. If any exceeds
    the configured cell threshold, return a steering block telling the model to
    aggregate/scope rather than SELECT *. Returns '' when no table is wide.
    Never raises — a missing/unreadable table is simply skipped."""
    from src.config import settings
    wide = []
    for s in schemas:
        try:
            nrows = con.execute(f'SELECT COUNT(*) FROM "{s.table}"').fetchone()[0]
        except Exception:
            continue
        ncols = len(s.columns)
        if nrows * ncols > settings.sql_wide_table_cell_threshold:
            wide.append(f'{s.table} (~{nrows} rows x {ncols} cols)')
    if not wide:
        return ""
    return ("\nNOTE: " + "; ".join(wide) + " — returning every row is unhelpful and will be "
            "truncated. Prefer aggregation (MIN/MAX/AVG with GROUP BY on low-cardinality "
            "columns such as locality/grade) or scope with WHERE/LIMIT to directly answer "
            "the question.")


_JUDGE_PROMPT = """You check whether SQL result rows answer a user's question.
Respond with ONLY a JSON object: {"helpful": true|false, "reason": "<short reason if not helpful>"}.
Mark helpful=false only when the rows clearly do not address the question (wrong entity,
wrong columns, off-topic). If they plausibly answer it, mark helpful=true."""


def _relevance_judge(gen, question: str, rows: list[dict]) -> tuple[bool, str]:
    """Ask the LLM whether a sample of rows answers the question. Fail-open:
    any parse/LLM problem returns (True, '') so we never block on the judge."""
    sample = json.dumps(rows[:5], default=str)
    try:
        raw = gen(system_prompt=_JUDGE_PROMPT,
                  user_prompt=f"Question: {question}\nResult rows (sample): {sample}",
                  temperature=0.0, max_tokens=256)
        m = re.search(r"\{.*\}", raw, re.DOTALL)
        data = json.loads(m.group(0)) if m else {}
        return bool(data.get("helpful", True)), str(data.get("reason", "") or "")
    except Exception:
        return True, ""


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


def generate_sql(schema_prompt: str, question: str, generate_fn=None,
                 *, extra_user_context: str = "", temperature: float = 0.0) -> str:
    """LLM text-to-SQL for one question + rendered schema prompt; returns the
    extracted SQL string (robust to prose/code-fence wrapping). ``extra_user_context``
    carries pre-flight steering or retry feedback; ``temperature`` is raised on
    retries so the model does not deterministically regenerate the same query."""
    gen = generate_fn or generate
    raw = gen(
        system_prompt=TEXT_TO_SQL_PROMPT.format(schema=schema_prompt),
        user_prompt=f"Question: {question}{extra_user_context}",
        temperature=temperature,
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


@dataclass
class SqlFitResult:
    """Outcome of the gate + bounded repair loop."""
    sql: str
    rows: list = field(default_factory=list)
    attempts: int = 0
    verdict: str = "satisfactory"


def _generate_run_fit(con, question: str, schemas, *, hints=None, generate_fn=None) -> SqlFitResult:
    """Pre-flight wide-table gate, then a bounded generate->run->classify loop.
    Returns the first satisfactory result, or the best valid (non-error) result
    after exhausting retries. Raises only if EVERY attempt errored (no valid
    result was ever produced) so callers' existing error paths still engage.

    Uses one read-only connection supplied by the caller. Synchronous."""
    from src.config import settings
    from src.ingestion.tabular_store import schema_prompt_with_values
    gen = generate_fn or generate
    allowed = {s.table for s in schemas}
    base_schema_prompt = schema_prompt_with_values(schemas, con, hints=hints)
    steering = _wide_table_steering(con, schemas)

    best = None            # (sql, rows) best valid result seen
    best_verdict = "error"
    extra = steering       # first attempt carries the gate steering
    last_error = ""
    max_attempts = settings.sql_repair_max_retries + 1

    for attempt in range(max_attempts):
        temperature = 0.0 if attempt == 0 else 0.3
        sql = generate_sql(base_schema_prompt, question, generate_fn=gen,
                           extra_user_context=extra, temperature=temperature)
        try:
            rows = run_sql(con, sql, allowed)
        except Exception as e:
            last_error = str(e)
            verdict = "error"
            extra = "\n" + _repair_feedback("error", rows=[], sql=sql,
                                            question=question, error=last_error)
            continue

        verdict = _classify_sql_result(rows)
        if verdict == "satisfactory":
            return SqlFitResult(sql=sql, rows=rows, attempts=attempt + 1, verdict=verdict)

        best = (sql, rows)          # valid but unsatisfactory — keep as fallback
        best_verdict = verdict
        if attempt < max_attempts - 1:
            reason = ""
            if settings.sql_relevance_judge_enabled:
                helpful, reason = _relevance_judge(gen, question, rows)
                if helpful:
                    reason = ""
            extra = "\n" + _repair_feedback(verdict, rows=rows, sql=sql,
                                            question=question, judge_reason=reason)

    if best is not None:
        return SqlFitResult(sql=best[0], rows=best[1], attempts=max_attempts, verdict=best_verdict)
    exc = RuntimeError(f"text-to-SQL produced no valid query: {last_error}")
    exc.last_sql = sql  # type: ignore[attr-defined]  # caller can surface this in the trace
    raise exc


def run_structured_lookup(question: str, schemas, query_type: str,
                          gate: list | None = None, generate_fn=None, hints=None) -> StructuredLookupTrace:
    """Generate + run SQL (with the bounded repair loop) and capture a full trace.
    Never raises: a failure is recorded as status='error' (with the SQL, if
    generated) and fell_back=True so the caller can fall back. Sync (run via
    asyncio.to_thread from async callers)."""
    from src.ingestion.tabular_store import (
        connect_tabular, schema_context_for_synthesis)
    trace = StructuredLookupTrace(query_type=query_type, gate=gate)
    con = connect_tabular(read_only=True)
    try:
        fit = _generate_run_fit(con, question, schemas, hints=hints, generate_fn=generate_fn)
        trace.sql = fit.sql
        # Carry the meaning of the queried table(s) forward to the synthesizer.
        referenced = [s for s in schemas if s.table in fit.sql] or list(schemas)
        trace.schema_context = schema_context_for_synthesis(referenced, hints=hints)
        trace.status = "ran"
        trace.rows = fit.rows
        trace.row_count = len(fit.rows)
        trace.sample_rows = fit.rows[:5]
        trace.fell_back = fit.attempts > 1  # signal the loop had to retry
    except Exception as e:
        trace.status = "error"
        trace.error = str(e)
        trace.fell_back = True
        # Surface the last-attempted SQL even on all-error runs so callers can log it.
        if not trace.sql:
            trace.sql = getattr(e, "last_sql", "")
    finally:
        con.close()
    return trace


def structured_sql_rows(question: str, schemas, generate_fn=None, hints=None) -> list[dict]:
    """Generate SQL (with the bounded repair loop) and run it against the tabular
    DuckDB, restricted to ``schemas`` as the allowlist. One read-only connection;
    raises only if no valid query could be produced. Synchronous (run via
    ``asyncio.to_thread`` from async callers)."""
    from src.ingestion.tabular_store import connect_tabular
    con = connect_tabular(read_only=True)
    try:
        fit = _generate_run_fit(con, question, schemas, hints=hints, generate_fn=generate_fn)
        return fit.rows
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

    _boosts = {}
    try:
        _boosts = await get_feedback_boosts(qv, user_groups)
        chunks = apply_feedback_boosts_to_chunks(chunks, _boosts)
    except Exception:
        pass

    return {"sql_results": trace.rows, "retrieved_chunks": chunks,
            "structured_trace": trace.to_dict(), "feedback_boosts": _boosts}
