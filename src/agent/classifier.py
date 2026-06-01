import logging

from src.agent.state import AgentState, QueryType
from src.config import settings
from src.generation.llm_client import generate, parse_json_response
from src.retrieval.strategy_memory import get_best_strategy

logger = logging.getLogger(__name__)

CLASSIFICATION_PROMPT = """You are a query classifier for a document knowledge base. Classify the user's question into exactly one type and identify sub-tasks.

Query types:
- lookup: Question about a SPECIFIC entity, person, company, contract, document, policy, or fact. Example: "What does policy 4.2 say?", "Tell me about the contract awarded to Acme Corp", "What did John Smith say?"
- sweep: Exhaustive search needing ALL matching items across many documents. Use when the question says "all", "every", "list", "total", "how many", or needs complete coverage. Example: "What are all the contracts?", "What was the total value of all awards?", "How many contracts were awarded in January?"
- analytical: Question requiring SQL against a structured database (only if database tables exist). Example: "What was Q3 revenue from the finance database?"
- cross_reference: Question spanning multiple source types (e.g., compare database data against a policy). Example: "Does our spending comply with policy?"
- temporal: Question about changes over time or date-bounded searches. Example: "What changed last month?"
- metadata: Question ABOUT the documents/files themselves (the catalog) rather than their content — counts, lists, upload dates, datasets, categories, who uploaded, or which files mention a term. Example: "How many PDFs do we have?", "When was the pay doc uploaded?", "Which files mention officers?", "What datasets exist?", "List files uploaded in May".

IMPORTANT:
- If the question asks about a SPECIFIC named entity (person, company, organization, contract number), use LOOKUP even if it mentions "contract" or "award". LOOKUP is for targeted searches; SWEEP is for exhaustive collection.
- If the question asks for "all", "every", "total", or "sum" of items from documents, use SWEEP not analytical. Only use analytical if a structured database is explicitly needed.
- If the question asks about a specific DATE (e.g. "on Jan 30th", "on February 5"), use SWEEP — the system has date-based document filtering for sweep queries.
- Use METADATA only for questions ABOUT the files (catalog: counts, dates, datasets, filenames, which-files-mention). A question answered by the CONTENT of a file (e.g. "what does the pay doc SAY about officers?", "what is the pay for an O-4?") is NOT metadata — use lookup/analytical.

Respond with ONLY valid JSON:
{"query_type": "<type>", "sub_tasks": ["<task1>", "<task2>"]}"""


_MAX_NOTE_CHARS = 200


def _hint_note(rh) -> str:
    """Compact, length-capped domain note for a table, built from its resolved
    hints: table notes first, then the distinct glossary meanings (e.g. the human
    labels behind coded values). Lets the classifier recognize what a generically
    profiled table actually holds."""
    parts = list(dict.fromkeys(n for n in rh.table_notes if n))  # order-preserving dedup
    meanings: list[str] = []
    for col_map in rh.column_glossaries.values():
        for meaning in col_map.values():
            if meaning and meaning not in meanings:
                meanings.append(meaning)
    if meanings:
        parts.append(", ".join(meanings))
    return "; ".join(parts)[:_MAX_NOTE_CHARS]


def format_available_tables(schemas, hints=None) -> str:
    """One '- <table>: <description>' line per schema, sorted by table name for a
    stable (run-to-run identical) classifier prompt. When ``hints`` (table ->
    ResolvedHints) supplies a note for a table, it is appended after an em dash.
    With ``hints`` None/empty the output is byte-identical to before."""
    hints = hints or {}
    lines = []
    for s in sorted(schemas, key=lambda s: s.table):
        line = f"- {s.table}: {s.description}"
        rh = hints.get(s.table)
        note = _hint_note(rh) if rh is not None else ""
        if note:
            line += f" — {note}"
        lines.append(line)
    return "\n".join(lines)


def classify_query(state: AgentState, available_tables: str = "") -> dict:
    question = state["question"]
    system_prompt = CLASSIFICATION_PROMPT
    if available_tables:
        system_prompt += (
            "\n\nAvailable structured tables (queryable with SQL):\n"
            f"{available_tables}\n"
            "If the question asks for specific values, totals, or filtered rows that "
            "these tables contain, classify it as ANALYTICAL."
        )
    response = generate(
        system_prompt=system_prompt,
        user_prompt=f"Question: {question}",
        temperature=0.0,
        max_tokens=1024,
    )
    try:
        parsed = parse_json_response(response)
        query_type = QueryType(parsed["query_type"])
        sub_tasks = parsed.get("sub_tasks", [question])
        logger.info("Classified %r -> %s (tables_available=%s)",
                    question, query_type.value, bool(available_tables))
    except (Exception,):
        query_type = QueryType.LOOKUP
        sub_tasks = [question]
        logger.warning("Classification parse failed for %r; defaulting to LOOKUP. Raw: %r",
                       question, response)
    return {"query_type": query_type, "sub_tasks": sub_tasks}


async def _resolve_hints_for_classifier(schemas) -> dict:
    """Fail-open hint resolution for the classifier table view. Mirrors the call
    retrieve_analytical uses; returns {} on any error so classification never breaks."""
    try:
        from src.agent.strategies.structured import resolve_hints_for_schemas
        from src.api.routes_ingest import get_hint_store, get_metadata_store
        return await resolve_hints_for_schemas(schemas, get_hint_store(), get_metadata_store())
    except Exception:
        logger.warning("Classifier hint resolution failed; using bare table descriptions", exc_info=True)
        return {}


def _classify_node_factory(schema_registry):
    """Build an async LangGraph 'classify' node: LLM classification, then a
    confidence-gated soft override from Strategy Memory."""
    async def classify_node(state: AgentState) -> dict:
        import asyncio
        # Live sub-step reporter for async-status visibility; no-op when absent
        # (sync path / tests). Fires synchronously mid-node so progress shows in
        # real time instead of only after the node completes.
        progress = state.get("progress") or (lambda *a, **k: None)
        available = ""
        if schema_registry is not None:
            progress("classify.hints")
            schemas = schema_registry.list_for_user(state.get("user_groups", ["ALL"]))
            hints = await _resolve_hints_for_classifier(schemas)
            available = format_available_tables(schemas, hints)
        # classify_query makes a blocking LLM call — run it off the event loop
        # (the old sync node was run by LangGraph in a threadpool).
        progress("classify.llm")
        result = await asyncio.to_thread(classify_query, state, available)
        llm_pick = result["query_type"]

        memory_decision = {"llm_pick": str(llm_pick), "overrode": False, "reason": "disabled"}
        if settings.strategy_memory_enabled:
            progress("classify.strategy")
            try:
                best = await get_best_strategy(state["question"])
                memory_decision["reason"] = "no record"
                if best:
                    memory_decision.update({
                        "memory_best": best["strategy"], "count": best["count"],
                        "margin": best["margin"], "reason": "below gate",
                    })
                    try:
                        mem_type = QueryType(best["strategy"])
                    except ValueError:
                        mem_type = None
                    if mem_type is not None and mem_type == llm_pick:
                        memory_decision["reason"] = "agreed"
                    elif (mem_type is not None
                            and best["count"] >= settings.strategy_memory_min_runs
                            and best["margin"] >= settings.strategy_memory_margin):
                        if llm_pick in (QueryType.ANALYTICAL, QueryType.METADATA):
                            # ANALYTICAL/METADATA are capability-gated picks: ANALYTICAL is chosen only when a
                            # relevant structured table is registered + ACL-visible; METADATA is chosen only
                            # when a catalog capability is available. A learned prior
                            # (trainable by cited-but-unhelpful answers)
                            # must not veto it. Memory relearns once analytical/metadata runs.
                            memory_decision["reason"] = "protected"
                            logger.info("Strategy memory suppressed: %s capability "
                                        "pick protected (memory wanted %s, n=%d, margin=%.0f%%)",
                                        llm_pick, mem_type, best["count"], best["margin"] * 100)
                        else:
                            result["query_type"] = mem_type
                            memory_decision["overrode"] = True
                            memory_decision["reason"] = "override"
                            logger.info("Strategy memory override: %s -> %s (n=%d, margin=%.0f%%)",
                                        llm_pick, mem_type, best["count"], best["margin"] * 100)
                    # else: reason stays "below gate" (memory differs but a gate failed,
                    # or mem_type is None/invalid)
            except Exception as e:
                logger.warning("Strategy memory lookup failed, keeping LLM pick: %s", e)
                memory_decision["reason"] = "error"

        result["strategy_memory"] = memory_decision if settings.strategy_memory_enabled else None
        progress("classify.done", {"kind": "classification", "data": {
            "query_type": str(result["query_type"]),
            "reason": result.get("reason", ""),
            "sub_tasks": result.get("sub_tasks", []),
            "strategy_memory": result["strategy_memory"],
        }})
        return result
    return classify_node
