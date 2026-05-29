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

IMPORTANT:
- If the question asks about a SPECIFIC named entity (person, company, organization, contract number), use LOOKUP even if it mentions "contract" or "award". LOOKUP is for targeted searches; SWEEP is for exhaustive collection.
- If the question asks for "all", "every", "total", or "sum" of items from documents, use SWEEP not analytical. Only use analytical if a structured database is explicitly needed.
- If the question asks about a specific DATE (e.g. "on Jan 30th", "on February 5"), use SWEEP — the system has date-based document filtering for sweep queries.

Respond with ONLY valid JSON:
{"query_type": "<type>", "sub_tasks": ["<task1>", "<task2>"]}"""


def format_available_tables(schemas) -> str:
    """One '- <table>: <description>' line per schema, sorted by table name for
    a stable (run-to-run identical) classifier prompt."""
    return "\n".join(
        f"- {s.table}: {s.description}" for s in sorted(schemas, key=lambda s: s.table)
    )


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


def _classify_node_factory(schema_registry):
    """Build an async LangGraph 'classify' node: LLM classification, then a
    confidence-gated soft override from Strategy Memory."""
    async def classify_node(state: AgentState) -> dict:
        import asyncio
        available = ""
        if schema_registry is not None:
            schemas = schema_registry.list_for_user(state.get("user_groups", ["ALL"]))
            available = format_available_tables(schemas)
        # classify_query makes a blocking LLM call — run it off the event loop
        # (the old sync node was run by LangGraph in a threadpool).
        result = await asyncio.to_thread(classify_query, state, available)
        llm_pick = result["query_type"]

        memory_decision = {"llm_pick": str(llm_pick), "overrode": False, "reason": "disabled"}
        if settings.strategy_memory_enabled:
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

        result["strategy_memory"] = memory_decision
        return result
    return classify_node
