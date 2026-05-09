import json
from src.agent.state import AgentState, QueryType
from src.generation.llm_client import generate, parse_json_response

MAX_RETRIEVAL_ATTEMPTS = 3

EVALUATION_PROMPT = """You are evaluating whether retrieved context is sufficient to answer a question.

Question: {question}

Retrieved context:
{context}

Is this context sufficient? If NOT, suggest a reformulated search query that might find better results.

Respond with ONLY JSON:
{{"sufficient": true/false, "reason": "brief explanation", "reformulated_query": "alternative search terms if not sufficient"}}"""


def evaluate_context(state: AgentState) -> dict:
    retrieval_attempts = state.get("retrieval_attempts", 0)
    chunks = state.get("retrieved_chunks", [])
    sql_results = state.get("sql_results", [])

    # Hard stop after max attempts
    if retrieval_attempts >= MAX_RETRIEVAL_ATTEMPTS:
        return {"needs_reretrieval": False}

    # No context at all — retry with different approach
    if not chunks and not sql_results:
        query_type = state.get("query_type")
        # Escalate strategy: lookup → sweep, sweep → cross_reference
        if query_type == QueryType.LOOKUP:
            return {"needs_reretrieval": True, "query_type": QueryType.SWEEP}
        elif query_type == QueryType.SWEEP:
            return {"needs_reretrieval": True, "query_type": QueryType.CROSS_REFERENCE}
        return {"needs_reretrieval": True}

    # Build context summary for evaluation
    context_parts = [c.text for c in chunks[:5]]
    if sql_results:
        context_parts.append(f"SQL results: {json.dumps(sql_results[:5])}")
    context = "\n\n".join(context_parts)

    response = generate(
        system_prompt="You evaluate retrieved context sufficiency.",
        user_prompt=EVALUATION_PROMPT.format(question=state["question"], context=context),
        temperature=0.0,
        max_tokens=512,
    )

    try:
        parsed = parse_json_response(response)
        sufficient = parsed.get("sufficient", True)
        if not sufficient:
            reformulated = parsed.get("reformulated_query", "")
            query_type = state.get("query_type")
            # Escalate strategy on retry
            new_type = query_type
            if query_type == QueryType.LOOKUP:
                new_type = QueryType.SWEEP
            elif query_type == QueryType.SWEEP:
                new_type = QueryType.CROSS_REFERENCE

            result = {"needs_reretrieval": True, "query_type": new_type}
            if reformulated:
                result["question"] = reformulated
                result["reformulated_query"] = reformulated
            return result
    except Exception:
        pass

    return {"needs_reretrieval": False}
