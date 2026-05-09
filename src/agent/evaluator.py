import json
from src.agent.state import AgentState
from src.generation.llm_client import generate, parse_json_response

MAX_RETRIEVAL_ATTEMPTS = 3

EVALUATION_PROMPT = """You are evaluating whether retrieved context is sufficient to answer a question.

Question: {question}

Retrieved context:
{context}

Is this context sufficient to answer the question? Respond with ONLY valid JSON:
{{"sufficient": true/false, "reason": "brief explanation"}}"""


def evaluate_context(state: AgentState) -> dict:
    retrieval_attempts = state.get("retrieval_attempts", 0)
    chunks = state.get("retrieved_chunks", [])
    sql_results = state.get("sql_results", [])

    if retrieval_attempts >= MAX_RETRIEVAL_ATTEMPTS:
        return {"needs_reretrieval": False}
    if not chunks and not sql_results:
        return {"needs_reretrieval": True}

    context_parts = [c.text for c in chunks[:5]]
    if sql_results:
        context_parts.append(f"SQL results: {json.dumps(sql_results[:5])}")
    context = "\n\n".join(context_parts)

    response = generate(
        system_prompt="You evaluate retrieved context sufficiency.",
        user_prompt=EVALUATION_PROMPT.format(question=state["question"], context=context),
        temperature=0.0,
        max_tokens=128,
    )

    try:
        parsed = parse_json_response(response)
        sufficient = parsed.get("sufficient", True)
    except Exception:
        sufficient = True
    return {"needs_reretrieval": not sufficient}
