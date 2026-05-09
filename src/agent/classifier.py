from src.agent.state import AgentState, QueryType
from src.generation.llm_client import generate, parse_json_response

CLASSIFICATION_PROMPT = """You are a query classifier for a document knowledge base. Classify the user's question into exactly one type and identify sub-tasks.

Query types:
- lookup: Direct question about a specific document, policy, or fact. Example: "What does policy 4.2 say?"
- sweep: Exhaustive search across many documents for a pattern. Example: "What questions did Mike ask in all meetings?"
- analytical: Question requiring data from a database or spreadsheet (numbers, aggregations). Example: "What was Q3 revenue?"
- cross_reference: Question spanning multiple source types (e.g., compare database data against a policy). Example: "Does our spending comply with policy?"
- temporal: Question about changes over time or date-bounded searches. Example: "What changed last month?"

Respond with ONLY valid JSON:
{"query_type": "<type>", "sub_tasks": ["<task1>", "<task2>"]}"""

def classify_query(state: AgentState) -> dict:
    question = state["question"]
    response = generate(
        system_prompt=CLASSIFICATION_PROMPT,
        user_prompt=f"Question: {question}",
        temperature=0.0,
        max_tokens=256,
    )
    try:
        parsed = parse_json_response(response)
        query_type = QueryType(parsed["query_type"])
        sub_tasks = parsed.get("sub_tasks", [question])
    except (Exception,):
        query_type = QueryType.LOOKUP
        sub_tasks = [question]
    return {"query_type": query_type, "sub_tasks": sub_tasks}
