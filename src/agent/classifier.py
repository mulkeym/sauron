from src.agent.state import AgentState, QueryType
from src.generation.llm_client import generate, parse_json_response

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

def classify_query(state: AgentState) -> dict:
    question = state["question"]
    response = generate(
        system_prompt=CLASSIFICATION_PROMPT,
        user_prompt=f"Question: {question}",
        temperature=0.0,
        max_tokens=1024,
    )
    try:
        parsed = parse_json_response(response)
        query_type = QueryType(parsed["query_type"])
        sub_tasks = parsed.get("sub_tasks", [question])
    except (Exception,):
        query_type = QueryType.LOOKUP
        sub_tasks = [question]
    return {"query_type": query_type, "sub_tasks": sub_tasks}
