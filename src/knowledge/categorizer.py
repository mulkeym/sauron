from dataclasses import dataclass, field
from src.generation.llm_client import generate, parse_json_response

CATEGORIZATION_PROMPT = """You are a document categorizer. Given a document's filename, type, and text preview, classify it into one of the existing categories OR propose a new category.

Existing categories:
{categories}

Rules:
- If the document fits an existing category, use it
- If no existing category fits, propose a new one
- Respond with ONLY valid JSON

For existing category match:
{{"category": "<name>", "confidence": 0.0-1.0, "is_new": false}}

For new category proposal:
{{"category": "<proposed_name>", "confidence": 0.0-1.0, "is_new": true, "description": "<what this category covers>", "suggested_acl_groups": ["<group1>"], "suggested_keywords": ["<keyword1>"]}}"""


@dataclass
class CategorizationResult:
    category: str
    confidence: float
    is_new: bool
    description: str = ""
    suggested_acl_groups: list[str] = field(default_factory=list)
    suggested_keywords: list[str] = field(default_factory=list)


def categorize_document(filename, doc_type, text_preview, metadata_store):
    import asyncio
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None
    if loop and loop.is_running():
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor() as pool:
            categories = pool.submit(asyncio.run, metadata_store.list_categories()).result()
    else:
        categories = asyncio.run(metadata_store.list_categories())

    cat_descriptions = []
    for cat in categories:
        keywords = ", ".join(cat.routing_keywords[:5]) if cat.routing_keywords else ""
        cat_descriptions.append(f"- {cat.name}: {cat.description[:60]} ({keywords})")
    categories_text = "\n".join(cat_descriptions) if cat_descriptions else "No existing categories."

    response = generate(
        system_prompt=CATEGORIZATION_PROMPT.format(categories=categories_text),
        user_prompt=f"Filename: {filename}\nType: {doc_type}\nPreview: {text_preview[:300]}",
        temperature=0.0, max_tokens=512,
    )
    try:
        parsed = parse_json_response(response)
        return CategorizationResult(
            category=parsed["category"], confidence=parsed.get("confidence", 0.5),
            is_new=parsed.get("is_new", False), description=parsed.get("description", ""),
            suggested_acl_groups=parsed.get("suggested_acl_groups", []),
            suggested_keywords=parsed.get("suggested_keywords", []),
        )
    except Exception:
        return CategorizationResult(category="uncategorized", confidence=0.0, is_new=False)
