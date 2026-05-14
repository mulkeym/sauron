import logging
from dataclasses import dataclass, field
from src.generation.llm_client import generate, parse_json_response

logger = logging.getLogger(__name__)

CATEGORIZATION_PROMPT = """You are a document categorizer for a government/enterprise knowledge base. Given a document's filename, type, and text preview, classify it into one of the existing categories OR propose a new category.

Existing categories:
{categories}

Rules:
- If the document fits an existing category, use it EXACTLY AS LISTED
- If no existing category fits, propose a new one with:
  - A snake_case name (e.g., "network_configs", "security_audits")
  - A clear description of what the category covers
  - Suggested ACL groups (who should access these documents)
  - 5-7 routing keywords (terms that indicate a document belongs in this category)
  - A NARA GRS (General Records Schedule) number if applicable

NARA GRS Reference (use the most applicable):
- 1.1: Financial transaction records (contracts, procurement, invoices)
- 2.1: Employee records
- 2.6: Training records
- 3.1: IT operations and management (SOPs, procedures)
- 3.2: IT security and compliance (ATO, STIG, RMF)
- 4.2: Privacy and compliance (HIPAA, PII)
- 5.1: Administrative records (meetings, memos, correspondence)
- 5.3: Continuity planning (COOP, disaster recovery)
- 5.6: Security management (access control, clearances)
- 5.7: Directives and policy issuances
- 5.8: Help desk and incident records
- 6.1: Mission records (permanent)
- 6.3: Technical reference (architecture, designs, configs)
- 6.4: Input/output and source records

Respond with ONLY valid JSON:

For existing category match:
{{"category": "<exact_name_from_list>", "confidence": 0.0-1.0, "is_new": false}}

For new category proposal:
{{"category": "<snake_case_name>", "confidence": 0.0-1.0, "is_new": true, "description": "<what this category covers>", "suggested_acl_groups": ["<group1>", "<group2>"], "suggested_keywords": ["<kw1>", "<kw2>", "<kw3>", "<kw4>", "<kw5>"], "suggested_grs": "<GRS number>"}}"""


@dataclass
class CategorizationResult:
    category: str
    confidence: float
    is_new: bool
    description: str = ""
    suggested_acl_groups: list[str] = field(default_factory=list)
    suggested_keywords: list[str] = field(default_factory=list)
    suggested_grs: str = ""


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
            acl_groups = pool.submit(asyncio.run, metadata_store.get_acl_group_names()).result()
    else:
        categories = asyncio.run(metadata_store.list_categories())
        acl_groups = asyncio.run(metadata_store.get_acl_group_names())

    cat_names = [cat.name for cat in categories]
    cat_descriptions = []
    for cat in categories:
        keywords = ", ".join(cat.routing_keywords[:5]) if cat.routing_keywords else ""
        grs = f" [GRS {cat.grs_number}]" if cat.grs_number else ""
        cat_descriptions.append(f"- {cat.name}{grs}: {cat.description[:80]} ({keywords})")
    categories_text = "\n".join(cat_descriptions) if cat_descriptions else "No existing categories."

    acl_list = ", ".join(acl_groups) if acl_groups else "it_support, engineering, finance, executives"

    response = generate(
        system_prompt=CATEGORIZATION_PROMPT.format(categories=categories_text),
        user_prompt=f"Filename: {filename}\nType: {doc_type}\nPreview: {text_preview[:500]}\n\nAvailable ACL groups (choose ONLY from these): {acl_list}",
        temperature=0.0, max_tokens=2048,
    )
    try:
        parsed = parse_json_response(response)
        category = parsed.get("category", "").strip()
        is_new = parsed.get("is_new", False)

        # If LLM returned a close match to existing category, use the exact name
        if not is_new and category not in cat_names and cat_names:
            lower_cat = category.lower()
            for exact_name in cat_names:
                if exact_name.lower() == lower_cat:
                    logger.info(f"Fixed category name from '{category}' to '{exact_name}'")
                    category = exact_name
                    break

        return CategorizationResult(
            category=category, confidence=parsed.get("confidence", 0.5),
            is_new=is_new, description=parsed.get("description", ""),
            suggested_acl_groups=parsed.get("suggested_acl_groups", []),
            suggested_keywords=parsed.get("suggested_keywords", []),
            suggested_grs=parsed.get("suggested_grs", ""),
        )
    except Exception as e:
        logger.error(f"Categorization parse error for {filename}: {e}")
        return CategorizationResult(category="uncategorized", confidence=0.0, is_new=False)
