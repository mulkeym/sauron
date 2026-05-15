from __future__ import annotations
import logging
from dataclasses import dataclass, field
from src.generation.llm_client import generate, parse_json_response

logger = logging.getLogger(__name__)

BASE_EXTRACTION_PROMPT = """Extract important entities and their relationships from the text. Return ONLY valid JSON.

IMPORTANT:
- Focus on substantive content entities — people, organizations, specific items, amounts, locations
- IGNORE website navigation, social media links, menu items, footers, headers, breadcrumbs
- IGNORE generic labels like "News", "Help Center", "Contact", "Resources", "Publications"
- Only extract entities that carry real informational value
{domain_guidance}
Entity types: {entity_types}
Relationship types: {relationship_types}

{examples}
Return format: {{"entities": [{{"name": "...", "type": "..."}}], "relationships": [{{"source": "...", "target": "...", "type": "..."}}]}}"""

# Domain-specific extraction guidance keyed by category
CATEGORY_PROFILES = {
    "contracts": {
        "guidance": "This is a CONTRACT document. Focus on: contractor/company names, awarding agencies, contract numbers, award amounts, work locations, performance periods, and competition details.",
        "entity_types": "contractor, agency, contract, financial_amount, location, person, organization, project, date",
        "relationship_types": "awarded_to, contracted_by, located_in, funds, manages, competed_by, modifies, supports",
        "examples": 'GOOD entities: "BL Harbert International LLC" (contractor), "Naval Air Systems Command" (agency), "$171,506,091" (financial_amount), "W91278-26-C-A004" (contract), "Birmingham, Alabama" (location)\nBAD entities: "Facebook", "News", "Help Center"',
    },
    "budget_finance": {
        "guidance": "This is a FINANCIAL document. Focus on: budget line items, funding amounts, fiscal years, programs, cost centers, and obligations.",
        "entity_types": "organization, financial_amount, program, fiscal_year, cost_center, person, regulation, date",
        "relationship_types": "funds, allocated_to, obligated_by, manages, references, part_of, reports_to",
        "examples": 'GOOD entities: "$2.5M" (financial_amount), "FY2026" (fiscal_year), "Program Element 0603461A" (program)\nBAD entities: "Home", "Contact Us"',
    },
    "financial_data": {
        "guidance": "This is a FINANCIAL DATA document. Focus on: budget items, revenue figures, cost breakdowns, fiscal periods, and responsible entities.",
        "entity_types": "organization, financial_amount, program, fiscal_year, cost_center, person, date",
        "relationship_types": "funds, allocated_to, reports_to, manages, references, part_of",
        "examples": 'GOOD entities: "Q3 Revenue" (financial_amount), "FY2026" (fiscal_year)\nBAD entities: "Sheet1", "Total"',
    },
    "it_policies": {
        "guidance": "This is an IT POLICY document. Focus on: systems, compliance frameworks, security controls, responsible roles, and accreditation status.",
        "entity_types": "system, regulation, organization, person, control, accreditation, location, date",
        "relationship_types": "governs, requires, implements, authorized_by, references, part_of, manages",
        "examples": 'GOOD entities: "NIST 800-53" (regulation), "MHS Genesis" (system), "ISSM" (person)\nBAD entities: "Table of Contents", "Appendix"',
    },
    "hipaa_compliance": {
        "guidance": "This is a HIPAA/PRIVACY document. Focus on: data types (PHI, PII), safeguards, responsible roles, systems handling data, and breach procedures.",
        "entity_types": "regulation, data_type, safeguard, system, person, organization, procedure, date",
        "relationship_types": "protects, governs, requires, handles, references, implements, reports_to",
        "examples": 'GOOD entities: "HIPAA Security Rule" (regulation), "PHI" (data_type), "Privacy Officer" (person)\nBAD entities: "Page 1", "Footer"',
    },
    "meeting_notes": {
        "guidance": "This is a MEETING document. Focus on: attendees, action items, decisions made, topics discussed, and deadlines.",
        "entity_types": "person, organization, action_item, decision, topic, date, system, project",
        "relationship_types": "assigned_to, decided_by, discussed, references, due_by, reports_to, part_of",
        "examples": 'GOOD entities: "John Smith" (person), "migrate to cloud" (action_item), "2026-03-15" (date)\nBAD entities: "Agenda", "Minutes"',
    },
}

# Default profile for categories without specific guidance
DEFAULT_PROFILE = {
    "guidance": "Extract the most important entities and their relationships from this document content.",
    "entity_types": "organization, person, contractor, agency, contract, location, project, financial_amount, regulation, system, date",
    "relationship_types": "awarded_to, contracted_by, located_in, funds, manages, references, part_of, related_to, modifies, supports, governs, requires",
    "examples": 'GOOD entities: named organizations, specific people, dollar amounts, locations, systems, regulations\nBAD entities: "Facebook", "X", "Instagram", "Help Center", "News", "Contact"',
}


def _build_prompt(category: str = "", category_description: str = "", category_keywords: list[str] | None = None) -> str:
    """Build a category-aware extraction prompt.

    Uses hardcoded profiles for known categories, otherwise auto-generates
    guidance from the category description and keywords stored in the DB.
    """
    if category in CATEGORY_PROFILES:
        profile = CATEGORY_PROFILES[category]
    elif category and category != "uncategorized" and (category_description or category_keywords):
        # Auto-generate guidance from category metadata
        kw_str = ", ".join(category_keywords[:10]) if category_keywords else ""
        desc = category_description or category
        profile = {
            "guidance": f"This is a {category.upper().replace('_', ' ')} document ({desc}). Focus on extracting the most important entities specific to this domain. Keywords: {kw_str}",
            "entity_types": DEFAULT_PROFILE["entity_types"],
            "relationship_types": DEFAULT_PROFILE["relationship_types"],
            "examples": DEFAULT_PROFILE["examples"],
        }
    else:
        profile = DEFAULT_PROFILE

    return BASE_EXTRACTION_PROMPT.format(
        domain_guidance=f"\nDOMAIN CONTEXT: {profile['guidance']}\n",
        entity_types=profile["entity_types"],
        relationship_types=profile["relationship_types"],
        examples=profile["examples"],
    )


@dataclass
class ExtractionResult:
    entities: list[dict] = field(default_factory=list)
    relationships: list[dict] = field(default_factory=list)
    sections: list[dict] = field(default_factory=list)


def extract_entities(text: str, category: str = "", category_description: str = "", category_keywords: list[str] | None = None) -> ExtractionResult:
    if not text.strip():
        return ExtractionResult()

    # Skip chunks that look like website navigation / boilerplate
    nav_indicators = ["[Resources]", "[Contact]", "breadcrumb", "Skip to main", "cookie policy"]
    lower_text = text.lower()
    nav_count = sum(1 for ind in nav_indicators if ind.lower() in lower_text)
    if nav_count >= 2:
        logger.debug("Skipping navigation/boilerplate chunk for entity extraction")
        return ExtractionResult()

    prompt = _build_prompt(category, category_description, category_keywords)

    try:
        response = generate(system_prompt=prompt, user_prompt=text[:4000], temperature=0.0, max_tokens=4096)
        parsed = parse_json_response(response)
        entities = parsed.get("entities", [])
        relationships = parsed.get("relationships", [])
        sections = parsed.get("sections", [])

        # Filter out junk entities
        junk_names = {"x", "facebook", "instagram", "twitter", "youtube", "linkedin",
                      "news", "help center", "contact", "resources", "publications",
                      "home", "about", "search", "menu", "footer", "header",
                      "privacy", "terms", "cookie", "sitemap", "login", "sign in"}
        entities = [e for e in entities if isinstance(e, dict)
                    and e.get("name", "").lower().strip() not in junk_names
                    and len(e.get("name", "")) > 1]

        logger.info(f"Extracted {len(entities)} entities, {len(relationships)} relationships (category: {category or 'default'})")
        return ExtractionResult(entities=entities, relationships=relationships, sections=sections)
    except Exception as e:
        logger.error(f"Entity extraction failed: {e}", exc_info=True)
        return ExtractionResult()
