import logging
from dataclasses import dataclass, field
from src.generation.llm_client import generate, parse_json_response

logger = logging.getLogger(__name__)

EXTRACTION_PROMPT = """Extract important entities and their relationships from the text. Return ONLY valid JSON.

IMPORTANT:
- Focus on substantive content entities: companies, agencies, people, contracts, financial amounts, locations, projects
- IGNORE website navigation, social media links, menu items, footers, headers, breadcrumbs
- IGNORE generic labels like "News", "Help Center", "Contact", "Resources", "Publications"
- Only extract entities that carry real informational value

Entity types: organization, person, contractor, contract, agency, location, project, financial_amount, regulation, system, date
Relationship types: awarded_to, contracted_by, located_in, funds, manages, references, part_of, related_to, modifies, supports

Examples of GOOD entities: "BL Harbert International LLC" (contractor), "Naval Air Systems Command" (agency), "$171,506,091" (financial_amount), "W91278-26-C-A004" (contract)
Examples of BAD entities (skip these): "Facebook", "X", "Instagram", "Help Center", "News", "Contact"

{"entities": [{"name": "...", "type": "..."}], "relationships": [{"source": "...", "target": "...", "type": "..."}]}"""


@dataclass
class ExtractionResult:
    entities: list[dict] = field(default_factory=list)
    relationships: list[dict] = field(default_factory=list)
    sections: list[dict] = field(default_factory=list)


def extract_entities(text: str) -> ExtractionResult:
    if not text.strip():
        return ExtractionResult()

    # Skip chunks that look like website navigation / boilerplate
    nav_indicators = ["[Resources]", "[Contact]", "breadcrumb", "Skip to main", "cookie policy"]
    lower_text = text.lower()
    nav_count = sum(1 for ind in nav_indicators if ind.lower() in lower_text)
    if nav_count >= 2:
        logger.debug("Skipping navigation/boilerplate chunk for entity extraction")
        return ExtractionResult()

    try:
        response = generate(system_prompt=EXTRACTION_PROMPT, user_prompt=text[:4000], temperature=0.0, max_tokens=4096)
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

        logger.info(f"Extracted {len(entities)} entities, {len(relationships)} relationships")
        return ExtractionResult(entities=entities, relationships=relationships, sections=sections)
    except Exception as e:
        logger.error(f"Entity extraction failed: {e}", exc_info=True)
        return ExtractionResult()
