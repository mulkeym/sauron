from dataclasses import dataclass, field
from src.generation.llm_client import generate, parse_json_response

EXTRACTION_PROMPT = """Extract entities, relationships, and document sections from the following text.

Entity types: person, organization, policy, project, date, system, location, document_section
Relationship types: references, governs, authored_by, allocated_to, requires, part_of, related_to

Respond with ONLY valid JSON:
{
  "entities": [{"name": "...", "type": "..."}],
  "relationships": [{"source": "...", "target": "...", "type": "..."}],
  "sections": [{"name": "...", "parent": null}]
}

If no entities are found, return empty arrays."""


@dataclass
class ExtractionResult:
    entities: list[dict] = field(default_factory=list)
    relationships: list[dict] = field(default_factory=list)
    sections: list[dict] = field(default_factory=list)


def extract_entities(text: str) -> ExtractionResult:
    if not text.strip():
        return ExtractionResult()
    response = generate(system_prompt=EXTRACTION_PROMPT, user_prompt=text[:3000], temperature=0.0, max_tokens=1024)
    try:
        parsed = parse_json_response(response)
        return ExtractionResult(entities=parsed.get("entities", []), relationships=parsed.get("relationships", []), sections=parsed.get("sections", []))
    except Exception:
        return ExtractionResult()
