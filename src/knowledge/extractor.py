from dataclasses import dataclass, field
from src.generation.llm_client import generate, parse_json_response

EXTRACTION_PROMPT = """Extract entities and relationships from the text. Return ONLY JSON, no explanation.

Types: person, organization, policy, project, date, system, location
Relationships: references, governs, authored_by, allocated_to, requires, part_of, related_to

{"entities": [{"name": "...", "type": "..."}], "relationships": [{"source": "...", "target": "...", "type": "..."}], "sections": []}"""


@dataclass
class ExtractionResult:
    entities: list[dict] = field(default_factory=list)
    relationships: list[dict] = field(default_factory=list)
    sections: list[dict] = field(default_factory=list)


def extract_entities(text: str) -> ExtractionResult:
    if not text.strip():
        return ExtractionResult()
    response = generate(system_prompt=EXTRACTION_PROMPT, user_prompt=text[:2000], temperature=0.0, max_tokens=2048)
    try:
        parsed = parse_json_response(response)
        return ExtractionResult(entities=parsed.get("entities", []), relationships=parsed.get("relationships", []), sections=parsed.get("sections", []))
    except Exception:
        return ExtractionResult()
