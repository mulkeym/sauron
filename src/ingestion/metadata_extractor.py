"""Extract structured metadata from document text via LLM."""
import json
import logging

from src.config import settings
from src.generation.llm_client import generate

logger = logging.getLogger(__name__)

EXTRACTION_PROMPT = """Extract structured metadata from this document. Return ONLY valid JSON with these fields.
Leave fields as empty arrays [] if not found. The summary field should be a string, not an array.
Be thorough — include ALL items found.

Fields:
- summary: string, 2-4 sentence summary of the document's purpose and key content
- entities: named entities (companies, agencies, programs, systems)
- people: person names mentioned
- organizations: organizations, departments, agencies
- locations: cities, states, countries, facilities
- dates: dates, fiscal years, time periods mentioned
- amounts: dollar amounts, quantities, percentages
- identifiers: contract numbers, policy numbers, regulation citations, system IDs
- topics: key topics and subject areas (3-8 keywords)
- procedures: processes, procedures, rules described
- action_items: action items, requirements, deadlines
- key_facts: important specific facts, conditions, or findings

Document ({filename}):
{text}"""

EMPTY_METADATA = {
    "summary": "",
    "entities": [],
    "people": [],
    "organizations": [],
    "locations": [],
    "dates": [],
    "amounts": [],
    "identifiers": [],
    "topics": [],
    "procedures": [],
    "action_items": [],
    "key_facts": [],
}


def extract_metadata(text: str, filename: str) -> dict:
    """Extract structured metadata from document text.

    Returns a dict with all metadata fields. On failure, returns EMPTY_METADATA.
    """
    if not settings.metadata_extraction_enabled:
        return dict(EMPTY_METADATA)

    # Truncate to configured max length
    doc_text = text[:settings.metadata_max_doc_length]

    try:
        raw = generate(
            system_prompt="You extract structured metadata from documents. Return ONLY valid JSON.",
            user_prompt=EXTRACTION_PROMPT.format(filename=filename, text=doc_text),
            temperature=0.0,
            max_tokens=4096,
        )
    except Exception as e:
        logger.warning(f"Metadata extraction LLM call failed for {filename}: {e}")
        return dict(EMPTY_METADATA)

    return _parse_metadata_response(raw, filename)


def _parse_metadata_response(raw: str, filename: str) -> dict:
    """Parse LLM response into metadata dict with fallback to json_repair."""
    import re

    # Strip markdown fences and preamble
    cleaned = re.sub(r"```(?:json)?\s*", "", raw).replace("```", "").strip()
    # Remove any text before the first {
    brace_idx = cleaned.find("{")
    if brace_idx > 0:
        cleaned = cleaned[brace_idx:]

    # Try standard json.loads first
    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError:
        # Fallback: json_repair (already installed via LightRAG)
        try:
            import json_repair
            parsed = json_repair.loads(cleaned)
        except Exception as e:
            logger.warning(f"Metadata JSON parse failed for {filename}: {e}")
            logger.debug(f"Raw response: {raw[:500]}")
            return dict(EMPTY_METADATA)

    if not isinstance(parsed, dict):
        logger.warning(f"Metadata extraction returned non-dict for {filename}")
        return dict(EMPTY_METADATA)

    # Normalize: ensure all expected fields exist with correct types
    result = dict(EMPTY_METADATA)
    for key in EMPTY_METADATA:
        if key in parsed:
            if key == "summary":
                result["summary"] = str(parsed["summary"]) if parsed["summary"] else ""
            elif isinstance(parsed[key], list):
                result[key] = [str(item) for item in parsed[key] if item]
            else:
                result[key] = []
    return result


def merge_chunk_metadata(chunk_results: list[dict]) -> dict:
    """Merge metadata extracted from multiple chunks into one."""
    merged = dict(EMPTY_METADATA)
    summaries = []

    for chunk_meta in chunk_results:
        for key in EMPTY_METADATA:
            if key == "summary":
                if chunk_meta.get("summary"):
                    summaries.append(chunk_meta["summary"])
            else:
                existing = set(merged.get(key, []))
                for item in chunk_meta.get(key, []):
                    if item and item not in existing:
                        merged[key].append(item)
                        existing.add(item)

    # Use longest summary as the merged summary
    if summaries:
        merged["summary"] = max(summaries, key=len)

    return merged
