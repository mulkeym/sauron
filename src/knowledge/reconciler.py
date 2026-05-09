# src/knowledge/reconciler.py
"""Entity reconciliation — finds duplicate entities and merges or proposes merges."""
from src.config import settings
from src.db.metadata import MetadataStore
from src.generation.llm_client import generate, parse_json_response

RECONCILIATION_PROMPT = """You are comparing two entities from a knowledge graph to determine if they refer to the same real-world thing.

Entity A: "{name_a}" (type: {type_a})
Entity B: "{name_b}" (type: {type_b})

Consider:
- Are these the same entity? (e.g., "ONR" and "Office of Naval Research" are the same)
- Acronyms, abbreviations, spelling variations, case differences
- Different names for the same concept

Respond with ONLY valid JSON:
{{"is_same": true/false, "confidence": 0.0-1.0, "reason": "brief explanation", "canonical_name": "preferred name if same"}}"""


async def reconcile_entities(metadata_store: MetadataStore) -> dict:
    """Scan all entities for potential duplicates. Auto-merge high confidence, propose others for review."""
    entities = await metadata_store.list_entities(limit=1000)
    if len(entities) < 2:
        return {"auto_merged": 0, "proposed": 0, "scanned": 0}

    auto_merged = 0
    proposed = 0
    scanned = 0
    already_checked = set()

    # Group by type for faster comparison
    by_type: dict[str, list] = {}
    for e in entities:
        by_type.setdefault(e.entity_type, []).append(e)

    for entity_type, group in by_type.items():
        for i, a in enumerate(group):
            for b in group[i + 1:]:
                pair_key = (min(a.id, b.id), max(a.id, b.id))
                if pair_key in already_checked:
                    continue
                already_checked.add(pair_key)

                # Quick pre-filter: skip if names are very different length
                if abs(len(a.name) - len(b.name)) > max(len(a.name), len(b.name)) * 0.7:
                    continue

                # Quick check: exact match after normalization
                if a.name.lower().strip() == b.name.lower().strip():
                    await metadata_store.merge_entities(a.id, b.id)
                    await metadata_store.add_merge_proposal(
                        a.id, a.name, b.id, b.name, 1.0,
                        "Exact match (case-insensitive)", status="auto_merged",
                    )
                    auto_merged += 1
                    continue

                # LLM check for potential matches
                import asyncio
                response = await asyncio.to_thread(
                    generate,
                    system_prompt=RECONCILIATION_PROMPT.format(
                        name_a=a.name, type_a=a.entity_type,
                        name_b=b.name, type_b=b.entity_type,
                    ),
                    user_prompt="Are these the same entity?",
                    temperature=0.0, max_tokens=256,
                )
                scanned += 1

                try:
                    parsed = parse_json_response(response)
                    if not parsed.get("is_same", False):
                        continue

                    confidence = parsed.get("confidence", 0.0)
                    reason = parsed.get("reason", "")

                    if confidence >= settings.entity_merge_auto_threshold:
                        await metadata_store.merge_entities(a.id, b.id)
                        await metadata_store.add_merge_proposal(
                            a.id, a.name, b.id, b.name, confidence,
                            reason, status="auto_merged",
                        )
                        auto_merged += 1
                    elif confidence >= settings.entity_merge_review_threshold:
                        await metadata_store.add_merge_proposal(
                            a.id, a.name, b.id, b.name, confidence, reason,
                        )
                        proposed += 1
                except Exception:
                    continue

    return {"auto_merged": auto_merged, "proposed": proposed, "scanned": scanned}
