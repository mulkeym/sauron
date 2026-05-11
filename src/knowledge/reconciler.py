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


class ReconciliationStatus:
    """Tracks reconciliation progress for UI polling."""
    def __init__(self):
        self.running = False
        self.total_pairs = 0
        self.scanned = 0
        self.auto_merged = 0
        self.proposed = 0
        self.skipped = 0
        self.current_pair = ""
        self.error = ""
        self.done = False
        self.stop_requested = False

    @property
    def progress_pct(self):
        if self.total_pairs == 0:
            return 0
        return int((self.scanned + self.skipped + self.auto_merged) / self.total_pairs * 100)


# Global status for UI polling
_status = ReconciliationStatus()


def get_reconciliation_status() -> ReconciliationStatus:
    return _status


def stop_reconciliation():
    _status.stop_requested = True


async def reconcile_entities(metadata_store: MetadataStore) -> dict:
    """Scan all entities for potential duplicates. Auto-merge high confidence, propose others for review."""
    global _status
    _status = ReconciliationStatus()
    _status.running = True

    entities = await metadata_store.list_entities(limit=1000)
    if len(entities) < 2:
        _status.running = False
        _status.done = True
        return {"auto_merged": 0, "proposed": 0, "scanned": 0}

    already_checked = set()

    # Group by type for faster comparison
    by_type: dict[str, list] = {}
    for e in entities:
        by_type.setdefault(e.entity_type, []).append(e)

    # Count total pairs for progress tracking
    total_pairs = 0
    for group in by_type.values():
        n = len(group)
        total_pairs += n * (n - 1) // 2
    _status.total_pairs = total_pairs

    for entity_type, group in by_type.items():
        for i, a in enumerate(group):
            if _status.stop_requested:
                break
            for b in group[i + 1:]:
                if _status.stop_requested:
                    break

                pair_key = (min(a.id, b.id), max(a.id, b.id))
                if pair_key in already_checked:
                    continue
                already_checked.add(pair_key)

                _status.current_pair = f"{a.name} vs {b.name}"

                # Quick pre-filter: skip if names are very different length
                if abs(len(a.name) - len(b.name)) > max(len(a.name), len(b.name)) * 0.7:
                    _status.skipped += 1
                    continue

                # Quick check: exact match after normalization
                if a.name.lower().strip() == b.name.lower().strip():
                    await metadata_store.merge_entities(a.id, b.id)
                    await metadata_store.add_merge_proposal(
                        a.id, a.name, b.id, b.name, 1.0,
                        "Exact match (case-insensitive)", status="auto_merged",
                    )
                    _status.auto_merged += 1
                    continue

                # LLM check for potential matches
                import asyncio
                try:
                    response = await asyncio.to_thread(
                        generate,
                        system_prompt=RECONCILIATION_PROMPT.format(
                            name_a=a.name, type_a=a.entity_type,
                            name_b=b.name, type_b=b.entity_type,
                        ),
                        user_prompt="Are these the same entity?",
                        temperature=0.0, max_tokens=1024,
                    )
                except Exception as e:
                    _status.scanned += 1
                    continue

                _status.scanned += 1

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
                        _status.auto_merged += 1
                    elif confidence >= settings.entity_merge_review_threshold:
                        await metadata_store.add_merge_proposal(
                            a.id, a.name, b.id, b.name, confidence, reason,
                        )
                        _status.proposed += 1
                except Exception:
                    continue

    _status.running = False
    _status.done = True
    _status.current_pair = ""
    return {"auto_merged": _status.auto_merged, "proposed": _status.proposed, "scanned": _status.scanned}
