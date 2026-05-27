from __future__ import annotations
from dataclasses import dataclass, field

_PROV_RANK = {"curated": 3, "learned": 2, "auto": 1}


@dataclass
class ResolvedHints:
    column_glossaries: dict[str, dict[str, str]] = field(default_factory=dict)  # col -> {code: meaning}
    column_notes: dict[str, str] = field(default_factory=dict)                  # col -> note
    table_notes: list[str] = field(default_factory=list)


def _better(a, b) -> bool:
    """True if hint ``a`` outranks hint ``b`` (provenance, then confidence, then recency)."""
    ra, rb = _PROV_RANK.get(a.provenance, 0), _PROV_RANK.get(b.provenance, 0)
    if ra != rb:
        return ra > rb
    if (a.confidence or 0) != (b.confidence or 0):
        return (a.confidence or 0) > (b.confidence or 0)
    return bool((a.created_at or 0) and (b.created_at or 0) and a.created_at >= b.created_at)


def resolve_hints(table_schema, doc_record, hint_store) -> ResolvedHints:
    """Hints applicable to ``table_schema`` given its owning ``doc_record``. Pure;
    fail-safe (missing doc / malformed payloads are skipped, never raised)."""
    out = ResolvedHints()
    if doc_record is None:
        return out
    col_names = {c.name for c in table_schema.columns}

    scopes = [("category", getattr(doc_record, "category", "") or ""),
              ("dataset", str(getattr(doc_record, "dataset_id", "") or ""))]
    hints = []
    for st, sv in scopes:
        if sv:
            hints.extend(hint_store.for_scope(st, sv))

    # Dedup non-table hints per (hint_type, target_column), keeping the best.
    best: dict[tuple, object] = {}
    for h in hints:
        try:
            if h.hint_type == "table_note":
                text = (h.payload or {}).get("text", "")
                if text:
                    out.table_notes.append(text)
                continue
            if h.target_column not in col_names:
                continue
            key = (h.hint_type, h.target_column)
            if key not in best or _better(h, best[key]):
                best[key] = h
        except Exception:
            continue

    for (hint_type, col), h in best.items():
        if hint_type == "value_glossary" and isinstance(h.payload, dict):
            out.column_glossaries[col] = {str(k): str(v) for k, v in h.payload.items()}
        elif hint_type == "column_note":
            text = (h.payload or {}).get("text", "")
            if text:
                out.column_notes[col] = text
    return out
