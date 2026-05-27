from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class SchemaHint:
    """One piece of curated (or, later, auto/learned) domain knowledge attached to
    a source/collection, injected into the text-to-SQL schema prompt.

    scope_type/scope_value bind the hint to a ``category`` (name) or ``dataset``
    (id as string). ``target_column`` names the column the hint applies to
    (None for ``table_note``); matched by name across every table in scope.
    ``hint_type`` is one of ``value_glossary`` | ``column_note`` | ``table_note``.
    For value_glossary, ``payload`` is ``{code: meaning}``; for notes,
    ``{"text": ...}``. ``provenance`` is ``curated`` | ``auto`` | ``learned``."""
    scope_type: str
    scope_value: str
    hint_type: str
    target_column: str | None
    payload: dict
    provenance: str = "curated"
    confidence: float = 1.0
    id: int | None = None
    created_at: datetime | None = None
    created_by: str = ""


class HintStore:
    """In-memory store of SchemaHints, indexed by (scope_type, scope_value).
    Parallels SchemaRegistry; loaded at startup from the metadata store."""

    def __init__(self):
        self._by_scope: dict[tuple[str, str], list[SchemaHint]] = {}

    def register(self, hint: SchemaHint) -> None:
        self._by_scope.setdefault((hint.scope_type, hint.scope_value), []).append(hint)

    def for_scope(self, scope_type: str, scope_value: str) -> list[SchemaHint]:
        return list(self._by_scope.get((scope_type, scope_value), []))

    def clear(self) -> None:
        self._by_scope.clear()
