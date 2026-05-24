"""Helpers for LightRAG entity-extraction output, kept free of the `lightrag`
import so they can be unit-tested without the (container-only) dependency.

LightRAG parses extraction output into records of the form::

    entity<|#|>name<|#|>type<|#|>description
    relation<|#|>src<|#|>tgt<|#|>keywords<|#|>description

Small local models (e.g. gemma) intermittently emit a field short — most often
omitting the entity `type` or the relation `keywords` — and LightRAG's parser
silently discards those records ("LLM output format error; found 3/4 fields").
`repair_extraction_format` reinserts a sane default so real entities survive.
"""
from __future__ import annotations

import os

# Must match LightRAG's DEFAULT_TUPLE_DELIMITER (lightrag.prompt).
TUPLE_DELIMITER = "<|#|>"

# Default filler values for fields the LLM dropped.
_DEFAULT_ENTITY_TYPE = "category"
_DEFAULT_RELATION_KEYWORDS = "related"

# File extensions whose content is tabular/numeric and yields no graph entities.
# Running KG extraction on them only wastes LLM time and triggers worker timeouts.
GRAPH_SKIP_EXTENSIONS = {".xls", ".xlsx", ".xlsm", ".xlsb", ".csv", ".tsv"}


def _repair_record(line: str) -> str:
    """Repair a single record line; return it unchanged if not a short record."""
    stripped = line.strip()
    if not stripped:
        return line
    parts = stripped.split(TUPLE_DELIMITER)
    kind = parts[0].strip().lower()

    # Entity expects 4 fields. A 3-field record means the LLM dropped the type
    # (observed: name + description present, type missing). Insert a default type.
    if kind == "entity" and len(parts) == 3:
        parts.insert(2, _DEFAULT_ENTITY_TYPE)
        return TUPLE_DELIMITER.join(parts)

    # Relation expects 5 fields. A 4-field record means the LLM dropped the
    # keywords field (description present). Insert default keywords before it.
    if kind in ("relation", "relationship") and len(parts) == 4:
        parts.insert(3, _DEFAULT_RELATION_KEYWORDS)
        return TUPLE_DELIMITER.join(parts)

    return line


def repair_extraction_format(text: str) -> str:
    """Repair malformed entity/relation records in raw LLM extraction output.

    Operates line by line, touching only lines that are short entity/relation
    records. Valid records and all non-record lines (prose, delimiters) pass
    through unchanged, so this is safe to apply to any LLM response.
    """
    if not text or TUPLE_DELIMITER not in text:
        return text
    return "\n".join(_repair_record(line) for line in text.split("\n"))


def should_skip_graph(filename: str | None) -> bool:
    """True when a file's type is tabular/numeric and should bypass KG extraction."""
    if not filename:
        return False
    return os.path.splitext(filename)[1].lower() in GRAPH_SKIP_EXTENSIONS
