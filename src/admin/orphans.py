"""Decision logic for purging orphaned vector-store chunks.

Kept as a pure function so the safety guards can be unit-tested without the
web app, stores, or auth. The route handles I/O; this decides what to do.
"""
from __future__ import annotations


def plan_orphan_purge(
    metadata_doc_ids: set[str],
    chunk_doc_ids: set[str],
    ingestion_active: bool,
) -> tuple[str, set[str]]:
    """Decide which chunk doc_ids are safe to purge as orphans.

    Returns (status, orphan_ids):
      - "refused_active": ingestion is running; in-flight jobs legitimately
        have chunks before their metadata commit, so we can't tell those from
        orphans. orphan_ids is empty.
      - "refused_empty_metadata": metadata has zero documents. We cannot
        distinguish "every chunk is an orphan" from "metadata failed to load",
        so refuse rather than risk wiping the whole store. orphan_ids is empty.
      - "ok": orphan_ids = chunk doc_ids with no metadata document row.
    """
    if ingestion_active:
        return ("refused_active", set())
    if not metadata_doc_ids:
        return ("refused_empty_metadata", set())
    return ("ok", chunk_doc_ids - metadata_doc_ids)
