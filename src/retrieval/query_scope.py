"""Resolve document permissions before retrieval or answer-cache reuse."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

from src.config import settings


@dataclass(frozen=True)
class QueryScope:
    doc_ids: tuple[str, ...]
    revision: str


def resolve_query_scope_sync(user_groups, metadata_store=None, **kwargs):
    import asyncio
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(resolve_query_scope(user_groups, metadata_store, **kwargs))
    import concurrent.futures
    with concurrent.futures.ThreadPoolExecutor() as pool:
        return pool.submit(asyncio.run, resolve_query_scope(user_groups, metadata_store, **kwargs)).result()


def scoped_schemas(registry, state):
    from src.ingestion.tabular_store import DUCKDB_DATABASE, duckdb_table_name
    schemas = registry.list_for_user(state.get("user_groups", []))
    allowed = state.get("allowed_doc_ids")
    if allowed is None:
        return schemas
    prefixes = tuple(duckdb_table_name(d, "") for d in allowed)
    return [s for s in schemas
            if (s.database == DUCKDB_DATABASE and s.table.startswith(prefixes))
            or (s.database != DUCKDB_DATABASE and not state.get("dataset_id"))]


async def resolve_query_scope(user_groups, metadata_store=None, *, dataset_id=0,
                              allowed_doc_ids=None, mode="full", answer_profile=None) -> QueryScope:
    if metadata_store is None:
        from src.api.routes_ingest import get_metadata_store
        metadata_store = get_metadata_store()
    # Read the authoritative catalog, not stale ACL values in vector/graph indexes.
    docs = await metadata_store.list_documents()
    allowed = set(allowed_doc_ids) if allowed_doc_ids is not None else None
    visible = [d for d in docs
               if ("ALL" in user_groups or set(user_groups).intersection(d.acl_groups))
               and (not dataset_id or d.dataset_id == dataset_id)
               and (allowed is None or d.doc_id in allowed)]
    doc_ids = tuple(sorted(d.doc_id for d in visible))
    # Invalidate conservatively on any catalog change, including deletions,
    # source edits, ACL changes, and newly added documents. Never cache a result
    # across a changed retrieval/model configuration or evidence format.
    fields = ("doc_id", "content_hash", "filename", "doc_type", "acl_groups",
              "dataset_id", "category", "source_url", "chunk_count", "summary",
              "metadata_tags", "created_at")
    records = [{k: getattr(d, k, None) for k in fields} for d in docs]
    # LanceDB versions change on index content/ACL mutations even when catalog
    # metadata happens to be unchanged. Opening an existing table does not load models.
    index_revision = None
    from pathlib import Path
    if Path(settings.lancedb_path).exists():
        import lancedb
        db = lancedb.connect(settings.lancedb_path)
        if settings.lancedb_table_name in db.table_names():
            index_revision = db.open_table(settings.lancedb_table_name).version
    from src.agent.synthesizer import SYSTEM_PROMPT, USER_PROMPT_TEMPLATE
    if answer_profile is None:
        from src.agent.profiles import active_snapshot
        answer_profile = active_snapshot()
    payload = {
        "format": 2, "documents": sorted(records, key=lambda d: d["doc_id"]),
        "groups": sorted(set(user_groups)), "allowed": doc_ids,
        "dataset_id": dataset_id, "mode": mode,
        "index_revision": index_revision,
        "settings": settings.model_dump(),
        "prompts": [SYSTEM_PROMPT, USER_PROMPT_TEMPLATE],
        "answer_profile": answer_profile,
    }
    revision = hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()
    return QueryScope(doc_ids, revision)
