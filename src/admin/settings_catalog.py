"""Complete, typed admin settings catalog and durable partial updates."""
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import get_args, get_origin, Literal

from src.config import Settings, settings

SETTINGS_PATH = Path("data/settings.json")
SECRET_FIELDS = {"admin_password", "api_keys", "vllm_api_key", "jwt_secret_key",
                 "mcp_openwebui_jwt_secret", "sharepoint_client_secret", "registered_databases", "database_url"}
RESTART_FIELDS = {
    "database_url", "lancedb_path", "lancedb_table_name", "tabular_duckdb_path",
    "embedding_mode", "embedding_model_name", "embedding_dimension", "embedding_api_url",
    "rerank_model", "audit_log_path", "mcp_enabled", "mcp_path", "mcp_server_name",
    "mcp_stateless_http", "mcp_port", "mcp_alt_port", "registered_databases",
    "max_parallel_ingestion", "max_parallel_async_query", "max_async_query_jobs",
    "async_query_timeout_seconds", "async_query_ttl_seconds", "llm_concurrency",
    "kg_chunk_token_size", "kg_chunk_overlap_token_size",
}


def configured_values():
    values = settings.model_dump()
    if SETTINGS_PATH.exists():
        try:
            saved = json.loads(SETTINGS_PATH.read_text())
            for name in RESTART_FIELDS:
                if name in saved:
                    values[name] = saved[name]
        except (OSError, ValueError):
            pass
    return values


def prepare_update(form):
    def optional(name):
        if name not in form:
            return None
        return form.getlist(name)[-1] if hasattr(form, "getlist") else form[name]
    values = configured_values()
    fields = Settings.model_fields
    for name in fields:
        if name not in form:
            continue
        raw = form.getlist(name)[-1] if hasattr(form, "getlist") else form[name]
        raw = str(raw).strip()
        if not raw:
            if name in {"admin_username", "admin_password", "api_keys", "vllm_base_url", "vllm_model_name", "embedding_mode", "embedding_api_url", "embedding_model_name"}:
                continue
            if name in SECRET_FIELDS and optional("keep_blank_secrets") == "true":
                continue
            if fields[name].annotation in (int, float, bool):
                continue
        values[name] = raw
    for name in SECRET_FIELDS - {"admin_password", "jwt_secret_key", "database_url"}:
        if optional("clear_" + name) == "true":
            values[name] = ""
    # Validation happens before any live value or persisted file changes.
    validated = Settings.model_validate(values)
    if validated.chunk_overlap >= validated.chunk_size:
        raise ValueError("Chunk overlap must be smaller than chunk size.")
    if validated.kg_chunk_overlap_token_size >= validated.kg_chunk_token_size:
        raise ValueError("Graph chunk overlap must be smaller than graph chunk size.")
    if validated.entity_merge_review_threshold > validated.entity_merge_auto_threshold:
        raise ValueError("Entity review threshold must not exceed auto-merge threshold.")
    return validated.model_dump()


def apply_live(values):
    for name, value in values.items():
        if name not in RESTART_FIELDS:
            setattr(settings, name, value)


def persist_settings(values):
    SETTINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".settings-", dir=SETTINGS_PATH.parent)
    try:
        with os.fdopen(fd, "w") as handle:
            json.dump(values, handle, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, SETTINGS_PATH)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def settings_catalog():
    values = configured_values()
    groups = {}
    for name, info in Settings.model_fields.items():
        group = ("Models and embeddings" if name.startswith(("vllm_", "embedding_", "llm_", "rerank_"))
                 else "Retrieval and answers" if name.startswith(("query_cache_", "answer_", "feedback_", "prf_", "strategy_", "sql_", "map_"))
                 else "Document processing" if name.startswith(("extraction_", "figure_", "kg_", "chunk_", "metadata_", "entity_"))
                 else "System, access and integrations")
        choices = list(get_args(info.annotation)) if get_origin(info.annotation) is Literal else []
        limits = {key: getattr(m, key) for m in info.metadata
                  for key in ("ge", "le") if hasattr(m, key)}
        groups.setdefault(group, []).append({
            "name": name, "label": info.title or name.replace("_", " ").capitalize(),
            "description": info.description or "",
            "value": "" if name in SECRET_FIELDS else values[name],
            "secret": name in SECRET_FIELDS,
            "configured": bool(values[name]) if name in SECRET_FIELDS else False,
            "kind": "boolean" if info.annotation is bool else "number" if info.annotation in (int, float) else "text",
            "step": "1" if info.annotation is int else "any", "choices": choices,
            "minimum": limits.get("ge"), "maximum": limits.get("le"),
            "restart": name in RESTART_FIELDS,
            "pending": name in RESTART_FIELDS and values[name] != getattr(settings, name),
        })
    return groups
