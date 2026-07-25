# src/auth/api_key.py
"""API key validation for service clients (applications).

Keys may come from:
  1. DB-backed application keys (hashed in api_key_records) — preferred
  2. Legacy settings.api_keys comma-separated list — still accepted for migration

The in-memory hash cache is loaded at startup and refreshed when keys change.
"""
from __future__ import annotations

import hashlib
import logging
import secrets
from dataclasses import dataclass
from datetime import datetime, timezone

from src.config import settings

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ApiKeyContext:
    """Resolved service identity for a valid API key."""
    source: str  # "db" | "legacy"
    application_id: int | None = None
    application_name: str | None = None
    key_id: int | None = None
    key_prefix: str | None = None


# key_hash -> ApiKeyContext for active DB keys
_db_key_cache: dict[str, ApiKeyContext] = {}


def hash_api_key(key: str) -> str:
    return hashlib.sha256(key.encode("utf-8")).hexdigest()


def generate_api_key() -> str:
    """Return a new opaque secret (show once to the admin)."""
    return f"sk_{secrets.token_urlsafe(32)}"


def key_prefix(key: str, n: int = 12) -> str:
    if not key:
        return ""
    return key[:n] + ("…" if len(key) > n else "")


def set_db_key_cache(entries: dict[str, ApiKeyContext]) -> None:
    """Replace the in-memory DB key cache (called after load/seed/revoke)."""
    global _db_key_cache
    _db_key_cache = dict(entries)
    logger.info("API key cache loaded: %d DB key(s)", len(_db_key_cache))


def get_db_key_cache_size() -> int:
    return len(_db_key_cache)


def resolve_api_key(key: str) -> ApiKeyContext | None:
    """Return context if key is valid, else None."""
    if not key:
        return None
    # Prefer DB keys
    ctx = _db_key_cache.get(hash_api_key(key))
    if ctx is not None:
        return ctx
    # Legacy settings allowlist
    if key in settings.api_key_list:
        return ApiKeyContext(source="legacy", application_name="legacy-settings")
    return None


def validate_api_key(key: str) -> bool:
    return resolve_api_key(key) is not None


async def reload_api_key_cache(store) -> int:
    """Load active non-revoked keys from the metadata store into the cache."""
    rows = await store.list_active_api_key_hashes()
    entries: dict[str, ApiKeyContext] = {}
    for row in rows:
        entries[row["key_hash"]] = ApiKeyContext(
            source="db",
            application_id=row["application_id"],
            application_name=row["application_name"],
            key_id=row["key_id"],
            key_prefix=row.get("key_prefix"),
        )
    set_db_key_cache(entries)
    return len(entries)


async def touch_api_key_usage(store, key: str) -> None:
    """Best-effort last_used_at update for DB keys."""
    ctx = resolve_api_key(key)
    if not ctx or ctx.source != "db" or not ctx.key_id:
        return
    try:
        await store.touch_api_key(ctx.key_id)
    except Exception as e:
        logger.debug("touch_api_key failed: %s", e)


async def seed_legacy_api_keys(store) -> int:
    """Import settings.api_keys into a 'legacy' application if DB has no keys yet.

    Does not remove settings.api_keys — both continue to work. Idempotent:
    skips secrets already present by hash.
    """
    from src.config import settings as cfg

    existing = await store.count_api_keys()
    keys = cfg.api_key_list
    if not keys:
        await reload_api_key_cache(store)
        return 0

    app = await store.get_api_application_by_name("legacy")
    if app is None:
        app = await store.add_api_application(
            name="legacy",
            display_name="Legacy (settings.api_keys)",
            description="Keys imported from API_KEYS / Security settings string for migration.",
        )
    if app is None:
        await reload_api_key_cache(store)
        return 0

    created = 0
    for raw in keys:
        h = hash_api_key(raw)
        if await store.get_api_key_by_hash(h):
            continue
        await store.add_api_key_record(
            application_id=app.id,
            key_hash=h,
            key_prefix=key_prefix(raw),
            label="imported-from-settings",
        )
        created += 1

    # Even if DB already had keys, ensure cache is warm
    if existing == 0 and created == 0 and keys:
        # All were already present
        pass
    await reload_api_key_cache(store)
    if created:
        logger.info("Seeded %d legacy API key(s) into api_key_records", created)
    return created
