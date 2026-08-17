from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

SOURCE_LABELS = {
    "mcp": "MCP",
    "rest": "REST",
    "openai": "OpenAI",
    "playground": "Playground",
}


async def record_query_activity(
    *,
    source: str,
    tool: str,
    username: str = "",
    user_groups: list | None = None,
    query_text: str = "",
    strategy: str = "",
    duration_seconds: float = 0.0,
    status: str = "ok",
    cache_hit: bool = False,
    error: str = "",
    store=None,
) -> None:
    try:
        if store is None:
            from src.api.routes_ingest import get_metadata_store
            store = get_metadata_store()
        await store.add_query_activity(
            source=source,
            tool=tool,
            username=username,
            user_groups=user_groups,
            query_text=query_text,
            strategy=strategy,
            duration_seconds=duration_seconds,
            status=status,
            cache_hit=cache_hit,
            error=error,
        )
    except Exception as exc:
        logger.warning("Failed to record query activity: %s", exc)


@dataclass
class QueryActivitySpan:
    source: str
    tool: str
    username: str = ""
    user_groups: list = field(default_factory=list)
    query_text: str = ""
    strategy: str = ""
    cache_hit: bool = False
    status: str = "ok"
    error: str = ""
    store: object | None = None
    _start: float = 0.0

    async def __aenter__(self) -> QueryActivitySpan:
        self._start = time.time()
        return self

    async def __aexit__(self, exc_type, exc, tb) -> bool:
        if exc_type is not None:
            self.status = "error"
            self.error = str(exc)[:200]
        await record_query_activity(
            source=self.source,
            tool=self.tool,
            username=self.username,
            user_groups=self.user_groups,
            query_text=self.query_text,
            strategy=self.strategy,
            duration_seconds=round(time.time() - self._start, 2),
            status=self.status,
            cache_hit=self.cache_hit,
            error=self.error,
            store=self.store,
        )
        return False


def query_activity_span(
    *,
    source: str,
    tool: str,
    username: str = "",
    user_groups: list | None = None,
    query_text: str = "",
    store=None,
) -> QueryActivitySpan:
    return QueryActivitySpan(
        source=source,
        tool=tool,
        username=username,
        user_groups=list(user_groups or []),
        query_text=query_text,
        store=store,
    )


def apply_result_to_span(span: QueryActivitySpan, result, *, default_strategy: str = "") -> None:
    if default_strategy and not span.strategy:
        span.strategy = default_strategy
    if not isinstance(result, dict):
        return
    err = result.get("error")
    if err:
        span.status = "error"
        span.error = str(err)[:200]
    if result.get("query_type"):
        span.strategy = str(result["query_type"])
    if result.get("cached"):
        span.cache_hit = True
        if not span.strategy:
            span.strategy = "cache"


def format_activity_row(row) -> dict:
    created = getattr(row, "created_at", None)
    when = created.strftime("%m-%d %H:%M") if created is not None else "—"
    source = getattr(row, "source", "")
    tool = getattr(row, "tool", "")
    label = SOURCE_LABELS.get(source, source or "—")
    groups = [str(g) for g in (getattr(row, "user_groups", None) or []) if g]
    if not groups:
        groups_s = "—"
    elif len(groups) <= 3:
        groups_s = ", ".join(groups)
    else:
        groups_s = ", ".join(groups[:3]) + f", +{len(groups) - 3}"
    question = (getattr(row, "query_text", None) or "").strip()
    if not question:
        question_s = "—"
    elif len(question) <= 80:
        question_s = question
    else:
        question_s = question[:80] + "..."
    status = getattr(row, "status", "ok") or "ok"
    if status == "ok" and getattr(row, "cache_hit", False):
        status = "cache"
    return {
        "when": when,
        "type": f"{label} · {tool}",
        "strategy": getattr(row, "strategy", "") or "—",
        "username": getattr(row, "username", "") or "—",
        "groups": groups_s,
        "question": question_s,
        "duration": f"{float(getattr(row, 'duration_seconds', 0) or 0):.1f}s",
        "status": status,
    }
