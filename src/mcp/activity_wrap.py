from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any

from src.audit.activity import apply_result_to_span, query_activity_span
from src.mcp.auth import current_mcp_context


async def run_logged_mcp_tool(
    *,
    tool: str,
    query_text: str,
    fn: Callable[[], Awaitable[Any]],
    default_strategy: str = "",
) -> Any:
    ctx = current_mcp_context()
    async with query_activity_span(
        source="mcp",
        tool=tool,
        username=ctx.username,
        user_groups=list(ctx.groups),
        query_text=query_text,
    ) as span:
        result = await fn()
        apply_result_to_span(span, result, default_strategy=default_strategy)
        return result
