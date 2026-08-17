import pytest
import pytest_asyncio

from src.audit.activity import (
    apply_result_to_span,
    format_activity_row,
    query_activity_span,
    record_query_activity,
)
from src.db.metadata import MetadataStore


@pytest_asyncio.fixture
async def store():
    s = MetadataStore("sqlite+aiosqlite:///:memory:")
    await s.init()
    yield s


@pytest.mark.asyncio
async def test_record_and_list_via_helper(store):
    await record_query_activity(
        source="mcp", tool="ask", username="mike",
        user_groups=["finance"], query_text="pto?", store=store,
    )
    rows = await store.list_recent_query_activity(10)
    assert len(rows) == 1
    assert rows[0].username == "mike"


@pytest.mark.asyncio
async def test_record_swallows_store_errors():
    class Boom:
        async def add_query_activity(self, **kwargs):
            raise RuntimeError("db down")

    await record_query_activity(source="rest", tool="query", store=Boom())


@pytest.mark.asyncio
async def test_span_records_success_and_duration(store):
    async with query_activity_span(
        source="rest", tool="query", username="mike",
        user_groups=["finance"], query_text="expense policy?", store=store,
    ) as span:
        span.strategy = "lookup"
        span.cache_hit = False
    rows = await store.list_recent_query_activity(1)
    assert rows[0].status == "ok"
    assert rows[0].strategy == "lookup"
    assert rows[0].duration_seconds >= 0


@pytest.mark.asyncio
async def test_span_records_error_and_reraises(store):
    with pytest.raises(ValueError, match="boom"):
        async with query_activity_span(
            source="mcp", tool="ask", username="mike",
            query_text="q", store=store,
        ):
            raise ValueError("boom")
    rows = await store.list_recent_query_activity(1)
    assert rows[0].status == "error"
    assert rows[0].error == "boom"
    assert rows[0].duration_seconds >= 0


def test_apply_result_to_span_error_dict():
    span = query_activity_span(source="mcp", tool="lookup_document", query_text="missing.pdf")
    apply_result_to_span(span, {"error": "Document 'missing.pdf' not found."})
    assert span.status == "error"
    assert "not found" in span.error


def test_apply_result_to_span_forwards_query_type_and_cache():
    span = query_activity_span(source="mcp", tool="ask", query_text="q")
    apply_result_to_span(span, {"query_type": "lookup", "cached": True, "answer": "ok"})
    assert span.cache_hit is True
    assert span.strategy == "lookup"


def test_apply_result_to_span_default_strategy_when_no_query_type():
    span = query_activity_span(source="mcp", tool="query_database", query_text="q")
    apply_result_to_span(span, {"sql": "SELECT 1", "results": []}, default_strategy="structured")
    assert span.strategy == "structured"
    assert span.status == "ok"


def test_format_activity_row():
    class Row:
        created_at = __import__("datetime").datetime(2026, 8, 16, 14, 5)
        source = "mcp"
        tool = "ask"
        strategy = "lookup"
        username = "mike"
        user_groups = ["finance", "executives", "engineering", "clinical", "medical"]
        query_text = "x" * 90
        duration_seconds = 12.44
        status = "ok"
        cache_hit = False

    d = format_activity_row(Row())
    assert d["when"] == "08-16 14:05"
    assert d["type"] == "MCP · ask"
    assert d["strategy"] == "lookup"
    assert d["username"] == "mike"
    assert d["groups"] == "finance, executives, engineering, +2"
    assert d["question"].endswith("...")
    assert len(d["question"]) == 83  # 80 + "..."
    assert d["duration"] == "12.4s"
    assert d["status"] == "ok"


def test_format_activity_row_cache_and_blanks():
    class Row:
        created_at = None
        source = "playground"
        tool = "playground"
        strategy = ""
        username = ""
        user_groups = []
        query_text = ""
        duration_seconds = 0.0
        status = "ok"
        cache_hit = True

    d = format_activity_row(Row())
    assert d["type"] == "Playground · playground"
    assert d["strategy"] == "—"
    assert d["username"] == "—"
    assert d["groups"] == "—"
    assert d["question"] == "—"
    assert d["status"] == "cache"
