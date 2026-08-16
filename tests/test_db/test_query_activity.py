import pytest
import pytest_asyncio

from src.db.metadata import MetadataStore


@pytest_asyncio.fixture
async def store():
    s = MetadataStore("sqlite+aiosqlite:///:memory:")
    await s.init()
    yield s


@pytest.mark.asyncio
async def test_add_and_list_newest_first(store):
    await store.add_query_activity(source="mcp", tool="ask", username="mike", query_text="first")
    await store.add_query_activity(source="rest", tool="query", username="bob", query_text="second")
    rows = await store.list_recent_query_activity(10)
    assert [r.query_text for r in rows] == ["second", "first"]
    assert rows[0].source == "rest"
    assert rows[0].tool == "query"
    assert rows[0].username == "bob"
    assert rows[0].status == "ok"
    assert rows[0].cache_hit is False


@pytest.mark.asyncio
async def test_list_caps_at_limit(store):
    for i in range(12):
        await store.add_query_activity(source="mcp", tool="ask", query_text=f"q{i}")
    rows = await store.list_recent_query_activity(10)
    assert len(rows) == 10
    assert rows[0].query_text == "q11"
    assert rows[-1].query_text == "q2"


@pytest.mark.asyncio
async def test_truncates_query_text_and_error(store):
    row = await store.add_query_activity(
        source="mcp",
        tool="ask",
        query_text="x" * 600,
        status="error",
        error="e" * 300,
        user_groups=["finance", "executives"],
        strategy="lookup",
        duration_seconds=1.25,
        cache_hit=True,
    )
    assert len(row.query_text) == 500
    assert len(row.error) == 200
    assert row.user_groups == ["finance", "executives"]
    assert row.strategy == "lookup"
    assert row.duration_seconds == 1.25
    assert row.cache_hit is True
