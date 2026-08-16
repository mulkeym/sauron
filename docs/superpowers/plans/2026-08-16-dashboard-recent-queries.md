# Dashboard Recent Queries Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Persist every finished Playground / REST / OpenAI / MCP tool call and show the last 10 on the admin dashboard under the stat cards.

**Architecture:** New `query_activity` SQLite table (not `query_metrics`, not the unused JSONL audit logger). A best-effort `query_activity_span` writes one row at the outermost surface. `RAGResponse.query_type` carries classifier / cache strategy so REST, OpenAI, and MCP `ask`-family rows can fill the strategy column. The dashboard `SELECT`s the last 10 on page load.

**Tech Stack:** FastAPI, Jinja2, SQLAlchemy 2 async + aiosqlite, pytest / pytest-asyncio, existing admin table CSS.

## Global Constraints

- `source` values are exactly `mcp` / `rest` / `openai` / `playground`.
- `tool` values are exactly: `ask`, `summarize_topic`, `compare`, `query_database`, `summarize_documents`, `search_documents`, `lookup_document`, `list_sources`, `list_documents`, `search_meetings`, `search_knowledge_graph`, `query`, `query_async`, `chat.completions`, `playground`.
- Type column renders as `{Surface} · {tool}` with labels `MCP` / `REST` / `OpenAI` / `Playground` and a middle dot `·`.
- Store `query_text` at most 500 chars; store `error` at most 200 chars; no stack traces in `error`.
- `status` is `ok` or `error`. UI status is `cache` when `status == "ok"` and `cache_hit` is true.
- `duration_seconds` is processing wall time, not time-in-queue.
- Log at the outermost surface only — never inside `ask()` or `agent_query`.
- Do not log MCP `get_result` or MCP resources.
- Do not change `query_metrics`, Settings query-metrics, or `/admin/audit`.
- Insert failures must not fail the user request. Activity-read failures must not blank the stat cards.
- Empty copy: `No queries recorded yet.` Read-error copy: `Unable to load recent queries.`
- Groups: more than 3 → first three names + `+N` (5 groups → `finance, executives, engineering, +2`).
- Question display: first 80 chars + ellipsis; `—` if blank.
- When column: `MM-DD HH:MM`.
- Time column: `{duration:.1f}s`.
- No live polling, pagination, or answer preview.

## File map

| File | Role |
|---|---|
| `src/db/models.py` | New `QueryActivity` model |
| `src/db/metadata.py` | Import model; `add_query_activity` / `list_recent_query_activity` |
| `src/audit/activity.py` | **Create.** `record_query_activity`, `query_activity_span`, display formatters, `apply_result_to_span` |
| `src/generation/rag_chain.py` | `RAGResponse.query_type`; cache path sets `"cache"` |
| `src/agent/graph.py` | Copy classify `query_type` onto `RAGResponse` |
| `src/mcp/tools_high.py` | Return `query_type` + `cached` from `ask` / `summarize_topic` / `compare` |
| `src/mcp/tools_low.py` | Forward those fields on `query_database` → `ask` fallback |
| `src/admin/routes.py` | Dashboard load; Playground terminal-state recording |
| `src/admin/templates/dashboard.html` | Recent queries table |
| `src/api/routes_query.py` | Span around `POST /api/v1/query` |
| `src/api/query_jobs.py` | Record on async worker complete / fail |
| `src/api/routes_openai_compat.py` | Span around chat completions |
| `src/mcp/server.py` | Span around every MCP tool except `get_result`; convert sync handlers to `async def` |
| `tests/test_db/test_query_activity.py` | **Create.** Store insert / list / caps |
| `tests/test_audit/test_activity.py` | **Create.** Helper + span + formatters |
| `tests/test_generation/test_agent_query_streamed.py` | Cache `query_type="cache"` |
| `tests/test_agent/test_graph.py` | Streamed / invoke responses carry `query_type` |
| `tests/test_mcp/test_tools_high.py` | `ask` return dict includes `query_type` / `cached` |
| `tests/test_mcp/test_tools_low.py` | Fallback forwards `query_type` / `cached` |
| `tests/test_admin/test_routes.py` | Dashboard empty / row / read-failure; fix existing dashboard mocks |
| `tests/test_api/test_routes_query.py` | REST route writes an activity row |
| `tests/test_api/test_query_jobs.py` | Async worker writes ok and error rows |
| `tests/test_mcp/test_server.py` | MCP wrappers record; `get_result` does not |

---

### Task 1: QueryActivity model and store methods

**Files:**
- Modify: `src/db/models.py` (append after `QueryMetrics`, around line 205)
- Modify: `src/db/metadata.py` (import list at lines 6–10; add two methods near other `add_`/`list_` methods)
- Test: `tests/test_db/test_query_activity.py`

**Interfaces:**
- Consumes: existing `Base`, `MetadataStore.init()` → `create_all`
- Produces:
  - `class QueryActivity` table `query_activity`
  - `MetadataStore.add_query_activity(*, source: str, tool: str, username: str = "", user_groups: list | None = None, query_text: str = "", strategy: str = "", duration_seconds: float = 0.0, status: str = "ok", cache_hit: bool = False, error: str = "") -> QueryActivity`
  - `MetadataStore.list_recent_query_activity(self, limit: int = 10) -> list[QueryActivity]`
  - Truncation: `query_text[:500]`, `error[:200]` inside `add_query_activity`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_db/test_query_activity.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `source .venv/bin/activate && python -m pytest tests/test_db/test_query_activity.py -v`

Expected: FAIL — `QueryActivity` / `add_query_activity` not defined.

- [ ] **Step 3: Implement the model and store methods**

In `src/db/models.py`, add after `QueryMetrics`:

```python
class QueryActivity(Base):
    __tablename__ = "query_activity"
    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
    )
    source: Mapped[str] = mapped_column(String, nullable=False)
    tool: Mapped[str] = mapped_column(String, nullable=False)
    username: Mapped[str] = mapped_column(String, default="")
    user_groups: Mapped[list] = mapped_column(JSON, default=list)
    query_text: Mapped[str] = mapped_column(String, default="")
    strategy: Mapped[str] = mapped_column(String, default="")
    duration_seconds: Mapped[float] = mapped_column(default=0.0)
    status: Mapped[str] = mapped_column(String, default="ok")
    cache_hit: Mapped[bool] = mapped_column(default=False)
    error: Mapped[str] = mapped_column(String, default="")
```

In `src/db/metadata.py`, add `QueryActivity` to the `src.db.models` import.

Add methods on `MetadataStore` (near the other `add_`/`list_` methods is fine):

```python
    async def add_query_activity(
        self,
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
    ):
        from src.db.models import QueryActivity
        record = QueryActivity(
            source=source,
            tool=tool,
            username=username or "",
            user_groups=list(user_groups or []),
            query_text=(query_text or "")[:500],
            strategy=strategy or "",
            duration_seconds=duration_seconds,
            status=status or "ok",
            cache_hit=bool(cache_hit),
            error=(error or "")[:200],
        )
        async with self.session_factory() as session:
            session.add(record)
            await session.commit()
            await session.refresh(record)
        return record

    async def list_recent_query_activity(self, limit: int = 10):
        from src.db.models import QueryActivity
        async with self.session_factory() as session:
            result = await session.execute(
                select(QueryActivity)
                .order_by(QueryActivity.created_at.desc(), QueryActivity.id.desc())
                .limit(limit)
            )
            return list(result.scalars().all())
```

`select` is already imported in this file.

- [ ] **Step 4: Run tests to verify they pass**

Run: `source .venv/bin/activate && python -m pytest tests/test_db/test_query_activity.py -v`

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/db/models.py src/db/metadata.py tests/test_db/test_query_activity.py
git commit -m "feat: add query_activity table and store methods"
```

---

### Task 2: Activity helper, span, and display formatters

**Files:**
- Create: `src/audit/activity.py`
- Test: `tests/test_audit/test_activity.py`

**Interfaces:**
- Consumes: `MetadataStore.add_query_activity` / `list_recent_query_activity` from Task 1
- Produces:
  - `async def record_query_activity(*, source, tool, username="", user_groups=None, query_text="", strategy="", duration_seconds=0.0, status="ok", cache_hit=False, error="", store=None) -> None`
  - `class QueryActivitySpan` with writable `strategy: str`, `cache_hit: bool`, `status: str`, `error: str`
  - `def query_activity_span(*, source: str, tool: str, username: str = "", user_groups: list | None = None, query_text: str = "", store=None) -> QueryActivitySpan` — async context manager
  - `def apply_result_to_span(span: QueryActivitySpan, result, *, default_strategy: str = "") -> None`
  - `def format_activity_row(row) -> dict` with keys `when`, `type`, `strategy`, `username`, `groups`, `question`, `duration`, `status`
  - `SOURCE_LABELS = {"mcp": "MCP", "rest": "REST", "openai": "OpenAI", "playground": "Playground"}`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_audit/test_activity.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `source .venv/bin/activate && python -m pytest tests/test_audit/test_activity.py -v`

Expected: FAIL — `src.audit.activity` cannot be imported.

- [ ] **Step 3: Implement `src/audit/activity.py`**

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `source .venv/bin/activate && python -m pytest tests/test_audit/test_activity.py tests/test_db/test_query_activity.py -v`

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/audit/activity.py tests/test_audit/test_activity.py
git commit -m "feat: add query activity recorder, span, and row formatters"
```

---

### Task 3: Plumb `query_type` through RAGResponse and MCP helpers

**Files:**
- Modify: `src/generation/rag_chain.py` (`RAGResponse` at line 25; cache return at lines 112–113)
- Modify: `src/agent/graph.py` (`run_agent` ~305, `run_agent_streamed` ~347, `run_agent_with_trace` ~409)
- Modify: `src/mcp/tools_high.py` (return dicts of `ask`, `summarize_topic`, `compare`)
- Modify: `src/mcp/tools_low.py` (`query_database` fallback return ~64)
- Test: `tests/test_generation/test_agent_query_streamed.py`
- Test: `tests/test_agent/test_graph.py`
- Test: `tests/test_mcp/test_tools_high.py`
- Test: `tests/test_mcp/test_tools_low.py`

**Interfaces:**
- Consumes: nothing from Tasks 1–2
- Produces:
  - `RAGResponse.query_type: str = ""`
  - Cache accept → `query_type="cache"` and `cached=True`
  - Graph finish → `query_type=str(state.query_type)` or `""`
  - `ask` / `summarize_topic` / `compare` return include `"query_type"` and `"cached"`
  - `query_database` fallback return includes those keys from `ask`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_generation/test_agent_query_streamed.py`:

```python
@pytest.mark.asyncio
async def test_streamed_cache_hit_sets_query_type_cache():
    cached = {"answer": "cached!", "citations": [], "cached_query": "old q"}
    decision = MagicMock(accepted=True, cached=cached, query_vector=None)
    with patch("src.generation.rag_chain.judged_cache_lookup", new_callable=AsyncMock, return_value=decision):
        out = await agent_query_streamed("q", ["finance"], MagicMock(), MagicMock(), None)
    assert out.cached is True
    assert out.query_type == "cache"
```

In `tests/test_agent/test_graph.py`, add (keep the file's existing imports; add `RAGResponse` if missing):

```python
@pytest.mark.asyncio
async def test_run_agent_streamed_copies_query_type():
    from src.agent.graph import run_agent_streamed
    from src.agent.state import QueryType
    from src.generation.rag_chain import RAGResponse

    class FakeGraph:
        async def astream(self, initial, stream_mode="updates"):
            yield {"classify": {"query_type": QueryType.LOOKUP, "answer": "hi", "citations": []}}

    with patch("src.agent.graph.create_agent_graph", return_value=FakeGraph()):
        out = await run_agent_streamed(
            question="q", user_groups=["finance"],
            vector_store=MagicMock(), schema_registry=MagicMock(),
        )
    assert isinstance(out, RAGResponse)
    assert out.query_type == "lookup"
```

If `test_graph.py` already patches `create_agent_graph` with a different FakeGraph shape, write this test in a new function that only patches that call.

Append to `tests/test_mcp/test_tools_high.py` inside `test_ask_quick` (or add a new test):

```python
@pytest.mark.asyncio
async def test_ask_forwards_query_type_and_cached():
    from src.mcp.tools_high import ask
    resp = _mock_rag_response()
    resp.query_type = "lookup"
    resp.cached = True
    with patch("src.mcp.tools_high.agent_query", new_callable=AsyncMock, return_value=resp):
        result = await ask(
            question="What is policy 4.2?",
            user_groups=["finance"],
            vector_store=MagicMock(),
            schema_registry=MagicMock(),
        )
    assert result["query_type"] == "lookup"
    assert result["cached"] is True
```

Append to `tests/test_mcp/test_tools_low.py`:

```python
@pytest.mark.asyncio
async def test_query_database_fallback_forwards_query_type():
    from src.mcp.tools_low import query_database
    from src.db.schema_registry import SchemaRegistry

    registry = SchemaRegistry()
    with patch(
        "src.mcp.tools_high.ask",
        new_callable=AsyncMock,
        return_value={"answer": "from docs", "citations": [], "query_type": "metadata", "cached": False},
    ):
        result = await query_database(
            question="how many files?",
            user_groups=["finance"],
            schema_registry=registry,
            vector_store=MagicMock(),
        )
    assert result["query_type"] == "metadata"
    assert result["cached"] is False
    assert result.get("error") in (None, "")
```

- [ ] **Step 2: Run the new tests to verify they fail**

Run: `source .venv/bin/activate && python -m pytest tests/test_generation/test_agent_query_streamed.py::test_streamed_cache_hit_sets_query_type_cache tests/test_agent/test_graph.py::test_run_agent_streamed_copies_query_type tests/test_mcp/test_tools_high.py::test_ask_forwards_query_type_and_cached tests/test_mcp/test_tools_low.py::test_query_database_fallback_forwards_query_type -v`

Expected: FAIL — `query_type` missing or empty.

- [ ] **Step 3: Implement plumbing**

`src/generation/rag_chain.py` — `RAGResponse`:

```python
@dataclass
class RAGResponse:
    answer: str
    citations: list[Citation]
    cached: bool = False
    cached_query: str | None = None
    query_type: str = ""
```

Cache-hit return (same function `_agent_query_streamed_bound`):

```python
        return RAGResponse(
            answer=cached["answer"], citations=citations,
            cached=True, cached_query=cached.get("cached_query"),
            query_type="cache",
        )
```

`src/agent/graph.py` — helper at module level and use it at all three `RAGResponse(...)` constructions (`run_agent`, `run_agent_streamed`, `run_agent_with_trace`):

```python
def _query_type_str(value) -> str:
    if value is None:
        return ""
    return str(value)
```

`run_agent`:

```python
    return RAGResponse(
        answer=result.get("answer", "I could not find any relevant information."),
        citations=result.get("citations", []),
        query_type=_query_type_str(result.get("query_type")),
    )
```

`run_agent_streamed`:

```python
    return RAGResponse(
        answer=final_state.get("answer", "I could not find any relevant information."),
        citations=final_state.get("citations", []),
        query_type=_query_type_str(final_state.get("query_type")),
    )
```

`run_agent_with_trace` response construction:

```python
    response = RAGResponse(
        answer=result.get("answer", "I could not find any relevant information."),
        citations=result.get("citations", []),
        query_type=_query_type_str(result.get("query_type") or trace.query_type),
    )
```

`src/mcp/tools_high.py` — add to `ask` return dict:

```python
        "query_type": response.query_type,
        "cached": response.cached,
```

Same two keys on `summarize_topic` and `compare` return dicts.

`src/mcp/tools_low.py` — fallback return inside `query_database`:

```python
            return {
                "sql": "",
                "results": [],
                "answer": result.get("answer", ""),
                "citations": result.get("citations", []),
                "query_type": result.get("query_type", ""),
                "cached": bool(result.get("cached")),
            }
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `source .venv/bin/activate && python -m pytest tests/test_generation/test_agent_query_streamed.py tests/test_agent/test_graph.py tests/test_mcp/test_tools_high.py tests/test_mcp/test_tools_low.py -v`

Expected: PASS (including pre-existing tests in those files).

- [ ] **Step 5: Commit**

```bash
git add src/generation/rag_chain.py src/agent/graph.py src/mcp/tools_high.py src/mcp/tools_low.py \
  tests/test_generation/test_agent_query_streamed.py tests/test_agent/test_graph.py \
  tests/test_mcp/test_tools_high.py tests/test_mcp/test_tools_low.py
git commit -m "feat: plumb query_type through RAGResponse and MCP helpers"
```

---

### Task 4: Dashboard recent-queries section

**Files:**
- Modify: `src/admin/routes.py` (`dashboard`, lines 104–133)
- Modify: `src/admin/templates/dashboard.html`
- Modify: `tests/test_admin/test_routes.py` (existing `test_dashboard_loads` plus new tests)
- Modify: `tests/test_admin/test_settings_shell.py` (`test_top_nav_drops_moved_links` mocks the dashboard store)

**Interfaces:**
- Consumes: `MetadataStore.list_recent_query_activity(limit: int = 10)` (Task 1), `format_activity_row(row) -> dict` (Task 2)
- Produces: dashboard context keys `activity: list[dict]` and `activity_error: bool`

- [ ] **Step 1: Write the failing tests**

Update `test_dashboard_loads` in `tests/test_admin/test_routes.py` so the mocked store returns an empty list (otherwise `await store.list_recent_query_activity(10)` is an unconfigured AsyncMock). Add the three new tests in the same file. Patch `_is_authenticated` so the handler actually renders `dashboard.html` instead of the login page.

```python
def _dashboard_store(activity=None, activity_exc=None):
    store = AsyncMock()
    store.list_documents.return_value = [MagicMock() for _ in range(5)]
    store.list_categories.return_value = [MagicMock() for _ in range(3)]
    store.list_proposals.return_value = [MagicMock(), MagicMock()]
    if activity_exc is not None:
        store.list_recent_query_activity.side_effect = activity_exc
    else:
        store.list_recent_query_activity.return_value = activity if activity is not None else []
    return store


def test_dashboard_loads(client):
    with patch("src.admin.routes._is_authenticated", return_value=True), \
         patch("src.admin.routes.get_metadata_store", return_value=_dashboard_store()):
        resp = client.get("/admin/")
    assert resp.status_code == 200
    assert "Dashboard" in resp.text
    assert "Recent queries" in resp.text
    assert "No queries recorded yet." in resp.text


def test_dashboard_shows_activity_row(client):
    from datetime import datetime, timezone
    row = MagicMock()
    row.created_at = datetime(2026, 8, 16, 14, 5, tzinfo=timezone.utc)
    row.source = "mcp"
    row.tool = "ask"
    row.strategy = "lookup"
    row.username = "mike"
    row.user_groups = ["finance"]
    row.query_text = "What is the PTO policy?"
    row.duration_seconds = 3.2
    row.status = "ok"
    row.cache_hit = False
    with patch("src.admin.routes._is_authenticated", return_value=True), \
         patch("src.admin.routes.get_metadata_store", return_value=_dashboard_store([row])):
        resp = client.get("/admin/")
    assert resp.status_code == 200
    assert "MCP · ask" in resp.text
    assert "mike" in resp.text
    assert "What is the PTO policy?" in resp.text
    assert "lookup" in resp.text
    assert "ok" in resp.text


def test_dashboard_activity_read_failure_keeps_stat_cards(client):
    with patch("src.admin.routes._is_authenticated", return_value=True), \
         patch("src.admin.routes.get_metadata_store", return_value=_dashboard_store(activity_exc=RuntimeError("db"))):
        resp = client.get("/admin/")
    assert resp.status_code == 200
    assert "Documents" in resp.text
    assert "Unable to load recent queries." in resp.text
    assert "No queries recorded yet." not in resp.text
```

In `tests/test_admin/test_settings_shell.py`, inside `test_top_nav_drops_moved_links`, add:

```python
        store.list_recent_query_activity.return_value = []
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `source .venv/bin/activate && python -m pytest tests/test_admin/test_routes.py::test_dashboard_loads tests/test_admin/test_routes.py::test_dashboard_shows_activity_row tests/test_admin/test_routes.py::test_dashboard_activity_read_failure_keeps_stat_cards -v`

Expected: FAIL — `Recent queries` not in the page (or `list_recent_query_activity` missing and the empty-state assertion fails).

- [ ] **Step 3: Implement dashboard route + template**

Replace the end of `dashboard` in `src/admin/routes.py` so activity is loaded in its own try/except **after** the existing stat queries:

```python
    activity = []
    activity_error = False
    try:
        from src.audit.activity import format_activity_row
        rows = await store.list_recent_query_activity(10)
        activity = [format_activity_row(r) for r in rows]
    except Exception:
        activity_error = True

    return templates.TemplateResponse(request, "dashboard.html", {
        "doc_count": len(docs), "category_count": len(categories),
        "pending_proposals": len(proposals), "entity_count": entity_count,
        "vector_count": vector_count,
        "activity": activity, "activity_error": activity_error,
    })
```

Replace `src/admin/templates/dashboard.html` with:

```html
{% extends "base.html" %}
{% block title %}Dashboard - SAURON{% endblock %}
{% block content %}
<h1>Dashboard</h1>
<div class="stats-grid">
    <div class="stat-card"><div class="stat-value">{{ doc_count }}</div><div class="stat-label">Documents</div></div>
    <div class="stat-card"><div class="stat-value">{{ category_count }}</div><div class="stat-label">Categories</div></div>
    <div class="stat-card"><div class="stat-value">{{ pending_proposals }}</div><div class="stat-label">Pending Proposals</div></div>
    <div class="stat-card"><div class="stat-value">{{ vector_count }}</div><div class="stat-label">Vector Chunks</div></div>
    <div class="stat-card"><div class="stat-value">{{ entity_count }}</div><div class="stat-label">Graph Entities</div></div>
</div>

<div class="settings-section" style="margin-top:1.5rem;">
    <h2>Recent queries</h2>
    {% if activity_error %}
    <p>Unable to load recent queries.</p>
    {% elif not activity %}
    <p>No queries recorded yet.</p>
    {% else %}
    <table>
        <thead>
            <tr>
                <th>When</th>
                <th>Type</th>
                <th>Strategy</th>
                <th>User</th>
                <th>Groups</th>
                <th>Question</th>
                <th>Time</th>
                <th>Status</th>
            </tr>
        </thead>
        <tbody>
            {% for row in activity %}
            <tr>
                <td>{{ row.when }}</td>
                <td>{{ row.type }}</td>
                <td>{{ row.strategy }}</td>
                <td>{{ row.username }}</td>
                <td>{{ row.groups }}</td>
                <td>{{ row.question }}</td>
                <td>{{ row.duration }}</td>
                <td>{% if row.status == "error" %}<span class="status-err">error</span>{% elif row.status == "cache" %}cache{% else %}ok{% endif %}</td>
            </tr>
            {% endfor %}
        </tbody>
    </table>
    {% endif %}
</div>
{% endblock %}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `source .venv/bin/activate && python -m pytest tests/test_admin/test_routes.py tests/test_admin/test_settings_shell.py -v`

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/admin/routes.py src/admin/templates/dashboard.html tests/test_admin/test_routes.py tests/test_admin/test_settings_shell.py
git commit -m "feat: show last 10 query activity rows on the dashboard"
```

---

### Task 5: Instrument REST sync, REST async, and OpenAI-compat

**Files:**
- Modify: `src/api/routes_query.py` (`query`, lines 15–29)
- Modify: `src/api/query_jobs.py` (`_worker_loop`, lines 172–213)
- Modify: `src/api/routes_openai_compat.py` (`chat_completions`, from identity resolution through `agent_query`)
- Test: `tests/test_api/test_routes_query.py`
- Test: `tests/test_api/test_query_jobs.py`

**Interfaces:**
- Consumes: `query_activity_span`, `record_query_activity` (Task 2); `RAGResponse.query_type` / `cached` (Task 3)
- Produces:
  - `POST /api/v1/query` writes `source="rest"`, `tool="query"`
  - Async worker writes `source="rest"`, `tool="query_async"` on complete **and** fail; timer starts when processing starts
  - `POST /v1/chat/completions` writes `source="openai"`, `tool="chat.completions"`; username is JWT username or `""`

- [ ] **Step 1: Write the failing tests**

The span calls `record_query_activity` as a global in `src.audit.activity`. Patch `src.audit.activity.record_query_activity`. Do not use a real SQLite store inside `TestClient` — the helper patch is enough.

Append to `tests/test_api/test_routes_query.py`:

```python
def test_query_records_activity(client, auth_headers):
    mock_response = RAGResponse(answer="ok", citations=[], query_type="lookup", cached=False)
    with patch("src.api.routes_query.agent_query", new_callable=AsyncMock, return_value=mock_response), \
         patch("src.api.routes_query.get_vector_store", return_value=MagicMock()), \
         patch("src.api.routes_query.get_schema_registry", return_value=MagicMock()), \
         patch("src.api.routes_query.get_metadata_store", return_value=MagicMock()), \
         patch("src.audit.activity.record_query_activity", new_callable=AsyncMock) as rec:
        resp = client.post(
            "/api/v1/query",
            json={"question": "What is the expense policy?"},
            headers=auth_headers,
        )
    assert resp.status_code == 200
    rec.assert_awaited()
    kwargs = rec.await_args.kwargs
    assert kwargs["source"] == "rest"
    assert kwargs["tool"] == "query"
    assert kwargs["username"] == "mike"
    assert kwargs["user_groups"] == ["finance"]
    assert kwargs["query_text"] == "What is the expense policy?"
    assert kwargs["strategy"] == "lookup"
    assert kwargs["status"] == "ok"
    assert kwargs["duration_seconds"] >= 0
```

Append to `tests/test_api/test_query_jobs.py`:

```python
@pytest.mark.asyncio
async def test_worker_records_activity_on_success():
    from src.generation.rag_chain import RAGResponse
    q = QueryJobQueue()
    rec = AsyncMock()
    with patch("src.api.query_jobs.agent_query_streamed", new_callable=AsyncMock, return_value=RAGResponse(answer="ok", citations=[], query_type="sweep", cached=False)), \
         patch("src.audit.activity.record_query_activity", rec):
        await q.start_worker(MagicMock(), MagicMock(), MagicMock())
        token = q.enqueue(question="pay rate?", username="mike", groups=["finance"])
        for _ in range(50):
            if q.get_job(token).status == QueryStatus.COMPLETE:
                break
            await asyncio.sleep(0.02)
    rec.assert_awaited()
    kwargs = rec.await_args.kwargs
    assert kwargs["source"] == "rest"
    assert kwargs["tool"] == "query_async"
    assert kwargs["username"] == "mike"
    assert kwargs["user_groups"] == ["finance"]
    assert kwargs["query_text"] == "pay rate?"
    assert kwargs["strategy"] == "sweep"
    assert kwargs["status"] == "ok"


@pytest.mark.asyncio
async def test_worker_records_activity_on_failure():
    q = QueryJobQueue()
    rec = AsyncMock()
    with patch("src.api.query_jobs.agent_query_streamed", new_callable=AsyncMock, side_effect=RuntimeError("boom")), \
         patch("src.audit.activity.record_query_activity", rec):
        await q.start_worker(MagicMock(), MagicMock(), MagicMock())
        token = q.enqueue(question="pay rate?", username="mike", groups=["finance"])
        for _ in range(50):
            if q.get_job(token).status == QueryStatus.FAILED:
                break
            await asyncio.sleep(0.02)
    rec.assert_awaited()
    kwargs = rec.await_args.kwargs
    assert kwargs["status"] == "error"
    assert kwargs["tool"] == "query_async"
    assert kwargs["error"]
```

The implementation below imports `record_query_activity` inside `_worker_loop`, so patch `src.audit.activity.record_query_activity`.

There is no existing OpenAI-compat test file. Add `tests/test_api/test_routes_openai_compat.py`:

```python
from unittest.mock import AsyncMock, MagicMock, patch
from fastapi.testclient import TestClient
from src.auth.jwt import create_token
from src.generation.rag_chain import RAGResponse
from src.main import create_app


def test_chat_completions_records_activity():
    token = create_token(username="mike", groups=["finance"])
    headers = {"Authorization": f"Bearer {token}", "X-API-Key": "test-key-1"}
    rec = AsyncMock()
    with patch("src.api.routes_openai_compat.agent_query", new_callable=AsyncMock, return_value=RAGResponse(answer="hi", citations=[], query_type="lookup", cached=False)), \
         patch("src.api.routes_openai_compat.get_vector_store", return_value=MagicMock()), \
         patch("src.api.routes_openai_compat.get_schema_registry", return_value=MagicMock()), \
         patch("src.audit.activity.record_query_activity", rec):
        resp = TestClient(create_app()).post(
            "/v1/chat/completions",
            json={"model": "sauron", "messages": [{"role": "user", "content": "hello"}]},
            headers=headers,
        )
    assert resp.status_code == 200
    rec.assert_awaited()
    kwargs = rec.await_args.kwargs
    assert kwargs["source"] == "openai"
    assert kwargs["tool"] == "chat.completions"
    assert kwargs["username"] == "mike"
    assert kwargs["user_groups"] == ["finance"]
    assert kwargs["query_text"] == "hello"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `source .venv/bin/activate && python -m pytest tests/test_api/test_routes_query.py::test_query_records_activity tests/test_api/test_query_jobs.py::test_worker_records_activity_on_success tests/test_api/test_query_jobs.py::test_worker_records_activity_on_failure tests/test_api/test_routes_openai_compat.py::test_chat_completions_records_activity -v`

Expected: FAIL — `record_query_activity` not awaited.

- [ ] **Step 3: Instrument the three surfaces**

`src/api/routes_query.py` — wrap `query`:

```python
from src.audit.activity import query_activity_span

@router.post("/query", response_model=QueryResponse)
async def query(payload: QueryRequest, http: Request, user: UserContext = Depends(require_auth)):
    async with query_activity_span(
        source="rest", tool="query",
        username=user.username, user_groups=list(user.groups),
        query_text=payload.question,
    ) as span:
        result = await agent_query(
            question=payload.question, user_groups=user.groups,
            vector_store=get_vector_store(), schema_registry=get_schema_registry(),
            metadata_store=get_metadata_store(),
            skip_cache=payload.skip_cache,
            session_headers=http.headers, agent_id=user.username,
        )
        span.strategy = result.query_type or ("cache" if result.cached else "")
        span.cache_hit = bool(result.cached)
        return QueryResponse(
            answer=result.answer,
            citations=[CitationResponse(doc_id=c.doc_id, filename=c.filename, doc_type=c.doc_type, chunk_index=c.chunk_index, page=c.page, snippet=c.snippet, relevance=c.relevance, figure_id=c.figure_id, section_title=c.section_title, caption=c.caption, slide=c.slide) for c in result.citations],
            cached=result.cached,
            cached_query=result.cached_query,
        )
```

Do **not** wrap `query_async` enqueue.

`src/api/query_jobs.py` — in `_worker_loop`, time processing only. After the existing `complete` / `fail` calls, record. Keep the current fail messages (`"Query timed out"` / `_GENERIC_ERROR`):

```python
            t0 = time.time()
            try:
                vector_store, schema_registry, metadata_store = self._stores
                job.status = QueryStatus.PROCESSING
                result = await asyncio.wait_for(
                    agent_query_streamed(
                        question=job.question, user_groups=job.groups,
                        vector_store=vector_store, schema_registry=schema_registry,
                        metadata_store=metadata_store,
                        step_callback=lambda node, detail=None, _t=token: self.update_step(_t, node, detail),
                        skip_cache=job.skip_cache,
                        session_id=job.session_id, agent_id=job.agent_id,
                    ),
                    timeout=self._job_timeout,
                )
                citation_dicts = [
                    {"doc_id": c.doc_id, "filename": c.filename, "doc_type": c.doc_type,
                     "chunk_index": c.chunk_index, "page": c.page, "snippet": c.snippet,
                     "relevance": c.relevance, "figure_id": c.figure_id,
                     "section_title": c.section_title, "caption": c.caption,
                     "slide": c.slide}
                    for c in result.citations
                ]
                self.complete(token, answer=result.answer, citations=citation_dicts,
                              cached=result.cached, cached_query=result.cached_query)
                strategy = result.query_type or ""
                if not strategy and job.classification:
                    strategy = str(job.classification.get("query_type") or "")
                from src.audit.activity import record_query_activity
                await record_query_activity(
                    source="rest", tool="query_async",
                    username=job.username, user_groups=job.groups,
                    query_text=job.question, strategy=strategy,
                    duration_seconds=round(time.time() - t0, 2),
                    status="ok", cache_hit=bool(result.cached),
                )
            except asyncio.TimeoutError:
                logger.error(f"Async query {token} timed out after {self._job_timeout}s")
                self.fail(token, "Query timed out")
                from src.audit.activity import record_query_activity
                await record_query_activity(
                    source="rest", tool="query_async",
                    username=job.username, user_groups=job.groups,
                    query_text=job.question,
                    duration_seconds=round(time.time() - t0, 2),
                    status="error", error="Query timed out",
                )
            except Exception as e:
                logger.error(f"Async query {token} failed: {e}\n{traceback.format_exc()}")
                self.fail(token, _GENERIC_ERROR)
                from src.audit.activity import record_query_activity
                await record_query_activity(
                    source="rest", tool="query_async",
                    username=job.username, user_groups=job.groups,
                    query_text=job.question,
                    duration_seconds=round(time.time() - t0, 2),
                    status="error", error=_GENERIC_ERROR,
                )
            self._queue.task_done()
```

`src/api/routes_openai_compat.py` — wrap the `agent_query` call. `username` is `agent_id or ""` (JWT username, else empty). `user_groups` is the list actually used:

```python
    from src.audit.activity import query_activity_span

    async with query_activity_span(
        source="openai", tool="chat.completions",
        username=agent_id or "", user_groups=list(user_groups),
        query_text=question,
    ) as span:
        result = await agent_query(
            question=question,
            user_groups=user_groups,
            vector_store=get_vector_store(),
            schema_registry=get_schema_registry(),
            session_headers=http.headers,
            agent_id=agent_id,
        )
        span.strategy = result.query_type or ("cache" if result.cached else "")
        span.cache_hit = bool(result.cached)
```

Leave the OpenAI response-shaping code below the span.

- [ ] **Step 4: Run tests to verify they pass**

Run: `source .venv/bin/activate && python -m pytest tests/test_api/test_routes_query.py tests/test_api/test_query_jobs.py tests/test_api/test_routes_openai_compat.py -v`

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/api/routes_query.py src/api/query_jobs.py src/api/routes_openai_compat.py \
  tests/test_api/test_routes_query.py tests/test_api/test_query_jobs.py tests/test_api/test_routes_openai_compat.py
git commit -m "feat: record query activity for REST and OpenAI-compat"
```

---

### Task 6: Instrument Playground

**Files:**
- Modify: `src/admin/routes.py` (`playground_start` `run_query` ~813–1417; `playground_query` ~1484–1548)
- Test: `tests/test_admin/test_routes.py`

**Interfaces:**
- Consumes: `query_activity_span` (Task 2)
- Produces: one row per terminal Playground job (`source="playground"`, `tool="playground"`). Not on status poll or SSE. Cache hit → `cache_hit=True`, `strategy="cache"`. Graph success → `strategy` from classify `query_type`. Errors → `status="error"` without re-raising out of `run_query`.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_admin/test_routes.py`. Drive the legacy `POST /api/playground/query` path (sync handler, easy to assert) and assert `record_query_activity` is awaited. Do not try to unit-test the entire `playground_start` background task in this step — after the test exists, implement **both** handlers.

```python
def test_playground_query_records_activity(client):
    from src.generation.rag_chain import RAGResponse
    from src.agent.graph import AgentTrace
    rec = AsyncMock()
    result = RAGResponse(answer="hi", citations=[], query_type="lookup")
    trace = AgentTrace(query_type="lookup", total_time=1.2)
    store = AsyncMock()
    store.resolve_play_user_groups.return_value = ["finance"]
    with patch("src.admin.routes._is_authenticated", return_value=True), \
         patch("src.admin.routes.get_metadata_store", return_value=store), \
         patch("src.admin.routes.get_vector_store", return_value=MagicMock()), \
         patch("src.admin.routes.get_schema_registry", return_value=MagicMock()), \
         patch("src.agent.graph.run_agent_with_trace", new_callable=AsyncMock, return_value=(result, trace)), \
         patch("src.audit.activity.record_query_activity", rec):
        resp = client.post(
            "/api/playground/query",
            data={"question": "What is PTO?", "play_user": "mike"},
        )
    assert resp.status_code == 200
    rec.assert_awaited()
    kwargs = rec.await_args.kwargs
    assert kwargs["source"] == "playground"
    assert kwargs["tool"] == "playground"
    assert kwargs["username"] == "mike"
    assert kwargs["user_groups"] == ["finance"]
    assert kwargs["query_text"] == "What is PTO?"
    assert kwargs["strategy"] == "lookup"
    assert kwargs["status"] == "ok"
```

If `playground_query` does not call `_require_login` today, the `_is_authenticated` patch is still harmless.

- [ ] **Step 2: Run the test to verify it fails**

Run: `source .venv/bin/activate && python -m pytest tests/test_admin/test_routes.py::test_playground_query_records_activity -v`

Expected: FAIL — `record_query_activity` not awaited.

- [ ] **Step 3: Instrument both Playground handlers**

`playground_query` (`POST /api/playground/query`):

```python
    from src.audit.activity import query_activity_span

    async with query_activity_span(
        source="playground", tool="playground",
        username=play_user, user_groups=user_groups,
        query_text=question,
    ) as span:
        try:
            from src.agent.graph import run_agent_with_trace
            result, trace = await run_agent_with_trace(...)
            span.strategy = trace.query_type or result.query_type or ""
            # existing HTML return unchanged
            return HTMLResponse(...)
        except Exception as e:
            span.status = "error"
            span.error = str(e)[:200]
            import traceback
            return HTMLResponse(f'<div class="status-err">Error: {e}<br><pre style="font-size:0.75rem;">{traceback.format_exc()}</pre></div>')
```

`playground_start` — wrap the body of `run_query` (do not wrap the `create_task` / return). Open the span immediately inside `run_query`. Set span fields at the existing terminal points; do **not** re-raise:

- Cache-hit complete (today around the `_playground_jobs[query_id] = {..., "step": "complete"}` after the cache HTML): `span.cache_hit = True` and `span.strategy = "cache"`.
- Graph complete (the existing `"step": "complete"` after synthesis): `span.strategy = query_type or ""`.
- Existing `except Exception as e` that sets `"step": "error"`: also `span.status = "error"` and `span.error = str(e)[:200]`. Do not re-raise (the task must keep swallowing so the poll UI still works).

Do not add a span to `playground_status` or `playground_stream`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `source .venv/bin/activate && python -m pytest tests/test_admin/test_routes.py -v`

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/admin/routes.py tests/test_admin/test_routes.py
git commit -m "feat: record query activity for playground queries"
```

---

### Task 7: Instrument every MCP tool except `get_result`

**Files:**
- Modify: `src/mcp/server.py` (every `@mcp.tool()` handler except `tool_get_result`)
- Test: `tests/test_mcp/test_server.py`

**Interfaces:**
- Consumes: `query_activity_span`, `apply_result_to_span` (Task 2); `query_type`/`cached` on MCP helper returns (Task 3); `current_mcp_context()`
- Produces: one activity row per listed tool; `tool_get_result` unchanged; sync handlers become `async def` and call the existing sync implementations

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_mcp/test_server.py`:

```python
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from src.mcp.auth import MCPContext
from src.mcp.server import create_mcp_server


def _ctx():
    return MCPContext(username="mike", groups=["finance"], api_key="test-key-1")


@pytest.mark.asyncio
async def test_logged_mcp_wrapper_records_ask():
    from src.mcp.activity_wrap import run_logged_mcp_tool
    rec = AsyncMock()
    with patch("src.mcp.auth.current_mcp_context", return_value=_ctx()), \
         patch("src.audit.activity.record_query_activity", rec):
        result = await run_logged_mcp_tool(
            tool="ask",
            query_text="What is PTO?",
            fn=AsyncMock(return_value={"answer": "ok", "query_type": "lookup", "cached": False}),
        )
    assert result["answer"] == "ok"
    rec.assert_awaited()
    kwargs = rec.await_args.kwargs
    assert kwargs["source"] == "mcp"
    assert kwargs["tool"] == "ask"
    assert kwargs["username"] == "mike"
    assert kwargs["user_groups"] == ["finance"]
    assert kwargs["query_text"] == "What is PTO?"
    assert kwargs["strategy"] == "lookup"
    assert kwargs["status"] == "ok"


@pytest.mark.asyncio
async def test_logged_mcp_wrapper_records_error_dict():
    from src.mcp.activity_wrap import run_logged_mcp_tool
    rec = AsyncMock()
    with patch("src.mcp.auth.current_mcp_context", return_value=_ctx()), \
         patch("src.audit.activity.record_query_activity", rec):
        result = await run_logged_mcp_tool(
            tool="lookup_document",
            query_text="missing.pdf",
            fn=AsyncMock(return_value={"error": "Document 'missing.pdf' not found."}),
        )
    assert "error" in result
    kwargs = rec.await_args.kwargs
    assert kwargs["status"] == "error"
    assert "not found" in kwargs["error"]


def test_get_result_is_not_wrapped():
    import inspect
    from src.mcp import server as mcp_server
    source = inspect.getsource(mcp_server.create_mcp_server)
    get_idx = source.index("async def tool_get_result")
    ask_idx = source.index("run_logged_mcp_tool")
    assert "run_logged_mcp_tool" not in source[get_idx:get_idx + 800]
    assert ask_idx < get_idx  # other tools are wrapped; get_result is not
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `source .venv/bin/activate && python -m pytest tests/test_mcp/test_server.py::test_logged_mcp_wrapper_records_ask tests/test_mcp/test_server.py::test_logged_mcp_wrapper_records_error_dict tests/test_mcp/test_server.py::test_get_result_is_not_wrapped -v`

Expected: FAIL — `src.mcp.activity_wrap` missing.

- [ ] **Step 3: Implement the wrapper module and wire every tool**

Create `src/mcp/activity_wrap.py`:

```python
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
```

In `src/mcp/server.py`, import `run_logged_mcp_tool`. Change each tool as follows. Keep the existing docstrings. Convert sync handlers to `async def`. Leave `tool_get_result` and the three `@mcp.resource` handlers untouched.

```python
    @mcp.tool()
    async def tool_ask(question: str, depth: str = "thorough", context: str = "") -> dict:
        """THIS IS THE PRIMARY TOOL — use it for ANY question about document content, contracts, policies, people, companies, awards, or facts. It searches all documents, enriches with knowledge graph data, and generates a comprehensive cited answer. Use this FIRST before trying other tools. Only use tool_list_documents or tool_lookup_document for browsing/reading specific files."""
        async def _run():
            return await ask(
                question=question,
                user_groups=user_groups(),
                vector_store=vector_store,
                schema_registry=schema_registry,
                metadata_store=metadata_store,
                depth=depth,
                context=context or None,
            )
        return await run_logged_mcp_tool(tool="ask", query_text=question, fn=_run)

    @mcp.tool()
    async def tool_summarize_topic(topic: str, format: str = "brief") -> dict:
        """Summarize a specific topic by searching across all documents and generating a summary with source references."""
        async def _run():
            return await summarize_topic(
                topic=topic,
                user_groups=user_groups(),
                vector_store=vector_store,
                schema_registry=schema_registry,
                metadata_store=metadata_store,
                format=format,
            )
        return await run_logged_mcp_tool(tool="summarize_topic", query_text=topic, fn=_run)

    @mcp.tool()
    async def tool_compare(item_a: str, item_b: str) -> dict:
        """Compare and contrast two items, policies, or topics by searching the documents for both and listing differences."""
        async def _run():
            return await compare(
                item_a=item_a,
                item_b=item_b,
                user_groups=user_groups(),
                vector_store=vector_store,
                schema_registry=schema_registry,
                metadata_store=metadata_store,
            )
        return await run_logged_mcp_tool(
            tool="compare", query_text=f"{item_a} vs {item_b}", fn=_run,
        )

    @mcp.tool()
    async def tool_search_documents(query: str, doc_type: str = "", top_k: int = 10) -> list[dict]:
        """Low-level search returning raw text snippets. For ANSWERING questions, use tool_ask instead — it provides better results with knowledge graph enrichment and cited answers. Only use this tool when you need raw document snippets for your own analysis, or to find doc_ids for tool_lookup_document."""
        async def _run():
            return search_documents(
                query=query,
                user_groups=user_groups(),
                vector_store=vector_store,
                doc_type=doc_type or None,
                top_k=top_k,
            )
        return await run_logged_mcp_tool(tool="search_documents", query_text=query, fn=_run)

    @mcp.tool()
    async def tool_query_database(question: str) -> dict:
        """Query a structured SQL database only. For ANSWERING questions about document content, contracts, policies, or facts, use tool_ask instead. Only use this when you specifically need to run a SQL query against a registered database."""
        async def _run():
            return await query_database(
                question=question,
                user_groups=user_groups(),
                schema_registry=schema_registry,
                vector_store=vector_store,
                metadata_store=metadata_store,
            )
        return await run_logged_mcp_tool(
            tool="query_database", query_text=question, fn=_run, default_strategy="structured",
        )

    @mcp.tool()
    async def tool_lookup_document(doc_id: str) -> dict:
        """Read the full content of a document. Accepts either a doc_id (UUID) or a filename. Use this when the user asks to read, view, summarize, or display a specific file. You can pass the filename directly (e.g., 'sample.pdf') or a doc_id from tool_list_documents."""
        async def _run():
            return lookup_document(
                doc_id=doc_id,
                user_groups=user_groups(),
                vector_store=vector_store,
            )
        return await run_logged_mcp_tool(tool="lookup_document", query_text=doc_id, fn=_run)

    @mcp.tool()
    async def tool_search_meetings(topic: str = "", speaker: str = "", type_filter: str = "") -> list[dict]:
        """Search meeting transcripts. Filter by speaker name, topic, or utterance type (question, statement, action_item). Use this to find what someone said in meetings or to find specific discussions."""
        q = " / ".join(p for p in (topic, speaker, type_filter) if p)
        async def _run():
            return search_meetings(
                user_groups=user_groups(),
                vector_store=vector_store,
                topic=topic or None,
                speaker=speaker or None,
                type_filter=type_filter or None,
            )
        return await run_logged_mcp_tool(tool="search_meetings", query_text=q, fn=_run)

    @mcp.tool()
    async def tool_list_sources() -> list[dict]:
        """List all available document categories and their document counts. Shows what knowledge sources exist in the system (e.g., finance_policies, it_runbooks, meeting_notes)."""
        async def _run():
            return list_sources(user_groups=user_groups(), metadata_store=metadata_store)
        return await run_logged_mcp_tool(tool="list_sources", query_text="", fn=_run)

    @mcp.tool()
    async def tool_summarize_documents(category: str = "") -> dict:
        """Read and summarize every document in a category. Returns a list of filenames with a 2-3 sentence summary of each. Use this when the user asks to summarize multiple documents, summarize a category, or wants an overview of what's in a group of files. For a single file, use tool_lookup_document instead."""
        async def _run():
            return summarize_documents(
                category=category or "uncategorized",
                user_groups=user_groups(),
                vector_store=vector_store,
                metadata_store=metadata_store,
            )
        return await run_logged_mcp_tool(
            tool="summarize_documents",
            query_text=category,
            fn=_run,
        )

    @mcp.tool()
    async def tool_list_documents(category: str = "") -> list[dict]:
        """List all documents with filenames, doc_ids, types, and categories. Use this FIRST when the user asks about what files exist, what's in a category, or wants to see uncategorized documents. Filter by category name (e.g., 'meeting_notes', 'finance_policies', 'uncategorized'). To read a file's content after listing, pass its doc_id or filename to tool_lookup_document."""
        async def _run():
            groups = user_groups()
            if category:
                return list_documents_in_category(category=category, user_groups=groups, metadata_store=metadata_store)
            docs = await metadata_store.list_documents(None if "ALL" in groups else groups)
            return [{"doc_id": d.doc_id, "filename": d.filename, "doc_type": d.doc_type, "category": d.category or "uncategorized", "chunk_count": d.chunk_count, "uploaded_by": d.uploaded_by} for d in docs]
        return await run_logged_mcp_tool(tool="list_documents", query_text=category, fn=_run)

    @mcp.tool()
    async def tool_search_knowledge_graph(query: str, entity_type: str = "") -> dict:
        """Search the knowledge graph for an entity (person, policy, project, organization, etc.) and find all related entities, relationships, and source documents. Use this to understand how concepts are connected across documents. Example: 'TOEE 26' returns related projects, people, and organizations."""
        async def _run():
            return await search_knowledge_graph(
                query=query,
                user_groups=user_groups(),
                metadata_store=metadata_store,
                entity_type=entity_type or None,
            )
        return await run_logged_mcp_tool(tool="search_knowledge_graph", query_text=query, fn=_run)
```

Leave `tool_get_result` exactly as it is today.

- [ ] **Step 4: Run tests to verify they pass**

Run: `source .venv/bin/activate && python -m pytest tests/test_mcp/test_server.py tests/test_mcp/test_tools_high.py tests/test_mcp/test_tools_low.py tests/test_mcp/test_http.py -v`

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/mcp/activity_wrap.py src/mcp/server.py tests/test_mcp/test_server.py
git commit -m "feat: record query activity for MCP tools"
```

---

## Self-review (spec coverage)

| Spec requirement | Task |
|---|---|
| `query_activity` table + create_all import | 1 |
| Best-effort insert, 500/200 caps | 1–2 |
| Span times, records errors, does not swallow | 2 |
| Display format (type, groups +N, cache status, 80-char question) | 2, 4 |
| `RAGResponse.query_type` / cache / graph | 3 |
| MCP helper return `query_type` + `cached`; `query_database` fallback | 3 |
| Dashboard last 10, empty + read-error copy, stat cards isolated | 4 |
| REST `/query`, async complete/fail (processing time), OpenAI | 5 |
| Playground terminal only (start + legacy query); not poll/SSE | 6 |
| All MCP tools except `get_result` / resources | 7 |
| Outermost-only (no log inside `ask` / `agent_query`) | 5–7 |
| No `query_metrics` / audit JSONL / auto-refresh | not in any task |
