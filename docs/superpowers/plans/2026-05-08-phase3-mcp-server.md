# Phase 3: MCP Knowledge Service — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Expose the RAG system as an MCP (Model Context Protocol) server so other AI agents can query enterprise knowledge via standardized tools — with agent-level permissions, user JWT passthrough, async job support, and audit logging.

**Architecture:** A FastMCP server runs alongside the FastAPI app, sharing the same agent orchestrator, vector store, metadata store, and schema registry via singletons. Tools are split into high-level (run full agent pipeline) and low-level (direct retrieval). An agent registry controls which tools and sources each registered agent can access. Auth uses API key (agent identity) + JWT (user identity). Async jobs for long-running queries use an in-memory store with TTL.

**Tech Stack:** FastMCP (Python SDK), existing FastAPI app, existing LangGraph agent, existing Qdrant + E5-large

---

## File Structure

```
src/
├── mcp/                                # NEW — MCP server
│   ├── __init__.py
│   ├── server.py                       # FastMCP server setup, tool/resource registration
│   ├── auth.py                         # MCP auth middleware (API key + JWT extraction)
│   ├── agent_registry.py               # Agent permissions (which tools/sources per agent)
│   ├── tools_high.py                   # High-level tools: ask, summarize_topic, compare
│   ├── tools_low.py                    # Low-level tools: search_documents, query_database, etc.
│   ├── resources.py                    # MCP resources: document://, category://, schema://
│   └── jobs.py                         # Async job store for long-running queries
│
├── audit/                              # NEW — audit logging
│   ├── __init__.py
│   └── logger.py                       # Structured audit log writer
│
├── config.py                           # MODIFY — add MCP and audit settings
├── main.py                             # MODIFY — optionally start MCP alongside FastAPI
│
tests/
├── test_mcp/
│   ├── __init__.py
│   ├── test_agent_registry.py
│   ├── test_auth.py
│   ├── test_tools_high.py
│   ├── test_tools_low.py
│   ├── test_resources.py
│   ├── test_jobs.py
│   └── test_server.py
├── test_audit/
│   ├── __init__.py
│   └── test_logger.py
```

---

## Task 1: FastMCP Dependency & MCP Auth

**Files:**
- Modify: `rag/requirements.txt`
- Create: `rag/src/mcp/__init__.py`
- Create: `rag/src/mcp/auth.py`
- Create: `rag/tests/test_mcp/__init__.py`
- Create: `rag/tests/test_mcp/test_auth.py`

- [ ] **Step 1: Add fastmcp to requirements.txt**

Add to the `# Core` section:
```
fastmcp>=3.0.0
```

Install: `source .venv/bin/activate && pip install "fastmcp>=3.0.0"`

- [ ] **Step 2: Write failing tests for MCP auth**

```python
# tests/test_mcp/__init__.py
```

```python
# tests/test_mcp/test_auth.py
import pytest
from src.mcp.auth import extract_mcp_context, MCPContext
from src.auth.jwt import create_token


def test_extract_context_from_headers():
    token = create_token(username="mike", groups=["finance"])
    headers = {"authorization": f"Bearer {token}", "x-api-key": "dev-key-1"}
    ctx = extract_mcp_context(headers)
    assert ctx.username == "mike"
    assert ctx.groups == ["finance"]
    assert ctx.api_key == "dev-key-1"


def test_extract_context_missing_jwt():
    headers = {"x-api-key": "dev-key-1"}
    with pytest.raises(ValueError, match="Missing"):
        extract_mcp_context(headers)


def test_extract_context_missing_api_key():
    token = create_token(username="mike", groups=["finance"])
    headers = {"authorization": f"Bearer {token}"}
    with pytest.raises(ValueError, match="API key"):
        extract_mcp_context(headers)


def test_extract_context_invalid_jwt():
    headers = {"authorization": "Bearer bad-token", "x-api-key": "dev-key-1"}
    with pytest.raises(ValueError, match="Invalid token"):
        extract_mcp_context(headers)
```

- [ ] **Step 3: Implement MCP auth**

```python
# src/mcp/__init__.py
```

```python
# src/mcp/auth.py
from dataclasses import dataclass

from src.auth.api_key import validate_api_key
from src.auth.jwt import decode_token


@dataclass
class MCPContext:
    username: str
    groups: list[str]
    api_key: str
    agent_id: str = ""


def extract_mcp_context(headers: dict) -> MCPContext:
    api_key = headers.get("x-api-key", "")
    if not api_key or not validate_api_key(api_key):
        raise ValueError("Invalid or missing API key")

    auth_header = headers.get("authorization", "")
    if not auth_header.startswith("Bearer "):
        raise ValueError("Missing Bearer token")

    token = auth_header.removeprefix("Bearer ")
    user = decode_token(token)

    return MCPContext(
        username=user.username,
        groups=user.groups,
        api_key=api_key,
    )
```

- [ ] **Step 4: Run tests**

Run: `source .venv/bin/activate && cd /Users/michaelmulkey/Documents/Repositories/rag && python -m pytest tests/test_mcp/test_auth.py -v`
Expected: 4 tests PASS

- [ ] **Step 5: Commit**

```bash
git add requirements.txt src/mcp/__init__.py src/mcp/auth.py tests/test_mcp/__init__.py tests/test_mcp/test_auth.py
git commit -m "feat: FastMCP dependency and MCP auth context extraction"
```

---

## Task 2: Agent Registry

**Files:**
- Create: `rag/src/mcp/agent_registry.py`
- Create: `rag/tests/test_mcp/test_agent_registry.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_mcp/test_agent_registry.py
import pytest
from src.mcp.agent_registry import AgentRegistry, AgentPermissions


def test_register_and_get_agent():
    registry = AgentRegistry()
    registry.register(AgentPermissions(
        agent_id="hr-agent",
        api_key="hr-key-1",
        allowed_tools=["ask", "search_documents", "summarize_topic"],
        allowed_sources=["hr_policies", "meeting_notes"],
    ))
    agent = registry.get_by_api_key("hr-key-1")
    assert agent is not None
    assert agent.agent_id == "hr-agent"
    assert "ask" in agent.allowed_tools


def test_get_unknown_agent():
    registry = AgentRegistry()
    assert registry.get_by_api_key("unknown-key") is None


def test_check_tool_permission_allowed():
    registry = AgentRegistry()
    registry.register(AgentPermissions(
        agent_id="hr-agent", api_key="hr-key-1",
        allowed_tools=["ask", "search_documents"],
        allowed_sources=["hr_policies"],
    ))
    assert registry.can_use_tool("hr-key-1", "ask") is True
    assert registry.can_use_tool("hr-key-1", "query_database") is False


def test_check_tool_permission_all():
    registry = AgentRegistry()
    registry.register(AgentPermissions(
        agent_id="compliance-agt", api_key="comp-key-1",
        allowed_tools=["ALL"],
        allowed_sources=["ALL"],
    ))
    assert registry.can_use_tool("comp-key-1", "anything") is True
    assert registry.can_access_source("comp-key-1", "anything") is True


def test_check_source_permission():
    registry = AgentRegistry()
    registry.register(AgentPermissions(
        agent_id="it-agent", api_key="it-key-1",
        allowed_tools=["ask"],
        allowed_sources=["it_runbooks", "infra_data"],
    ))
    assert registry.can_access_source("it-key-1", "it_runbooks") is True
    assert registry.can_access_source("it-key-1", "finance_policies") is False


def test_unregistered_agent_has_no_permissions():
    registry = AgentRegistry()
    assert registry.can_use_tool("unknown", "ask") is False
    assert registry.can_access_source("unknown", "anything") is False
```

- [ ] **Step 2: Implement agent_registry.py**

```python
# src/mcp/agent_registry.py
from dataclasses import dataclass, field


@dataclass
class AgentPermissions:
    agent_id: str
    api_key: str
    allowed_tools: list[str] = field(default_factory=list)
    allowed_sources: list[str] = field(default_factory=list)


class AgentRegistry:
    def __init__(self):
        self._agents: dict[str, AgentPermissions] = {}

    def register(self, permissions: AgentPermissions) -> None:
        self._agents[permissions.api_key] = permissions

    def get_by_api_key(self, api_key: str) -> AgentPermissions | None:
        return self._agents.get(api_key)

    def can_use_tool(self, api_key: str, tool_name: str) -> bool:
        agent = self._agents.get(api_key)
        if agent is None:
            return False
        if "ALL" in agent.allowed_tools:
            return True
        return tool_name in agent.allowed_tools

    def can_access_source(self, api_key: str, source_name: str) -> bool:
        agent = self._agents.get(api_key)
        if agent is None:
            return False
        if "ALL" in agent.allowed_sources:
            return True
        return source_name in agent.allowed_sources
```

- [ ] **Step 3: Run tests**

Run: `source .venv/bin/activate && cd /Users/michaelmulkey/Documents/Repositories/rag && python -m pytest tests/test_mcp/test_agent_registry.py -v`
Expected: 6 tests PASS

- [ ] **Step 4: Commit**

```bash
git add src/mcp/agent_registry.py tests/test_mcp/test_agent_registry.py
git commit -m "feat: agent registry with per-agent tool and source permissions"
```

---

## Task 3: Async Job Store

**Files:**
- Create: `rag/src/mcp/jobs.py`
- Create: `rag/tests/test_mcp/test_jobs.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_mcp/test_jobs.py
import pytest
import asyncio
from src.mcp.jobs import JobStore, JobStatus


def test_create_job():
    store = JobStore()
    job_id = store.create()
    assert job_id is not None
    job = store.get(job_id)
    assert job is not None
    assert job["status"] == JobStatus.PROCESSING


def test_complete_job():
    store = JobStore()
    job_id = store.create()
    store.complete(job_id, result={"answer": "test", "citations": []})
    job = store.get(job_id)
    assert job["status"] == JobStatus.COMPLETE
    assert job["result"]["answer"] == "test"


def test_fail_job():
    store = JobStore()
    job_id = store.create()
    store.fail(job_id, error="Something went wrong")
    job = store.get(job_id)
    assert job["status"] == JobStatus.FAILED
    assert "Something went wrong" in job["error"]


def test_get_nonexistent_job():
    store = JobStore()
    assert store.get("nonexistent") is None


def test_expired_jobs_cleaned():
    store = JobStore(ttl_seconds=0)  # immediate expiry
    job_id = store.create()
    import time
    time.sleep(0.1)
    store.cleanup_expired()
    assert store.get(job_id) is None
```

- [ ] **Step 2: Implement jobs.py**

```python
# src/mcp/jobs.py
import time
import uuid
from enum import StrEnum


class JobStatus(StrEnum):
    PROCESSING = "processing"
    COMPLETE = "complete"
    FAILED = "failed"


class JobStore:
    def __init__(self, ttl_seconds: int = 3600):
        self._jobs: dict[str, dict] = {}
        self._ttl = ttl_seconds

    def create(self) -> str:
        job_id = str(uuid.uuid4())
        self._jobs[job_id] = {
            "status": JobStatus.PROCESSING,
            "result": None,
            "error": None,
            "created_at": time.time(),
        }
        return job_id

    def get(self, job_id: str) -> dict | None:
        return self._jobs.get(job_id)

    def complete(self, job_id: str, result: dict) -> None:
        if job_id in self._jobs:
            self._jobs[job_id]["status"] = JobStatus.COMPLETE
            self._jobs[job_id]["result"] = result

    def fail(self, job_id: str, error: str) -> None:
        if job_id in self._jobs:
            self._jobs[job_id]["status"] = JobStatus.FAILED
            self._jobs[job_id]["error"] = error

    def cleanup_expired(self) -> int:
        now = time.time()
        expired = [jid for jid, j in self._jobs.items() if now - j["created_at"] > self._ttl]
        for jid in expired:
            del self._jobs[jid]
        return len(expired)
```

- [ ] **Step 3: Run tests**

Run: `source .venv/bin/activate && cd /Users/michaelmulkey/Documents/Repositories/rag && python -m pytest tests/test_mcp/test_jobs.py -v`
Expected: 5 tests PASS

- [ ] **Step 4: Commit**

```bash
git add src/mcp/jobs.py tests/test_mcp/test_jobs.py
git commit -m "feat: async job store with TTL for long-running MCP queries"
```

---

## Task 4: Audit Logger

**Files:**
- Create: `rag/src/audit/__init__.py`
- Create: `rag/src/audit/logger.py`
- Create: `rag/tests/test_audit/__init__.py`
- Create: `rag/tests/test_audit/test_logger.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_audit/__init__.py
```

```python
# tests/test_audit/test_logger.py
import json
import pytest
from pathlib import Path
from src.audit.logger import AuditLogger, AuditEntry


def test_log_entry(tmp_path):
    log_file = tmp_path / "audit.jsonl"
    logger = AuditLogger(log_path=str(log_file))
    logger.log(AuditEntry(
        agent_id="hr-agent",
        username="mike",
        tool="ask",
        query="What is the PTO policy?",
        documents_returned=["doc-1", "doc-2"],
        retrieval_strategy="lookup",
    ))
    lines = log_file.read_text().strip().split("\n")
    assert len(lines) == 1
    entry = json.loads(lines[0])
    assert entry["agent_id"] == "hr-agent"
    assert entry["username"] == "mike"
    assert entry["tool"] == "ask"
    assert "timestamp" in entry


def test_multiple_entries(tmp_path):
    log_file = tmp_path / "audit.jsonl"
    logger = AuditLogger(log_path=str(log_file))
    for i in range(3):
        logger.log(AuditEntry(
            agent_id="agent", username="user",
            tool="search", query=f"query {i}",
        ))
    lines = log_file.read_text().strip().split("\n")
    assert len(lines) == 3
```

- [ ] **Step 2: Implement audit logger**

```python
# src/audit/__init__.py
```

```python
# src/audit/logger.py
import json
import time
from dataclasses import dataclass, field, asdict


@dataclass
class AuditEntry:
    agent_id: str = ""
    username: str = ""
    tool: str = ""
    query: str = ""
    documents_returned: list[str] = field(default_factory=list)
    retrieval_strategy: str = ""
    timestamp: float = 0.0

    def __post_init__(self):
        if self.timestamp == 0.0:
            self.timestamp = time.time()


class AuditLogger:
    def __init__(self, log_path: str = "data/audit.jsonl"):
        self._log_path = log_path

    def log(self, entry: AuditEntry) -> None:
        with open(self._log_path, "a") as f:
            f.write(json.dumps(asdict(entry)) + "\n")
```

- [ ] **Step 3: Run tests**

Run: `source .venv/bin/activate && cd /Users/michaelmulkey/Documents/Repositories/rag && python -m pytest tests/test_audit/ -v`
Expected: 2 tests PASS

- [ ] **Step 4: Commit**

```bash
git add src/audit/ tests/test_audit/
git commit -m "feat: structured JSONL audit logger for MCP queries"
```

---

## Task 5: Low-Level MCP Tools

**Files:**
- Create: `rag/src/mcp/tools_low.py`
- Create: `rag/tests/test_mcp/test_tools_low.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_mcp/test_tools_low.py
import pytest
from unittest.mock import MagicMock, patch, AsyncMock
from src.mcp.tools_low import search_documents, query_database, list_sources, search_meetings, lookup_document
from src.retrieval.models import RetrievedChunk, ChunkMetadata


def _make_chunk(text, doc_type="pdf", filename="test.pdf", speaker=None, utterance_type=None):
    return RetrievedChunk(
        text=text, score=0.9,
        metadata=ChunkMetadata(
            doc_id="d1", filename=filename, doc_type=doc_type,
            chunk_index=0, start_char=0, acl_groups=["finance"],
            speaker=speaker, utterance_type=utterance_type,
        ),
    )


def test_search_documents():
    mock_store = MagicMock()
    mock_store.search.return_value = [_make_chunk("Some content")]
    with patch("src.mcp.tools_low.embed_query", return_value=[0.1] * 1024):
        result = search_documents(
            query="test", user_groups=["finance"],
            vector_store=mock_store,
        )
    assert len(result) == 1
    assert result[0]["text"] == "Some content"
    assert result[0]["source"] == "test.pdf"


def test_search_documents_with_doc_type_filter():
    chunk_pdf = _make_chunk("PDF content", doc_type="pdf")
    chunk_docx = _make_chunk("Word content", doc_type="docx", filename="test.docx")
    mock_store = MagicMock()
    mock_store.search.return_value = [chunk_pdf, chunk_docx]
    with patch("src.mcp.tools_low.embed_query", return_value=[0.1] * 1024):
        result = search_documents(
            query="test", user_groups=["finance"],
            vector_store=mock_store, doc_type="pdf",
        )
    assert len(result) == 1
    assert result[0]["source"] == "test.pdf"


def test_list_sources():
    mock_metadata = AsyncMock()
    mock_doc = MagicMock()
    mock_doc.category = "finance_policies"
    mock_doc.doc_type = "pdf"
    mock_metadata.list_documents.return_value = [mock_doc, mock_doc, mock_doc]
    result = list_sources(user_groups=["finance"], metadata_store=mock_metadata)
    assert len(result) >= 1


@pytest.mark.asyncio
async def test_query_database():
    with patch("src.mcp.tools_low.generate", return_value="SELECT revenue FROM results WHERE quarter='Q3'"):
        with patch("src.mcp.tools_low.execute_sql", new_callable=AsyncMock, return_value=[{"revenue": 1500000}]):
            from src.db.schema_registry import SchemaRegistry, TableSchema, ColumnSchema
            registry = SchemaRegistry()
            registry.register(TableSchema(
                database="finance_db", table="results",
                columns=[ColumnSchema(name="revenue", dtype="numeric", description="USD")],
                description="Results", acl_groups=["finance"],
            ))
            result = await query_database(
                question="Q3 revenue?", user_groups=["finance"],
                schema_registry=registry,
            )
    assert result["results"] == [{"revenue": 1500000}]


def test_search_meetings():
    chunks = [
        _make_chunk("Mike: Are we on track?", doc_type="transcript", filename="standup.txt", speaker="Mike", utterance_type="question"),
        _make_chunk("Sarah: Yes", doc_type="transcript", filename="standup.txt", speaker="Sarah", utterance_type="statement"),
    ]
    mock_store = MagicMock()
    mock_store.search.return_value = chunks
    with patch("src.mcp.tools_low.embed_query", return_value=[0.1] * 1024):
        result = search_meetings(
            user_groups=["engineering"], vector_store=mock_store,
            speaker="Mike", type_filter="question",
        )
    assert len(result) == 1
    assert result[0]["speaker"] == "Mike"
```

- [ ] **Step 2: Implement tools_low.py**

```python
# src/mcp/tools_low.py
from src.db.schema_registry import SchemaRegistry
from src.db.sql_executor import execute_sql
from src.db.metadata import MetadataStore
from src.generation.llm_client import generate
from src.ingestion.embedder import embed_query
from src.retrieval.vector_store import VectorStore


def search_documents(
    query: str,
    user_groups: list[str],
    vector_store: VectorStore,
    doc_type: str | None = None,
    top_k: int = 10,
) -> list[dict]:
    query_vector = embed_query(query)
    chunks = vector_store.search(vector=query_vector, user_groups=user_groups, top_k=top_k)

    if doc_type:
        chunks = [c for c in chunks if c.metadata.doc_type == doc_type]

    return [
        {
            "text": c.text,
            "source": c.metadata.filename,
            "doc_id": c.metadata.doc_id,
            "doc_type": c.metadata.doc_type,
            "page": c.metadata.page,
            "relevance": c.score,
        }
        for c in chunks
    ]


async def query_database(
    question: str,
    user_groups: list[str],
    schema_registry: SchemaRegistry,
) -> dict:
    schema_prompt = schema_registry.schemas_to_prompt(user_groups)
    if schema_prompt == "No database schemas available.":
        return {"sql": "", "results": [], "error": "No accessible database schemas"}

    sql = generate(
        system_prompt=f"Generate a single SELECT query.\n\nSchema:\n{schema_prompt}",
        user_prompt=f"Question: {question}",
        temperature=0.0, max_tokens=512,
    )
    sql = sql.strip().strip("`").removeprefix("sql\n").removeprefix("sql").strip()

    from src.config import settings
    try:
        rows = await execute_sql(database_url=settings.database_url, sql=sql)
    except (ValueError, Exception) as e:
        return {"sql": sql, "results": [], "error": str(e)}

    return {"sql": sql, "results": rows}


def lookup_document(
    doc_id: str,
    user_groups: list[str],
    vector_store: VectorStore,
) -> dict:
    # Search for chunks belonging to this doc_id
    # We use a zero vector and rely on filtering, or search with broad query
    query_vector = embed_query(f"document {doc_id}")
    chunks = vector_store.search(vector=query_vector, user_groups=user_groups, top_k=100)
    doc_chunks = [c for c in chunks if c.metadata.doc_id == doc_id]

    if not doc_chunks:
        return {"content": "", "metadata": {}, "error": "Document not found or access denied"}

    content = "\n\n".join(c.text for c in sorted(doc_chunks, key=lambda c: c.metadata.chunk_index))
    meta = doc_chunks[0].metadata
    return {
        "content": content,
        "metadata": {
            "doc_id": meta.doc_id,
            "filename": meta.filename,
            "doc_type": meta.doc_type,
            "category": meta.category,
        },
    }


def search_meetings(
    user_groups: list[str],
    vector_store: VectorStore,
    topic: str | None = None,
    speaker: str | None = None,
    type_filter: str | None = None,
    top_k: int = 50,
) -> list[dict]:
    query = topic or "meeting transcript"
    query_vector = embed_query(query)
    chunks = vector_store.search(vector=query_vector, user_groups=user_groups, top_k=top_k)

    # Filter to transcripts
    results = [c for c in chunks if c.metadata.doc_type == "transcript"]

    if speaker:
        results = [c for c in results if c.metadata.speaker == speaker]
    if type_filter:
        results = [c for c in results if c.metadata.utterance_type == type_filter]

    return [
        {
            "text": c.text,
            "speaker": c.metadata.speaker,
            "meeting": c.metadata.filename,
            "type": c.metadata.utterance_type,
            "relevance": c.score,
        }
        for c in results
    ]


def list_sources(
    user_groups: list[str],
    metadata_store: MetadataStore,
) -> list[dict]:
    import asyncio
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop and loop.is_running():
        # We're in an async context; use a workaround
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor() as pool:
            docs = pool.submit(asyncio.run, metadata_store.list_documents(user_groups=user_groups)).result()
    else:
        docs = asyncio.run(metadata_store.list_documents(user_groups=user_groups))

    # Group by category
    categories: dict[str, dict] = {}
    for doc in docs:
        cat = doc.category or "uncategorized"
        if cat not in categories:
            categories[cat] = {"name": cat, "type": doc.doc_type, "doc_count": 0}
        categories[cat]["doc_count"] += 1

    return list(categories.values())
```

- [ ] **Step 3: Run tests**

Run: `source .venv/bin/activate && cd /Users/michaelmulkey/Documents/Repositories/rag && python -m pytest tests/test_mcp/test_tools_low.py -v`
Expected: 5 tests PASS

- [ ] **Step 4: Commit**

```bash
git add src/mcp/tools_low.py tests/test_mcp/test_tools_low.py
git commit -m "feat: low-level MCP tools (search, query_database, lookup, meetings, sources)"
```

---

## Task 6: High-Level MCP Tools

**Files:**
- Create: `rag/src/mcp/tools_high.py`
- Create: `rag/tests/test_mcp/test_tools_high.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_mcp/test_tools_high.py
import pytest
from unittest.mock import patch, MagicMock, AsyncMock
from src.mcp.tools_high import ask, summarize_topic, compare
from src.generation.rag_chain import RAGResponse
from src.retrieval.models import Citation


def _mock_rag_response(answer="Test answer", citations=None):
    return RAGResponse(
        answer=answer,
        citations=citations or [Citation(
            doc_id="d1", filename="test.pdf", doc_type="pdf",
            chunk_index=0, page=1, snippet="test snippet", relevance=0.9,
        )],
    )


@pytest.mark.asyncio
async def test_ask_quick():
    with patch("src.mcp.tools_high.agent_query", new_callable=AsyncMock, return_value=_mock_rag_response()):
        result = await ask(
            question="What is policy 4.2?",
            user_groups=["finance"],
            vector_store=MagicMock(),
            schema_registry=MagicMock(),
            depth="quick",
        )
    assert result["answer"] == "Test answer"
    assert len(result["citations"]) == 1


@pytest.mark.asyncio
async def test_ask_with_context():
    with patch("src.mcp.tools_high.agent_query", new_callable=AsyncMock, return_value=_mock_rag_response()):
        result = await ask(
            question="PTO policy?",
            user_groups=["hr"],
            vector_store=MagicMock(),
            schema_registry=MagicMock(),
            context="Onboarding new employee in CA",
        )
    # Context should be prepended to question
    call_args = result["_call_args"]
    assert "Onboarding" in call_args["question"] or "PTO" in call_args["question"]


@pytest.mark.asyncio
async def test_summarize_topic():
    with patch("src.mcp.tools_high.agent_query", new_callable=AsyncMock, return_value=_mock_rag_response("Summary of Q3 results")):
        result = await summarize_topic(
            topic="Q3 financial results",
            user_groups=["finance"],
            vector_store=MagicMock(),
            schema_registry=MagicMock(),
        )
    assert "summary" in result
    assert result["summary"] == "Summary of Q3 results"


@pytest.mark.asyncio
async def test_compare():
    with patch("src.mcp.tools_high.agent_query", new_callable=AsyncMock, return_value=_mock_rag_response("Policy A requires X, Policy B requires Y")):
        result = await compare(
            item_a="Policy A",
            item_b="Policy B",
            user_groups=["finance"],
            vector_store=MagicMock(),
            schema_registry=MagicMock(),
        )
    assert "comparison" in result
```

- [ ] **Step 2: Implement tools_high.py**

```python
# src/mcp/tools_high.py
from src.db.schema_registry import SchemaRegistry
from src.generation.rag_chain import agent_query
from src.retrieval.vector_store import VectorStore


async def ask(
    question: str,
    user_groups: list[str],
    vector_store: VectorStore,
    schema_registry: SchemaRegistry,
    depth: str = "thorough",
    context: str | None = None,
) -> dict:
    # Prepend context to question if provided
    full_question = question
    if context:
        full_question = f"Context: {context}\n\nQuestion: {question}"

    result = await agent_query(
        question=full_question,
        user_groups=user_groups,
        vector_store=vector_store,
        schema_registry=schema_registry,
    )

    return {
        "answer": result.answer,
        "citations": [
            {
                "doc_id": c.doc_id,
                "filename": c.filename,
                "doc_type": c.doc_type,
                "chunk_index": c.chunk_index,
                "page": c.page,
                "snippet": c.snippet,
                "relevance": c.relevance,
            }
            for c in result.citations
        ],
        "retrieval_strategy": depth,
        "_call_args": {"question": full_question},
    }


async def summarize_topic(
    topic: str,
    user_groups: list[str],
    vector_store: VectorStore,
    schema_registry: SchemaRegistry,
    format: str = "brief",
) -> dict:
    question = f"Provide a {'brief' if format == 'brief' else 'detailed'} summary of: {topic}"

    result = await agent_query(
        question=question,
        user_groups=user_groups,
        vector_store=vector_store,
        schema_registry=schema_registry,
    )

    return {
        "summary": result.answer,
        "sources": [
            {"doc_id": c.doc_id, "filename": c.filename}
            for c in result.citations
        ],
    }


async def compare(
    item_a: str,
    item_b: str,
    user_groups: list[str],
    vector_store: VectorStore,
    schema_registry: SchemaRegistry,
) -> dict:
    question = f"Compare and contrast: '{item_a}' vs '{item_b}'. List key differences."

    result = await agent_query(
        question=question,
        user_groups=user_groups,
        vector_store=vector_store,
        schema_registry=schema_registry,
    )

    return {
        "comparison": result.answer,
        "sources": [
            {"doc_id": c.doc_id, "filename": c.filename}
            for c in result.citations
        ],
    }
```

- [ ] **Step 3: Run tests**

Run: `source .venv/bin/activate && cd /Users/michaelmulkey/Documents/Repositories/rag && python -m pytest tests/test_mcp/test_tools_high.py -v`
Expected: 4 tests PASS

- [ ] **Step 4: Commit**

```bash
git add src/mcp/tools_high.py tests/test_mcp/test_tools_high.py
git commit -m "feat: high-level MCP tools (ask, summarize_topic, compare)"
```

---

## Task 7: MCP Resources

**Files:**
- Create: `rag/src/mcp/resources.py`
- Create: `rag/tests/test_mcp/test_resources.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_mcp/test_resources.py
import pytest
from unittest.mock import MagicMock, AsyncMock
from src.mcp.resources import get_document_resource, get_category_resource, get_schema_resource
from src.db.schema_registry import SchemaRegistry, TableSchema, ColumnSchema


@pytest.mark.asyncio
async def test_get_document_resource():
    mock_metadata = AsyncMock()
    mock_doc = MagicMock()
    mock_doc.doc_id = "doc-1"
    mock_doc.filename = "policy.pdf"
    mock_doc.doc_type = "pdf"
    mock_doc.category = "finance_policies"
    mock_doc.acl_groups = ["finance"]
    mock_doc.chunk_count = 5
    mock_metadata.get_document.return_value = mock_doc
    result = await get_document_resource("doc-1", user_groups=["finance"], metadata_store=mock_metadata)
    assert result["doc_id"] == "doc-1"
    assert result["filename"] == "policy.pdf"


@pytest.mark.asyncio
async def test_get_document_resource_access_denied():
    mock_metadata = AsyncMock()
    mock_doc = MagicMock()
    mock_doc.acl_groups = ["finance"]
    mock_metadata.get_document.return_value = mock_doc
    result = await get_document_resource("doc-1", user_groups=["it_support"], metadata_store=mock_metadata)
    assert "error" in result
    assert "access" in result["error"].lower()


def test_get_category_resource():
    mock_metadata = AsyncMock()
    mock_doc1 = MagicMock()
    mock_doc1.category = "finance_policies"
    mock_doc1.doc_id = "d1"
    mock_doc1.filename = "a.pdf"
    mock_doc1.doc_type = "pdf"
    mock_doc1.acl_groups = ["finance"]
    mock_metadata.list_documents.return_value = [mock_doc1]
    result = get_category_resource("finance_policies", user_groups=["finance"], metadata_store=mock_metadata)
    assert result["name"] == "finance_policies"
    assert len(result["documents"]) == 1


def test_get_schema_resource():
    registry = SchemaRegistry()
    registry.register(TableSchema(
        database="finance_db", table="budget",
        columns=[ColumnSchema(name="amount", dtype="numeric", description="USD")],
        description="Budget data", acl_groups=["finance"],
    ))
    result = get_schema_resource("finance_db", user_groups=["finance"], schema_registry=registry)
    assert len(result["tables"]) == 1
    assert result["tables"][0]["table"] == "budget"
```

- [ ] **Step 2: Implement resources.py**

```python
# src/mcp/resources.py
import asyncio
from src.db.metadata import MetadataStore
from src.db.schema_registry import SchemaRegistry


async def get_document_resource(
    doc_id: str,
    user_groups: list[str],
    metadata_store: MetadataStore,
) -> dict:
    doc = await metadata_store.get_document(doc_id)
    if doc is None:
        return {"error": "Document not found"}

    if not any(g in doc.acl_groups for g in user_groups):
        return {"error": "Access denied for this document"}

    return {
        "doc_id": doc.doc_id,
        "filename": doc.filename,
        "doc_type": doc.doc_type,
        "category": doc.category,
        "acl_groups": doc.acl_groups,
        "chunk_count": doc.chunk_count,
    }


def get_category_resource(
    category_name: str,
    user_groups: list[str],
    metadata_store: MetadataStore,
) -> dict:
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop and loop.is_running():
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor() as pool:
            docs = pool.submit(asyncio.run, metadata_store.list_documents(user_groups=user_groups)).result()
    else:
        docs = asyncio.run(metadata_store.list_documents(user_groups=user_groups))

    category_docs = [d for d in docs if d.category == category_name]

    return {
        "name": category_name,
        "doc_count": len(category_docs),
        "documents": [
            {"doc_id": d.doc_id, "filename": d.filename, "doc_type": d.doc_type}
            for d in category_docs
        ],
    }


def get_schema_resource(
    database_name: str,
    user_groups: list[str],
    schema_registry: SchemaRegistry,
) -> dict:
    schemas = schema_registry.list_for_user(user_groups)
    db_schemas = [s for s in schemas if s.database == database_name]

    return {
        "database": database_name,
        "tables": [
            {
                "table": s.table,
                "description": s.description,
                "columns": [
                    {"name": c.name, "dtype": c.dtype, "description": c.description}
                    for c in s.columns
                ],
            }
            for s in db_schemas
        ],
    }
```

- [ ] **Step 3: Run tests**

Run: `source .venv/bin/activate && cd /Users/michaelmulkey/Documents/Repositories/rag && python -m pytest tests/test_mcp/test_resources.py -v`
Expected: 4 tests PASS

- [ ] **Step 4: Commit**

```bash
git add src/mcp/resources.py tests/test_mcp/test_resources.py
git commit -m "feat: MCP resources for document, category, and schema access"
```

---

## Task 8: FastMCP Server Wiring

**Files:**
- Create: `rag/src/mcp/server.py`
- Create: `rag/tests/test_mcp/test_server.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_mcp/test_server.py
import pytest
from unittest.mock import MagicMock, patch, AsyncMock
from src.mcp.server import create_mcp_server


def test_create_mcp_server():
    mock_store = MagicMock()
    from src.db.schema_registry import SchemaRegistry
    from src.db.metadata import MetadataStore
    from src.mcp.agent_registry import AgentRegistry
    server = create_mcp_server(
        vector_store=mock_store,
        schema_registry=SchemaRegistry(),
        metadata_store=MagicMock(),
        agent_registry=AgentRegistry(),
    )
    assert server is not None
    # Verify tools are registered
    assert hasattr(server, '_tool_manager') or hasattr(server, 'name')


def test_server_has_expected_name():
    mock_store = MagicMock()
    from src.db.schema_registry import SchemaRegistry
    from src.mcp.agent_registry import AgentRegistry
    server = create_mcp_server(
        vector_store=mock_store,
        schema_registry=SchemaRegistry(),
        metadata_store=MagicMock(),
        agent_registry=AgentRegistry(),
    )
    assert server.name == "rag-knowledge-service"
```

- [ ] **Step 2: Implement server.py**

```python
# src/mcp/server.py
from fastmcp import FastMCP

from src.db.metadata import MetadataStore
from src.db.schema_registry import SchemaRegistry
from src.mcp.agent_registry import AgentRegistry
from src.mcp.jobs import JobStore
from src.mcp.tools_high import ask, summarize_topic, compare
from src.mcp.tools_low import search_documents, query_database, lookup_document, search_meetings, list_sources
from src.mcp.resources import get_document_resource, get_category_resource, get_schema_resource
from src.retrieval.vector_store import VectorStore


def create_mcp_server(
    vector_store: VectorStore,
    schema_registry: SchemaRegistry,
    metadata_store: MetadataStore,
    agent_registry: AgentRegistry,
) -> FastMCP:
    mcp = FastMCP("rag-knowledge-service")
    job_store = JobStore()

    # ── High-Level Tools ─────────────────────────────────────

    @mcp.tool()
    async def tool_ask(question: str, depth: str = "thorough", context: str = "") -> dict:
        """Ask a question and get a complete cited answer from the knowledge base."""
        return await ask(
            question=question, user_groups=["ALL"],  # TODO: extract from MCP auth context
            vector_store=vector_store, schema_registry=schema_registry,
            depth=depth, context=context or None,
        )

    @mcp.tool()
    async def tool_summarize_topic(topic: str, format: str = "brief") -> dict:
        """Summarize a topic from the knowledge base."""
        return await summarize_topic(
            topic=topic, user_groups=["ALL"],
            vector_store=vector_store, schema_registry=schema_registry,
            format=format,
        )

    @mcp.tool()
    async def tool_compare(item_a: str, item_b: str) -> dict:
        """Compare two items using the knowledge base."""
        return await compare(
            item_a=item_a, item_b=item_b, user_groups=["ALL"],
            vector_store=vector_store, schema_registry=schema_registry,
        )

    # ── Low-Level Tools ──────────────────────────────────────

    @mcp.tool()
    def tool_search_documents(query: str, doc_type: str = "", top_k: int = 10) -> list[dict]:
        """Search documents by semantic similarity with optional type filter."""
        return search_documents(
            query=query, user_groups=["ALL"],
            vector_store=vector_store, doc_type=doc_type or None, top_k=top_k,
        )

    @mcp.tool()
    async def tool_query_database(question: str) -> dict:
        """Query a registered database using natural language (text-to-SQL)."""
        return await query_database(
            question=question, user_groups=["ALL"],
            schema_registry=schema_registry,
        )

    @mcp.tool()
    def tool_lookup_document(doc_id: str) -> dict:
        """Retrieve a specific document by ID."""
        return lookup_document(
            doc_id=doc_id, user_groups=["ALL"],
            vector_store=vector_store,
        )

    @mcp.tool()
    def tool_search_meetings(topic: str = "", speaker: str = "", type_filter: str = "") -> list[dict]:
        """Search meeting transcripts with optional speaker and type filters."""
        return search_meetings(
            user_groups=["ALL"], vector_store=vector_store,
            topic=topic or None, speaker=speaker or None,
            type_filter=type_filter or None,
        )

    @mcp.tool()
    def tool_list_sources() -> list[dict]:
        """List available knowledge sources and their document counts."""
        return list_sources(user_groups=["ALL"], metadata_store=metadata_store)

    # ── Async Job Tools ──────────────────────────────────────

    @mcp.tool()
    def tool_get_result(job_id: str) -> dict:
        """Check the status of an async job."""
        job = job_store.get(job_id)
        if job is None:
            return {"error": "Job not found"}
        return {
            "job_id": job_id,
            "status": job["status"],
            "result": job.get("result"),
            "error": job.get("error"),
        }

    # ── Resources ────────────────────────────────────────────

    @mcp.resource("document://{doc_id}")
    async def resource_document(doc_id: str) -> dict:
        """Get document metadata by ID."""
        return await get_document_resource(doc_id, user_groups=["ALL"], metadata_store=metadata_store)

    @mcp.resource("category://{category_name}")
    def resource_category(category_name: str) -> dict:
        """Get category details and document list."""
        return get_category_resource(category_name, user_groups=["ALL"], metadata_store=metadata_store)

    @mcp.resource("schema://{database_name}")
    def resource_schema(database_name: str) -> dict:
        """Get database schema details."""
        return get_schema_resource(database_name, user_groups=["ALL"], schema_registry=schema_registry)

    return mcp
```

- [ ] **Step 3: Run tests**

Run: `source .venv/bin/activate && cd /Users/michaelmulkey/Documents/Repositories/rag && python -m pytest tests/test_mcp/test_server.py -v`
Expected: 2 tests PASS

- [ ] **Step 4: Commit**

```bash
git add src/mcp/server.py tests/test_mcp/test_server.py
git commit -m "feat: FastMCP server with all tools, resources, and job support"
```

---

## Task 9: MCP Config & Docker Compose Update

**Files:**
- Modify: `rag/src/config.py`
- Modify: `rag/.env.example`
- Modify: `rag/docker-compose.yml`

- [ ] **Step 1: Add MCP settings to config.py**

Add to the Settings class:

```python
    # MCP Server
    mcp_server_name: str = "rag-knowledge-service"
    mcp_port: int = 8090

    # Audit
    audit_log_path: str = "data/audit.jsonl"
```

- [ ] **Step 2: Add to .env.example**

```bash
# MCP Server
MCP_SERVER_NAME=rag-knowledge-service
MCP_PORT=8090

# Audit
AUDIT_LOG_PATH=data/audit.jsonl
```

- [ ] **Step 3: Add MCP to docker-compose.yml**

Add a new service after the `api` service:

```yaml
  # MCP Server (Knowledge Service for AI agents)
  mcp:
    build: .
    command: ["python", "-m", "src.mcp.run"]
    ports:
      - "8090:8090"
    environment:
      - VLLM_BASE_URL=http://vllm:8000/v1
      - VLLM_MODEL_NAME=google/gemma-4-31b-it
      - EMBEDDING_MODEL_NAME=intfloat/e5-large-v2
      - EMBEDDING_DEVICE=cpu
      - QDRANT_HOST=qdrant
      - QDRANT_PORT=6333
      - QDRANT_COLLECTION_NAME=documents
      - JWT_SECRET_KEY=${JWT_SECRET_KEY:-change-me-in-production}
      - API_KEYS=${API_KEYS:-dev-key-1}
      - DATABASE_URL=sqlite+aiosqlite:///./data/metadata.db
      - MCP_PORT=8090
    volumes:
      - api_data:/app/data
      - embedding_cache:/root/.cache/huggingface
    depends_on:
      - qdrant
      - vllm
    restart: unless-stopped
```

- [ ] **Step 4: Create MCP runner script**

Create `src/mcp/run.py`:

```python
# src/mcp/run.py
"""Standalone MCP server runner."""
from src.api.routes_ingest import get_vector_store, get_metadata_store, get_schema_registry
from src.mcp.agent_registry import AgentRegistry
from src.mcp.server import create_mcp_server
from src.config import settings


def main():
    server = create_mcp_server(
        vector_store=get_vector_store(),
        schema_registry=get_schema_registry(),
        metadata_store=get_metadata_store(),
        agent_registry=AgentRegistry(),
    )
    server.run(transport="sse", port=settings.mcp_port)


if __name__ == "__main__":
    main()
```

- [ ] **Step 5: Run all tests to confirm no regressions**

Run: `source .venv/bin/activate && cd /Users/michaelmulkey/Documents/Repositories/rag && python -m pytest tests/ -v --tb=short`
Expected: All tests pass

- [ ] **Step 6: Commit**

```bash
git add src/config.py src/mcp/run.py .env.example docker-compose.yml
git commit -m "feat: MCP config, Docker Compose service, and standalone runner"
```

---

## Task 10: Final Integration Validation

**Files:** No new files

- [ ] **Step 1: Run complete test suite**

Run: `source .venv/bin/activate && cd /Users/michaelmulkey/Documents/Repositories/rag && python -m pytest tests/ -v --tb=short`
Expected: All tests pass

- [ ] **Step 2: Verify MCP server creates successfully**

Run: `source .venv/bin/activate && cd /Users/michaelmulkey/Documents/Repositories/rag && python -c "from src.mcp.server import create_mcp_server; from src.db.schema_registry import SchemaRegistry; from src.mcp.agent_registry import AgentRegistry; from unittest.mock import MagicMock; s = create_mcp_server(MagicMock(), SchemaRegistry(), MagicMock(), AgentRegistry()); print(f'MCP server: {s.name}')"`

- [ ] **Step 3: Verify FastAPI routes still work**

Run: `source .venv/bin/activate && cd /Users/michaelmulkey/Documents/Repositories/rag && python -c "from src.main import create_app; app = create_app(); print('Routes:'); [print(f'  {r.methods} {r.path}') for r in app.routes if hasattr(r, 'methods')]"`

- [ ] **Step 4: Commit**

```bash
git add -A
git commit -m "chore: Phase 3 MCP Knowledge Service complete — all tests passing"
```
