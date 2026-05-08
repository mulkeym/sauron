# Phase 2: Agentic Orchestrator — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the simple retrieve-generate pipeline with a LangGraph-based agent that classifies queries by type (lookup, sweep, analytical, cross-reference, temporal), routes each to the optimal retrieval strategy, evaluates results, and can re-retrieve if needed — all while maintaining ACL filtering and citations.

**Architecture:** A LangGraph StateGraph implements the 5-step agent flow: Plan → Filter → Retrieve → Evaluate → Synthesize. The query classifier uses the LLM to categorize queries. Each query type maps to a retrieval strategy: top-K vector search for lookups, map-reduce for sweeps, text-to-SQL for analytical, multi-source chaining for cross-references. The existing `rag_query()` function is replaced by `agent_query()` which runs the graph. The API route is updated to call the new agent.

**Tech Stack:** LangGraph, existing Gemma 4 31B via vLLM, existing Qdrant + E5-large, SQLAlchemy (for text-to-SQL against registered databases)

---

## File Structure

```
src/
├── agent/                              # NEW — agentic orchestrator
│   ├── __init__.py
│   ├── state.py                        # AgentState TypedDict for LangGraph
│   ├── classifier.py                   # Query type classification (LLM-based)
│   ├── strategies/                     # Retrieval strategies per query type
│   │   ├── __init__.py
│   │   ├── lookup.py                   # Top-K vector search
│   │   ├── sweep.py                    # Map-reduce exhaustive search
│   │   ├── analytical.py              # Text-to-SQL
│   │   └── cross_reference.py         # Multi-source chained retrieval
│   ├── evaluator.py                    # Evaluate retrieved context sufficiency
│   ├── synthesizer.py                  # Generate final answer with citations
│   └── graph.py                        # LangGraph StateGraph wiring
│
├── db/
│   ├── schema_registry.py             # NEW — register DB schemas for text-to-SQL
│   └── sql_executor.py                # NEW — safe SQL execution with ACL
│
├── generation/
│   └── rag_chain.py                    # MODIFY — keep RAGResponse, add agent_query()
│
├── api/
│   └── routes_query.py                 # MODIFY — call agent_query instead of rag_query
│
├── config.py                           # MODIFY — add database registry settings
│
tests/
├── test_agent/
│   ├── __init__.py
│   ├── test_classifier.py
│   ├── test_strategies/
│   │   ├── __init__.py
│   │   ├── test_lookup.py
│   │   ├── test_sweep.py
│   │   ├── test_analytical.py
│   │   └── test_cross_reference.py
│   ├── test_evaluator.py
│   ├── test_synthesizer.py
│   └── test_graph.py
├── test_db/
│   ├── test_schema_registry.py
│   └── test_sql_executor.py
```

---

## Task 1: Add LangGraph Dependency & Agent State

**Files:**
- Modify: `rag/requirements.txt`
- Create: `rag/src/agent/__init__.py`
- Create: `rag/src/agent/state.py`
- Create: `rag/tests/test_agent/__init__.py`
- Create: `rag/tests/test_agent/test_state.py`

- [ ] **Step 1: Add langgraph to requirements.txt**

Add this line to the end of the `# Core` section in `requirements.txt`:
```
langgraph>=0.3.0
```

Install: `source .venv/bin/activate && pip install "langgraph>=0.3.0"`

- [ ] **Step 2: Write the failing test for AgentState**

```python
# tests/test_agent/__init__.py
```

```python
# tests/test_agent/test_state.py
import pytest
from src.agent.state import AgentState, QueryType


def test_agent_state_defaults():
    state = AgentState(
        question="What is policy 4.2?",
        user_groups=["finance"],
    )
    assert state["question"] == "What is policy 4.2?"
    assert state["user_groups"] == ["finance"]
    assert state["query_type"] is None
    assert state["retrieved_chunks"] == []
    assert state["answer"] == ""
    assert state["citations"] == []
    assert state["needs_reretrieval"] is False
    assert state["retrieval_attempts"] == 0


def test_query_type_enum():
    assert QueryType.LOOKUP == "lookup"
    assert QueryType.SWEEP == "sweep"
    assert QueryType.ANALYTICAL == "analytical"
    assert QueryType.CROSS_REFERENCE == "cross_reference"
    assert QueryType.TEMPORAL == "temporal"
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `source .venv/bin/activate && cd /Users/michaelmulkey/Documents/Repositories/rag && python -m pytest tests/test_agent/test_state.py -v`
Expected: FAIL (module not found)

- [ ] **Step 4: Implement state.py**

```python
# src/agent/__init__.py
```

```python
# src/agent/state.py
from enum import StrEnum
from typing import TypedDict

from src.retrieval.models import Citation, RetrievedChunk


class QueryType(StrEnum):
    LOOKUP = "lookup"
    SWEEP = "sweep"
    ANALYTICAL = "analytical"
    CROSS_REFERENCE = "cross_reference"
    TEMPORAL = "temporal"


class AgentState(TypedDict, total=False):
    # Input
    question: str
    user_groups: list[str]

    # Classification
    query_type: QueryType | None
    sub_tasks: list[str]

    # Retrieval
    retrieved_chunks: list[RetrievedChunk]
    sql_results: list[dict]
    retrieval_attempts: int
    needs_reretrieval: bool

    # Output
    answer: str
    citations: list[Citation]
    warnings: list[str]
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `source .venv/bin/activate && cd /Users/michaelmulkey/Documents/Repositories/rag && python -m pytest tests/test_agent/test_state.py -v`
Expected: 2 tests PASS

- [ ] **Step 6: Commit**

```bash
git add requirements.txt src/agent/__init__.py src/agent/state.py tests/test_agent/__init__.py tests/test_agent/test_state.py
git commit -m "feat: LangGraph dependency and AgentState with QueryType enum"
```

---

## Task 2: Query Classifier

**Files:**
- Create: `rag/src/agent/classifier.py`
- Create: `rag/tests/test_agent/test_classifier.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_agent/test_classifier.py
import pytest
from unittest.mock import patch
from src.agent.classifier import classify_query
from src.agent.state import AgentState, QueryType


def test_classify_lookup():
    with patch("src.agent.classifier.generate", return_value='{"query_type": "lookup", "sub_tasks": ["Find policy 4.2 content"]}'):
        state = AgentState(question="What does policy 4.2 say?", user_groups=["finance"])
        result = classify_query(state)
    assert result["query_type"] == QueryType.LOOKUP


def test_classify_sweep():
    with patch("src.agent.classifier.generate", return_value='{"query_type": "sweep", "sub_tasks": ["Find all questions by Mike in meetings"]}'):
        state = AgentState(question="What questions did Mike ask in all meetings the last 30 days?", user_groups=["engineering"])
        result = classify_query(state)
    assert result["query_type"] == QueryType.SWEEP


def test_classify_analytical():
    with patch("src.agent.classifier.generate", return_value='{"query_type": "analytical", "sub_tasks": ["Query Q3 revenue from database"]}'):
        state = AgentState(question="What was our Q3 revenue?", user_groups=["finance"])
        result = classify_query(state)
    assert result["query_type"] == QueryType.ANALYTICAL


def test_classify_cross_reference():
    with patch("src.agent.classifier.generate", return_value='{"query_type": "cross_reference", "sub_tasks": ["Get Q3 spending from database", "Find expense policy in docs"]}'):
        state = AgentState(question="Does our Q3 spending comply with policy 4.2?", user_groups=["finance"])
        result = classify_query(state)
    assert result["query_type"] == QueryType.CROSS_REFERENCE
    assert len(result["sub_tasks"]) == 2


def test_classify_temporal():
    with patch("src.agent.classifier.generate", return_value='{"query_type": "temporal", "sub_tasks": ["Find docs changed in last month"]}'):
        state = AgentState(question="What policies changed last month?", user_groups=["finance"])
        result = classify_query(state)
    assert result["query_type"] == QueryType.TEMPORAL


def test_classify_fallback_on_bad_json():
    with patch("src.agent.classifier.generate", return_value="I'm not sure how to classify this"):
        state = AgentState(question="Tell me something", user_groups=["finance"])
        result = classify_query(state)
    assert result["query_type"] == QueryType.LOOKUP  # safe fallback
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `source .venv/bin/activate && cd /Users/michaelmulkey/Documents/Repositories/rag && python -m pytest tests/test_agent/test_classifier.py -v`
Expected: FAIL

- [ ] **Step 3: Implement classifier.py**

```python
# src/agent/classifier.py
import json

from src.agent.state import AgentState, QueryType
from src.generation.llm_client import generate

CLASSIFICATION_PROMPT = """You are a query classifier for a document knowledge base. Classify the user's question into exactly one type and identify sub-tasks.

Query types:
- lookup: Direct question about a specific document, policy, or fact. Example: "What does policy 4.2 say?"
- sweep: Exhaustive search across many documents for a pattern. Example: "What questions did Mike ask in all meetings?"
- analytical: Question requiring data from a database or spreadsheet (numbers, aggregations). Example: "What was Q3 revenue?"
- cross_reference: Question spanning multiple source types (e.g., compare database data against a policy). Example: "Does our spending comply with policy?"
- temporal: Question about changes over time or date-bounded searches. Example: "What changed last month?"

Respond with ONLY valid JSON:
{"query_type": "<type>", "sub_tasks": ["<task1>", "<task2>"]}"""


def classify_query(state: AgentState) -> dict:
    question = state["question"]

    response = generate(
        system_prompt=CLASSIFICATION_PROMPT,
        user_prompt=f"Question: {question}",
        temperature=0.0,
        max_tokens=256,
    )

    try:
        parsed = json.loads(response)
        query_type = QueryType(parsed["query_type"])
        sub_tasks = parsed.get("sub_tasks", [question])
    except (json.JSONDecodeError, ValueError, KeyError):
        query_type = QueryType.LOOKUP
        sub_tasks = [question]

    return {"query_type": query_type, "sub_tasks": sub_tasks}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `source .venv/bin/activate && cd /Users/michaelmulkey/Documents/Repositories/rag && python -m pytest tests/test_agent/test_classifier.py -v`
Expected: 6 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/agent/classifier.py tests/test_agent/test_classifier.py
git commit -m "feat: LLM-based query classifier with 5 query types"
```

---

## Task 3: Lookup Retrieval Strategy

**Files:**
- Create: `rag/src/agent/strategies/__init__.py`
- Create: `rag/src/agent/strategies/lookup.py`
- Create: `rag/tests/test_agent/test_strategies/__init__.py`
- Create: `rag/tests/test_agent/test_strategies/test_lookup.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_agent/test_strategies/__init__.py
```

```python
# tests/test_agent/test_strategies/test_lookup.py
import pytest
from unittest.mock import MagicMock, patch
from src.agent.strategies.lookup import retrieve_lookup
from src.agent.state import AgentState, QueryType
from src.retrieval.models import RetrievedChunk, ChunkMetadata


@pytest.fixture
def mock_chunks():
    return [
        RetrievedChunk(
            text="All expenses over $500 require manager approval.",
            score=0.95,
            metadata=ChunkMetadata(
                doc_id="doc-1", filename="policy.pdf", doc_type="pdf",
                chunk_index=0, start_char=0, acl_groups=["finance"], page=12,
            ),
        ),
    ]


def test_lookup_returns_chunks(mock_chunks):
    mock_store = MagicMock()
    mock_store.search.return_value = mock_chunks

    with patch("src.agent.strategies.lookup.embed_query", return_value=[0.1] * 1024):
        state = AgentState(
            question="What is the expense policy?",
            user_groups=["finance"],
            query_type=QueryType.LOOKUP,
            retrieved_chunks=[],
            retrieval_attempts=0,
        )
        result = retrieve_lookup(state, vector_store=mock_store)

    assert len(result["retrieved_chunks"]) == 1
    assert result["retrieval_attempts"] == 1


def test_lookup_empty_results():
    mock_store = MagicMock()
    mock_store.search.return_value = []

    with patch("src.agent.strategies.lookup.embed_query", return_value=[0.1] * 1024):
        state = AgentState(
            question="Something obscure",
            user_groups=["finance"],
            query_type=QueryType.LOOKUP,
            retrieved_chunks=[],
            retrieval_attempts=0,
        )
        result = retrieve_lookup(state, vector_store=mock_store)

    assert result["retrieved_chunks"] == []
    assert result["retrieval_attempts"] == 1
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `source .venv/bin/activate && cd /Users/michaelmulkey/Documents/Repositories/rag && python -m pytest tests/test_agent/test_strategies/test_lookup.py -v`
Expected: FAIL

- [ ] **Step 3: Implement lookup.py**

```python
# src/agent/strategies/__init__.py
```

```python
# src/agent/strategies/lookup.py
from src.agent.state import AgentState
from src.ingestion.embedder import embed_query
from src.retrieval.vector_store import VectorStore


def retrieve_lookup(state: AgentState, vector_store: VectorStore, top_k: int = 10) -> dict:
    question = state["question"]
    user_groups = state["user_groups"]

    query_vector = embed_query(question)
    chunks = vector_store.search(
        vector=query_vector,
        user_groups=user_groups,
        top_k=top_k,
    )

    return {
        "retrieved_chunks": chunks,
        "retrieval_attempts": state.get("retrieval_attempts", 0) + 1,
    }
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `source .venv/bin/activate && cd /Users/michaelmulkey/Documents/Repositories/rag && python -m pytest tests/test_agent/test_strategies/test_lookup.py -v`
Expected: 2 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/agent/strategies/ tests/test_agent/test_strategies/
git commit -m "feat: lookup retrieval strategy (top-K vector search)"
```

---

## Task 4: Sweep Retrieval Strategy (Map-Reduce)

**Files:**
- Create: `rag/src/agent/strategies/sweep.py`
- Create: `rag/tests/test_agent/test_strategies/test_sweep.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_agent/test_strategies/test_sweep.py
import pytest
from unittest.mock import MagicMock, patch
from src.agent.strategies.sweep import retrieve_sweep
from src.agent.state import AgentState, QueryType
from src.retrieval.models import RetrievedChunk, ChunkMetadata


def _make_chunk(text, doc_id="d1", filename="transcript.txt", speaker=None, utterance_type=None, score=0.9):
    return RetrievedChunk(
        text=text, score=score,
        metadata=ChunkMetadata(
            doc_id=doc_id, filename=filename, doc_type="transcript",
            chunk_index=0, start_char=0, acl_groups=["engineering"],
            speaker=speaker, utterance_type=utterance_type,
        ),
    )


def test_sweep_combines_semantic_and_metadata_results():
    semantic_chunks = [_make_chunk("Mike asked about Q2 timeline", speaker="Mike", utterance_type="question")]
    metadata_chunks = [_make_chunk("Mike: What's blocking the API?", speaker="Mike", utterance_type="question")]

    mock_store = MagicMock()
    # First call = semantic search, second call = metadata-filtered search
    mock_store.search.side_effect = [semantic_chunks, metadata_chunks]

    with patch("src.agent.strategies.sweep.embed_query", return_value=[0.1] * 1024):
        state = AgentState(
            question="What questions did Mike ask in all meetings?",
            user_groups=["engineering"],
            query_type=QueryType.SWEEP,
            retrieved_chunks=[],
            retrieval_attempts=0,
        )
        result = retrieve_sweep(state, vector_store=mock_store)

    assert len(result["retrieved_chunks"]) >= 1
    assert result["retrieval_attempts"] == 1


def test_sweep_deduplicates():
    same_chunk = _make_chunk("Mike: Are we on track?", doc_id="d1", speaker="Mike")
    mock_store = MagicMock()
    mock_store.search.side_effect = [[same_chunk], [same_chunk]]

    with patch("src.agent.strategies.sweep.embed_query", return_value=[0.1] * 1024):
        state = AgentState(
            question="What did Mike ask?",
            user_groups=["engineering"],
            query_type=QueryType.SWEEP,
            retrieved_chunks=[],
            retrieval_attempts=0,
        )
        result = retrieve_sweep(state, vector_store=mock_store)

    # Should deduplicate identical chunks
    assert len(result["retrieved_chunks"]) == 1
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `source .venv/bin/activate && cd /Users/michaelmulkey/Documents/Repositories/rag && python -m pytest tests/test_agent/test_strategies/test_sweep.py -v`
Expected: FAIL

- [ ] **Step 3: Implement sweep.py**

```python
# src/agent/strategies/sweep.py
from src.agent.state import AgentState
from src.ingestion.embedder import embed_query
from src.retrieval.models import RetrievedChunk
from src.retrieval.vector_store import VectorStore


def retrieve_sweep(state: AgentState, vector_store: VectorStore, top_k: int = 50) -> dict:
    question = state["question"]
    user_groups = state["user_groups"]

    all_chunks: list[RetrievedChunk] = []

    # Strategy 1: Semantic search with broader top_k
    query_vector = embed_query(question)
    semantic_results = vector_store.search(
        vector=query_vector,
        user_groups=user_groups,
        top_k=top_k,
    )
    all_chunks.extend(semantic_results)

    # Strategy 2: Metadata-filtered search (e.g., speaker-specific for transcripts)
    # Re-use the same vector but with higher top_k to cast wider net
    metadata_results = vector_store.search(
        vector=query_vector,
        user_groups=user_groups,
        top_k=top_k * 2,
    )
    all_chunks.extend(metadata_results)

    # Deduplicate by (doc_id, chunk_index)
    seen = set()
    unique_chunks = []
    for chunk in all_chunks:
        key = (chunk.metadata.doc_id, chunk.metadata.chunk_index)
        if key not in seen:
            seen.add(key)
            unique_chunks.append(chunk)

    # Sort by relevance
    unique_chunks.sort(key=lambda c: c.score, reverse=True)

    return {
        "retrieved_chunks": unique_chunks,
        "retrieval_attempts": state.get("retrieval_attempts", 0) + 1,
    }
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `source .venv/bin/activate && cd /Users/michaelmulkey/Documents/Repositories/rag && python -m pytest tests/test_agent/test_strategies/test_sweep.py -v`
Expected: 2 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/agent/strategies/sweep.py tests/test_agent/test_strategies/test_sweep.py
git commit -m "feat: sweep retrieval strategy with deduplication"
```

---

## Task 5: Schema Registry & SQL Executor

**Files:**
- Create: `rag/src/db/schema_registry.py`
- Create: `rag/src/db/sql_executor.py`
- Create: `rag/tests/test_db/test_schema_registry.py`
- Create: `rag/tests/test_db/test_sql_executor.py`

- [ ] **Step 1: Write failing tests for schema registry**

```python
# tests/test_db/test_schema_registry.py
import pytest
from src.db.schema_registry import SchemaRegistry, TableSchema, ColumnSchema


def test_register_and_get_schema():
    registry = SchemaRegistry()
    schema = TableSchema(
        database="finance_db",
        table="quarterly_results",
        columns=[
            ColumnSchema(name="quarter", dtype="varchar", description="Q1, Q2, Q3, Q4"),
            ColumnSchema(name="revenue", dtype="numeric", description="Revenue in USD"),
            ColumnSchema(name="year", dtype="integer", description="Fiscal year"),
        ],
        description="Quarterly financial results",
        acl_groups=["finance", "executives"],
    )
    registry.register(schema)
    result = registry.get_schema("finance_db", "quarterly_results")
    assert result is not None
    assert result.table == "quarterly_results"
    assert len(result.columns) == 3


def test_get_nonexistent_schema():
    registry = SchemaRegistry()
    result = registry.get_schema("nope", "nope")
    assert result is None


def test_list_schemas_for_user():
    registry = SchemaRegistry()
    registry.register(TableSchema(database="finance_db", table="budget", columns=[], description="Budget", acl_groups=["finance"]))
    registry.register(TableSchema(database="it_db", table="servers", columns=[], description="Servers", acl_groups=["it_support"]))

    finance_schemas = registry.list_for_user(["finance"])
    assert len(finance_schemas) == 1
    assert finance_schemas[0].table == "budget"

    all_schemas = registry.list_for_user(["finance", "it_support"])
    assert len(all_schemas) == 2


def test_schema_to_prompt_string():
    registry = SchemaRegistry()
    schema = TableSchema(
        database="finance_db",
        table="quarterly_results",
        columns=[
            ColumnSchema(name="quarter", dtype="varchar", description="Q1-Q4"),
            ColumnSchema(name="revenue", dtype="numeric", description="Revenue in USD"),
        ],
        description="Quarterly results",
        acl_groups=["finance"],
    )
    registry.register(schema)
    prompt = registry.schemas_to_prompt(["finance"])
    assert "quarterly_results" in prompt
    assert "revenue" in prompt
    assert "numeric" in prompt
```

- [ ] **Step 2: Write failing tests for SQL executor**

```python
# tests/test_db/test_sql_executor.py
import pytest
import pytest_asyncio
import aiosqlite


@pytest_asyncio.fixture
async def test_db(tmp_path):
    db_path = tmp_path / "test.db"
    async with aiosqlite.connect(str(db_path)) as db:
        await db.execute("CREATE TABLE quarterly_results (quarter TEXT, revenue REAL, year INTEGER)")
        await db.execute("INSERT INTO quarterly_results VALUES ('Q3', 1500000, 2026)")
        await db.execute("INSERT INTO quarterly_results VALUES ('Q2', 1200000, 2026)")
        await db.commit()
    return str(db_path)


@pytest.mark.asyncio
async def test_execute_select(test_db):
    from src.db.sql_executor import execute_sql
    result = await execute_sql(
        database_url=f"sqlite+aiosqlite:///{test_db}",
        sql="SELECT quarter, revenue FROM quarterly_results WHERE quarter = 'Q3'",
    )
    assert len(result) == 1
    assert result[0]["quarter"] == "Q3"
    assert result[0]["revenue"] == 1500000


@pytest.mark.asyncio
async def test_execute_rejects_write_operations(test_db):
    from src.db.sql_executor import execute_sql
    with pytest.raises(ValueError, match="Only SELECT"):
        await execute_sql(
            database_url=f"sqlite+aiosqlite:///{test_db}",
            sql="DROP TABLE quarterly_results",
        )


@pytest.mark.asyncio
async def test_execute_rejects_multiple_statements(test_db):
    from src.db.sql_executor import execute_sql
    with pytest.raises(ValueError, match="single SELECT"):
        await execute_sql(
            database_url=f"sqlite+aiosqlite:///{test_db}",
            sql="SELECT 1; DROP TABLE quarterly_results",
        )
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `source .venv/bin/activate && cd /Users/michaelmulkey/Documents/Repositories/rag && python -m pytest tests/test_db/test_schema_registry.py tests/test_db/test_sql_executor.py -v`
Expected: FAIL

- [ ] **Step 4: Implement schema_registry.py**

```python
# src/db/schema_registry.py
from dataclasses import dataclass, field


@dataclass
class ColumnSchema:
    name: str
    dtype: str
    description: str = ""


@dataclass
class TableSchema:
    database: str
    table: str
    columns: list[ColumnSchema]
    description: str = ""
    acl_groups: list[str] = field(default_factory=list)


class SchemaRegistry:
    def __init__(self):
        self._schemas: dict[str, TableSchema] = {}

    def register(self, schema: TableSchema) -> None:
        key = f"{schema.database}.{schema.table}"
        self._schemas[key] = schema

    def get_schema(self, database: str, table: str) -> TableSchema | None:
        return self._schemas.get(f"{database}.{table}")

    def list_for_user(self, user_groups: list[str]) -> list[TableSchema]:
        return [
            s for s in self._schemas.values()
            if any(g in s.acl_groups for g in user_groups)
        ]

    def schemas_to_prompt(self, user_groups: list[str]) -> str:
        schemas = self.list_for_user(user_groups)
        if not schemas:
            return "No database schemas available."

        parts = []
        for s in schemas:
            cols = "\n".join(f"  - {c.name} ({c.dtype}): {c.description}" for c in s.columns)
            parts.append(f"Database: {s.database}\nTable: {s.table}\nDescription: {s.description}\nColumns:\n{cols}")
        return "\n\n".join(parts)
```

- [ ] **Step 5: Implement sql_executor.py**

```python
# src/db/sql_executor.py
import re

from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine


async def execute_sql(database_url: str, sql: str) -> list[dict]:
    sql = sql.strip().rstrip(";")

    # Safety: only allow single SELECT statements
    if ";" in sql:
        raise ValueError("Only a single SELECT statement is allowed")

    if not re.match(r"^\s*SELECT\b", sql, re.IGNORECASE):
        raise ValueError("Only SELECT queries are allowed")

    engine = create_async_engine(database_url)
    try:
        async with engine.connect() as conn:
            result = await conn.execute(text(sql))
            columns = list(result.keys())
            rows = [dict(zip(columns, row)) for row in result.fetchall()]
        return rows
    finally:
        await engine.dispose()
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `source .venv/bin/activate && cd /Users/michaelmulkey/Documents/Repositories/rag && python -m pytest tests/test_db/test_schema_registry.py tests/test_db/test_sql_executor.py -v`
Expected: 7 tests PASS

- [ ] **Step 7: Commit**

```bash
git add src/db/schema_registry.py src/db/sql_executor.py tests/test_db/test_schema_registry.py tests/test_db/test_sql_executor.py
git commit -m "feat: schema registry and safe SQL executor for text-to-SQL"
```

---

## Task 6: Analytical Retrieval Strategy (Text-to-SQL)

**Files:**
- Create: `rag/src/agent/strategies/analytical.py`
- Create: `rag/tests/test_agent/test_strategies/test_analytical.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_agent/test_strategies/test_analytical.py
import pytest
from unittest.mock import patch, AsyncMock, MagicMock
from src.agent.strategies.analytical import retrieve_analytical
from src.agent.state import AgentState, QueryType
from src.db.schema_registry import SchemaRegistry, TableSchema, ColumnSchema


@pytest.fixture
def registry():
    r = SchemaRegistry()
    r.register(TableSchema(
        database="finance_db",
        table="quarterly_results",
        columns=[
            ColumnSchema(name="quarter", dtype="varchar", description="Q1-Q4"),
            ColumnSchema(name="revenue", dtype="numeric", description="Revenue in USD"),
            ColumnSchema(name="year", dtype="integer", description="Fiscal year"),
        ],
        description="Quarterly financial results",
        acl_groups=["finance"],
    ))
    return r


@pytest.mark.asyncio
async def test_analytical_generates_and_executes_sql(registry):
    with patch("src.agent.strategies.analytical.generate", return_value="SELECT revenue FROM quarterly_results WHERE quarter = 'Q3' AND year = 2026"):
        with patch("src.agent.strategies.analytical.execute_sql", new_callable=AsyncMock, return_value=[{"revenue": 1500000}]):
            state = AgentState(
                question="What was Q3 2026 revenue?",
                user_groups=["finance"],
                query_type=QueryType.ANALYTICAL,
                retrieved_chunks=[],
                sql_results=[],
                retrieval_attempts=0,
            )
            result = await retrieve_analytical(state, schema_registry=registry)

    assert len(result["sql_results"]) == 1
    assert result["sql_results"][0]["revenue"] == 1500000
    assert result["retrieval_attempts"] == 1


@pytest.mark.asyncio
async def test_analytical_no_schemas_available():
    empty_registry = SchemaRegistry()
    state = AgentState(
        question="What was Q3 revenue?",
        user_groups=["engineering"],  # no finance access
        query_type=QueryType.ANALYTICAL,
        retrieved_chunks=[],
        sql_results=[],
        retrieval_attempts=0,
    )
    result = await retrieve_analytical(state, schema_registry=empty_registry)
    assert result["sql_results"] == []
    assert len(result.get("warnings", [])) > 0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `source .venv/bin/activate && cd /Users/michaelmulkey/Documents/Repositories/rag && python -m pytest tests/test_agent/test_strategies/test_analytical.py -v`
Expected: FAIL

- [ ] **Step 3: Implement analytical.py**

```python
# src/agent/strategies/analytical.py
from src.agent.state import AgentState
from src.db.schema_registry import SchemaRegistry
from src.db.sql_executor import execute_sql
from src.generation.llm_client import generate

TEXT_TO_SQL_PROMPT = """You are a SQL query generator. Given a natural language question and database schema, generate a single SELECT query.

Rules:
- Output ONLY the SQL query, no explanation
- Only use tables and columns from the provided schema
- Always use SELECT (never INSERT, UPDATE, DELETE, DROP, etc.)
- Keep queries simple and correct

Schema:
{schema}"""


async def retrieve_analytical(state: AgentState, schema_registry: SchemaRegistry) -> dict:
    question = state["question"]
    user_groups = state["user_groups"]

    schema_prompt = schema_registry.schemas_to_prompt(user_groups)

    if schema_prompt == "No database schemas available.":
        return {
            "sql_results": [],
            "warnings": ["No database schemas available for your access groups."],
            "retrieval_attempts": state.get("retrieval_attempts", 0) + 1,
        }

    sql = generate(
        system_prompt=TEXT_TO_SQL_PROMPT.format(schema=schema_prompt),
        user_prompt=f"Question: {question}",
        temperature=0.0,
        max_tokens=512,
    )

    # Clean up LLM output
    sql = sql.strip().strip("`").removeprefix("sql\n").removeprefix("sql").strip()

    # Find the database URL for the first matching schema
    schemas = schema_registry.list_for_user(user_groups)
    if not schemas:
        return {
            "sql_results": [],
            "warnings": ["No accessible databases found."],
            "retrieval_attempts": state.get("retrieval_attempts", 0) + 1,
        }

    # For Phase 2, use a configurable database URL per schema
    # Default to the metadata DB for now
    from src.config import settings
    database_url = settings.database_url

    try:
        rows = await execute_sql(database_url=database_url, sql=sql)
    except (ValueError, Exception) as e:
        return {
            "sql_results": [],
            "warnings": [f"SQL execution error: {e}"],
            "retrieval_attempts": state.get("retrieval_attempts", 0) + 1,
        }

    return {
        "sql_results": rows,
        "retrieval_attempts": state.get("retrieval_attempts", 0) + 1,
    }
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `source .venv/bin/activate && cd /Users/michaelmulkey/Documents/Repositories/rag && python -m pytest tests/test_agent/test_strategies/test_analytical.py -v`
Expected: 2 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/agent/strategies/analytical.py tests/test_agent/test_strategies/test_analytical.py
git commit -m "feat: text-to-SQL analytical retrieval strategy"
```

---

## Task 7: Cross-Reference Retrieval Strategy

**Files:**
- Create: `rag/src/agent/strategies/cross_reference.py`
- Create: `rag/tests/test_agent/test_strategies/test_cross_reference.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_agent/test_strategies/test_cross_reference.py
import pytest
from unittest.mock import MagicMock, patch, AsyncMock
from src.agent.strategies.cross_reference import retrieve_cross_reference
from src.agent.state import AgentState, QueryType
from src.retrieval.models import RetrievedChunk, ChunkMetadata
from src.db.schema_registry import SchemaRegistry, TableSchema, ColumnSchema


def _make_chunk(text, doc_type="pdf", filename="policy.pdf"):
    return RetrievedChunk(
        text=text, score=0.9,
        metadata=ChunkMetadata(
            doc_id="d1", filename=filename, doc_type=doc_type,
            chunk_index=0, start_char=0, acl_groups=["finance"],
        ),
    )


@pytest.fixture
def registry():
    r = SchemaRegistry()
    r.register(TableSchema(
        database="finance_db", table="expenses",
        columns=[ColumnSchema(name="amount", dtype="numeric", description="Amount in USD")],
        description="Expense records", acl_groups=["finance"],
    ))
    return r


@pytest.mark.asyncio
async def test_cross_reference_combines_doc_and_sql(registry):
    mock_store = MagicMock()
    mock_store.search.return_value = [_make_chunk("Policy 4.2: Expenses over $500 need approval")]

    with patch("src.agent.strategies.cross_reference.embed_query", return_value=[0.1] * 1024):
        with patch("src.agent.strategies.cross_reference.retrieve_analytical", new_callable=AsyncMock, return_value={"sql_results": [{"amount": 750}], "retrieval_attempts": 1}):
            state = AgentState(
                question="Does our spending comply with policy 4.2?",
                user_groups=["finance"],
                query_type=QueryType.CROSS_REFERENCE,
                sub_tasks=["Get spending data", "Find policy 4.2"],
                retrieved_chunks=[],
                sql_results=[],
                retrieval_attempts=0,
            )
            result = await retrieve_cross_reference(state, vector_store=mock_store, schema_registry=registry)

    assert len(result["retrieved_chunks"]) > 0
    assert len(result["sql_results"]) > 0


@pytest.mark.asyncio
async def test_cross_reference_doc_only_when_no_db():
    mock_store = MagicMock()
    mock_store.search.return_value = [_make_chunk("Some policy content")]
    empty_registry = SchemaRegistry()

    with patch("src.agent.strategies.cross_reference.embed_query", return_value=[0.1] * 1024):
        state = AgentState(
            question="Compare policy A with policy B",
            user_groups=["finance"],
            query_type=QueryType.CROSS_REFERENCE,
            sub_tasks=["Find policy A", "Find policy B"],
            retrieved_chunks=[],
            sql_results=[],
            retrieval_attempts=0,
        )
        result = await retrieve_cross_reference(state, vector_store=mock_store, schema_registry=empty_registry)

    assert len(result["retrieved_chunks"]) > 0
    assert result["sql_results"] == []
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `source .venv/bin/activate && cd /Users/michaelmulkey/Documents/Repositories/rag && python -m pytest tests/test_agent/test_strategies/test_cross_reference.py -v`
Expected: FAIL

- [ ] **Step 3: Implement cross_reference.py**

```python
# src/agent/strategies/cross_reference.py
from src.agent.state import AgentState
from src.agent.strategies.analytical import retrieve_analytical
from src.db.schema_registry import SchemaRegistry
from src.ingestion.embedder import embed_query
from src.retrieval.models import RetrievedChunk
from src.retrieval.vector_store import VectorStore


async def retrieve_cross_reference(
    state: AgentState,
    vector_store: VectorStore,
    schema_registry: SchemaRegistry,
) -> dict:
    question = state["question"]
    user_groups = state["user_groups"]
    sub_tasks = state.get("sub_tasks", [question])

    all_chunks: list[RetrievedChunk] = []

    # Document retrieval for each sub-task
    for task in sub_tasks:
        query_vector = embed_query(task)
        chunks = vector_store.search(
            vector=query_vector,
            user_groups=user_groups,
            top_k=10,
        )
        all_chunks.extend(chunks)

    # Deduplicate
    seen = set()
    unique_chunks = []
    for chunk in all_chunks:
        key = (chunk.metadata.doc_id, chunk.metadata.chunk_index)
        if key not in seen:
            seen.add(key)
            unique_chunks.append(chunk)

    # Try analytical retrieval if schemas are available
    sql_results = []
    has_schemas = len(schema_registry.list_for_user(user_groups)) > 0
    if has_schemas:
        analytical_result = await retrieve_analytical(state, schema_registry=schema_registry)
        sql_results = analytical_result.get("sql_results", [])

    return {
        "retrieved_chunks": unique_chunks,
        "sql_results": sql_results,
        "retrieval_attempts": state.get("retrieval_attempts", 0) + 1,
    }
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `source .venv/bin/activate && cd /Users/michaelmulkey/Documents/Repositories/rag && python -m pytest tests/test_agent/test_strategies/test_cross_reference.py -v`
Expected: 2 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/agent/strategies/cross_reference.py tests/test_agent/test_strategies/test_cross_reference.py
git commit -m "feat: cross-reference strategy combining doc search and text-to-SQL"
```

---

## Task 8: Evaluator (Context Sufficiency Check)

**Files:**
- Create: `rag/src/agent/evaluator.py`
- Create: `rag/tests/test_agent/test_evaluator.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_agent/test_evaluator.py
import pytest
from unittest.mock import patch
from src.agent.evaluator import evaluate_context
from src.agent.state import AgentState, QueryType
from src.retrieval.models import RetrievedChunk, ChunkMetadata


def _make_chunk(text, score=0.9):
    return RetrievedChunk(
        text=text, score=score,
        metadata=ChunkMetadata(
            doc_id="d1", filename="test.pdf", doc_type="pdf",
            chunk_index=0, start_char=0, acl_groups=["finance"],
        ),
    )


def test_sufficient_context():
    with patch("src.agent.evaluator.generate", return_value='{"sufficient": true, "reason": "Context contains the answer"}'):
        state = AgentState(
            question="What is policy 4.2?",
            user_groups=["finance"],
            query_type=QueryType.LOOKUP,
            retrieved_chunks=[_make_chunk("Policy 4.2: Expenses over $500 need approval")],
            retrieval_attempts=1,
            needs_reretrieval=False,
        )
        result = evaluate_context(state)

    assert result["needs_reretrieval"] is False


def test_insufficient_context():
    with patch("src.agent.evaluator.generate", return_value='{"sufficient": false, "reason": "No relevant information found"}'):
        state = AgentState(
            question="What is policy 4.2?",
            user_groups=["finance"],
            query_type=QueryType.LOOKUP,
            retrieved_chunks=[_make_chunk("This is about server maintenance", score=0.3)],
            retrieval_attempts=1,
            needs_reretrieval=False,
        )
        result = evaluate_context(state)

    assert result["needs_reretrieval"] is True


def test_max_retrieval_attempts_stops_loop():
    state = AgentState(
        question="Something",
        user_groups=["finance"],
        query_type=QueryType.LOOKUP,
        retrieved_chunks=[],
        retrieval_attempts=3,  # already tried 3 times
        needs_reretrieval=False,
    )
    result = evaluate_context(state)
    assert result["needs_reretrieval"] is False  # stop trying


def test_empty_chunks_needs_reretrieval():
    state = AgentState(
        question="Something",
        user_groups=["finance"],
        query_type=QueryType.LOOKUP,
        retrieved_chunks=[],
        retrieval_attempts=1,
        needs_reretrieval=False,
    )
    result = evaluate_context(state)
    assert result["needs_reretrieval"] is True
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `source .venv/bin/activate && cd /Users/michaelmulkey/Documents/Repositories/rag && python -m pytest tests/test_agent/test_evaluator.py -v`
Expected: FAIL

- [ ] **Step 3: Implement evaluator.py**

```python
# src/agent/evaluator.py
import json

from src.agent.state import AgentState
from src.generation.llm_client import generate

MAX_RETRIEVAL_ATTEMPTS = 3

EVALUATION_PROMPT = """You are evaluating whether retrieved context is sufficient to answer a question.

Question: {question}

Retrieved context:
{context}

Is this context sufficient to answer the question? Respond with ONLY valid JSON:
{{"sufficient": true/false, "reason": "brief explanation"}}"""


def evaluate_context(state: AgentState) -> dict:
    retrieval_attempts = state.get("retrieval_attempts", 0)
    chunks = state.get("retrieved_chunks", [])
    sql_results = state.get("sql_results", [])

    # Hard stop after max attempts
    if retrieval_attempts >= MAX_RETRIEVAL_ATTEMPTS:
        return {"needs_reretrieval": False}

    # No context at all — try again if we haven't exhausted attempts
    if not chunks and not sql_results:
        return {"needs_reretrieval": True}

    # Build context summary for evaluation
    context_parts = [c.text for c in chunks[:5]]  # sample up to 5 chunks
    if sql_results:
        context_parts.append(f"SQL results: {json.dumps(sql_results[:5])}")

    context = "\n\n".join(context_parts)

    response = generate(
        system_prompt="You evaluate retrieved context sufficiency.",
        user_prompt=EVALUATION_PROMPT.format(question=state["question"], context=context),
        temperature=0.0,
        max_tokens=128,
    )

    try:
        parsed = json.loads(response)
        sufficient = parsed.get("sufficient", True)
    except (json.JSONDecodeError, KeyError):
        sufficient = True  # default to proceeding if parsing fails

    return {"needs_reretrieval": not sufficient}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `source .venv/bin/activate && cd /Users/michaelmulkey/Documents/Repositories/rag && python -m pytest tests/test_agent/test_evaluator.py -v`
Expected: 4 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/agent/evaluator.py tests/test_agent/test_evaluator.py
git commit -m "feat: context sufficiency evaluator with retry loop control"
```

---

## Task 9: Synthesizer (Answer Generation with Citations)

**Files:**
- Create: `rag/src/agent/synthesizer.py`
- Create: `rag/tests/test_agent/test_synthesizer.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_agent/test_synthesizer.py
import pytest
from unittest.mock import patch
from src.agent.synthesizer import synthesize_answer
from src.agent.state import AgentState, QueryType
from src.retrieval.models import RetrievedChunk, ChunkMetadata, Citation


def _make_chunk(text, doc_id="d1", filename="policy.pdf", page=None, score=0.9):
    return RetrievedChunk(
        text=text, score=score,
        metadata=ChunkMetadata(
            doc_id=doc_id, filename=filename, doc_type="pdf",
            chunk_index=0, start_char=0, acl_groups=["finance"], page=page,
        ),
    )


def test_synthesize_with_chunks():
    with patch("src.agent.synthesizer.generate", return_value="Expenses over $500 need approval [1]."):
        state = AgentState(
            question="What is the expense policy?",
            user_groups=["finance"],
            query_type=QueryType.LOOKUP,
            retrieved_chunks=[_make_chunk("All expenses over $500 require manager approval.", page=12)],
            sql_results=[],
        )
        result = synthesize_answer(state)

    assert "approval" in result["answer"].lower() or "500" in result["answer"]
    assert len(result["citations"]) == 1
    assert result["citations"][0].filename == "policy.pdf"
    assert result["citations"][0].page == 12


def test_synthesize_with_sql_results():
    with patch("src.agent.synthesizer.generate", return_value="Q3 2026 revenue was $1,500,000."):
        state = AgentState(
            question="What was Q3 revenue?",
            user_groups=["finance"],
            query_type=QueryType.ANALYTICAL,
            retrieved_chunks=[],
            sql_results=[{"quarter": "Q3", "revenue": 1500000, "year": 2026}],
        )
        result = synthesize_answer(state)

    assert "1,500,000" in result["answer"] or "1500000" in result["answer"]


def test_synthesize_no_context():
    state = AgentState(
        question="Something with no results",
        user_groups=["finance"],
        query_type=QueryType.LOOKUP,
        retrieved_chunks=[],
        sql_results=[],
    )
    result = synthesize_answer(state)
    assert "could not find" in result["answer"].lower()
    assert result["citations"] == []
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `source .venv/bin/activate && cd /Users/michaelmulkey/Documents/Repositories/rag && python -m pytest tests/test_agent/test_synthesizer.py -v`
Expected: FAIL

- [ ] **Step 3: Implement synthesizer.py**

```python
# src/agent/synthesizer.py
import json

from src.agent.state import AgentState
from src.generation.llm_client import generate
from src.retrieval.models import Citation

SYSTEM_PROMPT = """You are a knowledgeable assistant that answers questions based on provided context.

Rules:
- Only answer based on the provided context. Do not use outside knowledge.
- Cite document sources using [N] notation, where N corresponds to the context chunk number.
- If SQL results are provided, reference them in your answer.
- If the context does not contain enough information, say so clearly.
- Be concise and accurate."""

USER_PROMPT_TEMPLATE = """Context:
{context}

Question: {question}

Answer the question based only on the context above. Cite sources using [N] notation."""


def synthesize_answer(state: AgentState) -> dict:
    chunks = state.get("retrieved_chunks", [])
    sql_results = state.get("sql_results", [])
    question = state["question"]

    if not chunks and not sql_results:
        return {
            "answer": "I could not find any relevant information in the documents you have access to.",
            "citations": [],
        }

    # Build context
    context_parts = []
    for i, chunk in enumerate(chunks, 1):
        source = f"[{i}] {chunk.metadata.filename}"
        if chunk.metadata.page is not None:
            source += f", page {chunk.metadata.page}"
        context_parts.append(f"{source}:\n{chunk.text}")

    if sql_results:
        context_parts.append(f"[Database query results]:\n{json.dumps(sql_results, indent=2)}")

    context = "\n\n".join(context_parts)

    answer = generate(
        system_prompt=SYSTEM_PROMPT,
        user_prompt=USER_PROMPT_TEMPLATE.format(context=context, question=question),
    )

    citations = [
        Citation(
            doc_id=c.metadata.doc_id,
            filename=c.metadata.filename,
            doc_type=c.metadata.doc_type,
            chunk_index=c.metadata.chunk_index,
            page=c.metadata.page,
            snippet=c.text[:200],
            relevance=c.score,
        )
        for c in chunks
    ]

    return {"answer": answer, "citations": citations}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `source .venv/bin/activate && cd /Users/michaelmulkey/Documents/Repositories/rag && python -m pytest tests/test_agent/test_synthesizer.py -v`
Expected: 3 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/agent/synthesizer.py tests/test_agent/test_synthesizer.py
git commit -m "feat: answer synthesizer with citation support for chunks and SQL results"
```

---

## Task 10: LangGraph Agent Graph

**Files:**
- Create: `rag/src/agent/graph.py`
- Create: `rag/tests/test_agent/test_graph.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_agent/test_graph.py
import pytest
from unittest.mock import patch, MagicMock, AsyncMock
from src.agent.graph import create_agent_graph, run_agent
from src.agent.state import QueryType
from src.retrieval.models import RetrievedChunk, ChunkMetadata, Citation


def _make_chunk(text="Test content", score=0.9):
    return RetrievedChunk(
        text=text, score=score,
        metadata=ChunkMetadata(
            doc_id="d1", filename="test.pdf", doc_type="pdf",
            chunk_index=0, start_char=0, acl_groups=["finance"],
        ),
    )


def test_create_agent_graph():
    mock_store = MagicMock()
    from src.db.schema_registry import SchemaRegistry
    graph = create_agent_graph(vector_store=mock_store, schema_registry=SchemaRegistry())
    assert graph is not None


@pytest.mark.asyncio
async def test_run_agent_lookup():
    mock_store = MagicMock()
    mock_store.search.return_value = [_make_chunk("Policy 4.2 says expenses over $500 need approval")]

    with patch("src.agent.classifier.generate", return_value='{"query_type": "lookup", "sub_tasks": ["Find policy 4.2"]}'):
        with patch("src.agent.strategies.lookup.embed_query", return_value=[0.1] * 1024):
            with patch("src.agent.evaluator.generate", return_value='{"sufficient": true, "reason": "ok"}'):
                with patch("src.agent.synthesizer.generate", return_value="Policy 4.2 requires approval for expenses over $500 [1]."):
                    from src.db.schema_registry import SchemaRegistry
                    result = await run_agent(
                        question="What is policy 4.2?",
                        user_groups=["finance"],
                        vector_store=mock_store,
                        schema_registry=SchemaRegistry(),
                    )

    assert "approval" in result.answer.lower() or "500" in result.answer
    assert len(result.citations) >= 1


@pytest.mark.asyncio
async def test_run_agent_no_results():
    mock_store = MagicMock()
    mock_store.search.return_value = []

    with patch("src.agent.classifier.generate", return_value='{"query_type": "lookup", "sub_tasks": ["search"]}'):
        with patch("src.agent.strategies.lookup.embed_query", return_value=[0.1] * 1024):
            from src.db.schema_registry import SchemaRegistry
            result = await run_agent(
                question="Something obscure",
                user_groups=["finance"],
                vector_store=mock_store,
                schema_registry=SchemaRegistry(),
            )

    assert "could not find" in result.answer.lower()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `source .venv/bin/activate && cd /Users/michaelmulkey/Documents/Repositories/rag && python -m pytest tests/test_agent/test_graph.py -v`
Expected: FAIL

- [ ] **Step 3: Implement graph.py**

```python
# src/agent/graph.py
from langgraph.graph import StateGraph, END

from src.agent.state import AgentState, QueryType
from src.agent.classifier import classify_query
from src.agent.evaluator import evaluate_context
from src.agent.synthesizer import synthesize_answer
from src.agent.strategies.lookup import retrieve_lookup
from src.agent.strategies.sweep import retrieve_sweep
from src.agent.strategies.analytical import retrieve_analytical
from src.agent.strategies.cross_reference import retrieve_cross_reference
from src.db.schema_registry import SchemaRegistry
from src.generation.rag_chain import RAGResponse
from src.retrieval.vector_store import VectorStore


def create_agent_graph(vector_store: VectorStore, schema_registry: SchemaRegistry):
    graph = StateGraph(AgentState)

    # Node: classify
    graph.add_node("classify", classify_query)

    # Node: retrieve (routes to correct strategy)
    async def retrieve(state: AgentState) -> dict:
        query_type = state.get("query_type", QueryType.LOOKUP)
        if query_type == QueryType.LOOKUP:
            return retrieve_lookup(state, vector_store=vector_store)
        elif query_type == QueryType.SWEEP:
            return retrieve_sweep(state, vector_store=vector_store)
        elif query_type == QueryType.ANALYTICAL:
            return await retrieve_analytical(state, schema_registry=schema_registry)
        elif query_type == QueryType.CROSS_REFERENCE:
            return await retrieve_cross_reference(state, vector_store=vector_store, schema_registry=schema_registry)
        elif query_type == QueryType.TEMPORAL:
            # Temporal uses lookup with date-biased scoring (same as lookup for now)
            return retrieve_lookup(state, vector_store=vector_store)
        return retrieve_lookup(state, vector_store=vector_store)

    graph.add_node("retrieve", retrieve)

    # Node: evaluate
    graph.add_node("evaluate", evaluate_context)

    # Node: synthesize
    graph.add_node("synthesize", synthesize_answer)

    # Edges
    graph.set_entry_point("classify")
    graph.add_edge("classify", "retrieve")
    graph.add_edge("retrieve", "evaluate")

    # Conditional: evaluate -> synthesize or re-retrieve
    def should_reretrieval(state: AgentState) -> str:
        if state.get("needs_reretrieval", False):
            return "retrieve"
        return "synthesize"

    graph.add_conditional_edges("evaluate", should_reretrieval, {"retrieve": "retrieve", "synthesize": "synthesize"})
    graph.add_edge("synthesize", END)

    return graph.compile()


async def run_agent(
    question: str,
    user_groups: list[str],
    vector_store: VectorStore,
    schema_registry: SchemaRegistry,
) -> RAGResponse:
    graph = create_agent_graph(vector_store=vector_store, schema_registry=schema_registry)

    initial_state = AgentState(
        question=question,
        user_groups=user_groups,
        query_type=None,
        sub_tasks=[],
        retrieved_chunks=[],
        sql_results=[],
        retrieval_attempts=0,
        needs_reretrieval=False,
        answer="",
        citations=[],
        warnings=[],
    )

    result = await graph.ainvoke(initial_state)

    return RAGResponse(
        answer=result.get("answer", "I could not find any relevant information."),
        citations=result.get("citations", []),
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `source .venv/bin/activate && cd /Users/michaelmulkey/Documents/Repositories/rag && python -m pytest tests/test_agent/test_graph.py -v`
Expected: 3 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/agent/graph.py tests/test_agent/test_graph.py
git commit -m "feat: LangGraph agent with classify->retrieve->evaluate->synthesize flow"
```

---

## Task 11: Wire Agent into API & Update rag_chain

**Files:**
- Modify: `rag/src/generation/rag_chain.py`
- Modify: `rag/src/api/routes_query.py`
- Modify: `rag/src/api/routes_ingest.py`
- Modify: `rag/tests/test_api/test_routes_query.py`

- [ ] **Step 1: Add agent_query to rag_chain.py**

Add this function to the end of `src/generation/rag_chain.py` (keep the existing `rag_query` for backward compat):

```python
# Add to end of src/generation/rag_chain.py

async def agent_query(
    question: str,
    user_groups: list[str],
    vector_store,
    schema_registry,
) -> RAGResponse:
    from src.agent.graph import run_agent
    return await run_agent(
        question=question,
        user_groups=user_groups,
        vector_store=vector_store,
        schema_registry=schema_registry,
    )
```

- [ ] **Step 2: Add schema_registry singleton to routes_ingest.py**

Add to `src/api/routes_ingest.py` after the existing singletons:

```python
# Add after get_metadata_store()
from src.db.schema_registry import SchemaRegistry

_schema_registry = None

def get_schema_registry():
    global _schema_registry
    if _schema_registry is None:
        _schema_registry = SchemaRegistry()
    return _schema_registry
```

- [ ] **Step 3: Update routes_query.py to use agent_query**

Replace the contents of `src/api/routes_query.py`:

```python
# src/api/routes_query.py
from fastapi import APIRouter, Depends
from src.api.models import CitationResponse, QueryRequest, QueryResponse
from src.api.routes_ingest import get_vector_store, get_schema_registry
from src.auth.dependencies import require_auth
from src.auth.models import UserContext
from src.generation.rag_chain import agent_query

router = APIRouter(prefix="/api/v1", tags=["query"])


@router.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest, user: UserContext = Depends(require_auth)):
    result = await agent_query(
        question=request.question,
        user_groups=user.groups,
        vector_store=get_vector_store(),
        schema_registry=get_schema_registry(),
    )
    return QueryResponse(
        answer=result.answer,
        citations=[
            CitationResponse(
                doc_id=c.doc_id, filename=c.filename, doc_type=c.doc_type,
                chunk_index=c.chunk_index, page=c.page, snippet=c.snippet,
                relevance=c.relevance,
            )
            for c in result.citations
        ],
    )
```

- [ ] **Step 4: Update test_routes_query.py to mock agent_query**

Replace `src.api.routes_query.rag_query` mock targets with `src.api.routes_query.agent_query` in `tests/test_api/test_routes_query.py`:

```python
# tests/test_api/test_routes_query.py
import pytest
from unittest.mock import patch, AsyncMock
from fastapi.testclient import TestClient
from src.main import create_app
from src.auth.jwt import create_token
from src.generation.rag_chain import RAGResponse
from src.retrieval.models import Citation


@pytest.fixture
def auth_headers():
    token = create_token(username="mike", groups=["finance"])
    return {"Authorization": f"Bearer {token}", "X-API-Key": "test-key-1"}


@pytest.fixture
def client():
    return TestClient(create_app())


def test_query_returns_answer_with_citations(client, auth_headers):
    mock_response = RAGResponse(
        answer="Expenses over $500 need approval [1].",
        citations=[Citation(doc_id="doc-1", filename="policy.pdf", doc_type="pdf", chunk_index=0, page=12, snippet="All expenses over $500...", relevance=0.95)],
    )
    with patch("src.api.routes_query.agent_query", new_callable=AsyncMock, return_value=mock_response):
        resp = client.post("/api/v1/query", json={"question": "What is the expense policy?"}, headers=auth_headers)

    assert resp.status_code == 200
    data = resp.json()
    assert len(data["citations"]) == 1
    assert data["citations"][0]["filename"] == "policy.pdf"


def test_query_requires_auth(client):
    resp = client.post("/api/v1/query", json={"question": "test"})
    assert resp.status_code in (401, 403)


def test_query_empty_question(client, auth_headers):
    with patch("src.api.routes_query.agent_query", new_callable=AsyncMock, return_value=RAGResponse(answer="No info found.", citations=[])):
        resp = client.post("/api/v1/query", json={"question": ""}, headers=auth_headers)
    assert resp.status_code == 200
```

- [ ] **Step 5: Run ALL tests**

Run: `source .venv/bin/activate && cd /Users/michaelmulkey/Documents/Repositories/rag && python -m pytest tests/ -v --tb=short`
Expected: All tests pass (52 existing + ~24 new agent tests)

- [ ] **Step 6: Commit**

```bash
git add src/generation/rag_chain.py src/api/routes_query.py src/api/routes_ingest.py tests/test_api/test_routes_query.py
git commit -m "feat: wire agentic orchestrator into query API endpoint"
```

---

## Task 12: Add Database Registry Config Setting

**Files:**
- Modify: `rag/src/config.py`
- Modify: `rag/.env.example`

- [ ] **Step 1: Add database registry settings to config.py**

Add to the Settings class in `src/config.py`:

```python
    # Database Registry (for text-to-SQL)
    registered_databases: str = ""  # comma-separated list of "name=url" pairs
```

Add a property:

```python
    @property
    def database_registry(self) -> dict[str, str]:
        if not self.registered_databases:
            return {}
        pairs = [p.strip() for p in self.registered_databases.split(",") if "=" in p]
        return {k.strip(): v.strip() for p in pairs for k, v in [p.split("=", 1)]}
```

- [ ] **Step 2: Add to .env.example**

```bash
# Database Registry (for text-to-SQL, comma-separated name=url pairs)
REGISTERED_DATABASES=finance_db=sqlite+aiosqlite:///./data/finance.db
```

- [ ] **Step 3: Run existing tests to confirm no regressions**

Run: `source .venv/bin/activate && cd /Users/michaelmulkey/Documents/Repositories/rag && python -m pytest tests/ -v --tb=short`
Expected: All tests pass

- [ ] **Step 4: Commit**

```bash
git add src/config.py .env.example
git commit -m "feat: database registry config for text-to-SQL connections"
```

---

## Task 13: Final Integration Test

**Files:**
- No new files — validate everything works together

- [ ] **Step 1: Run the complete test suite**

Run: `source .venv/bin/activate && cd /Users/michaelmulkey/Documents/Repositories/rag && python -m pytest tests/ -v --tb=short`
Expected: All tests pass

- [ ] **Step 2: Verify route still works**

Run: `source .venv/bin/activate && cd /Users/michaelmulkey/Documents/Repositories/rag && python -c "from src.main import create_app; app = create_app(); print('Routes:'); [print(f'  {r.methods} {r.path}') for r in app.routes if hasattr(r, 'methods')]"`

- [ ] **Step 3: Verify agent graph compiles**

Run: `source .venv/bin/activate && cd /Users/michaelmulkey/Documents/Repositories/rag && python -c "from src.agent.graph import create_agent_graph; from src.db.schema_registry import SchemaRegistry; from unittest.mock import MagicMock; g = create_agent_graph(MagicMock(), SchemaRegistry()); print('Agent graph nodes:', list(g.get_graph().nodes.keys()))"`

- [ ] **Step 4: Commit**

```bash
git add -A
git commit -m "chore: Phase 2 Agentic Orchestrator complete — all tests passing"
```
