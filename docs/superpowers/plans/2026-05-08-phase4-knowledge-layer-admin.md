# Phase 4: Knowledge Layer & Admin UI — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add LLM-powered auto-categorization to the ingestion pipeline, a category proposal/approval workflow, a knowledge registry with routing rules, and a web-based admin UI for managing documents, categories, ACLs, and audit logs.

**Architecture:** The ingestion pipeline gains a classification step that uses Gemma 4 to assign documents to existing categories or propose new ones. Proposed categories are stored in a new DB table and require admin approval. A Knowledge Registry wraps the existing SchemaRegistry + metadata store + category catalog into a unified interface with routing rules. The admin UI is a FastAPI-served set of HTML pages (Jinja2 templates + HTMX for interactivity) — lightweight, no separate frontend build step.

**Tech Stack:** Existing FastAPI + SQLAlchemy, Jinja2 templates, HTMX, existing Gemma 4 via vLLM

---

## File Structure

```
src/
├── knowledge/                          # NEW — knowledge layer
│   ├── __init__.py
│   ├── categorizer.py                  # LLM-based document auto-categorization
│   ├── registry.py                     # Knowledge Registry (sources, routing rules, relationships)
│   └── models.py                       # Category, RoutingRule, CategoryProposal DB models
│
├── admin/                              # NEW — admin web UI
│   ├── __init__.py
│   ├── routes.py                       # FastAPI routes for admin pages
│   ├── templates/                      # Jinja2 HTML templates
│   │   ├── base.html                   # Layout with nav
│   │   ├── dashboard.html              # Overview stats
│   │   ├── documents.html              # Document list + management
│   │   ├── categories.html             # Category list + routing rules
│   │   ├── proposals.html              # Pending category proposals (approve/reject)
│   │   └── audit.html                  # Audit log viewer
│   └── static/                         # Minimal CSS
│       └── style.css
│
├── db/
│   └── models.py                       # MODIFY — add Category, CategoryProposal tables
│
├── ingestion/
│   └── pipeline.py                     # MODIFY — add auto-categorization step
│
├── main.py                             # MODIFY — mount admin routes + serve static/templates
│
tests/
├── test_knowledge/
│   ├── __init__.py
│   ├── test_categorizer.py
│   └── test_registry.py
├── test_admin/
│   ├── __init__.py
│   └── test_routes.py
```

---

## Task 1: Category & Proposal DB Models

**Files:**
- Modify: `rag/src/db/models.py`
- Create: `rag/tests/test_db/test_category_models.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_db/test_category_models.py
import pytest
import pytest_asyncio
from src.db.metadata import MetadataStore


@pytest_asyncio.fixture
async def store():
    s = MetadataStore("sqlite+aiosqlite:///:memory:")
    await s.init()
    yield s


@pytest.mark.asyncio
async def test_add_and_get_category(store):
    from src.db.models import Category
    await store.add_category(name="finance_policies", description="Finance policy documents", acl_groups=["finance", "executives"], routing_keywords=["expense", "budget", "revenue"])
    cat = await store.get_category("finance_policies")
    assert cat is not None
    assert cat.name == "finance_policies"
    assert cat.acl_groups == ["finance", "executives"]
    assert "expense" in cat.routing_keywords


@pytest.mark.asyncio
async def test_list_categories(store):
    await store.add_category(name="finance", description="Finance", acl_groups=["finance"], routing_keywords=[])
    await store.add_category(name="it", description="IT", acl_groups=["it_support"], routing_keywords=[])
    cats = await store.list_categories()
    assert len(cats) == 2


@pytest.mark.asyncio
async def test_add_and_list_proposals(store):
    await store.add_proposal(proposed_name="legal_compliance", proposed_description="Legal docs", proposed_acl_groups=["legal"], proposed_keywords=["contract", "compliance"], proposed_by="system")
    proposals = await store.list_proposals(status="pending")
    assert len(proposals) == 1
    assert proposals[0].proposed_name == "legal_compliance"


@pytest.mark.asyncio
async def test_approve_proposal(store):
    await store.add_proposal(proposed_name="legal", proposed_description="Legal", proposed_acl_groups=["legal"], proposed_keywords=["legal"], proposed_by="system")
    proposals = await store.list_proposals(status="pending")
    proposal_id = proposals[0].id
    await store.approve_proposal(proposal_id, approved_by="admin")

    # Proposal status updated
    proposals = await store.list_proposals(status="pending")
    assert len(proposals) == 0

    # Category created
    cat = await store.get_category("legal")
    assert cat is not None


@pytest.mark.asyncio
async def test_reject_proposal(store):
    await store.add_proposal(proposed_name="spam", proposed_description="Bad", proposed_acl_groups=[], proposed_keywords=[], proposed_by="system")
    proposals = await store.list_proposals(status="pending")
    await store.reject_proposal(proposals[0].id, rejected_by="admin")
    assert len(await store.list_proposals(status="pending")) == 0
    assert len(await store.list_proposals(status="rejected")) == 1
```

- [ ] **Step 2: Add Category and CategoryProposal to db/models.py**

Add after the existing DocumentRecord class:

```python
class Category(Base):
    __tablename__ = "categories"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String, unique=True, nullable=False)
    description: Mapped[str] = mapped_column(String, default="")
    acl_groups: Mapped[list] = mapped_column(JSON, default=list)
    routing_keywords: Mapped[list] = mapped_column(JSON, default=list)
    doc_count: Mapped[int] = mapped_column(default=0)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))


class CategoryProposal(Base):
    __tablename__ = "category_proposals"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    proposed_name: Mapped[str] = mapped_column(String, nullable=False)
    proposed_description: Mapped[str] = mapped_column(String, default="")
    proposed_acl_groups: Mapped[list] = mapped_column(JSON, default=list)
    proposed_keywords: Mapped[list] = mapped_column(JSON, default=list)
    proposed_by: Mapped[str] = mapped_column(String, default="")
    status: Mapped[str] = mapped_column(String, default="pending")  # pending, approved, rejected
    reviewed_by: Mapped[str] = mapped_column(String, default="")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
```

- [ ] **Step 3: Add category/proposal methods to MetadataStore**

Add to `src/db/metadata.py`:

```python
from src.db.models import Base, DocumentRecord, Category, CategoryProposal

async def add_category(self, name, description, acl_groups, routing_keywords):
    record = Category(name=name, description=description, acl_groups=acl_groups, routing_keywords=routing_keywords)
    async with self.session_factory() as session:
        session.add(record)
        await session.commit()
        await session.refresh(record)
    return record

async def get_category(self, name):
    async with self.session_factory() as session:
        result = await session.execute(select(Category).where(Category.name == name))
        return result.scalar_one_or_none()

async def list_categories(self):
    async with self.session_factory() as session:
        result = await session.execute(select(Category))
        return list(result.scalars().all())

async def add_proposal(self, proposed_name, proposed_description, proposed_acl_groups, proposed_keywords, proposed_by):
    record = CategoryProposal(proposed_name=proposed_name, proposed_description=proposed_description, proposed_acl_groups=proposed_acl_groups, proposed_keywords=proposed_keywords, proposed_by=proposed_by)
    async with self.session_factory() as session:
        session.add(record)
        await session.commit()
        await session.refresh(record)
    return record

async def list_proposals(self, status="pending"):
    async with self.session_factory() as session:
        result = await session.execute(select(CategoryProposal).where(CategoryProposal.status == status))
        return list(result.scalars().all())

async def approve_proposal(self, proposal_id, approved_by):
    async with self.session_factory() as session:
        proposal = await session.get(CategoryProposal, proposal_id)
        if proposal:
            proposal.status = "approved"
            proposal.reviewed_by = approved_by
            # Create the category
            cat = Category(name=proposal.proposed_name, description=proposal.proposed_description, acl_groups=proposal.proposed_acl_groups, routing_keywords=proposal.proposed_keywords)
            session.add(cat)
            await session.commit()

async def reject_proposal(self, proposal_id, rejected_by):
    async with self.session_factory() as session:
        proposal = await session.get(CategoryProposal, proposal_id)
        if proposal:
            proposal.status = "rejected"
            proposal.reviewed_by = rejected_by
            await session.commit()
```

- [ ] **Step 4: Run tests**

Run: `source .venv/bin/activate && cd /Users/michaelmulkey/Documents/Repositories/rag && python -m pytest tests/test_db/test_category_models.py -v`
Expected: 6 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/db/models.py src/db/metadata.py tests/test_db/test_category_models.py
git commit -m "feat: Category and CategoryProposal DB models with approval workflow"
```

---

## Task 2: Auto-Categorizer

**Files:**
- Create: `rag/src/knowledge/__init__.py`
- Create: `rag/src/knowledge/categorizer.py`
- Create: `rag/tests/test_knowledge/__init__.py`
- Create: `rag/tests/test_knowledge/test_categorizer.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_knowledge/__init__.py
```

```python
# tests/test_knowledge/test_categorizer.py
import pytest
from unittest.mock import patch, AsyncMock, MagicMock
from src.knowledge.categorizer import categorize_document, CategorizationResult


@pytest.fixture
def mock_store():
    store = AsyncMock()
    cat1 = MagicMock()
    cat1.name = "finance_policies"
    cat1.description = "Finance policy documents"
    cat1.routing_keywords = ["expense", "budget", "revenue"]
    cat2 = MagicMock()
    cat2.name = "it_runbooks"
    cat2.description = "IT operational procedures"
    cat2.routing_keywords = ["server", "deploy", "outage"]
    store.list_categories.return_value = [cat1, cat2]
    return store


def test_categorize_matches_existing(mock_store):
    with patch("src.knowledge.categorizer.generate", return_value='{"category": "finance_policies", "confidence": 0.95, "is_new": false}'):
        result = categorize_document(
            filename="expense_policy_v3.pdf",
            doc_type="pdf",
            text_preview="All expenses over $500 require manager approval...",
            metadata_store=mock_store,
        )
    assert isinstance(result, CategorizationResult)
    assert result.category == "finance_policies"
    assert result.is_new is False
    assert result.confidence >= 0.9


def test_categorize_proposes_new(mock_store):
    with patch("src.knowledge.categorizer.generate", return_value='{"category": "legal_compliance", "confidence": 0.85, "is_new": true, "description": "Legal and compliance documents", "suggested_acl_groups": ["legal"], "suggested_keywords": ["contract", "compliance"]}'):
        result = categorize_document(
            filename="vendor_contract_2026.pdf",
            doc_type="pdf",
            text_preview="This vendor agreement is between...",
            metadata_store=mock_store,
        )
    assert result.category == "legal_compliance"
    assert result.is_new is True
    assert result.description == "Legal and compliance documents"
    assert "legal" in result.suggested_acl_groups


def test_categorize_fallback_on_bad_json(mock_store):
    with patch("src.knowledge.categorizer.generate", return_value="I can't categorize this"):
        result = categorize_document(
            filename="random.txt",
            doc_type="txt",
            text_preview="some random content",
            metadata_store=mock_store,
        )
    assert result.category == "uncategorized"
    assert result.is_new is False
```

- [ ] **Step 2: Implement categorizer.py**

```python
# src/knowledge/__init__.py
```

```python
# src/knowledge/categorizer.py
import json
from dataclasses import dataclass, field

from src.generation.llm_client import generate


CATEGORIZATION_PROMPT = """You are a document categorizer. Given a document's filename, type, and text preview, classify it into one of the existing categories OR propose a new category.

Existing categories:
{categories}

Rules:
- If the document fits an existing category, use it
- If no existing category fits, propose a new one
- Respond with ONLY valid JSON

For existing category match:
{{"category": "<name>", "confidence": 0.0-1.0, "is_new": false}}

For new category proposal:
{{"category": "<proposed_name>", "confidence": 0.0-1.0, "is_new": true, "description": "<what this category covers>", "suggested_acl_groups": ["<group1>"], "suggested_keywords": ["<keyword1>", "<keyword2>"]}}"""


@dataclass
class CategorizationResult:
    category: str
    confidence: float
    is_new: bool
    description: str = ""
    suggested_acl_groups: list[str] = field(default_factory=list)
    suggested_keywords: list[str] = field(default_factory=list)


def categorize_document(
    filename: str,
    doc_type: str,
    text_preview: str,
    metadata_store,
) -> CategorizationResult:
    import asyncio
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop and loop.is_running():
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor() as pool:
            categories = pool.submit(asyncio.run, metadata_store.list_categories()).result()
    else:
        categories = asyncio.run(metadata_store.list_categories())

    cat_descriptions = []
    for cat in categories:
        keywords = ", ".join(cat.routing_keywords) if cat.routing_keywords else "none"
        cat_descriptions.append(f"- {cat.name}: {cat.description} (keywords: {keywords})")

    categories_text = "\n".join(cat_descriptions) if cat_descriptions else "No existing categories."

    response = generate(
        system_prompt=CATEGORIZATION_PROMPT.format(categories=categories_text),
        user_prompt=f"Filename: {filename}\nType: {doc_type}\nPreview: {text_preview[:500]}",
        temperature=0.0,
        max_tokens=256,
    )

    try:
        parsed = json.loads(response)
        return CategorizationResult(
            category=parsed["category"],
            confidence=parsed.get("confidence", 0.5),
            is_new=parsed.get("is_new", False),
            description=parsed.get("description", ""),
            suggested_acl_groups=parsed.get("suggested_acl_groups", []),
            suggested_keywords=parsed.get("suggested_keywords", []),
        )
    except (json.JSONDecodeError, KeyError):
        return CategorizationResult(category="uncategorized", confidence=0.0, is_new=False)
```

- [ ] **Step 3: Run tests**

Run: `source .venv/bin/activate && cd /Users/michaelmulkey/Documents/Repositories/rag && python -m pytest tests/test_knowledge/test_categorizer.py -v`
Expected: 3 tests PASS

- [ ] **Step 4: Commit**

```bash
git add src/knowledge/ tests/test_knowledge/
git commit -m "feat: LLM-based document auto-categorizer with new category proposals"
```

---

## Task 3: Integrate Auto-Categorization into Ingestion Pipeline

**Files:**
- Modify: `rag/src/ingestion/pipeline.py`
- Modify: `rag/tests/test_ingestion/test_pipeline.py`

- [ ] **Step 1: Write failing test**

```python
# Add to tests/test_ingestion/test_pipeline.py

@pytest.mark.asyncio
async def test_ingest_auto_categorizes(mock_deps):
    vector_store, metadata_store, mock_embed = mock_deps
    metadata_store.list_categories = AsyncMock(return_value=[])
    metadata_store.add_proposal = AsyncMock()

    from src.knowledge.categorizer import CategorizationResult
    mock_result = CategorizationResult(category="finance_policies", confidence=0.9, is_new=False)

    with patch("src.ingestion.pipeline.embed_texts", mock_embed):
        with patch("src.ingestion.pipeline.categorize_document", return_value=mock_result):
            result = await ingest_document(
                file_path=FIXTURES / "sample.pdf",
                acl_groups=["finance"],
                uploaded_by="mike",
                vector_store=vector_store,
                metadata_store=metadata_store,
                auto_categorize=True,
            )
    assert result.doc_type == "pdf"
    # Category should be set from auto-categorization
    call_args = metadata_store.add_document.call_args
    assert call_args.kwargs.get("category", "") == "finance_policies" or call_args[1].get("category", "") == "finance_policies"
```

- [ ] **Step 2: Modify pipeline.py to add auto-categorization**

Update `ingest_document` in `src/ingestion/pipeline.py` to accept `auto_categorize=False` parameter. When True:
1. Call `categorize_document()` with the parsed text preview
2. If the result matches an existing category, use it
3. If it's a new category proposal, create a proposal in the metadata store and use "uncategorized" as the category for now

```python
# Updated ingest_document function
async def ingest_document(
    file_path,
    acl_groups,
    uploaded_by,
    vector_store,
    metadata_store,
    category="",
    chunk_size=512,
    chunk_overlap=50,
    auto_categorize=False,
):
    doc_id = str(uuid.uuid4())
    parsed = parse_document(file_path)

    # Auto-categorize if no category provided and auto_categorize is True
    if not category and auto_categorize:
        from src.knowledge.categorizer import categorize_document
        cat_result = categorize_document(
            filename=parsed.filename,
            doc_type=parsed.doc_type,
            text_preview=parsed.text[:500],
            metadata_store=metadata_store,
        )
        if cat_result.is_new:
            await metadata_store.add_proposal(
                proposed_name=cat_result.category,
                proposed_description=cat_result.description,
                proposed_acl_groups=cat_result.suggested_acl_groups,
                proposed_keywords=cat_result.suggested_keywords,
                proposed_by="auto-categorizer",
            )
            category = "uncategorized"
        else:
            category = cat_result.category

    chunks = chunk_text(parsed.text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    texts = [c.text for c in chunks]
    metadatas = [
        ChunkMetadata(
            doc_id=doc_id, filename=parsed.filename, doc_type=parsed.doc_type,
            chunk_index=c.index, start_char=c.start_char, acl_groups=acl_groups, category=category,
        )
        for c in chunks
    ]
    vectors = embed_texts(texts) if texts else []
    if vectors:
        vector_store.upsert(texts=texts, vectors=vectors, metadatas=metadatas)
    await metadata_store.add_document(
        doc_id=doc_id, filename=parsed.filename, doc_type=parsed.doc_type,
        acl_groups=acl_groups, chunk_count=len(chunks), uploaded_by=uploaded_by, category=category,
    )
    return IngestResult(doc_id=doc_id, filename=parsed.filename, doc_type=parsed.doc_type, chunk_count=len(chunks))
```

- [ ] **Step 3: Run tests**

Run: `source .venv/bin/activate && cd /Users/michaelmulkey/Documents/Repositories/rag && python -m pytest tests/test_ingestion/test_pipeline.py -v`
Expected: All pipeline tests PASS (existing + new)

- [ ] **Step 4: Commit**

```bash
git add src/ingestion/pipeline.py tests/test_ingestion/test_pipeline.py
git commit -m "feat: auto-categorization in ingestion pipeline with proposal creation"
```

---

## Task 4: Knowledge Registry

**Files:**
- Create: `rag/src/knowledge/registry.py`
- Create: `rag/tests/test_knowledge/test_registry.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_knowledge/test_registry.py
import pytest
from unittest.mock import AsyncMock, MagicMock
from src.knowledge.registry import KnowledgeRegistry


@pytest.fixture
def mock_deps():
    metadata_store = AsyncMock()
    from src.db.schema_registry import SchemaRegistry
    schema_registry = SchemaRegistry()
    return metadata_store, schema_registry


def test_get_routing_suggestion(mock_deps):
    metadata_store, schema_registry = mock_deps
    cat = MagicMock()
    cat.name = "finance_policies"
    cat.routing_keywords = ["expense", "budget", "revenue"]
    cat.acl_groups = ["finance"]
    metadata_store.list_categories.return_value = [cat]

    registry = KnowledgeRegistry(metadata_store=metadata_store, schema_registry=schema_registry)
    suggestions = registry.suggest_sources("What is our expense policy?", user_groups=["finance"])
    assert "finance_policies" in suggestions


def test_routing_no_match(mock_deps):
    metadata_store, schema_registry = mock_deps
    cat = MagicMock()
    cat.name = "finance_policies"
    cat.routing_keywords = ["expense", "budget"]
    cat.acl_groups = ["finance"]
    metadata_store.list_categories.return_value = [cat]

    registry = KnowledgeRegistry(metadata_store=metadata_store, schema_registry=schema_registry)
    suggestions = registry.suggest_sources("Tell me about server maintenance", user_groups=["finance"])
    # No keyword match — should return empty or all accessible
    assert isinstance(suggestions, list)


def test_routing_respects_acl(mock_deps):
    metadata_store, schema_registry = mock_deps
    cat_finance = MagicMock()
    cat_finance.name = "finance_policies"
    cat_finance.routing_keywords = ["expense"]
    cat_finance.acl_groups = ["finance"]
    cat_it = MagicMock()
    cat_it.name = "it_runbooks"
    cat_it.routing_keywords = ["server"]
    cat_it.acl_groups = ["it_support"]
    metadata_store.list_categories.return_value = [cat_finance, cat_it]

    registry = KnowledgeRegistry(metadata_store=metadata_store, schema_registry=schema_registry)
    # IT user asks about expenses — finance category not accessible
    suggestions = registry.suggest_sources("expense policy", user_groups=["it_support"])
    assert "finance_policies" not in suggestions


def test_get_all_sources(mock_deps):
    metadata_store, schema_registry = mock_deps
    cat = MagicMock()
    cat.name = "finance"
    cat.description = "Finance docs"
    cat.acl_groups = ["finance"]
    cat.routing_keywords = ["budget"]
    cat.doc_count = 10
    metadata_store.list_categories.return_value = [cat]

    from src.db.schema_registry import TableSchema, ColumnSchema
    schema_registry.register(TableSchema(database="fin_db", table="budget", columns=[], description="Budget", acl_groups=["finance"]))

    registry = KnowledgeRegistry(metadata_store=metadata_store, schema_registry=schema_registry)
    sources = registry.get_all_sources(user_groups=["finance"])
    assert len(sources) == 2  # 1 category + 1 database
```

- [ ] **Step 2: Implement registry.py**

```python
# src/knowledge/registry.py
import asyncio

from src.db.metadata import MetadataStore
from src.db.schema_registry import SchemaRegistry


class KnowledgeRegistry:
    def __init__(self, metadata_store: MetadataStore, schema_registry: SchemaRegistry):
        self._metadata_store = metadata_store
        self._schema_registry = schema_registry

    def _get_categories(self):
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None
        if loop and loop.is_running():
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as pool:
                return pool.submit(asyncio.run, self._metadata_store.list_categories()).result()
        return asyncio.run(self._metadata_store.list_categories())

    def suggest_sources(self, query: str, user_groups: list[str]) -> list[str]:
        categories = self._get_categories()
        query_lower = query.lower()
        suggestions = []

        for cat in categories:
            # ACL check
            if not any(g in cat.acl_groups for g in user_groups):
                continue
            # Keyword match
            for keyword in cat.routing_keywords:
                if keyword.lower() in query_lower:
                    suggestions.append(cat.name)
                    break

        return suggestions

    def get_all_sources(self, user_groups: list[str]) -> list[dict]:
        sources = []

        # Document categories
        categories = self._get_categories()
        for cat in categories:
            if any(g in cat.acl_groups for g in user_groups):
                sources.append({
                    "name": cat.name,
                    "type": "document_category",
                    "description": cat.description,
                    "routing_keywords": cat.routing_keywords,
                })

        # Database schemas
        db_schemas = self._schema_registry.list_for_user(user_groups)
        for schema in db_schemas:
            sources.append({
                "name": f"{schema.database}.{schema.table}",
                "type": "database",
                "description": schema.description,
            })

        return sources
```

- [ ] **Step 3: Run tests**

Run: `source .venv/bin/activate && cd /Users/michaelmulkey/Documents/Repositories/rag && python -m pytest tests/test_knowledge/test_registry.py -v`
Expected: 4 tests PASS

- [ ] **Step 4: Commit**

```bash
git add src/knowledge/registry.py tests/test_knowledge/test_registry.py
git commit -m "feat: Knowledge Registry with routing suggestions and source listing"
```

---

## Task 5: Admin UI — Templates & Static Assets

**Files:**
- Create: `rag/src/admin/__init__.py`
- Create: `rag/src/admin/templates/base.html`
- Create: `rag/src/admin/templates/dashboard.html`
- Create: `rag/src/admin/templates/documents.html`
- Create: `rag/src/admin/templates/categories.html`
- Create: `rag/src/admin/templates/proposals.html`
- Create: `rag/src/admin/templates/audit.html`
- Create: `rag/src/admin/static/style.css`

- [ ] **Step 1: Install Jinja2 (already included with FastAPI standard)**

Verify: `source .venv/bin/activate && python -c "import jinja2; print(jinja2.__version__)"`

- [ ] **Step 2: Create base template**

```html
<!-- src/admin/templates/base.html -->
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{% block title %}RAG Admin{% endblock %}</title>
    <link rel="stylesheet" href="/admin/static/style.css">
    <script src="https://unpkg.com/htmx.org@1.9.12"></script>
</head>
<body>
    <nav>
        <div class="nav-brand">RAG Knowledge Service</div>
        <div class="nav-links">
            <a href="/admin/">Dashboard</a>
            <a href="/admin/documents">Documents</a>
            <a href="/admin/categories">Categories</a>
            <a href="/admin/proposals">Proposals</a>
            <a href="/admin/audit">Audit Log</a>
        </div>
    </nav>
    <main>
        {% block content %}{% endblock %}
    </main>
</body>
</html>
```

- [ ] **Step 3: Create dashboard template**

```html
<!-- src/admin/templates/dashboard.html -->
{% extends "base.html" %}
{% block title %}Dashboard — RAG Admin{% endblock %}
{% block content %}
<h1>Dashboard</h1>
<div class="stats-grid">
    <div class="stat-card">
        <div class="stat-value">{{ doc_count }}</div>
        <div class="stat-label">Documents</div>
    </div>
    <div class="stat-card">
        <div class="stat-value">{{ category_count }}</div>
        <div class="stat-label">Categories</div>
    </div>
    <div class="stat-card">
        <div class="stat-value">{{ pending_proposals }}</div>
        <div class="stat-label">Pending Proposals</div>
    </div>
</div>
{% endblock %}
```

- [ ] **Step 4: Create documents template**

```html
<!-- src/admin/templates/documents.html -->
{% extends "base.html" %}
{% block title %}Documents — RAG Admin{% endblock %}
{% block content %}
<h1>Documents</h1>
<table>
    <thead>
        <tr><th>Filename</th><th>Type</th><th>Category</th><th>ACL Groups</th><th>Chunks</th><th>Uploaded By</th><th>Actions</th></tr>
    </thead>
    <tbody>
        {% for doc in documents %}
        <tr>
            <td>{{ doc.filename }}</td>
            <td>{{ doc.doc_type }}</td>
            <td>{{ doc.category or "uncategorized" }}</td>
            <td>{{ doc.acl_groups | join(", ") }}</td>
            <td>{{ doc.chunk_count }}</td>
            <td>{{ doc.uploaded_by }}</td>
            <td><button hx-delete="/admin/api/documents/{{ doc.doc_id }}" hx-confirm="Delete this document?" hx-target="closest tr" hx-swap="outerHTML">Delete</button></td>
        </tr>
        {% endfor %}
    </tbody>
</table>
{% endblock %}
```

- [ ] **Step 5: Create categories template**

```html
<!-- src/admin/templates/categories.html -->
{% extends "base.html" %}
{% block title %}Categories — RAG Admin{% endblock %}
{% block content %}
<h1>Categories</h1>
<table>
    <thead>
        <tr><th>Name</th><th>Description</th><th>ACL Groups</th><th>Routing Keywords</th></tr>
    </thead>
    <tbody>
        {% for cat in categories %}
        <tr>
            <td>{{ cat.name }}</td>
            <td>{{ cat.description }}</td>
            <td>{{ cat.acl_groups | join(", ") }}</td>
            <td>{{ cat.routing_keywords | join(", ") }}</td>
        </tr>
        {% endfor %}
    </tbody>
</table>
{% endblock %}
```

- [ ] **Step 6: Create proposals template**

```html
<!-- src/admin/templates/proposals.html -->
{% extends "base.html" %}
{% block title %}Proposals — RAG Admin{% endblock %}
{% block content %}
<h1>Pending Category Proposals</h1>
{% if proposals %}
<table>
    <thead>
        <tr><th>Proposed Name</th><th>Description</th><th>ACL Groups</th><th>Keywords</th><th>Proposed By</th><th>Actions</th></tr>
    </thead>
    <tbody>
        {% for p in proposals %}
        <tr id="proposal-{{ p.id }}">
            <td>{{ p.proposed_name }}</td>
            <td>{{ p.proposed_description }}</td>
            <td>{{ p.proposed_acl_groups | join(", ") }}</td>
            <td>{{ p.proposed_keywords | join(", ") }}</td>
            <td>{{ p.proposed_by }}</td>
            <td>
                <button hx-post="/admin/api/proposals/{{ p.id }}/approve" hx-target="#proposal-{{ p.id }}" hx-swap="outerHTML">Approve</button>
                <button hx-post="/admin/api/proposals/{{ p.id }}/reject" hx-target="#proposal-{{ p.id }}" hx-swap="outerHTML">Reject</button>
            </td>
        </tr>
        {% endfor %}
    </tbody>
</table>
{% else %}
<p>No pending proposals.</p>
{% endif %}
{% endblock %}
```

- [ ] **Step 7: Create audit template**

```html
<!-- src/admin/templates/audit.html -->
{% extends "base.html" %}
{% block title %}Audit Log — RAG Admin{% endblock %}
{% block content %}
<h1>Audit Log</h1>
<table>
    <thead>
        <tr><th>Timestamp</th><th>Agent</th><th>User</th><th>Tool</th><th>Query</th></tr>
    </thead>
    <tbody>
        {% for entry in entries %}
        <tr>
            <td>{{ entry.timestamp }}</td>
            <td>{{ entry.agent_id }}</td>
            <td>{{ entry.username }}</td>
            <td>{{ entry.tool }}</td>
            <td>{{ entry.query }}</td>
        </tr>
        {% endfor %}
    </tbody>
</table>
{% endblock %}
```

- [ ] **Step 8: Create CSS**

```css
/* src/admin/static/style.css */
* { margin: 0; padding: 0; box-sizing: border-box; }
body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; background: #f5f5f5; color: #1f2937; }
nav { background: #1f2937; color: white; padding: 1rem 2rem; display: flex; justify-content: space-between; align-items: center; }
.nav-brand { font-size: 1.2rem; font-weight: bold; }
.nav-links a { color: #93c5fd; text-decoration: none; margin-left: 1.5rem; }
.nav-links a:hover { color: white; }
main { max-width: 1200px; margin: 2rem auto; padding: 0 2rem; }
h1 { margin-bottom: 1.5rem; color: #1f2937; }
.stats-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 1rem; }
.stat-card { background: white; border-radius: 8px; padding: 1.5rem; box-shadow: 0 1px 3px rgba(0,0,0,0.1); text-align: center; }
.stat-value { font-size: 2rem; font-weight: bold; color: #2563eb; }
.stat-label { color: #6b7280; margin-top: 0.5rem; }
table { width: 100%; border-collapse: collapse; background: white; border-radius: 8px; overflow: hidden; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }
th { background: #f9fafb; padding: 0.75rem 1rem; text-align: left; font-weight: 600; border-bottom: 2px solid #e5e7eb; }
td { padding: 0.75rem 1rem; border-bottom: 1px solid #f3f4f6; }
tr:hover { background: #f9fafb; }
button { padding: 0.4rem 0.8rem; border: none; border-radius: 4px; cursor: pointer; font-size: 0.85rem; }
button { background: #2563eb; color: white; }
button:hover { background: #1d4ed8; }
button + button { margin-left: 0.5rem; background: #dc2626; }
button + button:hover { background: #b91c1c; }
p { color: #6b7280; }
```

- [ ] **Step 9: Commit**

```bash
git add src/admin/
git commit -m "feat: admin UI templates and static assets (dashboard, docs, categories, proposals, audit)"
```

---

## Task 6: Admin API Routes

**Files:**
- Create: `rag/src/admin/routes.py`
- Create: `rag/tests/test_admin/__init__.py`
- Create: `rag/tests/test_admin/test_routes.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_admin/__init__.py
```

```python
# tests/test_admin/test_routes.py
import pytest
from unittest.mock import patch, AsyncMock, MagicMock
from fastapi.testclient import TestClient


@pytest.fixture
def client():
    from src.main import create_app
    return TestClient(create_app())


def test_dashboard_loads(client):
    with patch("src.admin.routes.get_metadata_store") as mock_get:
        store = AsyncMock()
        store.list_documents.return_value = [MagicMock() for _ in range(5)]
        store.list_categories.return_value = [MagicMock() for _ in range(3)]
        store.list_proposals.return_value = [MagicMock(), MagicMock()]
        mock_get.return_value = store
        resp = client.get("/admin/")
    assert resp.status_code == 200
    assert "Dashboard" in resp.text


def test_documents_page(client):
    with patch("src.admin.routes.get_metadata_store") as mock_get:
        store = AsyncMock()
        doc = MagicMock()
        doc.doc_id = "d1"
        doc.filename = "test.pdf"
        doc.doc_type = "pdf"
        doc.category = "finance"
        doc.acl_groups = ["finance"]
        doc.chunk_count = 5
        doc.uploaded_by = "mike"
        store.list_documents.return_value = [doc]
        mock_get.return_value = store
        resp = client.get("/admin/documents")
    assert resp.status_code == 200
    assert "test.pdf" in resp.text


def test_proposals_page(client):
    with patch("src.admin.routes.get_metadata_store") as mock_get:
        store = AsyncMock()
        proposal = MagicMock()
        proposal.id = 1
        proposal.proposed_name = "legal"
        proposal.proposed_description = "Legal docs"
        proposal.proposed_acl_groups = ["legal"]
        proposal.proposed_keywords = ["contract"]
        proposal.proposed_by = "system"
        store.list_proposals.return_value = [proposal]
        mock_get.return_value = store
        resp = client.get("/admin/proposals")
    assert resp.status_code == 200
    assert "legal" in resp.text


def test_approve_proposal(client):
    with patch("src.admin.routes.get_metadata_store") as mock_get:
        store = AsyncMock()
        store.approve_proposal = AsyncMock()
        mock_get.return_value = store
        resp = client.post("/admin/api/proposals/1/approve")
    assert resp.status_code == 200
    store.approve_proposal.assert_called_once()


def test_reject_proposal(client):
    with patch("src.admin.routes.get_metadata_store") as mock_get:
        store = AsyncMock()
        store.reject_proposal = AsyncMock()
        mock_get.return_value = store
        resp = client.post("/admin/api/proposals/1/reject")
    assert resp.status_code == 200
    store.reject_proposal.assert_called_once()
```

- [ ] **Step 2: Implement admin routes**

```python
# src/admin/routes.py
import json
from pathlib import Path

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from src.api.routes_ingest import get_metadata_store

router = APIRouter(prefix="/admin", tags=["admin"])

TEMPLATE_DIR = Path(__file__).parent / "templates"
templates = Jinja2Templates(directory=str(TEMPLATE_DIR))


@router.get("/", response_class=HTMLResponse)
async def dashboard(request: Request):
    store = get_metadata_store()
    docs = await store.list_documents()
    categories = await store.list_categories()
    proposals = await store.list_proposals(status="pending")
    return templates.TemplateResponse("dashboard.html", {
        "request": request,
        "doc_count": len(docs),
        "category_count": len(categories),
        "pending_proposals": len(proposals),
    })


@router.get("/documents", response_class=HTMLResponse)
async def documents_page(request: Request):
    store = get_metadata_store()
    docs = await store.list_documents()
    return templates.TemplateResponse("documents.html", {"request": request, "documents": docs})


@router.get("/categories", response_class=HTMLResponse)
async def categories_page(request: Request):
    store = get_metadata_store()
    categories = await store.list_categories()
    return templates.TemplateResponse("categories.html", {"request": request, "categories": categories})


@router.get("/proposals", response_class=HTMLResponse)
async def proposals_page(request: Request):
    store = get_metadata_store()
    proposals = await store.list_proposals(status="pending")
    return templates.TemplateResponse("proposals.html", {"request": request, "proposals": proposals})


@router.get("/audit", response_class=HTMLResponse)
async def audit_page(request: Request):
    from src.config import settings
    entries = []
    log_path = Path(settings.audit_log_path)
    if log_path.exists():
        lines = log_path.read_text().strip().split("\n")
        for line in reversed(lines[-100:]):  # last 100 entries, newest first
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return templates.TemplateResponse("audit.html", {"request": request, "entries": entries})


@router.post("/api/proposals/{proposal_id}/approve")
async def approve_proposal(proposal_id: int):
    store = get_metadata_store()
    await store.approve_proposal(proposal_id, approved_by="admin")
    return HTMLResponse("<tr><td colspan='6'>Approved</td></tr>")


@router.post("/api/proposals/{proposal_id}/reject")
async def reject_proposal(proposal_id: int):
    store = get_metadata_store()
    await store.reject_proposal(proposal_id, rejected_by="admin")
    return HTMLResponse("<tr><td colspan='6'>Rejected</td></tr>")


@router.delete("/api/documents/{doc_id}")
async def delete_document(doc_id: str):
    store = get_metadata_store()
    await store.delete_document(doc_id)
    return HTMLResponse("")
```

- [ ] **Step 3: Mount admin routes and static files in main.py**

Update `src/main.py` to include admin router and static file serving:

```python
from contextlib import asynccontextmanager
from pathlib import Path
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from src.api.routes_auth import router as auth_router
from src.api.routes_ingest import router as ingest_router, get_metadata_store
from src.api.routes_query import router as query_router
from src.admin.routes import router as admin_router

ADMIN_STATIC = Path(__file__).parent / "admin" / "static"

@asynccontextmanager
async def lifespan(app: FastAPI):
    store = get_metadata_store()
    await store.init()
    yield

def create_app() -> FastAPI:
    app = FastAPI(title="RAG Knowledge Service", description="Agentic RAG system with document-level access control", version="0.1.0", lifespan=lifespan)
    app.include_router(auth_router)
    app.include_router(ingest_router)
    app.include_router(query_router)
    app.include_router(admin_router)
    if ADMIN_STATIC.exists():
        app.mount("/admin/static", StaticFiles(directory=str(ADMIN_STATIC)), name="admin-static")
    return app

app = create_app()
```

- [ ] **Step 4: Run tests**

Run: `source .venv/bin/activate && cd /Users/michaelmulkey/Documents/Repositories/rag && python -m pytest tests/test_admin/ -v`
Expected: 5 tests PASS

- [ ] **Step 5: Run ALL tests**

Run: `source .venv/bin/activate && cd /Users/michaelmulkey/Documents/Repositories/rag && python -m pytest tests/ -v --tb=short`
Expected: All tests pass

- [ ] **Step 6: Commit**

```bash
git add src/admin/routes.py src/main.py tests/test_admin/
git commit -m "feat: admin UI routes with dashboard, documents, categories, proposals, and audit pages"
```

---

## Task 7: Final Integration Validation

**Files:** No new files

- [ ] **Step 1: Run complete test suite**

Run: `source .venv/bin/activate && cd /Users/michaelmulkey/Documents/Repositories/rag && python -m pytest tests/ -v --tb=short`
Expected: All tests pass

- [ ] **Step 2: Verify all routes**

Run: `source .venv/bin/activate && cd /Users/michaelmulkey/Documents/Repositories/rag && python -c "from src.main import create_app; app = create_app(); print('Routes:'); [print(f'  {r.methods} {r.path}') for r in app.routes if hasattr(r, 'methods')]"`

Expected routes include:
- API: `/api/v1/auth/token`, `/api/v1/ingest`, `/api/v1/documents`, `/api/v1/query`
- Admin: `/admin/`, `/admin/documents`, `/admin/categories`, `/admin/proposals`, `/admin/audit`
- Admin API: `/admin/api/proposals/{id}/approve`, `/admin/api/proposals/{id}/reject`, `/admin/api/documents/{doc_id}`

- [ ] **Step 3: Commit**

```bash
git add -A
git commit -m "chore: Phase 4 Knowledge Layer and Admin UI complete — all tests passing"
```
