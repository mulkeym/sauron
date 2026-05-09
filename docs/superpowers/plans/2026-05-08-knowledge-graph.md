# Knowledge Graph — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extract entities and relationships from every document at ingestion, store them in SQLite, expose via MCP tool, and provide an admin UI for browsing the knowledge graph.

**Architecture:** Three new SQLite tables (entities, entity_mentions, relationships) added to existing models. An LLM-based extractor runs on each chunk during ingestion. MetadataStore gets graph CRUD methods. A new MCP tool `tool_search_knowledge_graph` enables graph traversal. Admin UI gets a `/admin/knowledge-graph` page.

**Tech Stack:** Existing SQLAlchemy + SQLite, existing Gemma 4 31B via vLLM, existing FastMCP, existing Jinja2/HTMX admin UI

---

## File Structure

```
src/
├── db/
│   ├── models.py              # MODIFY — add Entity, EntityMention, Relationship models
│   └── metadata.py            # MODIFY — add graph CRUD methods
│
├── knowledge/
│   └── extractor.py           # NEW — LLM-based entity/relationship extraction
│
├── ingestion/
│   └── pipeline.py            # MODIFY — add extraction step after chunking
│
├── mcp/
│   ├── tools_low.py           # MODIFY — add search_knowledge_graph function
│   └── server.py              # MODIFY — register tool_search_knowledge_graph
│
├── admin/
│   ├── routes.py              # MODIFY — add knowledge graph page + API
│   └── templates/
│       ├── base.html          # MODIFY — add nav link
│       └── knowledge_graph.html  # NEW — entity list + detail view
│
tests/
├── test_knowledge/
│   └── test_extractor.py      # NEW
├── test_db/
│   └── test_graph_models.py   # NEW
├── test_mcp/
│   └── test_tools_graph.py    # NEW
```

---

## Task 1: Entity & Relationship DB Models

**Files:**
- Modify: `rag/src/db/models.py`
- Create: `rag/tests/test_db/test_graph_models.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_db/test_graph_models.py
import pytest
import pytest_asyncio
from src.db.metadata import MetadataStore


@pytest_asyncio.fixture
async def store():
    s = MetadataStore("sqlite+aiosqlite:///:memory:")
    await s.init()
    yield s


@pytest.mark.asyncio
async def test_add_and_search_entity(store):
    entity_id = await store.add_entity(name="Mike", entity_type="person", first_seen_doc_id="doc-1")
    assert entity_id is not None
    results = await store.search_entities("Mike")
    assert len(results) >= 1
    assert results[0].name == "Mike"


@pytest.mark.asyncio
async def test_add_entity_deduplicates(store):
    id1 = await store.add_entity(name="Policy 4.2", entity_type="policy", first_seen_doc_id="doc-1")
    id2 = await store.add_entity(name="Policy 4.2", entity_type="policy", first_seen_doc_id="doc-2")
    assert id1 == id2  # same entity, not duplicated


@pytest.mark.asyncio
async def test_add_mention(store):
    entity_id = await store.add_entity(name="Mike", entity_type="person", first_seen_doc_id="doc-1")
    await store.add_mention(entity_id=entity_id, doc_id="doc-1", chunk_index=0, context_snippet="Mike asked about...")
    await store.add_mention(entity_id=entity_id, doc_id="doc-2", chunk_index=3, context_snippet="Mike reviewed...")
    details = await store.get_entity_details(entity_id)
    assert len(details["mentions"]) == 2


@pytest.mark.asyncio
async def test_add_relationship(store):
    e1 = await store.add_entity(name="Policy 4.2", entity_type="policy", first_seen_doc_id="doc-1")
    e2 = await store.add_entity(name="expense reporting", entity_type="project", first_seen_doc_id="doc-1")
    await store.add_relationship(source_entity_id=e1, target_entity_id=e2, relationship_type="governs", doc_id="doc-1", context_snippet="Policy 4.2 governs expense reporting")
    details = await store.get_entity_details(e1)
    assert len(details["relationships"]) == 1
    assert details["relationships"][0]["related_entity"] == "expense reporting"
    assert details["relationships"][0]["relationship_type"] == "governs"


@pytest.mark.asyncio
async def test_list_entities_by_type(store):
    await store.add_entity(name="Mike", entity_type="person", first_seen_doc_id="doc-1")
    await store.add_entity(name="Policy 4.2", entity_type="policy", first_seen_doc_id="doc-1")
    await store.add_entity(name="Sarah", entity_type="person", first_seen_doc_id="doc-2")
    people = await store.list_entities(entity_type="person")
    assert len(people) == 2
    all_entities = await store.list_entities()
    assert len(all_entities) == 3


@pytest.mark.asyncio
async def test_search_entities_partial_match(store):
    await store.add_entity(name="Policy 4.2", entity_type="policy", first_seen_doc_id="doc-1")
    await store.add_entity(name="Policy 5.1", entity_type="policy", first_seen_doc_id="doc-1")
    results = await store.search_entities("policy")
    assert len(results) == 2


@pytest.mark.asyncio
async def test_delete_entities_for_doc(store):
    e1 = await store.add_entity(name="Mike", entity_type="person", first_seen_doc_id="doc-1")
    await store.add_mention(entity_id=e1, doc_id="doc-1", chunk_index=0, context_snippet="Mike")
    await store.delete_entities_for_doc("doc-1")
    details = await store.get_entity_details(e1)
    assert len(details["mentions"]) == 0
```

- [ ] **Step 2: Add models to db/models.py**

Add after CategoryProposal class:

```python
class Entity(Base):
    __tablename__ = "entities"
    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String, nullable=False)
    entity_type: Mapped[str] = mapped_column(String, nullable=False)
    first_seen_doc_id: Mapped[str] = mapped_column(String, default="")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    __table_args__ = (UniqueConstraint("name", "entity_type", name="uq_entity_name_type"),)


class EntityMention(Base):
    __tablename__ = "entity_mentions"
    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    entity_id: Mapped[int] = mapped_column(nullable=False)
    doc_id: Mapped[str] = mapped_column(String, nullable=False)
    chunk_index: Mapped[int] = mapped_column(default=0)
    context_snippet: Mapped[str] = mapped_column(String, default="")


class Relationship(Base):
    __tablename__ = "relationships"
    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    source_entity_id: Mapped[int] = mapped_column(nullable=False)
    target_entity_id: Mapped[int] = mapped_column(nullable=False)
    relationship_type: Mapped[str] = mapped_column(String, nullable=False)
    doc_id: Mapped[str] = mapped_column(String, default="")
    context_snippet: Mapped[str] = mapped_column(String, default="")
```

Add `from sqlalchemy import UniqueConstraint` to the imports in models.py.

- [ ] **Step 3: Add graph CRUD methods to MetadataStore**

Add these imports to metadata.py:
```python
from src.db.models import Base, DocumentRecord, Category, CategoryProposal, Entity, EntityMention, Relationship
```

Add these methods to the MetadataStore class:

```python
async def add_entity(self, name, entity_type, first_seen_doc_id):
    async with self.session_factory() as session:
        # Check for existing entity (dedup by name + type)
        result = await session.execute(
            select(Entity).where(Entity.name == name, Entity.entity_type == entity_type)
        )
        existing = result.scalar_one_or_none()
        if existing:
            return existing.id
        entity = Entity(name=name, entity_type=entity_type, first_seen_doc_id=first_seen_doc_id)
        session.add(entity)
        await session.commit()
        await session.refresh(entity)
        return entity.id

async def add_mention(self, entity_id, doc_id, chunk_index, context_snippet):
    async with self.session_factory() as session:
        mention = EntityMention(entity_id=entity_id, doc_id=doc_id, chunk_index=chunk_index, context_snippet=context_snippet[:200])
        session.add(mention)
        await session.commit()

async def add_relationship(self, source_entity_id, target_entity_id, relationship_type, doc_id, context_snippet=""):
    async with self.session_factory() as session:
        rel = Relationship(source_entity_id=source_entity_id, target_entity_id=target_entity_id, relationship_type=relationship_type, doc_id=doc_id, context_snippet=context_snippet[:200])
        session.add(rel)
        await session.commit()

async def search_entities(self, query, entity_type=None):
    async with self.session_factory() as session:
        stmt = select(Entity).where(Entity.name.ilike(f"%{query}%"))
        if entity_type:
            stmt = stmt.where(Entity.entity_type == entity_type)
        result = await session.execute(stmt)
        return list(result.scalars().all())

async def get_entity_details(self, entity_id):
    async with self.session_factory() as session:
        entity = await session.get(Entity, entity_id)
        if not entity:
            return {"entity": None, "mentions": [], "relationships": []}

        # Get mentions
        mentions_result = await session.execute(
            select(EntityMention).where(EntityMention.entity_id == entity_id)
        )
        mentions = [{"doc_id": m.doc_id, "chunk_index": m.chunk_index, "context_snippet": m.context_snippet} for m in mentions_result.scalars().all()]

        # Get relationships (as source or target)
        rels_as_source = await session.execute(
            select(Relationship).where(Relationship.source_entity_id == entity_id)
        )
        rels_as_target = await session.execute(
            select(Relationship).where(Relationship.target_entity_id == entity_id)
        )

        relationships = []
        for r in rels_as_source.scalars().all():
            target = await session.get(Entity, r.target_entity_id)
            relationships.append({"related_entity": target.name if target else "unknown", "entity_type": target.entity_type if target else "", "relationship_type": r.relationship_type, "direction": "outgoing", "doc_id": r.doc_id, "context": r.context_snippet})
        for r in rels_as_target.scalars().all():
            source = await session.get(Entity, r.source_entity_id)
            relationships.append({"related_entity": source.name if source else "unknown", "entity_type": source.entity_type if source else "", "relationship_type": r.relationship_type, "direction": "incoming", "doc_id": r.doc_id, "context": r.context_snippet})

        return {"entity": {"id": entity.id, "name": entity.name, "type": entity.entity_type}, "mentions": mentions, "relationships": relationships}

async def list_entities(self, entity_type=None, limit=100):
    async with self.session_factory() as session:
        stmt = select(Entity)
        if entity_type:
            stmt = stmt.where(Entity.entity_type == entity_type)
        stmt = stmt.limit(limit)
        result = await session.execute(stmt)
        return list(result.scalars().all())

async def delete_entities_for_doc(self, doc_id):
    async with self.session_factory() as session:
        # Delete mentions for this doc
        await session.execute(delete(EntityMention).where(EntityMention.doc_id == doc_id))
        # Delete relationships for this doc
        await session.execute(delete(Relationship).where(Relationship.doc_id == doc_id))
        await session.commit()
```

- [ ] **Step 4: Run tests**

Run: `source .venv/bin/activate && cd /Users/michaelmulkey/Documents/Repositories/rag && python -m pytest tests/test_db/test_graph_models.py -v`
Expected: 7 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/db/models.py src/db/metadata.py tests/test_db/test_graph_models.py
git commit -m "feat: Entity, EntityMention, Relationship DB models with graph CRUD"
```

---

## Task 2: Entity Extractor

**Files:**
- Create: `rag/src/knowledge/extractor.py`
- Create: `rag/tests/test_knowledge/test_extractor.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_knowledge/test_extractor.py
import pytest
from unittest.mock import patch
from src.knowledge.extractor import extract_entities, ExtractionResult


def test_extract_entities_from_text():
    mock_response = '{"entities": [{"name": "Mike", "type": "person"}, {"name": "Policy 4.2", "type": "policy"}], "relationships": [{"source": "Policy 4.2", "target": "expense reporting", "type": "governs"}], "sections": []}'
    with patch("src.knowledge.extractor.generate", return_value=mock_response):
        result = extract_entities("Policy 4.2 governs expense reporting. Mike reviewed it.")
    assert isinstance(result, ExtractionResult)
    assert len(result.entities) == 2
    assert result.entities[0]["name"] == "Mike"
    assert len(result.relationships) == 1
    assert result.relationships[0]["type"] == "governs"


def test_extract_entities_with_sections():
    mock_response = '{"entities": [{"name": "TOEE 26", "type": "project"}], "relationships": [], "sections": [{"name": "Section 4.2: Expense Reporting", "parent": null}]}'
    with patch("src.knowledge.extractor.generate", return_value=mock_response):
        result = extract_entities("Section 4.2: Expense Reporting...")
    assert len(result.sections) == 1
    assert "4.2" in result.sections[0]["name"]


def test_extract_entities_bad_json_returns_empty():
    with patch("src.knowledge.extractor.generate", return_value="I can't extract anything"):
        result = extract_entities("random text")
    assert result.entities == []
    assert result.relationships == []


def test_extract_entities_empty_text():
    result = extract_entities("")
    assert result.entities == []
```

- [ ] **Step 2: Implement extractor.py**

```python
# src/knowledge/extractor.py
from dataclasses import dataclass, field

from src.generation.llm_client import generate, parse_json_response

EXTRACTION_PROMPT = """Extract entities, relationships, and document sections from the following text.

Entity types: person, organization, policy, project, date, system, location, document_section
Relationship types: references, governs, authored_by, allocated_to, requires, part_of, related_to

Respond with ONLY valid JSON:
{
  "entities": [{"name": "...", "type": "..."}],
  "relationships": [{"source": "...", "target": "...", "type": "..."}],
  "sections": [{"name": "...", "parent": null}]
}

If no entities are found, return empty arrays."""


@dataclass
class ExtractionResult:
    entities: list[dict] = field(default_factory=list)
    relationships: list[dict] = field(default_factory=list)
    sections: list[dict] = field(default_factory=list)


def extract_entities(text: str) -> ExtractionResult:
    if not text.strip():
        return ExtractionResult()

    response = generate(
        system_prompt=EXTRACTION_PROMPT,
        user_prompt=text[:3000],
        temperature=0.0,
        max_tokens=1024,
    )

    try:
        parsed = parse_json_response(response)
        return ExtractionResult(
            entities=parsed.get("entities", []),
            relationships=parsed.get("relationships", []),
            sections=parsed.get("sections", []),
        )
    except Exception:
        return ExtractionResult()
```

- [ ] **Step 3: Run tests**

Run: `source .venv/bin/activate && cd /Users/michaelmulkey/Documents/Repositories/rag && python -m pytest tests/test_knowledge/test_extractor.py -v`
Expected: 4 tests PASS

- [ ] **Step 4: Commit**

```bash
git add src/knowledge/extractor.py tests/test_knowledge/test_extractor.py
git commit -m "feat: LLM-based entity and relationship extractor"
```

---

## Task 3: Integrate Extraction into Ingestion Pipeline

**Files:**
- Modify: `rag/src/ingestion/pipeline.py`
- Modify: `rag/tests/test_ingestion/test_pipeline.py`

- [ ] **Step 1: Add test**

Add to `tests/test_ingestion/test_pipeline.py`:

```python
@pytest.mark.asyncio
async def test_ingest_extracts_entities(mock_deps):
    vector_store, metadata_store, mock_embed = mock_deps
    metadata_store.add_entity = AsyncMock(return_value=1)
    metadata_store.add_mention = AsyncMock()
    metadata_store.add_relationship = AsyncMock()

    from src.knowledge.extractor import ExtractionResult
    mock_extraction = ExtractionResult(
        entities=[{"name": "Mike", "type": "person"}, {"name": "Policy 4.2", "type": "policy"}],
        relationships=[{"source": "Policy 4.2", "target": "expense reporting", "type": "governs"}],
        sections=[],
    )

    with patch("src.ingestion.pipeline.embed_texts", mock_embed):
        with patch("src.ingestion.pipeline.extract_entities", return_value=mock_extraction):
            result = await ingest_document(
                file_path=FIXTURES / "sample.pdf", acl_groups=["finance"],
                uploaded_by="mike", vector_store=vector_store, metadata_store=metadata_store,
            )
    assert result.chunk_count > 0
    metadata_store.add_entity.assert_called()
    metadata_store.add_mention.assert_called()
```

- [ ] **Step 2: Modify pipeline.py**

Add import at top:
```python
from src.knowledge.extractor import extract_entities
```

Add extraction step after embedding, before the return statement:

```python
    # Extract entities and relationships from each chunk
    for chunk in chunks:
        extraction = extract_entities(chunk.text)
        entity_id_map = {}
        for ent in extraction.entities:
            eid = await metadata_store.add_entity(
                name=ent["name"], entity_type=ent["type"], first_seen_doc_id=doc_id,
            )
            entity_id_map[ent["name"]] = eid
            await metadata_store.add_mention(
                entity_id=eid, doc_id=doc_id, chunk_index=chunk.index,
                context_snippet=chunk.text[:200],
            )
        for rel in extraction.relationships:
            source_id = entity_id_map.get(rel["source"])
            target_name = rel["target"]
            if source_id is None:
                continue
            target_id = entity_id_map.get(target_name)
            if target_id is None:
                target_id = await metadata_store.add_entity(
                    name=target_name, entity_type="unknown", first_seen_doc_id=doc_id,
                )
            await metadata_store.add_relationship(
                source_entity_id=source_id, target_entity_id=target_id,
                relationship_type=rel.get("type", "related_to"), doc_id=doc_id,
                context_snippet=rel.get("context", chunk.text[:100]),
            )
        for section in extraction.sections:
            await metadata_store.add_entity(
                name=section["name"], entity_type="document_section", first_seen_doc_id=doc_id,
            )
```

- [ ] **Step 3: Run tests**

Run: `source .venv/bin/activate && cd /Users/michaelmulkey/Documents/Repositories/rag && python -m pytest tests/test_ingestion/test_pipeline.py -v`
Expected: All pipeline tests PASS

- [ ] **Step 4: Commit**

```bash
git add src/ingestion/pipeline.py tests/test_ingestion/test_pipeline.py
git commit -m "feat: entity extraction integrated into ingestion pipeline"
```

---

## Task 4: MCP Tool — search_knowledge_graph

**Files:**
- Modify: `rag/src/mcp/tools_low.py`
- Modify: `rag/src/mcp/server.py`
- Create: `rag/tests/test_mcp/test_tools_graph.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_mcp/test_tools_graph.py
import pytest
from unittest.mock import AsyncMock, MagicMock
from src.mcp.tools_low import search_knowledge_graph


@pytest.mark.asyncio
async def test_search_knowledge_graph_finds_entity():
    mock_store = AsyncMock()
    entity = MagicMock()
    entity.id = 1
    entity.name = "TOEE 26"
    entity.entity_type = "project"
    mock_store.search_entities.return_value = [entity]
    mock_store.get_entity_details.return_value = {
        "entity": {"id": 1, "name": "TOEE 26", "type": "project"},
        "mentions": [{"doc_id": "doc-1", "chunk_index": 0, "context_snippet": "TOEE 26 is..."}],
        "relationships": [{"related_entity": "Counter-UxS", "entity_type": "project", "relationship_type": "requires", "direction": "outgoing", "doc_id": "doc-1", "context": ""}],
    }
    result = await search_knowledge_graph(query="TOEE 26", metadata_store=mock_store)
    assert result["entity"]["name"] == "TOEE 26"
    assert len(result["relationships"]) == 1


@pytest.mark.asyncio
async def test_search_knowledge_graph_not_found():
    mock_store = AsyncMock()
    mock_store.search_entities.return_value = []
    result = await search_knowledge_graph(query="nonexistent", metadata_store=mock_store)
    assert "error" in result or result["entity"] is None
```

- [ ] **Step 2: Add search_knowledge_graph to tools_low.py**

Add at the end of `src/mcp/tools_low.py`:

```python
async def search_knowledge_graph(
    query: str,
    metadata_store,
    entity_type: str | None = None,
) -> dict:
    entities = await metadata_store.search_entities(query, entity_type=entity_type)
    if not entities:
        return {"entity": None, "error": f"No entity found matching '{query}'", "suggestions": "Try a different name or use tool_list_documents to find documents first."}

    # Return details for the best match (first result)
    best = entities[0]
    details = await metadata_store.get_entity_details(best.id)

    # Also list other matches if multiple
    other_matches = [{"name": e.name, "type": e.entity_type} for e in entities[1:5]]

    return {
        "entity": details["entity"],
        "mentions_in": details["mentions"],
        "relationships": details["relationships"],
        "other_matches": other_matches,
    }
```

- [ ] **Step 3: Register in MCP server**

Add to `src/mcp/server.py` imports:
```python
from src.mcp.tools_low import search_documents, query_database, lookup_document, search_meetings, list_sources, list_documents_in_category, search_knowledge_graph
```

Add tool registration before `tool_get_result`:
```python
    @mcp.tool()
    async def tool_search_knowledge_graph(query: str, entity_type: str = "") -> dict:
        """Search the knowledge graph for an entity (person, policy, project, organization, etc.) and find all related entities, relationships, and source documents. Use this to understand how concepts are connected across documents. Example: 'TOEE 26' returns related projects, people, and organizations."""
        return await search_knowledge_graph(
            query=query, metadata_store=metadata_store,
            entity_type=entity_type or None,
        )
```

- [ ] **Step 4: Run tests**

Run: `source .venv/bin/activate && cd /Users/michaelmulkey/Documents/Repositories/rag && python -m pytest tests/test_mcp/test_tools_graph.py -v`
Expected: 2 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/mcp/tools_low.py src/mcp/server.py tests/test_mcp/test_tools_graph.py
git commit -m "feat: knowledge graph search MCP tool"
```

---

## Task 5: Admin UI — Knowledge Graph Page

**Files:**
- Create: `rag/src/admin/templates/knowledge_graph.html`
- Modify: `rag/src/admin/templates/base.html`
- Modify: `rag/src/admin/routes.py`

- [ ] **Step 1: Add nav link to base.html**

Add after the Playground link:
```html
            <a href="/admin/knowledge-graph">Knowledge Graph</a>
```

- [ ] **Step 2: Create knowledge_graph.html template**

```html
{% extends "base.html" %}
{% block title %}Knowledge Graph - RAG Admin{% endblock %}
{% block content %}
<h1>Knowledge Graph</h1>

<div class="settings-section">
    <div class="form-row">
        <div class="form-group">
            <label for="kg-search">Search Entities</label>
            <div class="input-with-button">
                <input type="text" id="kg-search" name="query" placeholder="e.g., TOEE 26, Mike, Policy 4.2">
                <button type="button" hx-post="/admin/api/knowledge-graph/search" hx-include="#kg-search, #kg-type-filter" hx-target="#kg-results" hx-swap="innerHTML">Search</button>
            </div>
        </div>
        <div class="form-group">
            <label for="kg-type-filter">Filter by type</label>
            <select id="kg-type-filter" name="entity_type">
                <option value="">All types</option>
                <option value="person">Person</option>
                <option value="organization">Organization</option>
                <option value="policy">Policy</option>
                <option value="project">Project</option>
                <option value="system">System</option>
                <option value="date">Date</option>
                <option value="location">Location</option>
                <option value="document_section">Document Section</option>
            </select>
        </div>
    </div>
</div>

<div id="kg-results">
{% if entities %}
<table>
    <thead><tr><th>Name</th><th>Type</th><th>First Seen In</th><th>Actions</th></tr></thead>
    <tbody>
        {% for e in entities %}
        <tr>
            <td>{{ e.name }}</td>
            <td>{{ e.entity_type }}</td>
            <td>{{ e.first_seen_doc_id }}</td>
            <td><button hx-post="/admin/api/knowledge-graph/details" hx-vals='{"entity_id": "{{ e.id }}"}' hx-target="#kg-results" hx-swap="innerHTML">View</button></td>
        </tr>
        {% endfor %}
    </tbody>
</table>
{% else %}
<p>No entities found. Upload documents to populate the knowledge graph.</p>
{% endif %}
</div>
{% endblock %}
```

- [ ] **Step 3: Add routes to admin/routes.py**

Add these routes to `src/admin/routes.py`:

```python
@router.get("/knowledge-graph", response_class=HTMLResponse)
async def knowledge_graph_page(request: Request):
    store = get_metadata_store()
    entities = await store.list_entities(limit=50)
    return templates.TemplateResponse(request, "knowledge_graph.html", {"entities": entities})


@router.post("/api/knowledge-graph/search")
async def search_knowledge_graph_api(query: str = Form(""), entity_type: str = Form("")):
    store = get_metadata_store()
    entities = await store.search_entities(query, entity_type=entity_type or None)
    rows = ""
    for e in entities:
        rows += f'<tr><td>{e.name}</td><td>{e.entity_type}</td><td>{e.first_seen_doc_id}</td><td><button hx-post="/admin/api/knowledge-graph/details" hx-vals=\'{{"entity_id": "{e.id}"}}\' hx-target="#kg-results" hx-swap="innerHTML">View</button></td></tr>'
    if not rows:
        return HTMLResponse(f'<p>No entities found matching "{query}".</p>')
    return HTMLResponse(f'<table><thead><tr><th>Name</th><th>Type</th><th>First Seen In</th><th>Actions</th></tr></thead><tbody>{rows}</tbody></table>')


@router.post("/api/knowledge-graph/details")
async def knowledge_graph_details(entity_id: int = Form(0)):
    store = get_metadata_store()
    details = await store.get_entity_details(entity_id)
    if not details["entity"]:
        return HTMLResponse("<p>Entity not found.</p>")

    entity = details["entity"]
    html = f'<div class="result-card"><h2>{entity["name"]} <span class="result-meta">({entity["type"]})</span></h2>'

    # Mentions
    html += f'<h3>Mentioned in ({len(details["mentions"])} documents)</h3>'
    if details["mentions"]:
        html += '<table><thead><tr><th>Document</th><th>Chunk</th><th>Context</th></tr></thead><tbody>'
        for m in details["mentions"]:
            html += f'<tr><td>{m["doc_id"][:12]}...</td><td>{m["chunk_index"]}</td><td>{m["context_snippet"][:100]}</td></tr>'
        html += '</tbody></table>'

    # Relationships
    html += f'<h3>Relationships ({len(details["relationships"])})</h3>'
    if details["relationships"]:
        html += '<table><thead><tr><th>Related Entity</th><th>Type</th><th>Relationship</th><th>Direction</th><th>Source Doc</th></tr></thead><tbody>'
        for r in details["relationships"]:
            html += f'<tr><td>{r["related_entity"]}</td><td>{r["entity_type"]}</td><td>{r["relationship_type"]}</td><td>{r["direction"]}</td><td>{r["doc_id"][:12]}...</td></tr>'
        html += '</tbody></table>'
    else:
        html += '<p>No relationships found.</p>'

    html += '<br><button hx-get="/admin/knowledge-graph" hx-target="#kg-results" hx-swap="innerHTML" hx-select="#kg-results">Back to list</button></div>'
    return HTMLResponse(html)
```

- [ ] **Step 4: Run all tests**

Run: `source .venv/bin/activate && cd /Users/michaelmulkey/Documents/Repositories/rag && python -m pytest tests/ -v --tb=short`
Expected: All tests pass

- [ ] **Step 5: Commit**

```bash
git add src/admin/templates/knowledge_graph.html src/admin/templates/base.html src/admin/routes.py
git commit -m "feat: admin UI knowledge graph page with entity search and detail view"
```

---

## Task 6: Final Integration Validation

**Files:** No new files

- [ ] **Step 1: Run complete test suite**

Run: `source .venv/bin/activate && cd /Users/michaelmulkey/Documents/Repositories/rag && python -m pytest tests/ -v --tb=short`
Expected: All tests pass

- [ ] **Step 2: Verify new routes**

Run: `source .venv/bin/activate && cd /Users/michaelmulkey/Documents/Repositories/rag && python -c "from src.main import create_app; app = create_app(); [print(f'  {r.methods} {r.path}') for r in app.routes if hasattr(r, 'methods') and 'knowledge' in r.path.lower()]"`

- [ ] **Step 3: Commit**

```bash
git add -A
git commit -m "chore: Knowledge Graph feature complete — all tests passing"
```
