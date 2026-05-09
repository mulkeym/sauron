# Knowledge Graph — Design Spec

**Date:** 2026-05-08
**Status:** Draft

---

## 1. Overview

Add a knowledge graph to the RAG system that extracts entities and relationships from every document at ingestion time, stores them in SQLite alongside existing tables, exposes them via MCP tools for the agent, and provides an admin UI for exploration.

### Goals

- Automatically extract entities (people, organizations, policies, projects, dates, systems) from every ingested document
- Map typed relationships between entities ("governs", "references", "authored by", "requires")
- Extract document section structure for hierarchical linking
- Enable graph-based queries via MCP tools ("what's connected to TOEE 26?")
- Provide an admin UI page to browse entities, relationships, and source documents

### Non-Goals

- External graph database (Neo4j) — using existing SQLite
- Fancy interactive graph visualization — table-based UI is sufficient
- Manual entity/relationship editing in the UI (can be added later)

---

## 2. DB Schema

Three new tables added to the existing SQLAlchemy models:

### entities

| Column | Type | Description |
|--------|------|-------------|
| id | Integer (PK) | Auto-increment |
| name | String | Normalized entity name (e.g., "Mike", "Policy 4.2", "TOEE 26") |
| entity_type | String | Type: person, organization, policy, project, date, system, location, document_section |
| first_seen_doc_id | String (FK) | Doc where this entity was first extracted |
| created_at | DateTime | When the entity was first seen |

Unique constraint on (name, entity_type) — same entity across documents merges into one row.

### entity_mentions

| Column | Type | Description |
|--------|------|-------------|
| id | Integer (PK) | Auto-increment |
| entity_id | Integer (FK) | References entities.id |
| doc_id | String (FK) | Which document this mention is in |
| chunk_index | Integer | Which chunk within the document |
| context_snippet | String | Short surrounding text for context (up to 200 chars) |

Tracks every occurrence of an entity across all documents.

### relationships

| Column | Type | Description |
|--------|------|-------------|
| id | Integer (PK) | Auto-increment |
| source_entity_id | Integer (FK) | References entities.id |
| target_entity_id | Integer (FK) | References entities.id |
| relationship_type | String | Type: references, governs, authored_by, allocated_to, requires, part_of, related_to |
| doc_id | String (FK) | Document where this relationship was found |
| context_snippet | String | Text evidence for the relationship |

---

## 3. Entity Extraction

### Extraction Process

At ingestion time, after parsing and chunking, the LLM processes each chunk to extract entities and relationships.

### LLM Prompt

The extraction prompt sends document text (up to 3000 chars per chunk) and requests structured JSON:

```json
{
  "entities": [
    {"name": "Mike", "type": "person"},
    {"name": "Policy 4.2", "type": "policy"},
    {"name": "TOEE 26", "type": "project"}
  ],
  "relationships": [
    {"source": "Policy 4.2", "target": "expense reporting", "type": "governs"},
    {"source": "TOEE 26", "target": "Counter-UxS", "type": "requires"}
  ],
  "sections": [
    {"name": "Section 4.2: Expense Reporting", "parent": null}
  ]
}
```

### Entity Types

- person
- organization
- policy
- project
- date
- system
- location
- document_section

### Relationship Types

- references — one document or entity references another
- governs — a policy governs a process or area
- authored_by — a document or section authored by a person
- allocated_to — budget or resources allocated to a department/project
- requires — a project or process requires a capability
- part_of — a section is part of a document, or an entity is part of an organization
- related_to — general association when no specific type fits

### Entity Deduplication

Entities are normalized by (lowercase name, entity_type). When extracting from a new document, existing entities are matched and reused rather than duplicated. New mentions are added to entity_mentions.

---

## 4. Integration with Ingestion Pipeline

The extraction step runs after parsing and chunking, before or in parallel with embedding:

```
Parse → Chunk → [Extract Entities] → Embed → Store
                      ↓
               entities table
               entity_mentions table
               relationships table
```

Extraction is automatic on every document upload. It uses the same LLM endpoint as the rest of the system.

### Performance

Each chunk requires one LLM call for extraction. For a document with 5 chunks, that's 5 additional LLM calls at ingestion time. This adds latency to uploads but runs only once per document.

---

## 5. MCP Tool

### tool_search_knowledge_graph

**Input:** entity name or query string
**Process:**
1. Search entities table for matching names (case-insensitive, partial match)
2. Find all relationships where the entity is source or target
3. For each related entity, include its type and the relationship type
4. Include source document references for each relationship

**Output:**
```json
{
  "entity": {"name": "TOEE 26", "type": "project"},
  "mentions_in": [
    {"doc_id": "...", "filename": "rfi.pdf", "count": 3}
  ],
  "relationships": [
    {
      "related_entity": "Counter-UxS",
      "entity_type": "project",
      "relationship": "requires",
      "direction": "outgoing",
      "source_doc": "rfi.pdf"
    },
    {
      "related_entity": "Office of Naval Research",
      "entity_type": "organization",
      "relationship": "authored_by",
      "direction": "incoming",
      "source_doc": "rfi.pdf"
    }
  ]
}
```

**Tool description:** "Search the knowledge graph for an entity and find all related entities, relationships, and source documents. Use this to understand how concepts, people, projects, and policies are connected across documents."

---

## 6. Admin UI

### /admin/knowledge-graph page

**Entity list view (default):**
- Table of all entities sorted by mention count (most referenced first)
- Columns: Name, Type, Mentions (count), First Seen In (filename)
- Filter by entity type (dropdown)
- Search box to filter by name

**Entity detail view (click an entity):**
- Entity name and type at top
- "Mentioned in" section: list of documents with mention count per doc
- "Relationships" section: table showing related entity, relationship type, direction, source document
- Each document name links to the document in the documents page

---

## 7. MetadataStore Extensions

Add these methods to MetadataStore:

- `add_entity(name, entity_type, first_seen_doc_id)` — upsert, returns entity id
- `add_mention(entity_id, doc_id, chunk_index, context_snippet)` — add a mention record
- `add_relationship(source_entity_id, target_entity_id, relationship_type, doc_id, context_snippet)` — add relationship
- `search_entities(query, entity_type=None)` — search by name, optional type filter
- `get_entity_details(entity_id)` — entity + all mentions + all relationships
- `list_entities(entity_type=None, limit=100)` — list entities sorted by mention count
- `delete_entities_for_doc(doc_id)` — remove all mentions and orphaned entities for a doc (used when re-indexing)
