# SAURON

**Structured Agentic Unified Retrieval Over Networks**

The all-seeing eye that finds everything in your documents.

SAURON is a self-hosted agentic RAG (Retrieval-Augmented Generation) system with document-level access control, multi-pass indexing, hybrid search, knowledge graph enrichment, and an admin UI. It integrates with any OpenAI-compatible LLM and embedding endpoint.

## Features

- **Hybrid Search** -- BM25 keyword + vector similarity with CrossEncoder reranking (LanceDB)
- **Multi-Pass Indexing** -- Documents stored at 4 chunk granularities (1K/2K/4K/8K chars)
- **Knowledge Graph** -- Category-aware entity/relationship extraction with LightRAG, 3D visualization, application and persona filtering
- **Agentic Pipeline** -- LangGraph orchestration with query classification, sub-task decomposition, and strategy selection
- **Document-Level RBAC** -- ACL group filtering enforced at the search layer, per-application isolation
- **Web Crawler** -- Multi-page crawling with Playwright fallback for bot-protected sites, auto-downloads PDFs/DOCX/PPTX
- **Application Workspaces** -- Organize documents, connectors, and queries by project with filtering across the UI
- **MCP Server** -- Model Context Protocol tools for OpenWebUI, Claude Code, and other AI agents
- **Admin Dashboard** -- Document management, ingestion queue, playground, knowledge graph explorer, web connector management
- **Streaming Answers** -- SSE-based token streaming in the playground
- **OpenAI-Compatible API** -- Drop-in `/v1/chat/completions` endpoint with citations
- **Docker Ready** -- Multi-stage build with health checks and shared data volumes

## Architecture

```
                         +------------------+
                         |   User / Client  |
                         +--------+---------+
                                  |
              +-------------------+-------------------+
              |                   |                   |
     +--------v-------+  +-------v--------+  +-------v--------+
     |   Admin UI     |  |   REST API     |  |   MCP Server   |
     |   :8080/admin  |  |   :8080/api    |  |   :8090 / 8091 |
     +--------+-------+  +-------+--------+  +-------+--------+
              |                   |                   |
              +-------------------+-------------------+
                                  |
                    +-------------v--------------+
                    |     Agent Graph (LangGraph) |
                    |                             |
                    |  1. Classify Query           |
                    |  2. Retrieve (strategy)      |
                    |  3. Enrich (knowledge graph) |
                    |  4. Synthesize (LLM)         |
                    +--+----------+----------+----+
                       |          |          |
              +--------v--+  +---v------+  +v-----------+
              | LanceDB   |  | Metadata |  | LLM / Embed|
              | (vectors) |  | (SQLite) |  | (external) |
              +-----------+  +----------+  +------------+
```

## Query Pipeline

When a user asks a question, SAURON runs a 4-step agent pipeline:

### Step 1: Classify

The LLM analyzes the question and selects a retrieval strategy:

| Strategy | When | Example |
|----------|------|---------|
| `lookup` | Specific fact or document | "What does policy 4.2 say?" |
| `sweep` | Exhaustive "find all" | "What contracts did the army award?" |
| `analytical` | Numbers, aggregations | "What was Q3 revenue?" |
| `cross_reference` | Compare across sources | "Does spending comply with policy?" |
| `temporal` | Time-based | "What changed last month?" |

The classifier also decomposes complex questions into **sub-tasks** that are searched in parallel.

### Step 2: Retrieve

Each strategy uses different chunk tiers and search methods:

| Strategy | Search Method | Chunk Tier | Approach |
|----------|--------------|------------|----------|
| Lookup | Hybrid + CrossEncoder | Medium (2K) | Precision search with window expansion |
| Sweep | Hybrid discovery, then full fetch | XLarge -> Large | Find docs first, then get all their content |
| Analytical | Text-to-SQL | N/A | LLM generates SQL, executes against DB |
| Cross-reference | Parallel sub-task search | Medium (2K) | Each sub-task searched independently |

Sub-task decomposition runs all sub-queries concurrently via `asyncio.gather()`.

### Step 3: Enrich

If the knowledge graph has entities, SAURON:
1. Extracts key entities from the question via LLM
2. Traverses the graph (depth=2) to find connected entities and relationships
3. Adds a synthetic "Knowledge Graph Context" chunk to the results
4. Pulls additional document chunks where graph entities were mentioned

This finds documents that vector search alone would miss.

### Step 4: Synthesize

The LLM generates a cited answer from all retrieved context:
1. Score-based filtering removes low-relevance chunks (no LLM call needed)
2. Context is built with source filenames for citation
3. LLM generates the answer (supports streaming via SSE)
4. Reasoning artifacts from thinking models are stripped
5. Citations are deduplicated to one per source document

## Multi-Pass Indexing

Every document is chunked and embedded at 4 granularities:

| Tier | Size | Overlap | Purpose |
|------|------|---------|---------|
| Small | 1,024 chars | 100 | Precise entity mentions |
| Medium | 2,048 chars | 200 | Balanced (default for lookup, entity extraction) |
| Large | 4,096 chars | 400 | Full paragraphs (sweep content retrieval) |
| XLarge | 8,192 chars | 800 | Complete sections (sweep document discovery) |

Each search query targets the optimal tier for its strategy, avoiding duplicate content across tiers.

## Hybrid Search Engine

SAURON uses LanceDB with three search tiers:

1. **Semantic search** -- vector similarity (dense embeddings)
2. **Hybrid search** -- semantic + BM25 keyword matching with RRF fusion
3. **Hybrid + CrossEncoder** -- adds neural reranking for highest quality

ACL filtering is enforced at the database level using `array_has_any()` on the `acl_groups` field.

**Window expansion** pulls neighboring chunks from the same document and tier after initial search, recovering context that chunking split apart.

## Knowledge Graph

Entity extraction is **category-aware** -- the extraction prompt adapts based on the document's category:

| Category | Entity Focus |
|----------|-------------|
| Contracts | Contractors, agencies, contract numbers, award amounts, locations |
| Budget/Finance | Budget items, fiscal years, programs, obligations |
| IT Policies | Systems, compliance frameworks, security controls |
| HIPAA | PHI/PII data types, safeguards, breach procedures |
| Meeting Notes | Attendees, action items, decisions, deadlines |

Entities are deduplicated at ingestion time with 3-tier matching (exact, case-insensitive, normalized). The reconciler can auto-merge high-confidence duplicates via LLM comparison.

Graph traversal uses SQL queries against the entity/relationship tables, not graph databases.

## Ingestion Pipeline

```
Parse -> Categorize -> Generate Summary -> Chunk (4 tiers) -> Embed -> Store -> Extract Entities
```

- **Parse**: PDF, DOCX, XLSX, CSV, Markdown, plain text, meeting transcripts
- **Categorize**: LLM matches against existing categories or proposes new ones
- **Summary**: LLM generates a 2-3 sentence document summary, prepended to every chunk for contextual embeddings
- **Chunk**: 4 tiers stored in parallel
- **Embed**: Batch or individual via OpenAI-compatible API
- **Store**: LanceDB with FTS index, scalar indexes on doc_id and acl_groups
- **Extract**: Category-aware entity/relationship extraction with knowledge graph building

The async ingestion queue shows live progress with entity/relationship counts.

## Web Crawler

SAURON can crawl websites and automatically ingest their content:

- **Multi-page crawling** with configurable depth (0-3 levels), URL pattern filtering, and max page limits
- **Multiple seed URLs** per connector -- a base URL plus additional URLs, all crawled at depth 0
- **File detection** -- automatically downloads linked PDFs, DOCX, PPTX, XLSX, and other file types
- **Playwright fallback** -- sites behind bot protection (Akamai, Cloudflare) are fetched with a headless Chromium browser; file downloads use in-page `fetch()` to carry session cookies
- **Content dedup** -- SHA-256 hashing prevents re-ingesting unchanged pages
- **Live progress** -- active crawl status (pages found/ingested, current URL) shown on the Queue page
- **Application assignment** -- crawled content is tagged with the connector's application and ACL groups

## Application Workspaces

Documents and connectors can be organized into **applications** (projects/workspaces):

- Each application has a name, description, owner, and default ACL groups
- Documents inherit ACL from their application's defaults
- The **Knowledge Graph** can be filtered by application (server-side entity filtering via document tracing)
- The **Playground** can be scoped to an application (restricts vector search to the app's documents via `doc_id IN` filtering)
- Filtering composes with persona/ACL filtering using intersection (AND) semantics

## Document-Level RBAC

Every document has an `acl_groups` field (e.g., `["finance", "executives"]`). When a user searches:

- Their groups are checked against each document's ACL
- `"ALL"` bypasses filtering (admin access)
- Filtering happens at the database level (LanceDB `array_has_any`)
- MCP tools default to `["ALL"]` (configure per-user in production)

ACL groups can be set during upload or inherited from the document's category.

## MCP Server

SAURON exposes tools via the Model Context Protocol for integration with OpenWebUI, Claude Code, and other AI agents:

| Tool | Purpose |
|------|---------|
| `tool_ask` | **Primary** -- answers any question with full RAG pipeline |
| `tool_summarize_topic` | Summarize a topic across all documents |
| `tool_compare` | Compare two items, policies, or topics |
| `tool_search_documents` | Low-level snippet search |
| `tool_query_database` | Text-to-SQL against registered databases |
| `tool_lookup_document` | Read full document content |
| `tool_list_documents` | Browse documents by category |
| `tool_search_knowledge_graph` | Find entity relationships |
| `tool_search_meetings` | Search meeting transcripts by speaker/topic |

Three transport modes:
- **SSE** (port 8090) -- direct MCP clients
- **OpenAPI via mcpo** (port 8091) -- OpenWebUI and REST clients
- **stdio** -- subprocess mode (Claude Code, mcpo)

## Admin Dashboard

| Page | Purpose |
|------|---------|
| Dashboard | KPIs: documents, categories, entities, vectors, proposals |
| Documents | Upload, edit category/ACL, bulk select/delete, sortable columns |
| Applications | Create and manage project workspaces with default ACL groups |
| Connectors | Web crawler configuration with inline editing, additional URLs, crawl-now button |
| Queue | Live ingestion progress with entity/relationship counts and active crawl status |
| Categories | Create, edit, manage document categories with NARA GRS mapping |
| Proposals | Approve/reject auto-categorization and entity merge proposals |
| Playground | Query testing with step trace, streaming answers, application and persona filters |
| Knowledge Graph | 3D entity visualization with application, persona, and type filtering |
| Settings | LLM/embedding endpoints, reconciliation, LanceDB config |
| Audit Log | JSONL audit trail of all operations |

## Quick Start

### Local Development

```bash
# Clone and setup
git clone https://github.com/mulkeym/rag.git
cd rag
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Configure
cp .env.example .env
# Edit .env with your LLM and embedding server URLs

# Run
uvicorn src.main:app --host 0.0.0.0 --port 8080

# MCP server (separate terminal)
mcpo --port 8091 -- python -m src.mcp.run_stdio
```

### Docker

```bash
cp .env.example .env
# Edit .env with your LLM and embedding server URLs

docker compose up -d
```

Services:
- **API + Admin UI**: http://localhost:8080
- **MCP (SSE)**: http://localhost:8090
- **MCP (OpenAPI)**: http://localhost:8091
- **OpenWebUI**: http://localhost:3000

### OpenWebUI Integration

1. Start SAURON with `docker compose up -d`
2. Open OpenWebUI at http://localhost:3000
3. Go to Settings -> MCP Servers -> OpenAPI
4. Enter URL: `http://mcpo:8091`
5. Click Connect -- tools are auto-discovered

## Configuration

See `.env.example` for all available settings. Key configuration:

```bash
# LLM (any OpenAI-compatible endpoint)
VLLM_BASE_URL=http://192.168.1.181:8080/v1
VLLM_MODEL_NAME=gemma-4-26B-A4B-it-UD-Q4_K_M.gguf
VLLM_REQUEST_TIMEOUT=300  # increase for thinking models

# Embeddings
EMBEDDING_API_URL=http://192.168.1.117:8080/v1
EMBEDDING_MODEL_NAME=llama-embed-nemotron-8b.Q8_0.gguf

# Vector store (embedded, no server needed)
LANCEDB_PATH=data/lancedb

# Auth
JWT_SECRET_KEY=change-me-in-production
API_KEYS=your-api-key-here
```

## Thinking Model Support

SAURON supports reasoning/thinking models (Gemma, Qwen, etc.) that use extended reasoning:

- Extracts content from `reasoning_content` field (llama.cpp format)
- Strips `<think>` blocks from responses
- All internal LLM calls use `max_tokens >= 1024` to leave room for reasoning
- Reasoning artifacts are stripped from final answers

## Tech Stack

| Component | Technology |
|-----------|-----------|
| API Framework | FastAPI |
| Agent Orchestration | LangGraph |
| Vector Store | LanceDB (embedded) |
| Knowledge Graph | LightRAG (file-based, zero infrastructure) |
| Metadata Store | SQLite + SQLAlchemy (async) |
| Embeddings | nomic-embed-text-v1 / any OpenAI-compatible |
| LLM | Any OpenAI-compatible endpoint |
| Web Crawling | requests + Playwright (headless Chromium fallback) |
| MCP | FastMCP |
| Admin UI | Jinja2 + HTMX |
| Graph Visualization | 3D Force Graph |
| Document Parsing | Unstructured, python-docx, openpyxl |
| Containerization | Docker + Docker Compose |

## License

MIT
