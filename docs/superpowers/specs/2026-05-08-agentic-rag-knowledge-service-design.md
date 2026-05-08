# Agentic RAG Knowledge Service — Design Spec

**Date:** 2026-05-08
**Status:** Draft

---

## 1. Overview

An on-premises, agentic RAG system that ingests unstructured and structured enterprise data (PDFs, Word docs, spreadsheets, databases, meeting transcripts) and exposes it as a knowledge service. The system supports direct user interaction via a chat UI and serves as an MCP server for other AI agents across the organization.

### Goals

- Answer questions about enterprise documents with citations to source material
- Act as a domain expert capable of reasoning across financial and procedural knowledge
- Enforce document-level access control tied to Active Directory groups
- Serve as a knowledge tool for other AI agents via MCP
- Automatically organize and categorize new documents as they are ingested
- Run entirely on-premises with US-origin AI models

### Non-Goals

- Taking actions or triggering workflows (Q&A only)
- Replacing existing databases or document management systems
- Real-time streaming ingestion (batch/scheduled is sufficient)
- External/customer-facing access

---

## 2. High-Level Architecture

```
+-------------------------------------------------------------+
|                    On-Premises Deployment                     |
|                                                              |
|  +------------------  Consumers  -----------------------+    |
|  |                                                      |    |
|  |  AI Agents  |  Dev Tools  |  Chat UI  |  REST API   |    |
|  |  (MCP)      |  (MCP)      |  (Web)    |  (HTTP)     |    |
|  +-----+------------+------------+-----------+----------+    |
|        |            |            |           |               |
|  +-----v------------v------------v-----------v----------+    |
|  |              API Gateway / Auth Layer                 |    |
|  |         (JWT + API key + mTLS + agent registry)      |    |
|  +---+----------+-----------+----------+----------+-----+    |
|      |          |           |          |          |          |
|  +---v---+ +----v----+ +---v-----+ +--v------+ +-v------+   |
|  |  MCP  | | OpenAI  | |  REST   | |  Chat   | | Admin  |   |
|  |Server | | Compat  | |  API    | |   UI    | |  UI    |   |
|  |(hi+lo)| |         | |         | |         | |        |   |
|  +---+---+ +----+----+ +---+-----+ +--+------+ +---+----+   |
|      |          |           |          |            |        |
|  +---v----------v-----------v----------v------------v---+    |
|  |               Agent Orchestrator Core                 |    |
|  |            (Python — LangGraph)                       |    |
|  |                                                       |    |
|  |  +-------------+ +--------------+ +---------------+   |    |
|  |  | Query       | | Knowledge    | | Retrieval     |   |    |
|  |  | Classifier  | | Registry     | | Strategy      |   |    |
|  |  |             | |              | | Router        |   |    |
|  |  +-------------+ +--------------+ +---------------+   |    |
|  +-------+--------------+-----------------+--------------+    |
|          |              |                 |                   |
|  +-------v----+ +-------v--------+ +-----v-----------+      |
|  |  Gemma 4   | |   Vector DB    | |   Databases     |      |
|  |  31B on    | |   (Qdrant)     | |   (PostgreSQL   |      |
|  |  vLLM      | |                | |    etc.)        |      |
|  |  A100/H100 | |  + E5-large    | |                 |      |
|  +------------+ +----------------+ +-----------------+      |
|                                                              |
|  +----------------------------------------------------------+|
|  |              Ingestion Pipeline                           ||
|  |  Upload/Watched Folder -> Parse -> Classify -> Chunk     ||
|  |  -> Embed -> Store (Vector DB + Metadata)                ||
|  +----------------------------------------------------------+|
|                                                              |
|  +----------------------------------------------------------+|
|  |  Supporting Services                                      ||
|  |  LDAP/AD | Audit Log | Monitoring | Admin UI              ||
|  +----------------------------------------------------------+|
+--------------------------------------------------------------+
```

---

## 3. AI Model Selection

All models must be produced by US-based companies.

| Component | Model | Origin | Rationale |
|-----------|-------|--------|-----------|
| Agent LLM | Gemma 4 31B | Google (US) | Native tool use, 256K context, #3 on Arena AI among open models, strong agentic benchmarks |
| Embeddings | E5-large | Microsoft (US) | Proven retrieval quality, self-hostable |
| Text-to-SQL | Gemma 4 31B (shared) | Google (US) | Single model simplifies infrastructure |
| Doc Classification | Gemma 4 31B (shared) | Google (US) | Used at ingestion for auto-categorization |

### Serving

- **vLLM** for LLM inference — continuous batching, OpenAI-compatible API, best NVIDIA throughput
- Gemma 4 31B fits on 1x A100 80GB comfortably (~10-15 tok/s per request)
- E5-large can share the GPU or run on CPU

---

## 4. Document Ingestion Pipeline

```
Upload / Watched Folder
        |
        v
  +-- Parser (per type) --+
  | PDF: digital extract   |
  |       or OCR (Tesseract)|
  | Word: python-docx       |
  | Spreadsheet: row/col    |
  |   + text summaries      |
  | Transcripts: speaker    |
  |   identification +      |
  |   type tagging           |
  +----------+-------------+
             |
             v
  +-- Classifier (Gemma 4) --+
  | Match to existing         |
  | category? -> assign       |
  | New category detected?    |
  | -> propose + queue for    |
  |    admin approval         |
  +----------+---------------+
             |
             v
  +-- Chunker -----------------+
  | 512 tokens, 50 token overlap|
  | Structure-aware boundaries  |
  | (don't split mid-paragraph, |
  |  respect headings)          |
  +----------+-----------------+
             |
             v
  +-- Embed + Store -----------+
  | E5-large embeddings         |
  | Store in Qdrant with:       |
  |  - source document ID       |
  |  - page/section             |
  |  - doc type / category      |
  |  - ACL groups               |
  |  - date metadata            |
  |  - speaker (transcripts)    |
  |  - chunk type (question,    |
  |    statement, action_item)  |
  +----------------------------+
```

### Document Formats Supported

- **PDF** — digital and scanned (with OCR)
- **Word** (.docx) — preserving heading structure
- **Spreadsheets** (.xlsx, .csv) — stored as both embedded text summaries and queryable tabular data
- **Meeting transcripts** — parsed with speaker identification, utterance type tagging (question, statement, action item)
- **Databases** — not ingested; schema registered and queried live via text-to-SQL

### Parsing Library

**Unstructured.io** (open source, US company) — handles PDF, Word, spreadsheets, and OCR in a unified pipeline.

### Meeting Transcript Enrichment

Transcripts receive special processing to enable sweep queries (e.g., "all questions Mike asked"):

```
Raw:
  "Mike: Are we on track for Q2?
   Sarah: Yes, but the API work is behind."

Stored as:
  chunk_1: {speaker: "Mike", type: "question",
            text: "Are we on track for Q2?",
            meeting: "Engineering Standup",
            date: "2026-04-10"}
  chunk_2: {speaker: "Sarah", type: "statement", ...}
```

### Re-indexing

- When a document is updated, old chunks for that document are deleted and re-ingested
- Scheduled scan of watched folders for new/modified files
- Manual upload via admin UI also supported

### Auto-Categorization

New documents are classified by the LLM at ingestion:

- **Known category match** — automatic assignment, no human needed
- **New category detected** — proposed with name, description, suggested routing rules, and suggested ACL groups. Queued for admin approval before going live

Admin approval is required for new categories because:
- ACL implications (new access control decisions)
- Routing rules affect answer quality for all users
- Prevents category sprawl

---

## 5. Knowledge Layer (Knowledge Registry)

A metadata and organizational layer that catalogs all data sources and helps the agent make informed retrieval decisions.

### Source Catalog

Each data source is registered with:

```
source: "finance_policies"
type: procedure_docs
format: PDF
retrieval_strategy: vector_search
acl_groups: [finance, executives]
description: "Corporate finance policies and compliance procedures"
doc_count: 47
last_updated: 2026-05-01
```

### Source Types

| Type | Retrieval Strategy | Examples |
|------|-------------------|----------|
| Text documents | Vector similarity search | Policies, manuals, reports |
| Spreadsheets | Vector search + tabular query | Budget sheets, inventories |
| Databases | Text-to-SQL (live query) | Financial DB, HR systems |
| Meeting transcripts | Metadata filter + vector search | All-hands, standups, reviews |

### Routing Rules

Keyword and semantic hints that help the agent pick the right sources:

- "revenue/expense/budget" -> finance sources
- "server/deploy/outage" -> IT sources
- "policy/procedure/compliance" -> procedure docs
- Ambiguous queries -> search across all permitted sources

### Cross-Source Relationships

- `finance_policies` REFERENCES `quarterly_financials`
- `it_runbooks` REFERENCES `infra_inventory`
- Enables the agent to follow references when a question spans sources

### Growth Over Time

The registry grows organically through auto-categorization:

```
Month 1:  [finance_policies] [it_runbooks] [quarterly_data]
Month 3:  + [vendor_contracts]    <- admin approved
Month 5:  + [hr_policies]         <- admin approved
Month 6:  + [legal_compliance]    <- admin approved
```

Admins can also merge overlapping categories and update routing rules via the admin UI.

---

## 6. Agent Orchestrator & Retrieval

The agent orchestrator is the core reasoning engine. It classifies queries, plans retrieval strategies, executes them, and synthesizes answers with citations.

### Query Flow

```
1. PLAN     - Analyze query, identify sub-tasks, classify query type
2. FILTER   - Load user's AD groups, build ACL filter
3. RETRIEVE - Execute retrieval strategy (parallel where possible)
4. EVALUATE - Sufficient context? Gaps? Contradictions?
5. SYNTHESIZE - Generate grounded answer with citations
```

### Query Type Classification

| Query Type | Example | Strategy |
|-----------|---------|----------|
| Lookup | "What does policy 4.2 say?" | Single vector search, return top chunks |
| Sweep | "All questions Mike asked in 30 days" | Exhaustive: metadata filter + map-reduce over docs |
| Analytical | "What was Q3 revenue?" | Text-to-SQL against database |
| Cross-reference | "Does spending comply with policy?" | Multi-source: database + procedure docs |
| Temporal | "What changed last month?" | Metadata date filter + comparison |

### Retrieval Strategies

**Lookup queries:** Standard top-K vector similarity search with ACL filtering.

**Sweep queries (exhaustive):** Basic RAG fails here because top-K misses results scattered across many documents.

1. Metadata filter narrows to relevant document set
2. Multi-strategy retrieval:
   - Keyword/structured search (speaker tags, question marks)
   - Semantic search with multiple reformulations
   - Full document scan via map-reduce (feed each doc to LLM, extract matches)
3. Merge and deduplicate results

**Analytical queries:** Text-to-SQL generation against registered database schemas. The agent reads the schema from the knowledge registry, generates SQL, executes it, and returns results with the query as citation.

**Cross-reference queries:** The agent identifies multiple relevant sources, retrieves from each using the appropriate strategy, then reasons across the combined context.

### Retrieval Tools Available to the Agent

- `search_documents(query, doc_type_filter)` — vector similarity search
- `query_database(natural_language_question)` — text-to-SQL
- `lookup_document(doc_id, section)` — fetch a specific known section
- `refine_search(original_query, feedback)` — reformulate and retry
- `search_meetings(speaker, topic, date_range, type)` — metadata-filtered transcript search

### Guardrails

- The agent only answers from retrieved context; never fabricates
- If it cannot find sufficient information, it says so explicitly
- Contradictions between sources are flagged to the user

### Framework

**LangGraph** — provides the most control over multi-step agentic flows, conditional branching (query classification -> strategy selection), parallel tool execution, and map-reduce patterns.

---

## 7. Access Control & Authentication

The ACL system is only as strong as the identity chain. Three layers of trust are enforced.

### Trust Model

```
+----------+    +--------------+    +-------------+
|  User    |--->|  Calling App |--->|  RAG API    |
| (human)  |    |  or Agent    |    |             |
|          |    |              |    |  Validates: |
| JWT from |    | + mTLS cert  |    |  1. mTLS   |
| AD auth  |    | + API key    |    |  2. API key |
|          |    | + user JWT   |    |  3. JWT     |
+----------+    +--------------+    +-------------+
```

### Layer 1: mTLS — App/Agent Identity

- Each registered application or agent receives a client certificate signed by the internal CA
- The RAG API only accepts connections with valid client certificates
- Proves: "this connection is from a known, registered application"

### Layer 2: API Key — App Authorization

- Each app/agent gets a unique API key
- Enables: per-app rate limiting, auditing, revocation
- Proves: "this app is authorized to use this API"

### Layer 3: JWT — User Identity

- User authenticates directly against LDAP/AD and receives a signed JWT
- The JWT is passed through with every API call (even when routed through another agent)
- The RAG API validates the JWT signature independently — never trusts a bare user_id
- JWT contains user identity and AD group memberships
- Proves: "this request is on behalf of this specific user with these AD groups"

### Key Principle

The RAG API **never accepts a bare user_id**. It always validates the token independently. The calling app/agent does not assert identity — it passes through a token that the RAG system verifies.

### Document-Level ACL Enforcement

- Every document is tagged with permitted AD groups at ingestion time
- At query time, the user's AD groups (from JWT) are resolved
- All retrieval operations filter to only return documents the user has access to
- ACL filtering happens at the vector DB level (metadata filter) for performance

### Agent-Level Permissions

When other AI agents consume the MCP server, access is double-gated:

| Control | What it restricts |
|---------|------------------|
| Agent permissions (from agent registry) | Which tools and sources the agent can access |
| User permissions (from JWT/AD groups) | Which documents are visible for this specific request |

Both must pass. An IT support agent calling on behalf of a finance user still cannot access finance-only tools if the agent is restricted to IT sources.

### Agent Registry

```
+----------------+----------------+--------------+
| Agent          | Allowed Tools  | Allowed      |
|                |                | Sources      |
+----------------+----------------+--------------+
| hr-agent       | ask, search,   | hr_policies, |
|                | summarize      | meeting_notes|
+----------------+----------------+--------------+
| compliance-agt | ALL            | ALL          |
+----------------+----------------+--------------+
| it-support-agt | ask, search,   | it_runbooks, |
|                | lookup         | infra_data   |
+----------------+----------------+--------------+
```

### Audit Trail

Every query is logged:

- **Which app/agent** called (from API key / mTLS cert)
- **Which user** the request was on behalf of (from JWT)
- **What was queried** (full query text)
- **Which documents were retrieved** and returned
- **Which retrieval strategy** was used
- **Timestamp**

Stored in Elasticsearch or PostgreSQL for compliance review and troubleshooting.

---

## 8. Integration Layer

The RAG agent core is consumed through four endpoints, all sharing the same orchestrator, auth, and audit infrastructure.

### 8.1 REST API

Standard HTTP API. Foundation for all other integrations.

```
POST /api/v1/query
{
  "question": "What is policy 4.2?",
  "session_id": "abc123"
}
Authorization: Bearer <JWT>
X-API-Key: <api_key>

Response:
{
  "answer": "Policy 4.2 states...",
  "citations": [...],
  "confidence": 0.94
}
```

### 8.2 OpenAI-Compatible API

Drop-in replacement for any tool built against the OpenAI chat completions format. A FastAPI middleware layer sits in front of vLLM: it receives the OpenAI-format request, runs the full RAG pipeline (query classification, retrieval, synthesis), and returns the result formatted as a chat completion response. vLLM serves the raw Gemma 4 model; the RAG logic is in the middleware, not in vLLM itself.

```
POST /v1/chat/completions
{
  "model": "internal-rag",
  "messages": [{"role": "user", "content": "..."}]
}
```

Compatible with: Open WebUI, LibreChat, Continue.dev, and any OpenAI-compatible client. Citations are included in the response content (inline markdown references) since the OpenAI format has no native citation field.

### 8.3 Chat UI

A self-hosted web interface for direct user interaction. Options:

- **Open WebUI** — self-hosted, OpenAI-compatible, good out of the box
- **Custom build** — if specific UX requirements emerge

User authenticates via LDAP, receives JWT, chat interface passes JWT with each request.

### 8.4 MCP Server (Knowledge Service)

The primary integration point for other AI systems. Exposes the RAG system as a set of MCP tools and resources.

**Transport:** SSE (Server-Sent Events) over HTTP — works through existing load balancers, supports mTLS.

#### High-Level Tools (Agent-Friendly)

These run the full agent orchestrator and return complete answers. Designed for consuming agents that want a synthesized result without orchestrating retrieval themselves.

```
ask(
  question: str,
  context: optional[str],        # why the agent is asking
  depth: "quick"|"thorough"|"exhaustive"
) -> {answer, citations[], confidence}

summarize_topic(
  topic: str,
  time_range: optional[date_range],
  format: "brief"|"detailed"
) -> {summary, sources[], key_points[]}

compare(
  item_a: str,
  item_b: str
) -> {comparison, differences[], sources[]}
```

#### Low-Level Tools (Precise Control)

For consuming agents that want to orchestrate their own retrieval logic.

```
search_documents(
  query: str,
  doc_type: optional[str],
  department: optional[str],
  date_from: optional[date],
  date_to: optional[date]
) -> [{text, source, page, relevance}]

query_database(
  question: str,
  database: optional[str]
) -> {sql, results, source_table}

lookup_document(
  doc_id: str,
  section: optional[str]
) -> {content, metadata}

search_meetings(
  speaker: optional[str],
  topic: optional[str],
  date_from: optional[date],
  date_to: optional[date],
  type: optional["question"|"action_item"|"statement"]
) -> [{text, speaker, meeting, date}]

list_sources(
) -> [{name, type, doc_count, description}]
```

#### MCP Resources

```
document://{doc_id}       -> Full document content + metadata
category://{category_name} -> Category description, doc list, schema
schema://{database_name}   -> Table definitions, relationships
```

#### Context Passing

Consuming agents can pass context about their workflow to improve retrieval quality:

```
ask(
  question: "What is the PTO policy?",
  context: "I am onboarding a new employee in engineering.
            I need the policy for exempt employees in California.",
  depth: "thorough"
)
```

#### Async Support for Long Operations

Sweep queries and exhaustive searches may take 30+ seconds. The MCP server supports async execution:

```
# Synchronous (quick queries)
result = ask(question="What is policy 4.2?", depth="quick")

# Asynchronous (exhaustive queries)
job = ask(question="All mentions of Mike in 30 days",
          depth="exhaustive", async=true)
-> {job_id: "abc123", status: "processing"}

# Poll for results (MCP clients poll; no callback/webhook needed)
get_result(job_id="abc123")
-> {status: "processing"}  # still working

get_result(job_id="abc123")
-> {status: "complete", answer: ..., citations: [...]}
```

Async jobs are stored server-side with a configurable TTL (default 1 hour). Clients poll via `get_result`. Webhook callbacks are not supported in v1 to keep the MCP server stateless beyond job storage.

#### Structured Response Format

All MCP responses are structured for agent consumption:

```json
{
  "answer": "PTO policy for exempt CA employees...",
  "confidence": 0.92,
  "citations": [
    {
      "doc_id": "hr-policy-2024-v3",
      "doc_title": "HR Policy Manual v3",
      "section": "4.1 - Paid Time Off",
      "page": 15,
      "snippet": "Exempt employees in California...",
      "relevance": 0.95
    }
  ],
  "sources_searched": ["hr_policies", "ca_compliance"],
  "retrieval_strategy": "lookup",
  "warnings": []
}
```

---

## 9. Technology Stack

| Component | Technology | Rationale |
|-----------|-----------|-----------|
| LLM | Gemma 4 31B | US-origin, native tool use, 256K context |
| LLM Serving | vLLM | Best NVIDIA throughput, continuous batching, OpenAI-compat built in |
| Embeddings | E5-large (Microsoft) | US-origin, strong retrieval quality |
| Vector DB | Qdrant | Self-hosted, Rust (fast), metadata filtering for ACLs, MIT license |
| Agent Framework | LangGraph | Best control over multi-step agentic flows, tool routing, map-reduce |
| API Layer | FastAPI (Python) | Async, auto-generates OpenAPI docs |
| MCP Server | FastMCP (Python SDK) | Official MCP SDK, SSE transport |
| Chat UI | Open WebUI (or custom) | Self-hosted, OpenAI-compatible |
| Doc Parsing | Unstructured.io | Handles PDF, Word, spreadsheets, OCR; open source, US company |
| Auth | LDAP + JWT (PyJWT) | Integrates with existing AD infrastructure |
| Audit/Logging | Elasticsearch or PostgreSQL | Queryable audit trail for compliance |
| Monitoring | Prometheus + Grafana | Standard, self-hosted |
| Orchestration | Docker Compose or Kubernetes | Docker Compose to start; Kubernetes for production scale |

---

## 10. Deployment Architecture

### Minimal Deployment (Start Here)

- **1x A100/H100 node** — Gemma 4 31B + E5-large embeddings via vLLM
- **1x application server** — Qdrant + FastAPI + ingestion pipeline + MCP server
- **Docker Compose** to manage all services
- Handles ~5-10 concurrent users

### Production Scale

- **2+ GPU nodes** behind a load balancer for LLM redundancy
- **Dedicated Qdrant cluster** (3 nodes for replication)
- **Kubernetes** for orchestration and auto-scaling
- **Separate ingestion workers** — bulk indexing during off-hours
- Dedicated Elasticsearch cluster for audit logs

### GPU Sizing

| Workload | Requirement |
|----------|------------|
| Gemma 4 31B inference | 1x A100 80GB, ~10-15 tok/s per request |
| E5-large embeddings | Can share GPU or run on CPU |
| Concurrent conversations | 1x A100 handles ~5-10 via continuous batching |
| Bulk ingestion | Can share inference GPU or use separate GPU during off-hours |

---

## 11. Data Flow Summary

### Query Path

```
User/Agent -> Auth (mTLS + API key + JWT)
           -> Agent Orchestrator
           -> Query Classification
           -> ACL Filter (AD groups from JWT)
           -> Knowledge Registry (source selection)
           -> Retrieval Strategy Router
              -> Vector search (Qdrant)
              -> Text-to-SQL (database)
              -> Map-reduce (sweep queries)
           -> Result synthesis with citations
           -> Structured response
           -> Audit log
```

### Ingestion Path

```
Document upload / watched folder scan
  -> Format detection
  -> Parser (Unstructured.io)
  -> LLM classifier (category assignment or proposal)
  -> Structure-aware chunking
  -> Embedding (E5-large)
  -> Vector DB storage with metadata + ACL tags
  -> Audit log
```

---

## 12. Implementation Phasing

This system should be built incrementally, with each phase delivering usable value:

| Phase | Scope | Delivers |
|-------|-------|----------|
| 1 | Core RAG: ingestion pipeline + vector search + REST API + chat UI | Users can upload docs and ask questions with citations |
| 2 | Agentic orchestrator: query classification, multi-strategy retrieval, text-to-SQL | Smarter answers, database integration, sweep queries |
| 3 | MCP server + agent-to-agent integration | Other AI systems can use the knowledge base |
| 4 | Knowledge layer auto-categorization + admin UI | Self-organizing corpus, category management |

Each phase builds on the previous. Auth (JWT + API key) should be present from Phase 1. mTLS and the agent registry are added in Phase 3 when external agents connect.

---

## 13. Open Questions

1. **Specific AD groups and document-to-group mapping** — needs input from IT/security team on existing group structure
2. **Existing database schemas** — which databases to connect and their access patterns
3. **Meeting transcript format** — what tool generates them, what format are they in, do they already have speaker tags
4. **Admin UI scope** — build custom or use an existing internal tools framework
5. **Backup and disaster recovery** — strategy for vector DB and metadata store
6. **Model update cadence** — process for evaluating and upgrading to newer Gemma versions
