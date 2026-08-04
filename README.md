# SAURON

**Structured Agentic Unified Retrieval Over Networks**

The all-seeing eye that finds everything in your documents.

SAURON is a self-hosted agentic RAG (Retrieval-Augmented Generation) system with document-level access control, multi-pass indexing, hybrid search, knowledge graph enrichment, and an admin UI. It integrates with any OpenAI-compatible LLM and embedding endpoint.

## Features

- **Hybrid Search** -- BM25 keyword + vector similarity with CrossEncoder reranking (LanceDB)
- **Multi-Pass Indexing** -- Documents stored at 4 chunk granularities (1K/2K/4K/8K chars)
- **Knowledge Graph** -- Category-aware entity/relationship extraction with LightRAG, 3D visualization, dataset and persona filtering
- **Agentic Pipeline** -- LangGraph orchestration with query classification, sub-task decomposition, and strategy selection
- **Document-Level RBAC** -- ACL group filtering enforced at the search layer, per-dataset isolation
- **Web Crawler** -- Multi-page crawling with Playwright fallback for bot-protected sites, auto-downloads PDFs/DOCX/PPTX
- **Dataset Workspaces** -- Organize documents, connectors, and queries by project with filtering across the UI
- **MCP Server** -- Model Context Protocol tools for OpenWebUI, Claude Code, and other AI agents
- **Admin Dashboard** -- Document management, ingestion queue, playground, knowledge graph explorer, web connector management
- **Streaming Answers** -- SSE-based token streaming in the playground
- **OpenAI-Compatible API** -- Drop-in `/v1/chat/completions` endpoint with citations
- **Application API Keys** -- DB-backed multi-app service credentials (hashed, revocable) for backends that call SAURON
- **Docker Ready** -- Multi-stage build with health checks; CI publishes to `ghcr.io/mulkeym/sauron`

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

The knowledge graph explorer offers two renderers selectable via a dropdown:
- **Cosmos (GPU)** -- GPU-accelerated 2D force graph via cosmos.gl, handles thousands of nodes smoothly. Click a node to highlight its connections (neighbors glow, connected links turn white, everything else dims).
- **3D Force Graph** -- WebGL 3D visualization with particle effects, camera rotation, and link animations.

## Ingestion Pipeline

```
Parse -> Categorize -> Generate Summary -> Chunk (4 tiers) -> Embed -> Store -> Extract Entities
```

- **Parse**: PDF, DOCX, XLSX, CSV, Markdown, plain text, meeting transcripts
- **Categorize**: LLM matches against existing categories or proposes new ones
- **Extract Metadata**: LLM extracts structured metadata (entities, people, organizations, locations, dates, amounts, identifiers, topics, procedures, action items, key facts) and a summary in a single call
- **Chunk**: 4 tiers stored in parallel, plus a dedicated summary embedding per document
- **Embed**: Batch or individual via OpenAI-compatible API
- **Store**: LanceDB with FTS index, scalar indexes on doc_id and acl_groups
- **Extract**: Category-aware entity/relationship extraction with knowledge graph building

The async ingestion queue shows live progress with entity/relationship counts.

## Document Metadata

Every document gets structured metadata extracted at ingestion time via a single LLM call:

- **summary** -- 2-4 sentence document overview (also embedded as a searchable vector)
- **entities, people, organizations, locations** -- named entities
- **dates, amounts, identifiers** -- structured data points
- **topics, procedures, action_items, key_facts** -- semantic content

This metadata speeds up sweep queries -- instead of the LLM reading every discovered document, the system first searches document summaries and metadata to narrow the list, then only reads the truly relevant ones. Documents not fully read still contribute their metadata as lightweight context to the synthesizer. Existing documents can be backfilled from Settings.

## Adaptive Retrieval

SAURON learns from past queries to improve future retrieval accuracy:

- **Relevance Feedback** -- after each query, logs which documents were cited, relevant, or irrelevant. Similar future queries boost useful docs and exclude irrelevant ones (~50% speed improvement on repeated patterns)
- **Pseudo-Relevance Feedback (PRF)** -- expands queries using key terms from top initial results (organizations, identifiers, topics). Improves first-time query recall without needing history
- **Strategy Memory** -- learns which retrieval strategy (sweep, lookup, etc.) works best for each query pattern. Tracks precision per pattern over time
- **Decay** -- feedback older than 90 days loses weight, preventing stale patterns from dominating
- **Metrics Dashboard** -- Settings page shows MAP Precision (% of docs the LLM read that were useful), query timing, and feedback signal counts

## Web Crawler

SAURON can crawl websites and automatically ingest their content:

- **Multi-page crawling** with configurable depth (0-3 levels), URL pattern filtering, and max page limits
- **Multiple seed URLs** per connector -- a base URL plus additional URLs, all crawled at depth 0
- **File detection** -- automatically downloads linked PDFs, DOCX, PPTX, XLSX, and other file types
- **Playwright fallback** -- sites behind bot protection (Akamai, Cloudflare) are fetched with a headless Chromium browser; file downloads use in-page `fetch()` to carry session cookies
- **Content dedup** -- SHA-256 hashing prevents re-ingesting unchanged pages
- **Live progress** -- active crawl status (pages found/ingested, current URL) shown on the Queue page
- **Dataset assignment** -- crawled content is tagged with the connector's dataset and ACL groups

## Dataset Workspaces

Documents and connectors can be organized into **datasets** (projects/workspaces):

- Each dataset has a name, description, owner, and default ACL groups
- Documents inherit ACL from their dataset's defaults
- The **Knowledge Graph** can be filtered by dataset (server-side entity filtering via document tracing)
- The **Playground** can be scoped to a dataset (restricts vector search to the dataset's documents via `doc_id IN` filtering)
- Filtering composes with persona/ACL filtering using intersection (AND) semantics

## Document-Level RBAC

Every document has an `acl_groups` field (e.g., `["finance", "executives"]`). When a user searches:

- Their groups are checked against each document's ACL
- `"ALL"` bypasses filtering (admin access)
- Filtering happens at the database level (LanceDB `array_has_any`)
- MCP tools default to `["ALL"]` (configure per-user in production)

ACL groups can be set during upload or inherited from the document's category.

### Embedded figures (images in PDFs, Word, Excel)

When `FIGURE_EXTRACTION_ENABLED=true` (default), ingest also extracts embedded images from:

| Format | Image sources |
|--------|----------------|
| **PDF** | Embedded images + full-page render for text-sparse pages |
| **Word (.docx)** | `word/media/*` (and relationship order when available) |
| **Excel (.xlsx/.xlsm)** | Sheet drawings + `xl/media/*` |
| **PowerPoint (.pptx)** | `ppt/media/*` collector ready (parser/chunk path still basic) |

For each region the pipeline:

1. Runs **OCR** (Tesseract) and classifies: table / network / process / text / other  
2. Applies a strategy (vision + Phase-A OCR identifier merge when useful):
   - **table** — markdown grid → DuckDB path  
   - **network** / **process** — structured diagram description  
   - **text_scan** — OCR only (linear prose)  
3. Merges figure prose into the text stream **before** multi-tier chunking  
4. Feeds the **same enriched text** to LightRAG (PDF/DOCX; Excel figures only when present)

Word body text still comes from paragraphs; figures are appended under `## Embedded figures`. Excel cell tables stay on the DuckDB path; chart/screenshot images get figure strategies and optional KG on that figure text only.

Requires the `tesseract` binary and a multimodal-capable model on `VLLM_BASE_URL`. Failures are fail-open.

### Managing groups, test users, and application API keys

Admin **Settings → Security** includes:

- **ACL Groups** — register, edit, and deactivate group names; discover unregistered groups from existing documents
- **Playground Personas** — create/edit lab test users (Mike, Bob, …) and their group memberships (not production auth)
- **Applications & API Keys** — named service clients (demo apps, OpenWebUI, MCP gateways). Keys are stored **hashed**; the full secret is shown **once** on create. Revoke or deactivate per app without redeploying.

The Playground and Knowledge Graph “act as / view as” dropdowns load from personas. If imported documents use groups no persona holds, the UI warns and offers to add those groups to Alice (or you can use **Custom groups…** in the Playground).

#### API authentication (apps calling SAURON)

Protected REST routes require **both**:

| Header | Purpose |
|--------|---------|
| `X-API-Key` | Service credential — application key from Security (or legacy `API_KEYS` env during migration) |
| `Authorization: Bearer <JWT>` | User identity + **ACL groups** for document filtering |

Mint a lab JWT:

```bash
curl -s -X POST http://localhost:8080/api/v1/auth/token \
  -H "Content-Type: application/json" \
  -d '{"username":"mike","password":"demo","groups":["finance","executives"]}'
```

Then query (sync or async):

```bash
curl -s -X POST http://localhost:8080/api/v1/query/async \
  -H "Content-Type: application/json" \
  -H "X-API-Key: <your-app-key>" \
  -H "Authorization: Bearer <jwt>" \
  -d '{"question":"What is the expense policy?"}'

# Optional: skip the query cache (force a full retrieval)
# -d '{"question":"...","skip_cache":true}'

# List datasets / upload into a dataset
curl -s http://localhost:8080/api/v1/datasets \
  -H "X-API-Key: <your-app-key>" -H "Authorization: Bearer <jwt>"

curl -s -X POST http://localhost:8080/api/v1/ingest/async \
  -H "X-API-Key: <your-app-key>" -H "Authorization: Bearer <jwt>" \
  -F "file=@./doc.pdf" -F "dataset_id=1"
```

**CORS:** SAURON allows browser origins on `localhost` / `127.0.0.1` for local demos. Production front-ends should call SAURON only from a **backend** (BFF), not the browser—use a dedicated application API key per client.

See [docs/API_APPLICATIONS.md](docs/API_APPLICATIONS.md) for multi-app key management.

## Backup & Restore

A single `.tar.gz` backup file contains everything needed to transport SAURON to another host:

- **metadata.db** -- all documents, categories, datasets, connectors, entities, relationships
- **lancedb/** -- vector embeddings and search indexes
- **lightrag/** -- knowledge graph (GraphML + JSON stores)
- **.env** -- configuration and secrets

Create and download backups from the **Settings** page. Restore by uploading a backup file -- current data is saved to `data.pre-restore/` before overwriting. Temp files (`_transactions/`, SQLite WAL, `.DS_Store`) are excluded to keep backups lean.

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
| Datasets | Create and manage project workspaces with default ACL groups |
| Connectors | Web crawler configuration with inline editing, additional URLs, crawl-now button |
| Queue | Live ingestion progress with entity/relationship counts and active crawl status |
| Categories | Create, edit, manage document categories with NARA GRS mapping |
| Proposals | Approve/reject auto-categorization and entity merge proposals |
| Playground | Query testing with step trace, streaming answers, dataset and persona filters |
| Knowledge Graph | GPU-accelerated (cosmos.gl) or 3D entity visualization with dataset, persona, and type filtering; click-to-highlight connections |
| Settings | LLM/embedding endpoints (incl. ignore SSL cert errors for private CAs), Security (ACL, personas, application API keys), backup & restore |
| Audit Log | JSONL audit trail of all operations |

## Quick Start

### Local Development

```bash
# Clone and setup
git clone https://github.com/mulkeym/sauron.git
cd sauron
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

**Build locally** (compose builds from the Dockerfile):

```bash
cp .env.example .env
# Edit .env with your LLM and embedding server URLs

docker compose up -d
```

**Corporate MITM / SSL errors during `docker compose build` (PyPI):**

1. Put your TLS inspection CA at `certs/Trusted_Root_CAs.pem`
2. Compose passes pip `--trusted-host` for public PyPI hosts (see `x-sauron-build`)
3. Optional proxy / private PyPI in `.env`:

```bash
# HTTP_PROXY=http://proxy.example:8080
# HTTPS_PROXY=http://proxy.example:8080
# PIP_INDEX_URL=https://pypi.internal.example/simple
# PIP_TRUSTED_HOST=pypi.internal.example files.internal.example
# TORCH_CPU_INDEX=https://pypi.internal.example/simple
# SAURON_PREFETCH_INSECURE_SSL=1
```

```bash
docker compose build --no-cache
docker compose up -d
```

Confirm in the build log: `certs/ contents:` lists `Trusted_Root_CAs.pem`,
`pip trusted-host args:` includes your hosts, and
`prefetch_hf_models: ALL models baked successfully`.

### Hugging Face models baked at build time

The image **must** download these during `docker build` / `compose build` so
runtime never calls Hugging Face:

| Model | Purpose |
|-------|---------|
| `nomic-ai/nomic-embed-text-v1` | Local embeddings |
| `cross-encoder/ms-marco-MiniLM-L-6-v2` | App reranker |
| `cross-encoder/ms-marco-TinyBERT-L-6` | LanceDB CrossEncoder default |
| `unstructuredio/yolo_x_layout` | PDF hi_res layout (YOLOX) |
| `microsoft/table-transformer-structure-recognition*` | PDF tables |

**Artifactory / corporate HF proxy** (recommended on work networks): set in `.env`
before `docker compose build`:

```bash
HF_ENDPOINT=https://artifactory.example.com/artifactory/api/huggingfaceml/huggingface-remote
HF_TOKEN=your-token
SAURON_PREFETCH_INSECURE_SSL=1   # if the proxy uses a private CA (also use certs/Trusted_Root_CAs.pem)
```

`huggingface_hub` honors `HF_ENDPOINT` + `HF_TOKEN` during the bake step. After
models are in the image, runtime stays offline (`HF_HUB_OFFLINE=1`).

If the hub is fully unreachable, pre-seed `hf-cache/` on a connected machine:

```bash
huggingface-cli download nomic-ai/nomic-embed-text-v1
cp -a ~/.cache/huggingface/. /path/to/sauron/hf-cache/
docker compose build
```

**Or pull the CI image** (published by GitHub Actions on every push to `master` and on `v*` tags):

```bash
docker pull ghcr.io/mulkeym/sauron:latest
# Tags also include sha-<short> and release versions (e.g. 1.0.0 from tag v1.0.0)
```

Kubernetes / Run:ai: use the Helm chart under [`charts/sauron`](charts/sauron) (defaults to the GHCR image).

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

## Security: dependency installs

For secure environments, install with security floors and a frozen lock:

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install -U 'pip>=26.1.2' 'setuptools>=83.0.0' wheel
# Prefer lock for reproducible builds:
pip install -r requirements.lock.txt
# Or resolve latest within security floors:
# pip install -r requirements.txt -c constraints-security.txt
pip-audit -r requirements.lock.txt   # expect: no known vulnerabilities
```

See `constraints-security.txt` for High/Critical CVE minimum versions.

### CPU-only PyTorch (Docker / air-gapped)

SAURON uses **torch** only for local CPU embeddings and CrossEncoder reranking. It does **not** need NVIDIA GPUs or CUDA in the app container.

The Dockerfile therefore installs **CPU-only** `torch` / `torchvision` from the official CPU wheel index before the rest of `requirements.txt`. That avoids multi-GB `nvidia-*` CUDA packages that default PyPI Linux torch wheels pull in.

```bash
# Local Linux installs (same idea as the Dockerfile):
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt -c constraints-security.txt

# Air-gapped Docker build: mirror the CPU index, then:
docker build --build-arg TORCH_CPU_INDEX=https://your-mirror/.../whl/cpu -t sauron .
```

### Custom root CAs (MITM / private PKI)

If outbound HTTPS is **TLS-inspected** (corporate MITM proxy re-signs traffic),
or you use private roots, place a PEM bundle at:

```text
certs/Trusted_Root_CAs.pem
```

For MITM, this should be the **inspection / SSL-decrypt CA** from your security
stack (not a random internal web server cert). Without it, `pip` during
`docker build` fails with `CERTIFICATE_VERIFY_FAILED` even though the network
has internet access.

At **image build** time the Dockerfile:

1. Copies `certs/` into the image (the directory always exists; the `.pem` is optional)
2. If `Trusted_Root_CAs.pem` is present and non-empty, installs it via `update-ca-certificates`
3. Sets `SSL_CERT_FILE`, `REQUESTS_CA_BUNDLE`, `CURL_CA_BUNDLE`, and `PIP_CERT` to the system CA bundle so **OS tools, pip, and Python** trust those roots

This runs in both the **builder** (pip → PyPI / pytorch.org through the proxy)
and **runtime** (LLM / embedding HTTPS through the same proxy). If the file is
missing, only the default public CA set is used.

```bash
# Typical MITM fix — no private PyPI required:
cp /path/to/corp-inspection-ca.pem certs/Trusted_Root_CAs.pem
docker build -t sauron .

# If the proxy is explicit (not transparent):
docker build -t sauron \
  --build-arg HTTPS_PROXY=http://proxy.example:8080 \
  --build-arg HTTP_PROXY=http://proxy.example:8080 \
  .
```

Prefer this over **Admin → Settings → Models → Ignore SSL certificate errors**.
The PEM is gitignored by default.

Builder `pip` also defaults `--trusted-host` for `pypi.org`,
`files.pythonhosted.org`, `pypi.python.org`, and `download.pytorch.org` so MITM
inspection cannot break package downloads even when the CA chain is incomplete.
Override with `--build-arg PIP_TRUSTED_HOST="..."`.

Hugging Face model prefetch uses **certifi**; the image merges the system CA
bundle into certifi so MITM roots apply there too. If prefetch still fails TLS:

```bash
docker build -t sauron --build-arg SAURON_PREFETCH_INSECURE_SSL=1 .
```

The image build also caches and offline-verifies every vocabulary supported by
the installed `tiktoken` package. LightRAG otherwise downloads its tokenizer
data from OpenAI blob storage the first time the knowledge graph is opened or
built. For a build host with no network, pre-populate `tiktoken-cache/`; the
Dockerfile copies it to the runtime `TIKTOKEN_CACHE_DIR`.

See `certs/README.md`.

## Configuration

See `.env.example` for all available settings. Key configuration:

```bash
# LLM (any OpenAI-compatible endpoint)
VLLM_BASE_URL=http://192.168.1.181:8080/v1
VLLM_MODEL_NAME=gemma-4-26B-A4B-it-UD-Q4_K_M.gguf
VLLM_REQUEST_TIMEOUT=300  # increase for thinking models

# Embeddings (default: local CPU, no GPU needed)
EMBEDDING_MODE=local
EMBEDDING_MODEL_NAME=nomic-ai/nomic-embed-text-v1
# Or point at an external OpenAI-compatible embedding provider:
# EMBEDDING_MODE=api
# EMBEDDING_API_URL=http://your-embedding-host:8000/v1

# TLS verification for outbound model API calls (LLM + embeddings).
# Set false when endpoints use private CAs or self-signed certs.
# Can also be toggled in Admin → Settings → Models (persists to data/settings.json).
SSL_VERIFY=true

# Vector store (embedded, no server needed)
LANCEDB_PATH=data/lancedb

# Auth
JWT_SECRET_KEY=change-me-in-production
# Bootstrap / legacy keys (also imported into Applications on startup)
API_KEYS=your-api-key-here
```

Prefer creating **per-application keys** in Admin → Security after first boot. Keep `JWT_SECRET_KEY` strong in production.

### Private CAs / self-signed model endpoints

If your LLM or embedding API uses an untrusted or private CA certificate:

1. Open **Admin → Settings → Models**
2. Check **Ignore SSL certificate errors**
3. Click **Save**

This sets `ssl_verify=false` for outbound model HTTP clients (LLM chat, embeddings, and admin connection tests). Prefer installing the private CA into the container/host trust store when possible; the ignore option is for trusted private networks only.

Equivalent env/Helm: `SSL_VERIFY=false` or `config.sslVerify: "false"`.

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
