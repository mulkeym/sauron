# MCP Server with OpenAPI (mcpo) Setup

This document explains how to run the RAG Knowledge Service MCP server as an OpenAPI endpoint, suitable for integration with **OpenWebUI** and other clients that expect standard HTTP/OpenAPI interfaces.

## Overview

The MCP server can be exposed in two ways:

1. **SSE (Server-Sent Events)** — Direct FastMCP server on port 8090
   - Best for: Direct FastMCP browser clients
   - Protocol: Custom SSE/FastMCP protocol
   - URL: `http://localhost:8090/sse`

2. **OpenAPI via mcpo** — MCP-to-OpenAPI proxy on port 8091 (recommended for OpenWebUI)
   - Best for: OpenWebUI, standard HTTP clients, REST APIs
   - Protocol: Standard OpenAPI/REST
   - URL: `http://localhost:8091`

## Quick Start

### 1. Install mcpo

```bash
source .venv/bin/activate
pip install mcpo
```

### 2. Start the MCP Server with mcpo

```bash
source .venv/bin/activate
mcpo --port 8091 -- python -m src.mcp.run_stdio
```

This will:
- Start your MCP server in stdio mode (standard MCP protocol)
- Wrap it with mcpo to expose OpenAPI endpoints
- Listen on `http://0.0.0.0:8091`

### 3. Verify it's working

```bash
# Check if running
lsof -i :8091

# View OpenAPI docs
curl http://localhost:8091/openapi.json | jq .

# Or open in browser: http://localhost:8091/docs
```

## OpenWebUI Integration

### Configuration

1. Open **OpenWebUI** settings
2. Go to **Settings** → **MCP Servers** (or **Admin** → **Settings** → **MCP Servers**)
3. Select **OpenAPI** mode
4. Enter URL: `http://<your-server-ip>:8091` (e.g., `http://10.10.10.115:8091`)
5. Click **Connect**

The server should discover all available tools:
- `tool_ask` — Ask questions about documents
- `tool_search_documents` — Search knowledge base
- `tool_lookup_document` — Read full documents
- `tool_search_knowledge_graph` — Search entities and relationships
- `tool_summarize_topic` — Summarize topics across documents
- `tool_compare` — Compare two items/policies
- And more...

### Browser Access

View the interactive OpenAPI documentation at:
```
http://<your-server-ip>:8091/docs
```

## MCP Server Runners

### `src/mcp/run.py` — SSE Transport (Original)

Runs the MCP server with SSE (Server-Sent Events) transport on port 8090.

```bash
python -m src.mcp.run
```

**Use case:** FastMCP browser clients, direct SSE consumers
**Endpoint:** `http://localhost:8090/sse`

### `src/mcp/run_stdio.py` — Stdio Transport (Standard MCP)

Runs the MCP server with stdio transport (standard MCP protocol).

```bash
python -m src.mcp.run_stdio
```

**Use case:** Direct stdio spawning (Claude Code, mcpo, etc.)
**Endpoint:** Communicates via stdin/stdout
**When to use:** Wrap with mcpo for OpenAPI, or use directly in subprocess mode

### `src/mcp/run_http.py` — HTTP Transport (Alternative)

Alias for SSE transport with configurable port.

```bash
python -m src.mcp.run_http
```

## Comparison

| Feature | SSE (8090) | OpenAPI via mcpo (8091) |
|---------|-----------|------------------------|
| Protocol | FastMCP/SSE | Standard OpenAPI/REST |
| Best for | FastMCP browser UI | OpenWebUI, REST clients |
| Port | 8090 | 8091 |
| Auth | Header-based | REST auth (via mcpo) |
| Interactive docs | `/sse` endpoint | `/docs` Swagger UI |
| Setup complexity | Low | Medium (requires mcpo) |

## Configuration

Edit `.env` or `src/config.py` to customize:

```python
# MCP Server
mcp_port: int = 8090              # SSE server port
mcp_alt_port: int = 8091          # Alternative HTTP port
mcp_server_name: str = "rag-knowledge-service"
```

## Troubleshooting

### mcpo fails to start: "No module named src.mcp.run_stdio"

Make sure you're in the project directory and have the venv activated:

```bash
cd /path/to/rag
source .venv/bin/activate
mcpo --port 8091 -- python -m src.mcp.run_stdio
```

### OpenWebUI shows "Connection Failed"

1. Check mcpo is running: `lsof -i :8091`
2. Test the endpoint: `curl http://localhost:8091/openapi.json`
3. Verify firewall/network allows port 8091
4. Check mcpo logs for errors

### IPv4 Issues

All LLM and embedding calls are forced to IPv4 to avoid VPN timeout issues. This is configured in:
- `src/generation/llm_client.py`
- `src/ingestion/embedder.py`
- `src/admin/routes.py`

If you get connection timeouts, ensure your remote vLLM server is accessible via IPv4 (use `curl -4 http://...` to test).

## References

- [mcpo GitHub](https://github.com/open-webui/mcpo)
- [OpenWebUI MCP Docs](https://docs.openwebui.com/features/extensibility/mcp/)
- [Model Context Protocol](https://modelcontextprotocol.io)
