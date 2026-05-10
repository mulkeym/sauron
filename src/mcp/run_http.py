"""MCP server runner for HTTP transport (network accessible).

Start with: python -m src.mcp.run_stdio
"""
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
    # Use SSE (Server-Sent Events) transport on port 8091 for network access
    port = getattr(settings, 'mcp_alt_port', 8091)
    print(f"Starting MCP server on port {port} (SSE transport)")
    server.run(transport="sse", port=port)


if __name__ == "__main__":
    main()
