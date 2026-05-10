"""MCP server runner for stdio transport (standard MCP protocol).

Start with: python -m src.mcp.run_stdio
Or with mcpo: mcpo -- python -m src.mcp.run_stdio
"""
from src.api.routes_ingest import get_vector_store, get_metadata_store, get_schema_registry
from src.mcp.agent_registry import AgentRegistry
from src.mcp.server import create_mcp_server


def main():
    server = create_mcp_server(
        vector_store=get_vector_store(),
        schema_registry=get_schema_registry(),
        metadata_store=get_metadata_store(),
        agent_registry=AgentRegistry(),
    )
    # Use stdio transport for standard MCP protocol (works with mcpo)
    server.run(transport="stdio")


if __name__ == "__main__":
    main()
