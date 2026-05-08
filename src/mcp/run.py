"""Standalone MCP server runner."""
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
    server.run(transport="sse", port=settings.mcp_port)


if __name__ == "__main__":
    main()
