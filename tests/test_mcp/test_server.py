import pytest
from unittest.mock import MagicMock
from src.mcp.server import create_mcp_server
from src.db.schema_registry import SchemaRegistry
from src.mcp.agent_registry import AgentRegistry
from src.config import settings


def test_create_mcp_server():
    server = create_mcp_server(
        vector_store=MagicMock(),
        schema_registry=SchemaRegistry(),
        metadata_store=MagicMock(),
        agent_registry=AgentRegistry(),
    )
    assert server is not None


def test_server_has_expected_name():
    server = create_mcp_server(
        vector_store=MagicMock(),
        schema_registry=SchemaRegistry(),
        metadata_store=MagicMock(),
        agent_registry=AgentRegistry(),
    )
    assert server.name == settings.mcp_server_name
