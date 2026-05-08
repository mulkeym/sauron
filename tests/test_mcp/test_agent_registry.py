# tests/test_mcp/test_agent_registry.py
import pytest
from src.mcp.agent_registry import AgentRegistry, AgentPermissions

def test_register_and_get_agent():
    registry = AgentRegistry()
    registry.register(AgentPermissions(agent_id="hr-agent", api_key="hr-key-1", allowed_tools=["ask", "search_documents", "summarize_topic"], allowed_sources=["hr_policies", "meeting_notes"]))
    agent = registry.get_by_api_key("hr-key-1")
    assert agent is not None
    assert agent.agent_id == "hr-agent"
    assert "ask" in agent.allowed_tools

def test_get_unknown_agent():
    registry = AgentRegistry()
    assert registry.get_by_api_key("unknown-key") is None

def test_check_tool_permission_allowed():
    registry = AgentRegistry()
    registry.register(AgentPermissions(agent_id="hr-agent", api_key="hr-key-1", allowed_tools=["ask", "search_documents"], allowed_sources=["hr_policies"]))
    assert registry.can_use_tool("hr-key-1", "ask") is True
    assert registry.can_use_tool("hr-key-1", "query_database") is False

def test_check_tool_permission_all():
    registry = AgentRegistry()
    registry.register(AgentPermissions(agent_id="compliance-agt", api_key="comp-key-1", allowed_tools=["ALL"], allowed_sources=["ALL"]))
    assert registry.can_use_tool("comp-key-1", "anything") is True
    assert registry.can_access_source("comp-key-1", "anything") is True

def test_check_source_permission():
    registry = AgentRegistry()
    registry.register(AgentPermissions(agent_id="it-agent", api_key="it-key-1", allowed_tools=["ask"], allowed_sources=["it_runbooks", "infra_data"]))
    assert registry.can_access_source("it-key-1", "it_runbooks") is True
    assert registry.can_access_source("it-key-1", "finance_policies") is False

def test_unregistered_agent_has_no_permissions():
    registry = AgentRegistry()
    assert registry.can_use_tool("unknown", "ask") is False
    assert registry.can_access_source("unknown", "anything") is False
