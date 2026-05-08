# src/mcp/agent_registry.py
from dataclasses import dataclass, field

@dataclass
class AgentPermissions:
    agent_id: str
    api_key: str
    allowed_tools: list[str] = field(default_factory=list)
    allowed_sources: list[str] = field(default_factory=list)

class AgentRegistry:
    def __init__(self):
        self._agents: dict[str, AgentPermissions] = {}

    def register(self, permissions: AgentPermissions) -> None:
        self._agents[permissions.api_key] = permissions

    def get_by_api_key(self, api_key: str) -> AgentPermissions | None:
        return self._agents.get(api_key)

    def can_use_tool(self, api_key: str, tool_name: str) -> bool:
        agent = self._agents.get(api_key)
        if agent is None:
            return False
        if "ALL" in agent.allowed_tools:
            return True
        return tool_name in agent.allowed_tools

    def can_access_source(self, api_key: str, source_name: str) -> bool:
        agent = self._agents.get(api_key)
        if agent is None:
            return False
        if "ALL" in agent.allowed_sources:
            return True
        return source_name in agent.allowed_sources
