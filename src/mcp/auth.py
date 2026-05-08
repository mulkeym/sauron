# src/mcp/auth.py
from dataclasses import dataclass
from src.auth.api_key import validate_api_key
from src.auth.jwt import decode_token

@dataclass
class MCPContext:
    username: str
    groups: list[str]
    api_key: str
    agent_id: str = ""

def extract_mcp_context(headers: dict) -> MCPContext:
    api_key = headers.get("x-api-key", "")
    if not api_key or not validate_api_key(api_key):
        raise ValueError("Invalid or missing API key")
    auth_header = headers.get("authorization", "")
    if not auth_header.startswith("Bearer "):
        raise ValueError("Missing Bearer token")
    token = auth_header.removeprefix("Bearer ")
    user = decode_token(token)
    return MCPContext(username=user.username, groups=user.groups, api_key=api_key)
