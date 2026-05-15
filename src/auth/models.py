from __future__ import annotations
# src/auth/models.py
from __future__ import annotations
from pydantic import BaseModel


class TokenPayload(BaseModel):
    sub: str  # username
    groups: list[str] = []  # AD group names
    exp: int | None = None


class UserContext(BaseModel):
    """Resolved user identity available in request handlers."""
    username: str
    groups: list[str]
