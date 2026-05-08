# src/auth/models.py
from pydantic import BaseModel


class TokenPayload(BaseModel):
    sub: str  # username
    groups: list[str] = []  # AD group names
    exp: int | None = None


class UserContext(BaseModel):
    """Resolved user identity available in request handlers."""
    username: str
    groups: list[str]
