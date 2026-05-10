# src/auth/jwt.py
from __future__ import annotations
from datetime import datetime, timezone, timedelta

import jwt

from src.auth.models import UserContext
from src.config import settings


def create_token(
    username: str,
    groups: list[str],
    expiration_minutes: int | None = None,
) -> str:
    if expiration_minutes is None:
        expiration_minutes = settings.jwt_expiration_minutes

    now = datetime.now(timezone.utc)
    payload = {
        "sub": username,
        "groups": groups,
        "iat": now,
        "exp": now + timedelta(minutes=expiration_minutes),
    }
    return jwt.encode(payload, settings.jwt_secret_key, algorithm=settings.jwt_algorithm)


def decode_token(token: str) -> UserContext:
    try:
        payload = jwt.decode(
            token,
            settings.jwt_secret_key,
            algorithms=[settings.jwt_algorithm],
        )
    except jwt.ExpiredSignatureError:
        raise ValueError("Token has expired")
    except jwt.InvalidTokenError:
        raise ValueError("Invalid token")

    return UserContext(
        username=payload["sub"],
        groups=payload.get("groups", []),
    )
