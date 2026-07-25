# src/auth/dependencies.py
from fastapi import Header, HTTPException

from src.auth.api_key import touch_api_key_usage, validate_api_key
from src.auth.jwt import decode_token
from src.auth.models import UserContext


async def require_auth(
    authorization: str = Header(default=""),
    x_api_key: str = Header(default="", alias="X-API-Key"),
) -> UserContext:
    # Validate API key (DB application keys or legacy settings list)
    if not validate_api_key(x_api_key):
        raise HTTPException(status_code=403, detail="Invalid or missing API key")

    # Best-effort last_used for DB keys (ignore failures)
    try:
        from src.api.routes_ingest import get_metadata_store
        await touch_api_key_usage(get_metadata_store(), x_api_key)
    except Exception:
        pass

    # Validate JWT (user identity + ACL groups)
    if not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing Bearer token")

    token = authorization.removeprefix("Bearer ")
    try:
        return decode_token(token)
    except ValueError as e:
        raise HTTPException(status_code=401, detail=str(e))
