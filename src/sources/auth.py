"""Strict identity for binary delivery; legacy trusted headers do not apply."""

from dataclasses import dataclass
import time
import logging

import jwt
from fastapi import HTTPException, Request

from src.auth.api_key import validate_api_key
from src.config import settings

AUDIENCE = "sauron-source-downloads"


@dataclass(frozen=True)
class DownloadIdentity:
    subject: str
    groups: list[str]


async def require_download_identity(request: Request, doc_id: str, revision: str):
    def reject(status, detail):
        logging.getLogger("sauron.source_access").info(
            "original_auth_denied doc_id=%r revision=%r status=%d",
            doc_id,
            revision,
            status,
        )
        return HTTPException(
            status, detail, headers={"Cache-Control": "private, no-store"}
        )

    if not validate_api_key(request.headers.get("x-api-key", "")):
        raise reject(403, "Invalid or missing application key")
    secret = settings.source_download_jwt_secret
    token = request.headers.get("x-sauron-download-identity", "")
    if len(secret) < 32 or not token:
        raise reject(401, "Verified download identity required")
    try:
        claims = jwt.decode(
            token,
            secret,
            algorithms=["HS256"],
            issuer="open-webui",
            audience=AUDIENCE,
            options={
                "require": [
                    "sub",
                    "iss",
                    "aud",
                    "iat",
                    "exp",
                    "groups",
                    "doc_id",
                    "revision",
                    "operation",
                ]
            },
        )
        if (
            not isinstance(claims["sub"], str)
            or not claims["sub"].strip()
            or type(claims["iat"]) is not int
            or type(claims["exp"]) is not int
            or not 0 < claims["exp"] - claims["iat"] <= 60
            or claims["iat"] < time.time() - 60
            or claims["aud"] != AUDIENCE
            or claims["operation"] != "read-original"
            or claims["doc_id"] != doc_id
            or claims["revision"] != revision
            or not isinstance(claims["groups"], list)
            or len(claims["groups"]) > 1000
            or any(
                not isinstance(g, str) or not g or len(g) > 256
                for g in claims["groups"]
            )
        ):
            raise ValueError("Invalid claims")
        # ALL is an existing privileged retrieval bypass, never a download grant.
        return DownloadIdentity(
            claims["sub"], [g for g in claims["groups"] if g != "ALL"]
        )
    except (jwt.InvalidTokenError, ValueError, TypeError):
        raise reject(401, "Invalid download identity") from None
