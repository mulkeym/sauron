"""Short-lived bearer capabilities for one immutable, previously authorized PNG.

Sign only after document authorization. No API key, user JWT, group names or
filesystem paths are exposed to the browser. Recheck the live document/asset
snapshot on every read; an ordinary content hash alone is not authorization.
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import json
import re
import secrets
import time

from src.config import settings

LINK_PATH = "/api/v1/figure-links/"
TOKEN_PATTERN = r"[A-Za-z0-9_-]{1,4000}\.[a-f0-9]{64}"
SHORT_TOKEN_PATTERN = r"s_[A-Za-z0-9_-]{22}"
# Invalid tokens on this exact read route still receive the link verifier's 404,
# rather than an unrelated request for an API key. Other routes stay protected.
TOKEN_ROUTE_PATTERN = r"[^/]{1,4100}"


def enabled():
    return bool(settings.figure_public_base_url and len(settings.figure_link_signing_secret) >= 32)


def _acl_hash(doc):
    return hashlib.sha256(json.dumps(sorted(set(doc.acl_groups)), separators=(",", ":")).encode()).hexdigest()


def _signature(encoded):
    return hmac.new(settings.figure_link_signing_secret.encode(),
                    b"sauron-figure-v1:" + encoded.encode("ascii"), hashlib.sha256).hexdigest()


def issue(doc, ref):
    """Build the signed snapshot kept server-side; also supports legacy links."""
    if not enabled():
        return None
    now = int(time.time())
    payload = {"v": 1, "doc": doc.doc_id, "figure": ref["figure_id"],
               "variant": ref["variant"], "sha": ref["sha256"],
               "revision": getattr(doc, "content_hash", "") or "", "acl": _acl_hash(doc),
               "iat": now, "exp": now + settings.figure_link_ttl_seconds}
    encoded = base64.urlsafe_b64encode(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).rstrip(b"=").decode()
    token = encoded + "." + _signature(encoded)
    if not re.fullmatch(TOKEN_PATTERN, token):
        return None
    return {"inline_url": settings.figure_public_base_url.rstrip("/") + LINK_PATH + token,
            "expires_at": payload["exp"]}


async def issue_short(doc, ref, metadata_store):
    """Caller has authorized the figure. Issue an unguessable 128-bit capability.

    Keep the signed snapshot in the shared metadata DB, not in a long URL that a
    host model must reproduce. The lookup survives worker/container restarts.
    """
    signed = issue(doc, ref)
    if signed is None:
        return None
    token = "s_" + secrets.token_urlsafe(16)
    await metadata_store.store_figure_link(hashlib.sha256(token.encode()).hexdigest(),
        signed["inline_url"].rsplit("/", 1)[-1], signed["expires_at"], int(time.time()))
    return {"inline_url": settings.figure_public_base_url.rstrip("/") + LINK_PATH + token,
            "expires_at": signed["expires_at"]}


def verify(token):
    if not enabled() or not re.fullmatch(TOKEN_PATTERN, token):
        raise ValueError("Invalid diagram link")
    encoded, signature = token.split(".")
    if not hmac.compare_digest(_signature(encoded), signature):
        raise ValueError("Invalid diagram link")
    try:
        payload = json.loads(base64.urlsafe_b64decode(encoded + "=" * (-len(encoded) % 4)))
    except (ValueError, UnicodeError) as exc:
        raise ValueError("Invalid diagram link") from exc
    now = time.time()
    if (not isinstance(payload, dict) or payload.get("v") != 1
            or type(payload.get("iat")) is not int or type(payload.get("exp")) is not int
            or not payload["iat"] <= now < payload["exp"]
            or not 0 < payload["exp"] - payload["iat"] <= settings.figure_link_ttl_seconds
            or payload.get("variant") not in {"preview", "full"}
            or any(not isinstance(payload.get(k), str) for k in ("doc", "figure", "sha", "revision", "acl"))):
        raise ValueError("Invalid diagram link")
    return payload


async def read(token, metadata_store):
    from src.figures.service import image_bytes
    if not enabled():
        raise ValueError("Invalid diagram link")
    if re.fullmatch(SHORT_TOKEN_PATTERN, token):
        record = await metadata_store.get_figure_link(hashlib.sha256(token.encode()).hexdigest())
        if record is None or time.time() >= record.expires_at:
            raise ValueError("Invalid diagram link")
        token = record.signed_capability
    payload = verify(token)
    doc = await metadata_store.get_document(payload["doc"])
    if (doc is None or _acl_hash(doc) != payload["acl"]
            or (getattr(doc, "content_hash", "") or "") != payload["revision"]):
        raise FileNotFoundError("Diagram not found")
    # The signed capability authorizes exactly this snapshot, independently of
    # browser cookies. No caller-supplied path, groups or variant is accepted.
    raw, current = await image_bytes(payload["doc"], payload["figure"], ["ALL"], metadata_store, payload["variant"])
    if current is None or current["sha256"] != payload["sha"]:
        raise FileNotFoundError("Diagram not found")
    verify(token)  # Do not start a response if the read crossed the expiry.
    return raw
