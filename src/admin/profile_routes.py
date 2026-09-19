"""Admin-only draft, publication, rollback and isolated answer preview APIs."""
from __future__ import annotations

import asyncio
import logging
import sqlite3
import time
from typing import Annotated

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from src.agent import profile_store
from src.agent.profiles import AnswerProfile, snapshot
from src.config import settings

router = APIRouter()
logger = logging.getLogger(__name__)


class VersionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    expected_version: int = Field(ge=0)


class DraftRequest(VersionRequest):
    config: AnswerProfile


class PreviewRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    config: AnswerProfile
    question: str = Field(min_length=1, max_length=5000)
    user_groups: list[Annotated[str, Field(min_length=1, max_length=200)]] = Field(default_factory=list, max_length=50)
    dataset_id: int = Field(default=0, ge=0)


class PromptRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    config: AnswerProfile


@router.post("/api/answer-profiles/{profile_id}/prompt")
def preview_prompt(profile_id: str, request: PromptRequest):
    from src.agent.synthesizer import get_system_prompt
    from src.agent.classifier import CLASSIFICATION_PROMPT
    return {"system_prompt": get_system_prompt(snapshot(profile_id, None, request.config)),
            "routing_prompt": CLASSIFICATION_PROMPT + "\n\nTeam routing guidance:\n" + request.config.routing_instructions}


def _store_call(fn, *args):
    try:
        return fn(*args)
    except profile_store.ProfileConflict as exc:
        raise HTTPException(409, str(exc)) from exc
    except KeyError as exc:
        raise HTTPException(404, str(exc)) from exc
    except (sqlite3.Error, OSError, ValueError) as exc:
        logger.exception("Answer profile storage unavailable")
        raise HTTPException(503, "Profile storage is unavailable. No change was applied.") from exc


@router.get("/api/answer-profiles")
def list_profiles():
    return _store_call(profile_store.read_book)


@router.post("/api/answer-profiles")
def create_profile(request: DraftRequest):
    book, profile_id = _store_call(profile_store.create_profile, request.config, request.expected_version)
    return {"book": book, "profile_id": profile_id}


@router.put("/api/answer-profiles/{profile_id}/draft")
def save_draft(profile_id: str, request: DraftRequest):
    return _store_call(profile_store.save_draft, profile_id, request.config, request.expected_version)


@router.post("/api/answer-profiles/{profile_id}/publish")
def publish_profile(profile_id: str, request: VersionRequest):
    return _store_call(profile_store.publish, profile_id, request.expected_version, settings.admin_username)


@router.post("/api/answer-profiles/{profile_id}/revisions/{revision}/activate")
def activate_revision(profile_id: str, revision: int, request: VersionRequest):
    return _store_call(profile_store.activate_revision, profile_id, revision, request.expected_version, settings.admin_username)


async def run_preview(profile_id: str, request: PreviewRequest):
    from src.agent.graph import create_agent_graph
    from src.agent.synthesizer import get_system_prompt
    from src.api.routes_ingest import get_metadata_store, get_vector_store, get_schema_registry

    profile = snapshot(profile_id, None, request.config)
    progress = []
    graph = create_agent_graph(get_vector_store(), get_schema_registry(), get_metadata_store())
    started = time.monotonic()
    # Runs the production graph with this request's draft snapshot. It does not
    # enter the cache/metrics/strategy-learning wrappers or change global settings.
    result = await graph.ainvoke({
        "question": request.question, "original_question": request.question,
        "user_groups": request.user_groups, "dataset_id": request.dataset_id,
        "answer_profile": profile, "preview": True,
        "retrieved_chunks": [], "sql_results": [], "retrieval_attempts": 0,
        "progress": lambda name, detail=None: progress.append(name),
    })
    from src.figures.service import answer_images
    images = await answer_images(request.question, result.get("citations", []), request.user_groups, get_metadata_store(), profile)
    return {"images": images, "answer": result.get("answer", ""), "warnings": result.get("warnings", []),
            "response_kind": result.get("response_kind", "answer"),
            "citations": [c.model_dump() for c in result.get("citations", [])],
            "evidence": result.get("preview_evidence", []),
            "query_type": str(result.get("query_type", "")),
            "reason": result.get("reason", ""),
            "steps": progress, "elapsed_seconds": round(time.monotonic() - started, 2),
            "profile": profile, "system_prompt": get_system_prompt(profile),
            "chunks_retrieved": len(result.get("retrieved_chunks", []))}


@router.post("/api/answer-profiles/{profile_id}/preview")
async def preview_profile(profile_id: str, request: PreviewRequest):
    book = _store_call(profile_store.read_book)
    if profile_id not in book["profiles"]:
        raise HTTPException(404, "Profile not found")
    try:
        async with asyncio.timeout(min(settings.async_query_timeout_seconds, 600)):
            return await run_preview(profile_id, request)
    except TimeoutError as exc:
        raise HTTPException(504, "Preview timed out. Try focused retrieval or a narrower question.") from exc
    except Exception as exc:
        logger.exception("Answer profile preview failed")
        raise HTTPException(502, "Preview failed. Check the configured model and data connections in Sauron's logs.") from exc
