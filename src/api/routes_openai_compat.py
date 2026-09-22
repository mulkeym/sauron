from src.citations import render_citations, citation_markdown
# src/api/routes_openai_compat.py
"""OpenAI-compatible /v1/chat/completions endpoint.

Wraps the RAG pipeline behind the OpenAI chat completions format so any
OpenAI-compatible client can query the knowledge base as a drop-in replacement.
"""
import time
import uuid

from fastapi import APIRouter, Header, HTTPException, Request
from pydantic import BaseModel

from src.api.routes_ingest import get_vector_store, get_schema_registry, get_metadata_store
from src.generation.rag_chain import agent_query

router = APIRouter(prefix="/v1", tags=["openai-compat"])


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str = "sauron"
    messages: list[ChatMessage]
    temperature: float = 0.1
    max_tokens: int = 2048
    stream: bool = False


@router.get("/models")
async def list_models():
    return {
        "object": "list",
        "data": [
            {
                "id": "sauron",
                "object": "model",
                "created": int(time.time()),
                "owned_by": "local",
            }
        ],
    }


@router.post("/chat/completions")
async def chat_completions(
    payload: ChatCompletionRequest,
    http: Request,
    authorization: str = Header(default=""),
    x_api_key: str = Header(default="", alias="X-API-Key"),
):
    # Application credentials never grant document access on their own. Use
    # the same user identity and configured forwarding rules as the MCP endpoint.
    from src.mcp.auth import extract_mcp_context, MCPAuthenticationError
    try:
        identity = extract_mcp_context(dict(http.headers))
    except MCPAuthenticationError as exc:
        raise HTTPException(status_code=exc.status_code, detail=str(exc)) from exc
    user_groups = identity.groups
    agent_id = identity.agent_id or identity.username

    # Extract the last user message as the question
    question = ""
    last_user_index = -1
    for index in range(len(payload.messages)-1, -1, -1):
        msg = payload.messages[index]
        if msg.role == "user":
            question = msg.content
            last_user_index = index
            break

    if not question:
        raise HTTPException(status_code=400, detail="No user message found")

    from src.audit.activity import query_activity_span

    async with query_activity_span(
        source="openai", tool="chat.completions",
        username=identity.username, user_groups=list(user_groups),
        query_text=question,
    ) as span:
        result = await agent_query(
            question=question,
            conversation=[{'role':m.role, 'content':m.content[:4000]} for m in payload.messages[:last_user_index][-8:]
                          if m.role in ('user', 'assistant')],
            user_groups=user_groups,
            vector_store=get_vector_store(),
            schema_registry=get_schema_registry(),
            metadata_store=get_metadata_store(),
            session_headers=http.headers,
            agent_id=agent_id,
        )
        span.strategy = result.query_type or ("cache" if result.cached else "")
        span.cache_hit = bool(result.cached)

    # Format citations as part of the response
    from src.figures.presentation import present_answer
    illustrated, images = await present_answer(result.answer, result.images, user_groups, get_metadata_store())
    answer = render_citations(illustrated, result.citations)
    if result.citations:
        sources = "\n\n---\n**Sources:**\n"
        for i, c in enumerate(result.citations, 1):
            sources += f"- {citation_markdown(c)}\n"
        answer += sources
    if result.warnings:
        answer += "\n\nEvidence limitations:\n" + "\n".join("- " + w for w in result.warnings)

    # Return OpenAI-compatible response
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
        "object": "chat.completion",
        "sauron_images": images,
        "created": int(time.time()),
        "model": payload.model,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": answer,
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": len(question.split()),
            "completion_tokens": len(answer.split()),
            "total_tokens": len(question.split()) + len(answer.split()),
        },
    }
