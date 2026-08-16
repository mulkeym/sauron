# src/api/routes_openai_compat.py
"""OpenAI-compatible /v1/chat/completions endpoint.

Wraps the RAG pipeline behind the OpenAI chat completions format so any
OpenAI-compatible client can query the knowledge base as a drop-in replacement.
"""
import time
import uuid

from fastapi import APIRouter, Header, HTTPException, Request
from pydantic import BaseModel

from src.api.routes_ingest import get_vector_store, get_schema_registry
from src.auth.api_key import validate_api_key
from src.auth.jwt import decode_token
from src.config import settings
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
    # Auth: try JWT first, fall back to API key only (for simple clients)
    user_groups = ["ALL"]
    agent_id = None
    if authorization.startswith("Bearer "):
        token = authorization.removeprefix("Bearer ")
        # Check if it's a JWT or just an API key used as bearer token
        try:
            user = decode_token(token)
            user_groups = user.groups
            agent_id = user.username
        except ValueError:
            # Might be an API key passed as bearer token (common with OpenAI clients)
            if not validate_api_key(token):
                raise HTTPException(status_code=401, detail="Invalid token")
    elif x_api_key:
        if not validate_api_key(x_api_key):
            raise HTTPException(status_code=403, detail="Invalid API key")
    else:
        raise HTTPException(status_code=401, detail="Missing authentication")

    # Extract the last user message as the question
    question = ""
    for msg in reversed(payload.messages):
        if msg.role == "user":
            question = msg.content
            break

    if not question:
        raise HTTPException(status_code=400, detail="No user message found")

    # Run the RAG pipeline
    result = await agent_query(
        question=question,
        user_groups=user_groups,
        vector_store=get_vector_store(),
        schema_registry=get_schema_registry(),
        session_headers=http.headers,
        agent_id=agent_id,
    )

    # Format citations as part of the response
    answer = result.answer
    if result.citations:
        sources = "\n\n---\n**Sources:**\n"
        for i, c in enumerate(result.citations, 1):
            page_info = f", page {c.page}" if c.page else ""
            sources += f"- [{i}] {c.filename}{page_info} (relevance: {c.relevance:.2f})\n"
        answer += sources

    # Return OpenAI-compatible response
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
        "object": "chat.completion",
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
