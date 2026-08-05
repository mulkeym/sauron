from fastapi import APIRouter, Depends, HTTPException
from src.api.models import (
    CitationResponse, QueryRequest, QueryResponse,
    AsyncQuerySubmitResponse, AsyncQueryStatusResponse,
)
from src.api.routes_ingest import get_vector_store, get_schema_registry, get_metadata_store
from src.auth.dependencies import require_auth
from src.auth.models import UserContext
from src.generation.rag_chain import agent_query
from src.api.query_jobs import query_queue, QueueFullError

router = APIRouter(prefix="/api/v1", tags=["query"])

@router.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest, user: UserContext = Depends(require_auth)):
    result = await agent_query(
        question=request.question, user_groups=user.groups,
        vector_store=get_vector_store(), schema_registry=get_schema_registry(),
        metadata_store=get_metadata_store(),
        skip_cache=request.skip_cache,
    )
    return QueryResponse(
        answer=result.answer,
        citations=[CitationResponse(doc_id=c.doc_id, filename=c.filename, doc_type=c.doc_type, chunk_index=c.chunk_index, page=c.page, snippet=c.snippet, relevance=c.relevance, figure_id=c.figure_id, section_title=c.section_title, caption=c.caption, slide=c.slide) for c in result.citations],
        cached=result.cached,
        cached_query=result.cached_query,
    )


@router.post("/query/async", response_model=AsyncQuerySubmitResponse)
async def query_async(request: QueryRequest, user: UserContext = Depends(require_auth)):
    """Submit a question for async processing. Returns a token to poll for status/result."""
    await query_queue.start_worker(get_vector_store(), get_schema_registry(), get_metadata_store())
    try:
        token = query_queue.enqueue(
            question=request.question,
            username=user.username,
            groups=user.groups,
            skip_cache=request.skip_cache,
        )
    except QueueFullError:
        raise HTTPException(status_code=503, detail="Query queue is full; please retry shortly")
    return AsyncQuerySubmitResponse(token=token, status="queued")


@router.get("/query/async/{token}", response_model=AsyncQueryStatusResponse)
async def query_async_status(token: str, user: UserContext = Depends(require_auth)):
    """Poll an async query by token. Owner-scoped: another user's token returns 404."""
    job = query_queue.get_job(token)
    if job is None or job.username != user.username:
        raise HTTPException(status_code=404, detail="Job not found")
    return AsyncQueryStatusResponse(
        token=job.token,
        status=str(job.status),
        step=job.step,
        steps=job.steps,
        classification=job.classification,
        answer=job.answer,
        citations=[CitationResponse(**c) for c in job.citations],
        cached=job.cached,
        cached_query=job.cached_query,
        error=job.error,
        created_at=job.created_at,
        completed_at=job.completed_at,
    )
