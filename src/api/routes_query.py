from fastapi import APIRouter, Depends
from src.api.models import CitationResponse, QueryRequest, QueryResponse
from src.api.routes_ingest import get_vector_store, get_schema_registry, get_metadata_store
from src.auth.dependencies import require_auth
from src.auth.models import UserContext
from src.generation.rag_chain import agent_query

router = APIRouter(prefix="/api/v1", tags=["query"])

@router.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest, user: UserContext = Depends(require_auth)):
    result = await agent_query(
        question=request.question, user_groups=user.groups,
        vector_store=get_vector_store(), schema_registry=get_schema_registry(),
        metadata_store=get_metadata_store(),
    )
    return QueryResponse(
        answer=result.answer,
        citations=[CitationResponse(doc_id=c.doc_id, filename=c.filename, doc_type=c.doc_type, chunk_index=c.chunk_index, page=c.page, snippet=c.snippet, relevance=c.relevance) for c in result.citations],
        cached=result.cached,
        cached_query=result.cached_query,
    )
