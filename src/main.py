from contextlib import asynccontextmanager
from fastapi import FastAPI
from src.api.routes_auth import router as auth_router
from src.api.routes_ingest import router as ingest_router, get_metadata_store
from src.api.routes_query import router as query_router

@asynccontextmanager
async def lifespan(app: FastAPI):
    store = get_metadata_store()
    await store.init()
    yield

def create_app() -> FastAPI:
    app = FastAPI(title="RAG Knowledge Service", description="Agentic RAG system with document-level access control", version="0.1.0", lifespan=lifespan)
    app.include_router(auth_router)
    app.include_router(ingest_router)
    app.include_router(query_router)
    return app

app = create_app()
