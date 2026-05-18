from contextlib import asynccontextmanager
from pathlib import Path
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from src.api.routes_auth import router as auth_router
from src.api.routes_ingest import router as ingest_router, get_metadata_store
from src.api.routes_query import router as query_router
from src.admin.routes import router as admin_router
from src.api.routes_openai_compat import router as openai_compat_router
from src.config import settings

# Disable SSL verification globally when ssl_verify=False (self-signed certs)
if not settings.ssl_verify:
    import os
    import urllib3
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
    os.environ["CURL_CA_BUNDLE"] = ""
    os.environ["REQUESTS_CA_BUNDLE"] = ""
    # For httpx (used by OpenAI SDK / LightRAG)
    os.environ["SSL_CERT_FILE"] = ""
    os.environ["HTTPX_SSL_VERIFY"] = "0"

ADMIN_STATIC = Path(__file__).parent / "admin" / "static"

@asynccontextmanager
async def lifespan(app: FastAPI):
    store = get_metadata_store()
    await store.init()
    # Initialize LightRAG knowledge graph
    try:
        from src.knowledge.graph_rag import get_lightrag
        await get_lightrag()
    except Exception as e:
        import logging
        logging.getLogger(__name__).warning(f"LightRAG init deferred: {e}")
    yield

def create_app() -> FastAPI:
    app = FastAPI(title="SAURON", description="Structured Agentic Unified Retrieval Over Networks", version="0.1.0", lifespan=lifespan)

    @app.get("/api/health")
    async def health():
        return {"status": "ok"}

    app.include_router(auth_router)
    app.include_router(ingest_router)
    app.include_router(query_router)
    app.include_router(openai_compat_router)
    app.include_router(admin_router)
    if ADMIN_STATIC.exists():
        app.mount("/admin/static", StaticFiles(directory=str(ADMIN_STATIC)), name="admin-static")
    return app

app = create_app()
