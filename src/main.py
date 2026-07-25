from contextlib import asynccontextmanager
from pathlib import Path
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from src.api.routes_auth import router as auth_router
from src.api.routes_ingest import router as ingest_router, get_metadata_store
from src.api.routes_query import router as query_router
from src.admin.routes import router as admin_router
from src.api.routes_openai_compat import router as openai_compat_router
from src.config import settings

# Disable SSL verification globally when ssl_verify=False (self-signed certs)
if not settings.ssl_verify:
    import urllib3
    import requests as _requests
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
    # Monkey-patch requests to default verify=False
    _orig_request = _requests.Session.request
    def _patched_request(self, *args, **kwargs):
        kwargs.setdefault("verify", False)
        return _orig_request(self, *args, **kwargs)
    _requests.Session.request = _patched_request
    # Monkey-patch httpx (used by OpenAI SDK / LightRAG)
    try:
        import httpx
        _orig_httpx_client_init = httpx.Client.__init__
        _orig_httpx_async_init = httpx.AsyncClient.__init__
        def _patched_httpx_init(self, *args, **kwargs):
            kwargs.setdefault("verify", False)
            return _orig_httpx_client_init(self, *args, **kwargs)
        def _patched_httpx_async_init(self, *args, **kwargs):
            kwargs.setdefault("verify", False)
            return _orig_httpx_async_init(self, *args, **kwargs)
        httpx.Client.__init__ = _patched_httpx_init
        httpx.AsyncClient.__init__ = _patched_httpx_async_init
    except ImportError:
        pass

ADMIN_STATIC = Path(__file__).parent / "admin" / "static"

@asynccontextmanager
async def lifespan(app: FastAPI):
    store = get_metadata_store()
    await store.init()
    # Crash recovery: LightRAG resumes PENDING/PROCESSING/FAILED docs on the
    # next ainsert. Drop any rows that no longer exist in SAURON metadata so a
    # killed mid-KG import cannot resurrect deleted PDFs beside new uploads.
    try:
        from src.knowledge.graph_rag import reconcile_lightrag_with_metadata, get_lightrag
        docs = await store.list_documents()
        live = {d.doc_id for d in docs if getattr(d, "doc_id", None)}
        result = await reconcile_lightrag_with_metadata(live)
        import logging
        logging.getLogger(__name__).info(f"LightRAG startup reconcile: {result}")
        if live:
            await get_lightrag()
    except Exception as e:
        import logging
        logging.getLogger(__name__).warning(f"LightRAG init/reconcile deferred: {e}")
    # Load persisted table schemas into the in-memory registry
    try:
        import logging
        from src.api.routes_ingest import get_schema_registry
        from src.ingestion.tabular_ingest import populate_schema_registry
        n = await populate_schema_registry(store, get_schema_registry())
        logging.getLogger(__name__).info(f"Loaded {n} persisted table schema(s) into the registry")
    except Exception as e:
        logging.getLogger(__name__).warning(f"Schema registry load deferred: {e}")
    # Load persisted table hints into the in-memory hint store
    try:
        from src.api.routes_ingest import get_hint_store
        from src.ingestion.tabular_ingest import populate_hint_store
        hn = await populate_hint_store(store, get_hint_store())
        logging.getLogger(__name__).info(f"Loaded {hn} table hint(s) into the hint store")
    except Exception as e:
        logging.getLogger(__name__).warning(f"Hint store load deferred: {e}")
    yield

def create_app() -> FastAPI:
    app = FastAPI(title="SAURON", description="Structured Agentic Unified Retrieval Over Networks", version="0.1.0", lifespan=lifespan)

    # Allow browser demos (e.g. Vite on :5173) to call the API cross-origin.
    # Without this, preflight OPTIONS fails with 405 and the browser reports "Failed to fetch".
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[
            "http://localhost:5173",
            "http://127.0.0.1:5173",
            "http://localhost:4173",
            "http://127.0.0.1:4173",
            "http://localhost:3000",
            "http://127.0.0.1:3000",
        ],
        allow_origin_regex=r"https?://(localhost|127\.0\.0\.1)(:\d+)?",
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
        expose_headers=["*"],
    )

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
