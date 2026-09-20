import asyncio
from contextlib import asynccontextmanager
from pathlib import Path
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from src.api.routes_auth import router as auth_router
from src.api.routes_ingest import (
    router as ingest_router,
    get_metadata_store,
    get_schema_registry,
    get_vector_store,
)
from src.api.routes_query import router as query_router
from src.admin.routes import router as admin_router
from src.api.routes_openai_compat import router as openai_compat_router
from src.config import settings
from src.auth.http import EndpointAuthenticationMiddleware

from src.ssl_config import apply_ssl_verify_setting

# Apply at import so early clients honor data/settings.json / env.
apply_ssl_verify_setting()

ADMIN_STATIC = Path(__file__).parent / "admin" / "static"

@asynccontextmanager
async def lifespan(app: FastAPI):
    store = get_metadata_store()
    await store.init()
    from src.figures.storage import FigureStore
    await FigureStore().reconcile(store)
    # Restore row ACLs after old admin edits or an interrupted permission update,
    # before accepting queries. No re-embedding or document re-ingestion needed.
    repaired = 0
    for doc in await store.list_documents():
        repaired += await asyncio.to_thread(get_vector_store().synchronize_document_acl,
                                           doc.doc_id, doc.acl_groups or [])
    if repaired:
        import logging
        logging.getLogger(__name__).info("Repaired vector permissions for %s documents", repaired)
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
    try:
        yield
    finally:
        from src.ingestion.queue import ingest_queue
        await ingest_queue.stop_worker()
        from src.ingestion.embedding_isolation import shutdown_embedding_worker
        await asyncio.to_thread(shutdown_embedding_worker)

def create_app() -> FastAPI:
    mcp_http_app = None
    app_lifespan = lifespan
    if settings.mcp_enabled:
        from fastmcp.utilities.lifespan import combine_lifespans
        from src.mcp.agent_registry import AgentRegistry
        from src.mcp.http import add_mcp_http_route, create_mcp_http_app
        from src.mcp.server import create_mcp_server

        mcp_server = create_mcp_server(
            vector_store=get_vector_store(),
            schema_registry=get_schema_registry(),
            metadata_store=get_metadata_store(),
            agent_registry=AgentRegistry(),
        )
        mcp_http_app = create_mcp_http_app(mcp_server)
        app_lifespan = combine_lifespans(lifespan, mcp_http_app.lifespan)

    app = FastAPI(
        title="SAURON",
        description="Structured Agentic Unified Retrieval Over Networks",
        version="0.1.0",
        lifespan=app_lifespan,
    )

    # Gate every route, including docs, health, token issuance, and mounted MCP.
    # CORS is added last so it can answer preflight without invoking endpoints.
    app.add_middleware(EndpointAuthenticationMiddleware)

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
    from src.api.routes_figures import router as figures_router, admin_router as admin_figures_router
    from src.api.routes_sources import router as sources_router
    app.include_router(sources_router)
    app.include_router(figures_router)
    app.include_router(admin_figures_router)
    app.include_router(openai_compat_router)
    app.include_router(admin_router)
    if ADMIN_STATIC.exists():
        app.mount("/admin/static", StaticFiles(directory=str(ADMIN_STATIC)), name="admin-static")
    if mcp_http_app is not None:
        # Register only the exact MCP path. A root mount would intercept normal
        # FastAPI fallback handling such as the /admin -> /admin/ redirect.
        add_mcp_http_route(app, mcp_http_app)
        app.state.mcp_http_app = mcp_http_app
    return app

app = create_app()
