"""Authenticated image retrieval; never expose a public image directory."""
from fastapi import APIRouter, Depends, HTTPException, Query, UploadFile, File
from fastapi.responses import Response
from src.auth.dependencies import require_auth
from src.auth.models import UserContext
from src.api.routes_ingest import get_metadata_store, get_vector_store
from src.figures.service import image_bytes, search_diagrams

router = APIRouter(prefix="/api/v1", tags=["figures"])
admin_router = APIRouter(prefix="/admin/api")


@router.get("/diagrams/search")
async def search(query: str = Query(min_length=1, max_length=5000), top_k: int = Query(5, ge=1, le=20),
                 doc_id: str | None = None, kind: str | None = None, user: UserContext = Depends(require_auth)):
    return await search_diagrams(query, user.groups, get_vector_store(), get_metadata_store(), top_k, doc_id, kind)


async def content(doc_id, figure_id, groups, variant):
    try:
        raw, ref = await image_bytes(doc_id, figure_id, groups, get_metadata_store(), variant)
    except (OSError, ValueError):
        raise HTTPException(404, "Figure not found") from None
    return Response(raw, media_type="image/png", headers={"Cache-Control": "private, no-store", "X-Content-Type-Options": "nosniff"})


@router.get("/documents/{doc_id}/figures/{figure_id}/content")
async def figure_content(doc_id: str, figure_id: str, variant: str = "preview", user: UserContext = Depends(require_auth)):
    return await content(doc_id, figure_id, user.groups, variant)


@admin_router.get("/figure-documents/{doc_id}/figures/{figure_id}/content")
async def admin_figure_content(doc_id: str, figure_id: str, variant: str = "preview"):
    # Entire /admin prefix is protected by EndpointAuthenticationMiddleware.
    return await content(doc_id, figure_id, ["ALL"], variant)


@admin_router.get("/figures/search")
async def admin_search(query: str = Query(min_length=1, max_length=5000), top_k: int = Query(5, ge=1, le=20), groups: str = "ALL"):
    return await search_diagrams(query, [g.strip() for g in groups.split(",") if g.strip()], get_vector_store(), get_metadata_store(), top_k)


@admin_router.post("/figure-documents/{doc_id}/backfill")
async def admin_backfill(doc_id: str, file: UploadFile = File(...)):
    from src.ingestion.uploads import save_upload
    from src.figures.backfill import backfill_figures
    path = await save_upload(file)
    try:
        return await backfill_figures(doc_id, path, get_metadata_store(), get_vector_store())
    except ValueError as exc:
        raise HTTPException(409, str(exc)) from None
    finally:
        path.unlink(missing_ok=True)


@admin_router.get("/figures/inventory")
async def inventory():
    store = get_metadata_store()
    docs = await store.list_documents()
    figures = await store.list_figures([d.doc_id for d in docs])
    result = []
    for doc in docs:
        own = [r for r in figures if r["doc_id"] == doc.doc_id]
        unique = {a["key"]: a["bytes"] for r in own for a in r.get("assets", {}).values()}
        result.append({"doc_id": doc.doc_id, "filename": doc.filename, "figures": len(own),
                       "bytes": sum(unique.values()), "analyzed": sum(r.get("analysis_status") == "complete" for r in own),
                       "can_backfill": bool(doc.content_hash) and not own,
                       "warnings": (doc.metadata_tags or {}).get("ingestion_warnings", [])})
    return result
