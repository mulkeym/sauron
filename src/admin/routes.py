import asyncio
import hashlib
import json
import logging
import secrets
import tempfile
from pathlib import Path
from typing import List
from fastapi import APIRouter, Request, Form, UploadFile, File, HTTPException
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from src.api.routes_ingest import get_metadata_store, get_vector_store, get_schema_registry
from src.config import settings
from src.ingestion.queue import ingest_queue

router = APIRouter(prefix="/admin", tags=["admin"])
TEMPLATE_DIR = Path(__file__).parent / "templates"
templates = Jinja2Templates(directory=str(TEMPLATE_DIR))

# Session management — simple signed cookie
_session_secret = secrets.token_hex(32)
_active_sessions: set[str] = set()


def _create_session() -> str:
    token = secrets.token_hex(32)
    _active_sessions.add(token)
    return token


def _is_authenticated(request: Request) -> bool:
    token = request.cookies.get("sauron_session", "")
    return token in _active_sessions


def _require_login(request: Request):
    """Check if request is authenticated, redirect to login if not."""
    if not _is_authenticated(request):
        return RedirectResponse(url="/admin/login", status_code=302)
    return None


def _format_structured_lookup(trace: dict) -> str:
    """Render a StructuredLookupTrace dict as playground step-detail HTML.
    Shared by the live-polling and final-trace render paths."""
    import html as _h, json as _json
    if not trace:
        return "<em>No structured lookup</em>"
    gate = trace.get("gate")
    parts = [f"<strong>Decision:</strong> {_h.escape(str(trace.get('query_type', '')))} → "
             + ("gate ran" if gate else "no gate (analytical)")]
    if gate:
        rows = "".join(
            f"<div>{_h.escape(str(t))} &nbsp; {float(score):.2f} {'&#10003;' if passed else '&#10007;'}</div>"
            for t, score, passed in gate)
        parts.append(f"<strong>Tables:</strong>{rows}")
    sql = trace.get("sql", "")
    if sql:
        parts.append("<strong>SQL:</strong><pre style=\"white-space:pre-wrap; background:#0f172a; "
                     f"padding:0.5rem; border-radius:4px;\">{_h.escape(sql)}</pre>")
    status = trace.get("status", "ran")
    if status == "skipped":
        note = " (used document search instead)" if trace.get("fell_back") else ""
        parts.append(f"<strong>Result:</strong> skipped — {_h.escape(trace.get('skip_reason', ''))}{note}")
    elif status == "error":
        fb = " (fell back to map-reduce)" if trace.get("fell_back") else ""
        parts.append(f"<strong>Result:</strong> error — {_h.escape(trace.get('error', ''))}{fb}")
    else:
        rc = trace.get("row_count", 0)
        if not rc:
            parts.append("<strong>Result:</strong> 0 rows (filter matched nothing)")
        else:
            sample = _h.escape(_json.dumps(trace.get("sample_rows", []), indent=2, default=str))
            parts.append(
                f"<strong>Result:</strong> {rc} rows"
                "<details style=\"margin-top:0.3rem;\"><summary style=\"cursor:pointer;\">view sample</summary>"
                f"<pre style=\"white-space:pre-wrap; background:#0f172a; padding:0.5rem; border-radius:4px;\">{sample}</pre></details>")
    return "<br>".join(parts)


@router.get("/login", response_class=HTMLResponse)
async def login_page(request: Request):
    return templates.TemplateResponse(request, "login.html", {"error": ""})


@router.post("/login")
async def login_submit(request: Request, username: str = Form(""), password: str = Form("")):
    if username == settings.admin_username and password == settings.admin_password:
        token = _create_session()
        response = RedirectResponse(url="/admin/", status_code=302)
        response.set_cookie("sauron_session", token, httponly=True, samesite="lax", max_age=86400)
        return response
    return templates.TemplateResponse(request, "login.html", {"error": "Invalid username or password"})


@router.get("/logout")
async def logout(request: Request):
    token = request.cookies.get("sauron_session", "")
    _active_sessions.discard(token)
    response = RedirectResponse(url="/admin/login", status_code=302)
    response.delete_cookie("sauron_session")
    return response

@router.get("/", response_class=HTMLResponse)
async def dashboard(request: Request):
    redirect = _require_login(request)
    if redirect:
        return redirect
    store = get_metadata_store()
    docs = await store.list_documents()
    categories = await store.list_categories()
    proposals = await store.list_proposals(status="pending")
    # Quick stats — avoid slow table scans on every page load
    entity_count = 0
    vector_count = 0
    try:
        from pathlib import Path
        kg_file = Path("data/lightrag/kv_store_full_entities.json")
        if kg_file.exists():
            entity_count = kg_file.stat().st_size // 50  # rough estimate from file size
    except Exception:
        pass
    try:
        vs = get_vector_store()
        vector_count = vs.table.count_rows()
    except Exception:
        pass

    return templates.TemplateResponse(request, "dashboard.html", {
        "doc_count": len(docs), "category_count": len(categories),
        "pending_proposals": len(proposals), "entity_count": entity_count,
        "vector_count": vector_count,
    })

@router.get("/documents", response_class=HTMLResponse)
async def documents_page(request: Request):
    redirect = _require_login(request)
    if redirect:
        return redirect
    store = get_metadata_store()
    docs = await store.list_documents()
    apps = await store.list_datasets()
    app_map = {a.id: a.name for a in apps}
    for doc in docs:
        doc._app_name = app_map.get(getattr(doc, 'dataset_id', 0), "")
    return templates.TemplateResponse(request, "documents.html", {"documents": docs, "datasets": apps})

@router.get("/categories", response_class=HTMLResponse)
async def categories_page(request: Request):
    redirect = _require_login(request)
    if redirect:
        return redirect
    store = get_metadata_store()
    categories = await store.list_categories()
    return templates.TemplateResponse(request, "categories.html", {"categories": categories, "active": "categories"})

@router.get("/proposals", response_class=HTMLResponse)
async def proposals_page(request: Request):
    redirect = _require_login(request)
    if redirect:
        return redirect
    store = get_metadata_store()
    proposals = await store.list_proposals(status="pending")
    return templates.TemplateResponse(request, "proposals.html", {"proposals": proposals, "active": "proposals"})

@router.get("/datasets", response_class=HTMLResponse)
async def datasets_page(request: Request):
    redirect = _require_login(request)
    if redirect:
        return redirect
    store = get_metadata_store()
    apps = await store.list_datasets(active_only=False)
    # Count docs per app
    docs = await store.list_documents(None)
    for app in apps:
        app.doc_count = sum(1 for d in docs if getattr(d, 'dataset_id', 0) == app.id)
    return templates.TemplateResponse(request, "datasets.html", {"datasets": apps})


@router.post("/api/datasets/create")
async def create_dataset(
    name: str = Form(""), slug: str = Form(""),
    description: str = Form(""), default_acl_groups: str = Form(""),
):
    if not name.strip() or not slug.strip():
        return HTMLResponse('<span class="status-err">Name and slug are required.</span>')

    slug_clean = slug.strip().lower().replace(" ", "-")
    acl = [g.strip() for g in default_acl_groups.split(",") if g.strip()]

    store = get_metadata_store()
    result = await store.add_dataset(
        name=name.strip(), slug=slug_clean, description=description.strip(),
        default_acl_groups=acl,
    )
    if result is None:
        return HTMLResponse(f'<span class="status-err">Slug "{slug_clean}" already exists.</span>')
    return HTMLResponse(f'<span class="status-ok">Dataset "{name}" created. Reload to see it.</span>')


@router.delete("/api/datasets/{app_id}")
async def deactivate_dataset(app_id: int):
    store = get_metadata_store()
    app = await store.get_dataset(app_id)
    if app:
        from sqlalchemy import update as sql_update
        from src.db.models import Dataset
        async with store.session_factory() as session:
            await session.execute(sql_update(Dataset).where(Dataset.id == app_id).values(active=False))
            await session.commit()
    return HTMLResponse(f'<tr><td colspan="7" style="color:#6b7280;">Deactivated</td></tr>')


@router.get("/connectors", response_class=HTMLResponse)
async def connectors_page(request: Request):
    redirect = _require_login(request)
    if redirect:
        return redirect
    store = get_metadata_store()
    connectors = await store.list_web_connectors(active_only=False)
    apps = await store.list_datasets()
    return templates.TemplateResponse(request, "connectors.html", {"connectors": connectors, "datasets": apps, "active": "connectors"})


@router.post("/api/connectors/create")
async def create_connector(
    name: str = Form(""), base_url: str = Form(""),
    dataset_id: int = Form(0), category: str = Form(""),
    acl_groups: str = Form(""), crawl_depth: int = Form(1),
    url_pattern: str = Form(""), max_pages: int = Form(100),
    additional_urls: str = Form(""), download_file_types: str = Form(""),
):
    if not name.strip() or not base_url.strip():
        return HTMLResponse('<span class="status-err">Name and URL are required.</span>')

    groups = [g.strip() for g in acl_groups.split(",") if g.strip()]
    extra_urls = [u.strip() for u in additional_urls.split("\n") if u.strip()]
    dl_types = [t.strip().lstrip(".") for t in download_file_types.split(",") if t.strip()]
    store = get_metadata_store()

    # Inherit ACL from dataset if not specified
    if not groups and dataset_id > 0:
        app = await store.get_dataset(dataset_id)
        if app and app.default_acl_groups:
            groups = app.default_acl_groups

    conn = await store.add_web_connector(
        name=name.strip(), base_url=base_url.strip(),
        dataset_id=dataset_id, category=category.strip(),
        acl_groups=groups, crawl_depth=crawl_depth,
        url_pattern=url_pattern.strip(), max_pages=max_pages,
        additional_urls=extra_urls, download_file_types=dl_types,
    )
    return HTMLResponse(f'<span class="status-ok">Connector "{name}" created. Reload to see it.</span>')


@router.post("/api/connectors/{connector_id}/update")
async def update_connector(
    connector_id: int,
    name: str = Form(""), base_url: str = Form(""),
    dataset_id: int = Form(0), category: str = Form(""),
    acl_groups: str = Form(""), crawl_depth: int = Form(1),
    url_pattern: str = Form(""), max_pages: int = Form(100),
    additional_urls: str = Form(""), download_file_types: str = Form(""),
):
    if not name.strip() or not base_url.strip():
        return HTMLResponse('<span class="status-err">Name and URL are required.</span>')

    groups = [g.strip() for g in acl_groups.split(",") if g.strip()]
    extra_urls = [u.strip() for u in additional_urls.split("\n") if u.strip()]
    dl_types = [t.strip().lstrip(".") for t in download_file_types.split(",") if t.strip()]
    store = get_metadata_store()

    if not groups and dataset_id > 0:
        app = await store.get_dataset(dataset_id)
        if app and app.default_acl_groups:
            groups = app.default_acl_groups

    await store.update_web_connector(
        connector_id,
        name=name.strip(), base_url=base_url.strip(),
        dataset_id=dataset_id, category=category.strip(),
        acl_groups=groups, crawl_depth=crawl_depth,
        url_pattern=url_pattern.strip(), max_pages=max_pages,
        additional_urls=extra_urls, download_file_types=dl_types,
    )
    return HTMLResponse(f'<span class="status-ok">Connector "{name}" updated. Reload to see changes.</span>')


_active_crawls: dict = {}  # connector_id -> {name, status, pages_found, pages_ingested, errors, started_at}


@router.post("/api/connectors/{connector_id}/crawl")
async def crawl_connector_now(connector_id: int):
    import asyncio, time
    store = get_metadata_store()
    conn = await store.get_web_connector(connector_id)
    if not conn:
        return HTMLResponse('<span class="status-err">Connector not found.</span>')

    if connector_id in _active_crawls and _active_crawls[connector_id]["status"] == "crawling":
        return HTMLResponse('<span style="color:#f59e0b;">Already crawling.</span>')

    _active_crawls[connector_id] = {
        "name": conn.name, "status": "crawling",
        "pages_found": 0, "pages_ingested": 0, "errors": [],
        "started_at": time.time(),
    }

    async def _run_crawl():
        from src.ingestion.web_crawler import crawl_connector
        await ingest_queue.start_worker(get_vector_store(), store)
        result = await crawl_connector(
            conn, store, ingest_queue, get_vector_store(),
            progress_callback=lambda stats: _active_crawls[connector_id].update(stats),
        )
        _active_crawls[connector_id]["status"] = "complete"
        _active_crawls[connector_id].update(result)
        logger_name = logging.getLogger(__name__)
        logger_name.info(f"Crawl complete: {result}")

    asyncio.create_task(_run_crawl())
    return HTMLResponse('<span style="color:#2563eb;">Crawling... check Queue for progress.</span>')


@router.get("/audit", response_class=HTMLResponse)
async def audit_page(request: Request):
    redirect = _require_login(request)
    if redirect:
        return redirect
    from src.config import settings
    entries = []
    log_path = Path(settings.audit_log_path)
    if log_path.exists():
        lines = log_path.read_text().strip().split("\n")
        for line in reversed(lines[-100:]):
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return templates.TemplateResponse(request, "audit.html", {"entries": entries, "active": "audit"})

async def _recategorize_uncategorized(store):
    """Sweep uncategorized documents and try to assign them to existing categories."""
    import asyncio
    from src.knowledge.categorizer import categorize_document
    docs = await store.list_documents(None)
    uncategorized = [d for d in docs if d.category == "uncategorized"]
    recategorized = 0
    for doc in uncategorized:
        try:
            # Get text preview from first chunk
            from src.retrieval.vector_store import VectorStore
            from src.ingestion.embedder import embed_query
            vs = VectorStore()
            vector = embed_query(f"document {doc.doc_id}")
            chunks = vs.search(vector=vector, user_groups=["ALL"], top_k=1)
            preview = chunks[0].text[:500] if chunks else doc.filename

            cat_result = await asyncio.to_thread(
                categorize_document,
                filename=doc.filename, doc_type=doc.doc_type,
                text_preview=preview, metadata_store=store,
            )
            if not cat_result.is_new and cat_result.category != "uncategorized" and cat_result.confidence >= 0.5:
                await store.update_document_category(doc.doc_id, cat_result.category)
                recategorized += 1
                logger.info(f"Re-categorized '{doc.filename}' → '{cat_result.category}' (confidence: {cat_result.confidence})")
        except Exception as e:
            logger.error(f"Re-categorization failed for {doc.filename}: {e}")
    return recategorized


@router.post("/api/proposals/{proposal_id}/approve")
async def approve_proposal(proposal_id: int):
    store = get_metadata_store()
    await store.approve_proposal(proposal_id, approved_by="admin")
    recategorized = await _recategorize_uncategorized(store)
    extra = f" Re-categorized {recategorized} document(s)." if recategorized else ""
    return HTMLResponse(f"<tr><td colspan='6'>Approved.{extra}</td></tr>")

@router.post("/api/proposals/{proposal_id}/reject")
async def reject_proposal(proposal_id: int):
    store = get_metadata_store()
    await store.reject_proposal(proposal_id, rejected_by="admin")
    return HTMLResponse("<tr><td colspan='6'>Rejected</td></tr>")

@router.post("/api/categories/add")
async def add_category(
    name: str = Form(""),
    description: str = Form(""),
    acl_groups: str = Form(""),
    routing_keywords: str = Form(""),
    grs_number: str = Form(""),
):
    if not name.strip():
        return HTMLResponse('<span class="status-err">Category name is required.</span>')
    store = get_metadata_store()
    existing = await store.get_category(name.strip())
    if existing:
        return HTMLResponse(f'<span class="status-err">Category "{name}" already exists.</span>')
    groups = [g.strip() for g in acl_groups.split(",") if g.strip()]
    keywords = [k.strip() for k in routing_keywords.split(",") if k.strip()]
    await store.add_category(name=name.strip(), description=description.strip(), acl_groups=groups, routing_keywords=keywords, grs_number=grs_number.strip())
    recategorized = await _recategorize_uncategorized(store)
    extra = f" Re-categorized {recategorized} uncategorized document(s)." if recategorized else ""
    return HTMLResponse(f'<span class="status-ok">Category "{name}" created.{extra} Reload to see it in the table.</span>')


@router.get("/api/categories/{name}/edit")
async def edit_category_form(name: str):
    store = get_metadata_store()
    cat = await store.get_category(name)
    if not cat:
        return HTMLResponse("<tr><td colspan='6'>Category not found</td></tr>")
    acl = ", ".join(cat.acl_groups) if cat.acl_groups else ""
    keywords = ", ".join(cat.routing_keywords) if cat.routing_keywords else ""
    grs = getattr(cat, "grs_number", "") or ""
    return HTMLResponse(f"""<tr id="cat-row-{name}">
        <form hx-post="/admin/api/categories/{name}/update" hx-target="#cat-row-{name}" hx-swap="outerHTML">
        <td><strong>{name}</strong></td>
        <td><input type="text" name="description" value="{cat.description}" style="width:100%;"></td>
        <td><input type="text" name="grs_number" value="{grs}" style="width:60px;"></td>
        <td><input type="text" name="acl_groups" value="{acl}" style="width:100%;"></td>
        <td><input type="text" name="routing_keywords" value="{keywords}" style="width:100%;"></td>
        <td>
            <button type="submit">Save</button>
            <button type="button" class="secondary" onclick="location.reload()">Cancel</button>
        </td>
        </form>
    </tr>""")


@router.post("/api/categories/{name}/update")
async def update_category(
    name: str,
    description: str = Form(""),
    grs_number: str = Form(""),
    acl_groups: str = Form(""),
    routing_keywords: str = Form(""),
):
    store = get_metadata_store()
    from sqlalchemy import select
    from src.db.models import Category
    async with store.session_factory() as session:
        result = await session.execute(select(Category).where(Category.name == name))
        cat = result.scalar_one_or_none()
        if not cat:
            return HTMLResponse(f"<tr><td colspan='6'>Category '{name}' not found</td></tr>")
        cat.description = description.strip()
        cat.grs_number = grs_number.strip()
        cat.acl_groups = [g.strip() for g in acl_groups.split(",") if g.strip()]
        cat.routing_keywords = [k.strip() for k in routing_keywords.split(",") if k.strip()]
        await session.commit()

    # Return the updated read-only row
    acl_display = ", ".join(cat.acl_groups)
    kw_display = ", ".join(cat.routing_keywords)
    return HTMLResponse(f"""<tr id="cat-row-{name}">
        <td>{name}</td>
        <td>{cat.description}</td>
        <td>{cat.grs_number or '-'}</td>
        <td>{acl_display}</td>
        <td>{kw_display}</td>
        <td>
            <button hx-get="/admin/api/categories/{name}/edit" hx-target="#cat-row-{name}" hx-swap="outerHTML">Edit</button>
            <button class="danger" hx-delete="/admin/api/categories/{name}" hx-confirm="Delete '{name}'?" hx-target="closest tr" hx-swap="outerHTML">Delete</button>
        </td>
    </tr>""")


@router.delete("/api/categories/{name}")
async def delete_category(name: str):
    store = get_metadata_store()
    from sqlalchemy import delete as sql_delete
    from src.db.models import Category
    async with store.session_factory() as session:
        await session.execute(sql_delete(Category).where(Category.name == name))
        await session.commit()
    return HTMLResponse("")


# ---- Table hints (extensible text-to-SQL domain knowledge) ----
# Impl functions are the testable core; routes are thin wrappers. Like the other
# admin /api routes, these rely on the /admin-prefix protection (no per-route auth).

async def create_hint_impl(scope_type, scope_value, hint_type, target_column,
                           payload, provenance="curated", confidence=1.0, created_by=""):
    """Persist one SchemaHint and register it live. Returns its id."""
    from src.api.routes_ingest import get_metadata_store, get_hint_store
    from src.db.hint_store import SchemaHint
    ms = get_metadata_store()
    hint = SchemaHint(scope_type=scope_type, scope_value=scope_value, hint_type=hint_type,
                      target_column=target_column, payload=payload or {},
                      provenance=provenance, confidence=confidence, created_by=created_by)
    hint.id = await ms.save_hint(hint)
    get_hint_store().register(hint)
    return hint.id


async def bulk_import_hints_impl(items, created_by=""):
    """Persist + register a list of hint dicts (see create_hint_impl args).
    Returns the count imported. Skips malformed entries."""
    n = 0
    for it in items:
        try:
            await create_hint_impl(
                scope_type=it["scope_type"], scope_value=it["scope_value"],
                hint_type=it["hint_type"], target_column=it.get("target_column"),
                payload=it.get("payload") or {}, provenance=it.get("provenance", "curated"),
                confidence=it.get("confidence", 1.0), created_by=created_by)
            n += 1
        except Exception:
            continue
    return n


async def delete_hint_impl(hint_id):
    """Delete a hint by id and rebuild the live HintStore from persistence."""
    from src.api.routes_ingest import get_metadata_store, get_hint_store
    from src.ingestion.tabular_ingest import populate_hint_store
    await get_metadata_store().delete_hint(int(hint_id))
    store = get_hint_store()
    store.clear()
    await populate_hint_store(get_metadata_store(), store)


@router.post("/api/hints")
async def create_hint_route(body: dict):
    hid = await create_hint_impl(
        scope_type=body["scope_type"], scope_value=body["scope_value"],
        hint_type=body["hint_type"], target_column=body.get("target_column"),
        payload=body.get("payload") or {}, provenance=body.get("provenance", "curated"),
        confidence=body.get("confidence", 1.0), created_by=body.get("created_by", "admin"))
    return {"id": hid}


@router.post("/api/hints/bulk")
async def bulk_import_hints_route(body: dict):
    return {"imported": await bulk_import_hints_impl(body.get("hints", []),
                                                     created_by=body.get("created_by", "admin"))}


@router.get("/api/hints")
async def list_hints_route():
    from src.api.routes_ingest import get_metadata_store
    hints = await get_metadata_store().load_all_hints()
    return {"hints": [vars(h) | {"created_at": str(h.created_at)} for h in hints]}


def _build_hints_view(hints, datasets):
    """Group SchemaHints by scope into a template-ready view-model. Pure/sync.
    Returns [{"scope_label": str, "hints": [hint-dict, ...]}, ...] with category
    scopes before dataset scopes. hint-dict carries hint_type/target_column/
    provenance/confidence, plus either entries+count (value_glossary) or text (notes)."""
    ds_name = {str(d.id): d.name for d in datasets}

    def scope_label(h):
        if h.scope_type == "dataset":
            name = ds_name.get(str(h.scope_value))
            return f"dataset = {name} (id {h.scope_value})" if name else f"dataset = {h.scope_value}"
        return f"category = {h.scope_value}"

    type_order = {"table_note": 0, "column_note": 1, "value_glossary": 2}
    groups: dict[str, list] = {}
    for h in hints:
        groups.setdefault(scope_label(h), []).append(h)

    out = []
    for label in sorted(groups, key=lambda l: (0 if l.startswith("category") else 1, l.lower())):
        hint_dicts = []
        for h in sorted(groups[label], key=lambda h: (type_order.get(h.hint_type, 9), h.target_column or "")):
            d = {"hint_type": h.hint_type, "target_column": h.target_column,
                 "provenance": h.provenance, "confidence": h.confidence}
            if h.hint_type == "value_glossary":
                payload = h.payload if isinstance(h.payload, dict) else {}
                d["entries"] = [{"code": k, "meaning": payload[k]} for k in sorted(payload)]
                d["count"] = len(d["entries"])
            else:
                d["text"] = h.payload.get("text", "") if isinstance(h.payload, dict) else ""
            hint_dicts.append(d)
        out.append({"scope_label": label, "hints": hint_dicts})
    return out


@router.get("/hints", response_class=HTMLResponse)
async def hints_page(request: Request):
    redirect = _require_login(request)
    if redirect:
        return redirect
    store = get_metadata_store()
    hints = await store.load_all_hints()
    datasets = await store.list_datasets(active_only=False)
    return templates.TemplateResponse(request, "hints.html",
                                      {"groups": _build_hints_view(hints, datasets), "active": "hints"})


@router.delete("/api/hints/{hint_id}")
async def delete_hint_route(hint_id: int):
    await delete_hint_impl(hint_id)
    return {"ok": True}


@router.get("/api/documents/{doc_id}/edit")
async def edit_document_form(doc_id: str):
    store = get_metadata_store()
    doc = await store.get_document(doc_id)
    if not doc:
        return HTMLResponse('<span class="status-err">Document not found</span>')
    cats = await store.list_categories()
    cat_options = "".join(
        f'<option value="{c.name}" {"selected" if c.name == doc.category else ""}>{c.name}</option>'
        for c in cats
    )
    cat_options = f'<option value="uncategorized" {"selected" if doc.category == "uncategorized" else ""}>uncategorized</option>' + cat_options
    acl_str = ", ".join(doc.acl_groups) if doc.acl_groups else ""
    return HTMLResponse(f"""<tr>
        <td>{doc.filename}</td><td>{doc.doc_type}</td>
        <td><select name="category" form="edit-{doc_id}">{cat_options}</select></td>
        <td><input type="text" name="acl_groups" form="edit-{doc_id}" value="{acl_str}" placeholder="group1, group2" style="width:100%;"></td>
        <td>{doc.chunk_count}</td><td>{doc.uploaded_by}</td>
        <td>
            <form id="edit-{doc_id}" hx-put="/admin/api/documents/{doc_id}" hx-target="closest tr" hx-swap="outerHTML" style="display:inline;">
                <button type="submit" class="small">Save</button>
            </form>
            <button class="small" hx-get="/admin/api/documents/{doc_id}/row" hx-target="closest tr" hx-swap="outerHTML">Cancel</button>
        </td>
    </tr>""")


@router.get("/api/documents/{doc_id}/row")
async def document_row(doc_id: str):
    store = get_metadata_store()
    doc = await store.get_document(doc_id)
    if not doc:
        return HTMLResponse("")
    acl = ", ".join(doc.acl_groups) if doc.acl_groups else ""
    return HTMLResponse(f"""<tr>
        <td>{doc.filename}</td><td>{doc.doc_type}</td>
        <td>{doc.category or "uncategorized"}</td>
        <td>{acl}</td>
        <td>{doc.chunk_count}</td><td>{doc.uploaded_by}</td>
        <td>
            <button class="small" hx-get="/admin/api/documents/{doc_id}/edit" hx-target="closest tr" hx-swap="outerHTML">Edit</button>
            <button class="danger small" hx-delete="/admin/api/documents/{doc_id}" hx-confirm="Delete this document?" hx-target="closest tr" hx-swap="outerHTML">Delete</button>
        </td>
    </tr>""")


@router.put("/api/documents/{doc_id}")
async def update_document(doc_id: str, category: str = Form(""), acl_groups: str = Form("")):
    store = get_metadata_store()
    groups = [g.strip() for g in acl_groups.split(",") if g.strip()]
    await store.update_document(doc_id, category=category or "uncategorized", acl_groups=groups)
    doc = await store.get_document(doc_id)
    acl = ", ".join(doc.acl_groups) if doc.acl_groups else ""
    return HTMLResponse(f"""<tr class="upload-ok">
        <td>{doc.filename}</td><td>{doc.doc_type}</td>
        <td>{doc.category or "uncategorized"}</td>
        <td>{acl}</td>
        <td>{doc.chunk_count}</td><td>{doc.uploaded_by}</td>
        <td>
            <button class="small" hx-get="/admin/api/documents/{doc_id}/edit" hx-target="closest tr" hx-swap="outerHTML">Edit</button>
            <button class="danger small" hx-delete="/admin/api/documents/{doc_id}" hx-confirm="Delete this document?" hx-target="closest tr" hx-swap="outerHTML">Delete</button>
        </td>
    </tr>""")


_kg_delete_queue: list[str] = []  # doc_ids pending KG cleanup


@router.delete("/api/documents/{doc_id}")
async def delete_document(doc_id: str):
    import asyncio
    store = get_metadata_store()
    # Remove from metadata DB
    await store.delete_document(doc_id)
    # Remove entity mentions and relationships for this doc
    await store.delete_entities_for_doc(doc_id)
    # Remove vector chunks from LanceDB
    vector_store = get_vector_store()
    vector_store.delete_by_doc_id(doc_id)
    # Remove any structured (DuckDB) tables + registered schemas for this doc
    from src.ingestion.tabular_ingest import cleanup_spreadsheet_tables
    await cleanup_spreadsheet_tables(doc_id, store, get_schema_registry())
    # Queue KG cleanup in background — adelete_by_doc_id is slow (rebuilds entities)
    _kg_delete_queue.append(doc_id)
    asyncio.create_task(_process_kg_deletes())
    return HTMLResponse("")


_kg_delete_running = False


async def _process_kg_deletes():
    """Process queued KG deletions in background. Purges if no docs remain."""
    global _kg_delete_running
    if _kg_delete_running:
        return  # already running, will pick up new items
    _kg_delete_running = True
    _logger = logging.getLogger(__name__)
    try:
        # Check if all documents were deleted — if so, just purge the whole graph
        store = get_metadata_store()
        remaining_docs = await store.list_documents()
        if not remaining_docs:
            _logger.info(f"KG cleanup: no documents remain, purging entire knowledge graph ({len(_kg_delete_queue)} queued deletes skipped)")
            _kg_delete_queue.clear()
            import shutil
            lightrag_dir = Path("data/lightrag")
            if lightrag_dir.exists():
                shutil.rmtree(str(lightrag_dir))
                lightrag_dir.mkdir(exist_ok=True)
            from src.knowledge import graph_rag
            graph_rag._rag_instance = None
            graph_rag._initialized = False
            return

        from src.knowledge.graph_rag import get_lightrag
        rag = await get_lightrag()
        while _kg_delete_queue:
            doc_id = _kg_delete_queue.pop(0)
            try:
                _logger.info(f"KG cleanup: deleting {doc_id} ({len(_kg_delete_queue)} remaining)")
                await asyncio.wait_for(rag.adelete_by_doc_id(doc_id), timeout=120)
            except asyncio.TimeoutError:
                _logger.warning(f"KG delete timed out for {doc_id}, skipping")
            except Exception as e:
                _logger.warning(f"KG delete failed for {doc_id}: {e}")
    finally:
        _kg_delete_running = False


@router.get("/playground", response_class=HTMLResponse)
async def playground_page(request: Request):
    redirect = _require_login(request)
    if redirect:
        return redirect
    store = get_metadata_store()
    apps = await store.list_datasets()
    return templates.TemplateResponse(request, "playground.html", {"datasets": apps})


_playground_jobs: dict = {}


@router.post("/api/playground/start")
async def playground_start(question: str = Form(""), play_user: str = Form("finance"), mode: str = Form("full"), app_id: int = Form(0), skip_cache: str = Form("false")):
    import uuid, asyncio
    from fastapi.responses import JSONResponse

    if not question.strip():
        return JSONResponse({"error": "No question"})

    query_id = str(uuid.uuid4())[:8]
    user_groups = [g.strip() for g in play_user.split(",") if g.strip()]

    # Look up doc_ids for the selected dataset, filtered by user's ACL groups
    allowed_doc_ids = None
    if app_id:
        store = get_metadata_store()
        docs = await store.list_documents(user_groups)
        allowed_doc_ids = [d.doc_id for d in docs if d.dataset_id == app_id]

    _playground_jobs[query_id] = {"step": "classify", "result_html": "", "error": "", "step_detail": "", "completed_steps": []}

    def _format_live_step(node_name, node_output, current_state):
        """Generate live step detail HTML for the polling UI."""
        import html as _h
        output = dict(node_output) if isinstance(node_output, dict) else {}

        if node_name == "classify":
            qt = output.get("query_type", "")
            subs = output.get("sub_tasks", [])
            detail = f"<strong>Strategy:</strong> {qt}"
            if subs:
                detail += "<br><strong>Sub-tasks:</strong> " + ", ".join(subs[:5])
            sm = output.get("strategy_memory") or {}
            if sm.get("overrode"):
                detail += f"<br><strong>Strategy memory override:</strong> {sm.get('llm_pick')} &rarr; {sm.get('memory_best')} (n={sm.get('count')}, margin={sm.get('margin')})"
            return detail
        elif node_name == "retrieve":
            rc = output.get("retrieved_chunks", [])
            SYNTHETIC = {"map-reduce", "knowledge-graph", "metadata-context"}
            regular = [c for c in rc if hasattr(c, 'metadata') and c.metadata.doc_id not in SYNTHETIC]
            synthetic = [c for c in rc if hasattr(c, 'metadata') and c.metadata.doc_id in SYNTHETIC]

            # Group regular chunks by document
            from collections import OrderedDict
            docs = OrderedDict()
            for c in regular:
                fn = c.metadata.filename
                if fn not in docs:
                    docs[fn] = []
                docs[fn].append(c)

            detail = f"<strong>Chunks retrieved:</strong> {len(regular)} from {len(docs)} documents"

            # Each document as a collapsible section with its chunks
            for fn, chunks_list in docs.items():
                chunks_list.sort(key=lambda c: c.metadata.chunk_index)
                top_score = max(c.score for c in chunks_list) if chunks_list else 0
                tier = getattr(chunks_list[0].metadata, 'chunk_size_tier', '')
                tier_label = f" [{tier}]" if tier else ''
                header = f"{fn}{tier_label} — {len(chunks_list)} chunks, relevance: {top_score:.2f}"
                chunks_html = ""
                for c in chunks_list:
                    snippet = _h.escape(c.text)
                    page = f" (page {c.metadata.page})" if c.metadata.page else ""
                    chunks_html += f'<div style="margin:0.3rem 0; padding:0.4rem 0.6rem; background:#1e293b; border-radius:4px; font-size:0.8rem;"><strong>Chunk {c.metadata.chunk_index}{page}</strong> <span style="opacity:0.6;">score: {c.score:.3f}</span><pre style="white-space:pre-wrap; margin:0.2rem 0 0 0; color:#e2e8f0;">{snippet}</pre></div>'
                detail += f'<details style="margin-top:0.4rem;"><summary style="cursor:pointer; font-weight:600; font-size:0.9rem;">{_h.escape(header)}</summary>{chunks_html}</details>'

            # Show map-reduce synthesis if present
            mr_chunks = [c for c in synthetic if c.metadata.doc_id == "map-reduce"]
            if mr_chunks:
                mr_text = mr_chunks[0].text
                detail += f'<br><details open><summary style="cursor:pointer; font-weight:600;">Map-Reduce Extraction ({len(mr_text):,} chars)</summary><pre style="font-size:0.8rem; white-space:pre-wrap; max-height:500px; overflow-y:auto; background:#1e293b; color:#e2e8f0; padding:0.5rem; border-radius:4px;">{_h.escape(mr_text)}</pre></details>'
            return detail
        elif node_name == "enrich":
            rc = output.get("retrieved_chunks", [])
            kg = [c for c in rc if hasattr(c, 'metadata') and c.metadata.filename == 'knowledge_graph']
            if kg:
                kg_text = kg[0].text
                return f'<details open><summary style="cursor:pointer; font-weight:600;">Knowledge Graph Context ({len(kg_text):,} chars)</summary><pre style="font-size:0.8rem; white-space:pre-wrap; max-height:500px; overflow-y:auto; background:#1e293b; color:#e2e8f0; padding:0.5rem; border-radius:4px;">{_h.escape(kg_text)}</pre></details>'
            return "<em>No graph enrichment</em>"
        elif node_name == "synthesize":
            return "<strong>Generating answer...</strong>"
        return ""

    async def run_query():
        try:
            import time, html as html_mod
            import asyncio as _asyncio

            # Check query cache first (unless skip_cache is set) — shared decision
            # (same embed -> lookup -> judge path the public query API uses).
            _skip_cache = skip_cache == "true"
            _playground_jobs[query_id]["step"] = "cache_check"
            from src.retrieval.query_cache import judged_cache_lookup, cache_store
            _decision = await judged_cache_lookup(question, user_groups, skip_cache=_skip_cache)
            query_vector = _decision.query_vector
            cached = _decision.cached
            cache_time = _decision.cache_time
            judge_time = _decision.judge_time
            judgment = _decision.judgment or {}
            cache_accepted = _decision.accepted

            if cached and not cache_accepted:
                # Judge rejected — clear cached so we fall through to full pipeline
                cached = None
                _playground_jobs[query_id]["step"] = "classify"

            if cache_accepted:
                from src.retrieval.models import Citation
                cached_age = time.time() - cached.get("cached_at", 0)
                if cached_age < 3600:
                    age_str = f"{int(cached_age)}s ago"
                elif cached_age < 86400:
                    age_str = f"{int(cached_age/3600)}h ago"
                else:
                    age_str = f"{int(cached_age/86400)}d ago"

                confidence_pct = int(judgment.get("confidence", 0) * 100)
                judge_reason = html_mod.escape(judgment.get("reason", ""))

                citations = cached.get("citations", [])
                citations_html = ""
                for i, c in enumerate(citations, 1):
                    page = f' &mdash; page {c.get("page", "")}' if c.get("page") else ''
                    citations_html += f'<div class="citation-card"><span class="filename">[{i}] {c.get("filename", "")}</span>{page}<span class="score"> &mdash; relevance: {c.get("relevance", 0):.2f}</span><div class="snippet">{c.get("snippet", "")[:300]}</div></div>'

                result_html = f"""<div class="trace-panel">
                <div class="trace-header">
                    <span style="color:#16a34a; font-weight:600;">&#9889; Cache Hit</span>
                    <span>Original query: <strong>{html_mod.escape(cached.get('cached_query', ''))}</strong></span>
                    <span>Cached: <strong>{age_str}</strong></span>
                    <span>Lookup: <strong>{cache_time}s</strong></span>
                </div>
                <div class="trace-step-wrap">
                    <div class="trace-step" style="background:#16a34a; color:white;">
                        <span>&#9889; Step 1 of 5: Check Cache — <strong>HIT</strong> (similarity ≥ 92%)</span>
                        <span class="trace-time">{cache_time}s</span>
                    </div>
                    <div class="trace-detail expanded" style="padding:0.5rem 1rem; font-size:0.85rem; background:#f0fdf4; border-left:3px solid #16a34a;">
                        <strong>LLM Judge:</strong> Applicable ({confidence_pct}% confidence)<br>
                        <strong>Reason:</strong> {judge_reason}<br>
                        <strong>Judge time:</strong> {judge_time}s
                    </div>
                </div>
                <div class="trace-step-wrap">
                    <div class="trace-step" style="opacity:0.4;">
                        <span>&#9898; Step 2 of 5: Classify Query — skipped</span>
                    </div>
                </div>
                <div class="trace-step-wrap">
                    <div class="trace-step" style="opacity:0.4;">
                        <span>&#9898; Step 3 of 5: Retrieve Documents — skipped</span>
                    </div>
                </div>
                <div class="trace-step-wrap">
                    <div class="trace-step" style="opacity:0.4;">
                        <span>&#9898; Step 4 of 5: Knowledge Graph Enrichment — skipped</span>
                    </div>
                </div>
                <div class="trace-step-wrap">
                    <div class="trace-step" style="opacity:0.4;">
                        <span>&#9898; Step 5 of 5: Generate Answer — skipped</span>
                    </div>
                </div>
                </div>
                <div class="result-card">
                    <div class="result-meta">Groups: {', '.join(user_groups)} | Source: Cache</div>
                    <div class="result-answer">{cached['answer']}</div>
                    <h3 style="margin-bottom:0.5rem; font-size:0.95rem;">Citations ({len(citations)})</h3>
                    {citations_html or '<p>No citations.</p>'}
                </div>"""

                _playground_jobs[query_id] = {"step": "complete", "result_html": result_html, "error": ""}
                # Log cache hit metric
                try:
                    from src.retrieval.metrics import QueryMetricsCollector
                    m = QueryMetricsCollector(query_text=question, user_groups=user_groups, cache_hit=True,
                        total_time_seconds=round(cache_time + judge_time, 2), docs_cited=len(citations))
                    m.log_summary()
                    await m.save()
                except Exception:
                    pass
                return

            # No cache hit — proceed with full pipeline
            _playground_jobs[query_id]["step"] = "classify"

            # Graph-only mode: bypass the full pipeline, use LightRAG directly
            if mode == "graph_only":
                _playground_jobs[query_id]["step"] = "enrich"
                from src.knowledge.graph_rag import query_graph
                start_time = time.time()
                result = await query_graph(question, mode="hybrid")
                elapsed = round(time.time() - start_time, 1)
                answer = result.get("context", "No graph data available.")

                result_html = f"""<div class="trace-panel">
                    <div class="trace-header">
                        <span>Mode: <strong>Knowledge Graph Only</strong></span>
                        <span>Total: <strong>{elapsed}s</strong></span>
                    </div>
                </div>
                <div class="result-card">
                    <div class="result-meta">Groups: {', '.join(user_groups)} | Source: LightRAG</div>
                    <div class="result-answer">{answer}</div>
                </div>"""
                _playground_jobs[query_id] = {"step": "complete", "result_html": result_html, "error": ""}
                return

            from src.agent.graph import create_agent_graph
            from src.agent.state import AgentState
            from langgraph.graph import END

            # Build graph WITHOUT synthesize — we'll stream that separately
            from src.agent.classifier import classify_query
            from src.agent.strategies.lookup import retrieve_lookup
            from src.agent.strategies.sweep import retrieve_sweep
            from src.agent.strategies.analytical import retrieve_analytical
            from src.agent.strategies.cross_reference import retrieve_cross_reference
            from src.agent.state import QueryType
            from langgraph.graph import StateGraph

            vs = get_vector_store()
            sr = get_schema_registry()
            ms = get_metadata_store()

            graph = create_agent_graph(vector_store=vs, schema_registry=sr, metadata_store=ms, include_synthesize=False)

            initial_state = AgentState(
                question=question, user_groups=user_groups, query_type=None, sub_tasks=[],
                retrieved_chunks=[], sql_results=[], retrieval_attempts=0,
                needs_reretrieval=False, answer="", citations=[], warnings=[],
                skip_graph=(mode == "vector_only"),
                dataset_id=app_id or 0,
                **({"allowed_doc_ids": allowed_doc_ids} if allowed_doc_ids else {}),
            )

            # Node execution order — retrieve and enrich run in parallel after classify.
            # merge is a no-op (instant), so we skip it in the UI and jump to synthesize.
            _STEP_ORDER = ["classify", "retrieve", "enrich", "merge", "synthesize"]

            steps_data = [{"step": "cache_check", "time": cache_time, "output": {"result": "miss"}}]
            _playground_jobs[query_id]["completed_steps"].append({"step": "cache_check", "time": cache_time, "detail": f"<strong>Result:</strong> miss ({cache_time}s)"})
            step_start = time.time()
            # Seed from initial_state so base keys (question, user_groups, ...) are
            # present: astream(stream_mode="updates") only emits per-node output
            # deltas, so node updates alone never carry the original question.
            final_state = dict(initial_state)
            # Show classify as the first active step
            _playground_jobs[query_id]["step"] = "classify"

            # Track per-node start times for parallel branches
            _node_starts = {}
            _completed_nodes = set()

            async for event in graph.astream(initial_state, stream_mode="updates"):
                now = time.time()
                for node_name, node_output in event.items():
                    # Use per-node start time if available, else fall back to step_start
                    node_elapsed = round(now - _node_starts.pop(node_name, step_start), 2)
                    output = dict(node_output) if isinstance(node_output, dict) else {}
                    _completed_nodes.add(node_name)

                    # Skip the merge node in UI — it's a no-op
                    if node_name != "merge":
                        # Structured lookup is computed inside the retrieve node; emit it as
                        # its own step (displayed before Retrieve) when a trace is present.
                        if node_name == "retrieve" and output.get("structured_trace"):
                            st = output["structured_trace"]
                            try:
                                sl_detail = _format_structured_lookup(st)
                            except Exception:
                                sl_detail = "<em>(structured lookup render error)</em>"
                            steps_data.append({"step": "structured_lookup", "time": 0.0, "output": {"structured_trace": st}})
                            _playground_jobs[query_id]["completed_steps"].append(
                                {"step": "structured_lookup", "time": 0.0, "detail": sl_detail})
                        step_entry = {"step": node_name, "time": node_elapsed, "detail": _format_live_step(node_name, node_output, final_state)}
                        steps_data.append({"step": node_name, "time": node_elapsed, "output": output})
                        _playground_jobs[query_id]["completed_steps"].append(step_entry)

                    # Determine what's running next
                    if node_name == "classify":
                        # After classify, retrieve + enrich start in parallel
                        _node_starts["retrieve"] = now
                        _node_starts["enrich"] = now
                        _playground_jobs[query_id]["step"] = "retrieve"
                        _playground_jobs[query_id]["step_detail"] = ""
                    elif node_name in ("retrieve", "enrich"):
                        # If both parallel branches done, next is synthesize
                        if "retrieve" in _completed_nodes and "enrich" in _completed_nodes:
                            _playground_jobs[query_id]["step"] = "synthesize"
                            _playground_jobs[query_id]["step_detail"] = ""
                        else:
                            # Show whichever parallel branch is still running
                            still_running = "enrich" if node_name == "retrieve" else "retrieve"
                            _playground_jobs[query_id]["step"] = still_running
                            _playground_jobs[query_id]["step_detail"] = ""
                    elif node_name == "merge":
                        _playground_jobs[query_id]["step"] = "synthesize"
                        _playground_jobs[query_id]["step_detail"] = ""
                    else:
                        _playground_jobs[query_id]["step_detail"] = _format_live_step(node_name, node_output, final_state)

                    step_start = now
                    final_state.update(node_output if isinstance(node_output, dict) else {})


            # Answer is produced by streaming (graph was built without synthesize).
            has_context = bool(final_state.get("retrieved_chunks")) or bool(final_state.get("sql_results"))
            answer = "I could not find any relevant information in the documents you have access to."
            synth_start = time.time()
            if has_context:
                from src.agent.synthesizer import build_synthesis_context
                _playground_jobs[query_id]["stream_context"] = {
                    "context": build_synthesis_context(final_state),
                    "question": question,
                }
                _playground_jobs[query_id]["stream_ready"] = True
                _playground_jobs[query_id]["step"] = "streaming"

                # The SSE endpoint (opened by the frontend) streams the answer and
                # stores it back as streamed_answer. Wait for it; fall back to a
                # non-streamed generate so a closed tab can never hang the job.
                for _ in range(1500):  # ~5 min at 0.2s
                    if _playground_jobs[query_id].get("streamed_answer") is not None:
                        break
                    await asyncio.sleep(0.2)
                streamed = _playground_jobs[query_id].get("streamed_answer")
                if streamed is not None:
                    answer = streamed
                else:
                    from src.agent.synthesizer import (
                        SYSTEM_PROMPT, USER_PROMPT_TEMPLATE, _strip_reasoning_artifacts)
                    from src.generation.llm_client import generate as _gen
                    from src.config import settings as _cfg
                    ctx = _playground_jobs[query_id]["stream_context"]["context"]
                    answer = _strip_reasoning_artifacts(_gen(
                        system_prompt=SYSTEM_PROMPT,
                        user_prompt=USER_PROMPT_TEMPLATE.format(context=ctx, question=question),
                        max_tokens=_cfg.llm_max_output_tokens))

            # Synthesize wasn't a graph node; build citations + a trace step here.
            from src.agent.synthesizer import build_citations
            final_state["citations"] = build_citations(final_state) if has_context else []
            synth_elapsed = round(time.time() - synth_start, 2)
            steps_data.append({"step": "synthesize", "time": synth_elapsed,
                               "output": {"answer": answer, "citations": final_state["citations"]}})
            _playground_jobs[query_id]["completed_steps"].append(
                {"step": "synthesize", "time": synth_elapsed,
                 "detail": f"<strong>Answer length:</strong> {len(answer)} chars<br><strong>Citations:</strong> {len(final_state['citations'])}"})

            total_time = sum(s["time"] for s in steps_data)
            chunks = final_state.get("retrieved_chunks", [])
            query_type = str(final_state.get("query_type", "lookup"))

            # Build trace with expandable step details
            step_labels = {"cache_check": "Check Cache", "classify": "Classify Query", "structured_lookup": "Structured Lookup", "retrieve": "Retrieve Documents", "enrich": "Knowledge Graph", "synthesize": "Generate Answer"}

            def format_step_detail(step_name, output):
                """Format step output for display."""
                if step_name == "cache_check":
                    result = output.get("result", "miss")
                    if result == "miss":
                        return "<strong>Cache:</strong> No match found — running full pipeline"
                    return f"<strong>Cache:</strong> {result}"
                elif step_name == "structured_lookup":
                    try:
                        return _format_structured_lookup(output.get("structured_trace", {}))
                    except Exception:
                        return "<em>(structured lookup render error)</em>"
                elif step_name == "classify":
                    qt = output.get("query_type", "")
                    subs = output.get("sub_tasks", [])
                    detail = f"<strong>Query Type:</strong> {qt}<br>"
                    if subs:
                        detail += "<strong>Sub-tasks:</strong><ul>" + "".join(f"<li>{s}</li>" for s in subs) + "</ul>"
                    sm = output.get("strategy_memory") or {}
                    if sm.get("overrode"):
                        detail += f"<br><strong>Strategy memory override:</strong> {sm.get('llm_pick')} &rarr; {sm.get('memory_best')} (n={sm.get('count')}, margin={sm.get('margin')})"
                    return detail
                elif step_name == "retrieve":
                    rc = output.get("retrieved_chunks", [])
                    SYNTHETIC = {"map-reduce", "knowledge-graph", "metadata-context"}
                    regular = [c for c in rc if hasattr(c, 'metadata') and c.metadata.doc_id not in SYNTHETIC]
                    synthetic = [c for c in rc if hasattr(c, 'metadata') and c.metadata.doc_id in SYNTHETIC]

                    from collections import OrderedDict
                    docs = OrderedDict()
                    for c in regular:
                        fn = c.metadata.filename
                        if fn not in docs:
                            docs[fn] = []
                        docs[fn].append(c)

                    detail = f"<strong>Chunks retrieved:</strong> {len(regular)} from {len(docs)} documents"

                    for fn, chunks_list in docs.items():
                        chunks_list.sort(key=lambda c: c.metadata.chunk_index)
                        top_score = max(c.score for c in chunks_list) if chunks_list else 0
                        tier = getattr(chunks_list[0].metadata, 'chunk_size_tier', '')
                        tier_label = f" [{tier}]" if tier else ''
                        header = f"{fn}{tier_label} — {len(chunks_list)} chunks, relevance: {top_score:.2f}"
                        chunks_html = ""
                        for c in chunks_list:
                            snippet = html_mod.escape(c.text)
                            page = f" (page {c.metadata.page})" if c.metadata.page else ""
                            chunks_html += f'<div style="margin:0.3rem 0; padding:0.4rem 0.6rem; background:#1e293b; border-radius:4px; font-size:0.8rem;"><strong>Chunk {c.metadata.chunk_index}{page}</strong> <span style="opacity:0.6;">score: {c.score:.3f}</span><pre style="white-space:pre-wrap; margin:0.2rem 0 0 0; color:#e2e8f0;">{snippet}</pre></div>'
                        detail += f'<details style="margin-top:0.4rem;"><summary style="cursor:pointer; font-weight:600; font-size:0.9rem;">{html_mod.escape(header)}</summary>{chunks_html}</details>'

                    mr_chunks = [c for c in synthetic if c.metadata.doc_id == "map-reduce"]
                    if mr_chunks:
                        mr_text = mr_chunks[0].text
                        detail += f'<br><details open><summary style="cursor:pointer; font-weight:600;">Map-Reduce Extraction ({len(mr_text):,} chars)</summary><pre style="font-size:0.8rem; white-space:pre-wrap; max-height:500px; overflow-y:auto; background:#1e293b; color:#e2e8f0; padding:0.5rem; border-radius:4px;">{html_mod.escape(mr_text)}</pre></details>'
                    return detail
                elif step_name == "enrich":
                    rc = output.get("retrieved_chunks", [])
                    kg_chunks = [c for c in rc if (c.metadata.filename if hasattr(c, 'metadata') else '') == 'knowledge_graph']
                    if kg_chunks:
                        text = kg_chunks[0].text if hasattr(kg_chunks[0], 'text') else ''
                        return f"<strong>Knowledge graph context added:</strong><pre style='font-size:0.8rem; white-space:pre-wrap;'>{html_mod.escape(text)}</pre>"
                    return "<em>No knowledge graph enrichment applied</em>"
                elif step_name == "evaluate":
                    needs = output.get("needs_reretrieval", False)
                    return f"<strong>Sufficient context:</strong> {'No — re-retrieving' if needs else 'Yes'}"
                elif step_name == "synthesize":
                    ans = output.get("answer", "")
                    cits = output.get("citations", [])
                    return f"<strong>Answer length:</strong> {len(ans)} chars<br><strong>Citations:</strong> {len(cits)}"
                return f"<pre style='font-size:0.75rem;'>{html_mod.escape(str(output)[:500])}</pre>"

            steps_html = ""
            for i, s in enumerate(steps_data):
                label = step_labels.get(s["step"], s["step"])
                detail = format_step_detail(s["step"], s["output"])
                step_id = f"step-detail-{query_id}-{i}"
                steps_html += f"""<div class="trace-step-wrap">
                    <div class="trace-step" onclick="document.getElementById('{step_id}').classList.toggle('expanded')" style="cursor:pointer;">
                        <span>&#9989; Step {i+1} of {len(steps_data)}: {label} &#9660;</span>
                        <span class="trace-time">{s['time']}s</span>
                    </div>
                    <div id="{step_id}" class="trace-detail">{detail}</div>
                </div>"""

            trace_html = f"""<div class="trace-panel">
                <div class="trace-header">
                    <span>Query Type: <strong>{query_type}</strong></span>
                    <span>Chunks: <strong>{len(chunks)}</strong></span>
                    <span>Total: <strong>{round(total_time, 1)}s</strong></span>
                </div>
                {steps_html}
            </div>"""

            # Citations: use the synthesizer's list (final_state["citations"]) as the
            # single source of truth — it already dedupes by doc, looks up source_url,
            # excludes synthetic chunks, AND adds structured/SQL source-document
            # citations. ANALYTICAL answers carry no chunks, so rebuilding from
            # `chunks` here (as this handler used to) dropped those SQL citations.
            citations = final_state.get("citations", [])

            citations_html = ""
            for i, c in enumerate(citations, 1):
                page = f' &mdash; page {c.page}' if c.page else ''
                if c.source_url:
                    name_display = f'<a href="{c.source_url}" target="_blank" style="color:#3b82f6;">[{i}] {c.filename}</a>'
                else:
                    name_display = f'[{i}] {c.filename}'
                citations_html += f'<div class="citation-card"><span class="filename">{name_display}</span>{page}<span class="score"> &mdash; relevance: {c.relevance:.2f}</span><div class="snippet">{c.snippet[:300]}</div></div>'

            result_html = f"""{trace_html}
            <div class="result-card">
                <div class="result-meta">Groups: {', '.join(user_groups)}</div>
                <div class="result-answer">{answer}</div>
                <h3 style="margin-bottom:0.5rem; font-size:0.95rem;">Citations ({len(citations)})</h3>
                {citations_html or '<p>No citations.</p>'}
            </div>"""

            _playground_jobs[query_id] = {"step": "complete", "result_html": result_html, "error": ""}

            # Cache the result for future queries
            try:
                citation_dicts = [
                    {"doc_id": c.doc_id, "filename": c.filename, "doc_type": c.doc_type,
                     "chunk_index": c.chunk_index, "page": c.page, "snippet": c.snippet,
                     "relevance": c.relevance}
                    for c in citations
                ]
                source_ids = list({c.doc_id for c in citations})
                if query_vector is not None:  # skip store when embed failed (fail-open)
                    cache_store(
                        query_text=question, query_vector=query_vector,
                        answer=answer, citations=citation_dicts,
                        user_groups=user_groups, source_doc_ids=source_ids,
                    )
            except Exception:
                pass

            # Log query metrics
            try:
                from src.retrieval.metrics import QueryMetricsCollector
                SYNTHETIC_IDS = {"map-reduce", "knowledge-graph", "metadata-context"}
                metrics = QueryMetricsCollector(
                    query_text=question,
                    query_type=query_type,
                    strategy_used=query_type,  # sweep, lookup, etc.
                    user_groups=user_groups,
                    docs_cited=len(citations),
                    answer_length=len(answer),
                    context_chars=len(context) if 'context' in dir() else 0,
                    cache_hit=False,
                    total_time_seconds=round(total_time, 2),
                )
                # Count docs from chunks
                all_doc_ids = {c.metadata.doc_id for c in chunks if c.metadata.doc_id not in SYNTHETIC_IDS}
                metrics.docs_discovered = len(all_doc_ids)

                # Check map-reduce results for MAP precision
                mr_chunks = [c for c in chunks if c.metadata.doc_id == "map-reduce"]
                if mr_chunks:
                    mr_text = mr_chunks[0].text
                    # Count "[filename]:" entries in map-reduce output
                    import re
                    mr_entries = re.findall(r'\[([^\]]+\.(?:md|pdf|docx))\]:', mr_text)
                    metrics.docs_relevant = len(set(mr_entries))
                    metrics.docs_map_read = metrics.docs_discovered  # approximate

                # Extract step timings
                for s in steps_data:
                    if s["step"] == "retrieve":
                        metrics.retrieval_time = s["time"]
                    elif s["step"] == "synthesize":
                        metrics.synthesis_time = s["time"]

                metrics.log_summary()
                await metrics.save()
            except Exception:
                pass

            # Log relevance feedback for adaptive retrieval
            try:
                from src.retrieval.feedback import log_feedback
                SYNTHETIC_IDS = {"map-reduce", "knowledge-graph", "metadata-context"}

                cited_ids = [c.doc_id for c in citations]

                # Build filename→doc_id map
                fn_to_docid = {}
                doc_fn_map = {}
                for c in chunks:
                    if c.metadata.doc_id not in SYNTHETIC_IDS:
                        fn_to_docid[c.metadata.filename] = c.metadata.doc_id
                        doc_fn_map[c.metadata.doc_id] = c.metadata.filename

                # Parse map-reduce to find relevant vs irrelevant docs
                mr_relevant_ids = []
                mr_irrelevant_ids = []
                mr_chunks = [c for c in chunks if c.metadata.doc_id == "map-reduce"]
                if mr_chunks:
                    import re as _re
                    mr_text = mr_chunks[0].text
                    mr_filenames = set(_re.findall(r'\[([^\]]+\.(?:md|pdf|docx))\]:', mr_text))
                    mr_relevant_ids = [fn_to_docid[fn] for fn in mr_filenames if fn in fn_to_docid]
                    # Discovered docs NOT in MAP-relevant results = irrelevant
                    # (Don't include cited_ids — citations come from sweep raw chunks too)
                    relevant_set = set(mr_relevant_ids)
                    mr_irrelevant_ids = [did for did in doc_fn_map if did not in relevant_set]
                    import logging as _fb_log
                    _fb_log.getLogger(__name__).info(
                        f"Feedback: {len(mr_relevant_ids)} relevant, {len(mr_irrelevant_ids)} irrelevant, {len(cited_ids)} cited"
                    )

                # Don't log irrelevant docs as cited — they appear in citations
                # from sweep raw chunks but MAP determined they're irrelevant
                irrelevant_set = set(mr_irrelevant_ids)
                clean_cited_ids = [did for did in cited_ids if did not in irrelevant_set]

                await log_feedback(
                    query_text=question,
                    query_vector=query_vector,
                    query_type=query_type,
                    user_groups=user_groups,
                    cited_doc_ids=clean_cited_ids,
                    relevant_doc_ids=mr_relevant_ids,
                    irrelevant_doc_ids=mr_irrelevant_ids,
                    doc_filenames=doc_fn_map,
                )
            except Exception:
                pass

            # Log strategy memory
            try:
                from src.retrieval.strategy_memory import log_strategy_result
                await log_strategy_result(
                    query_text=question,
                    query_type=query_type,
                    strategy_used=query_type,
                    docs_discovered=len({c.metadata.doc_id for c in chunks if c.metadata.doc_id not in {"map-reduce", "knowledge-graph", "metadata-context"}}),
                    docs_relevant=len(mr_relevant_ids) if mr_relevant_ids else 0,
                    docs_cited=len(citations),
                    answer_length=len(answer),
                    total_time_seconds=round(total_time, 2),
                )
            except Exception:
                pass

        except Exception as e:
            import traceback
            _playground_jobs[query_id] = {"step": "error", "result_html": "", "error": f"{e}\n{traceback.format_exc()}"}

    asyncio.create_task(run_query())
    return JSONResponse({"query_id": query_id})


@router.get("/api/playground/status/{query_id}")
async def playground_status(query_id: str):
    from fastapi.responses import JSONResponse
    job = _playground_jobs.get(query_id, {"step": "error", "error": "Not found"})
    return JSONResponse(job)


@router.get("/api/playground/stream/{query_id}")
async def playground_stream(query_id: str):
    """SSE endpoint that streams the synthesized answer token by token."""
    from fastapi.responses import StreamingResponse

    # Sync generator: Starlette runs it in a threadpool, so the blocking
    # generate_stream() (synchronous `requests` to the LLM) does NOT block the
    # asyncio event loop. Running it as an `async def` starved the loop and made
    # concurrent /status polls fail ("Failed to fetch") during long generations.
    def event_stream():
        import time

        # Wait for the job to reach synthesize step with context ready
        for _ in range(1500):  # ~5 min at 0.2s
            job = _playground_jobs.get(query_id, {})
            if job.get("stream_ready"):
                break
            if job.get("step") in ("complete", "error"):
                return
            time.sleep(0.2)

        context_data = job.get("stream_context")
        if not context_data:
            return

        from src.generation.llm_client import generate_stream
        from src.agent.synthesizer import SYSTEM_PROMPT, USER_PROMPT_TEMPLATE, _strip_reasoning_artifacts
        from src.config import settings as _cfg

        try:
            full_text = ""
            for token in generate_stream(
                system_prompt=SYSTEM_PROMPT,
                user_prompt=USER_PROMPT_TEMPLATE.format(
                    context=context_data["context"],
                    question=context_data["question"],
                ),
                max_tokens=_cfg.llm_max_output_tokens,
            ):
                full_text += token
                yield f"data: {json.dumps({'token': token})}\n\n"

            # Send cleaned final answer
            cleaned = _strip_reasoning_artifacts(full_text)
            yield f"data: {json.dumps({'done': True, 'answer': cleaned})}\n\n"

            # Store the final answer back in the job for citations
            _playground_jobs[query_id]["streamed_answer"] = cleaned
        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@router.post("/api/playground/query")
async def playground_query(question: str = Form(""), play_user: str = Form("finance")):
    if not question.strip():
        return HTMLResponse('<div class="status-err">Please enter a question.</div>')

    user_groups = [g.strip() for g in play_user.split(",") if g.strip()]

    try:
        from src.agent.graph import run_agent_with_trace
        result, trace = await run_agent_with_trace(
            question=question,
            user_groups=user_groups,
            vector_store=get_vector_store(),
            schema_registry=get_schema_registry(),
            metadata_store=get_metadata_store(),
        )

        # Build trace timeline
        step_labels = {
            "classify": "Classify Query",
            "retrieve": "Retrieve Documents",
            "enrich": "Knowledge Graph",
            "synthesize": "Generate Answer",
        }
        steps_html = ""
        for s in trace.steps:
            label = step_labels.get(s["step"], s["step"])
            status_icon = "&#9989;" if s["status"] == "done" else "&#x1F504;"
            steps_html += f'<div class="trace-step"><span>{status_icon} {label}</span><span class="trace-time">{s["time"]}s</span></div>'

        trace_html = f"""
        <div class="trace-panel">
            <div class="trace-header">
                <span>Query Type: <strong>{trace.query_type or 'lookup'}</strong></span>
                <span>Chunks: <strong>{trace.chunks_retrieved}</strong></span>
                <span>Total: <strong>{trace.total_time}s</strong></span>
            </div>
            <div class="trace-steps">{steps_html}</div>
        </div>"""

        citations_html = ""
        for i, c in enumerate(result.citations, 1):
            citations_html += f"""
            <div class="citation-card">
                <span class="filename">[{i}] {c.filename}</span>
                {f'<span class="score"> &mdash; page {c.page}</span>' if c.page else ''}
                <span class="score"> &mdash; relevance: {c.relevance:.2f}</span>
                <div class="snippet">{c.snippet[:300]}</div>
            </div>"""

        return HTMLResponse(f"""
        {trace_html}
        <div class="result-card">
            <div class="result-meta">Groups: {', '.join(user_groups)}</div>
            <div class="result-answer">{result.answer}</div>
            <h3 style="margin-bottom:0.5rem; font-size:0.95rem;">Citations ({len(result.citations)})</h3>
            {citations_html or '<p>No citations.</p>'}
        </div>""")
    except Exception as e:
        import traceback
        return HTMLResponse(f'<div class="status-err">Error: {e}<br><pre style="font-size:0.75rem;">{traceback.format_exc()}</pre></div>')


@router.post("/api/documents/upload")
async def bulk_upload(
    files: List[UploadFile] = File(...),
    acl_groups: str = Form(""),
    category: str = Form(""),
    dataset_id: int = Form(0),
    auto_categorize: str = Form(""),
    build_graph: str = Form(""),
):
    groups = [g.strip() for g in acl_groups.split(",") if g.strip()]
    do_auto_cat = auto_categorize == "true"
    do_build_graph = build_graph == "true"

    # Inherit defaults from dataset if selected
    if dataset_id > 0:
        store = get_metadata_store()
        app = await store.get_dataset(dataset_id)
        if app:
            if not groups and app.default_acl_groups:
                groups = app.default_acl_groups

    # Start the queue worker if not running
    await ingest_queue.start_worker(get_vector_store(), get_metadata_store())

    job_ids = []
    for file in files:
        suffix = Path(file.filename).suffix
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = tmp.name

        job_id = ingest_queue.enqueue(
            filename=file.filename, file_path=tmp_path,
            acl_groups=groups, uploaded_by="admin",
            category=category, dataset_id=dataset_id,
            auto_categorize=do_auto_cat, build_graph=do_build_graph,
        )
        job_ids.append((file.filename, job_id))

    results = "".join(
        f'<div class="upload-result upload-ok">{fname} — queued (job {jid})</div>'
        for fname, jid in job_ids
    )
    results += '<div style="margin-top:0.5rem;"><a href="/admin/queue">View Queue Status</a></div>'
    return HTMLResponse(results)


@router.get("/queue", response_class=HTMLResponse)
async def queue_page(request: Request):
    redirect = _require_login(request)
    if redirect:
        return redirect
    jobs = ingest_queue.list_jobs()
    return templates.TemplateResponse(request, "queue.html", {"jobs": jobs})


@router.get("/api/queue/status")
async def queue_status():
    import time as _time

    # Clean up completed crawls older than 5 minutes
    stale = [cid for cid, c in _active_crawls.items()
             if c["status"] == "complete" and _time.time() - c["started_at"] > 300]
    for cid in stale:
        del _active_crawls[cid]

    # Active crawls section
    crawl_html = ""
    for cid, crawl in _active_crawls.items():
        elapsed = int(_time.time() - crawl["started_at"])
        if crawl["status"] == "crawling":
            status = '<span style="color:#2563eb; font-weight:600;">Crawling</span>'
            current = f' — {crawl.get("current_url", "")}'
        elif crawl["status"] == "complete":
            status = '<span class="status-ok">Complete</span>'
            current = ""
        else:
            status = crawl["status"]
            current = ""
        crawl_html += f"""<tr>
            <td><strong>{crawl['name']}</strong></td>
            <td>{status}</td>
            <td>{crawl['pages_found']} found / {crawl['pages_ingested']} ingested</td>
            <td>{elapsed}s</td>
        </tr>"""

    if crawl_html:
        crawl_html = f"""<h3 style="margin-bottom:0.5rem;">Active Crawls</h3>
        <table style="margin-bottom:1.5rem;">
            <thead><tr><th>Connector</th><th>Status</th><th>Progress</th><th>Time</th></tr></thead>
            <tbody>{crawl_html}</tbody>
        </table>"""

    jobs = ingest_queue.list_jobs()
    if not jobs and not crawl_html:
        return HTMLResponse('<p>No ingestion jobs.</p>')
    if not jobs:
        return HTMLResponse(crawl_html + '<p>No ingestion jobs.</p>')

    rows = ""
    for job in jobs:
        if job.step == "complete":
            status = '<span class="status-ok">Complete</span>'
        elif job.step == "failed":
            status = '<span class="status-err">Failed</span>'
        elif job.step == "queued":
            status = "Queued"
        else:
            status = f'<span style="color: #2563eb; font-weight: 600;">{job.step}</span>'

        elapsed = ""
        if job.completed_at > 0:
            elapsed = f"{job.completed_at - job.created_at:.1f}s"
        elif job.step != "queued":
            import time
            elapsed = f"{time.time() - job.created_at:.0f}s..."

        error_row = ""
        if job.step == "failed":
            error_row = f'<tr><td colspan="7" class="status-err" style="font-size:0.85rem;">{job.error}</td></tr>'

        if job.step == 'complete':
            kg_info = f"{job.entity_count} / {job.relationship_count}"
        elif job.step == 'extracting_entities':
            kg_info = f'<span style="color: #2563eb;">{job.entity_count} / {job.relationship_count}...</span>'
        else:
            kg_info = '-'

        rows += f"""<tr>
            <td>{job.filename}</td><td>{status}</td><td>{job.progress}</td>
            <td>{job.uploaded_by}</td><td>{job.category or '-'}</td>
            <td>{job.chunk_count if job.chunk_count > 0 else '-'}</td>
            <td>{kg_info}</td>
            <td>{elapsed}</td>
        </tr>{error_row}"""

    return HTMLResponse(crawl_html + f"""<table>
        <thead><tr><th>File</th><th>Status</th><th>Progress</th><th>Uploaded By</th><th>Category</th><th>Chunks</th><th>Entities / Rels</th><th>Time</th></tr></thead>
        <tbody>{rows}</tbody>
    </table>""")


@router.get("/knowledge-graph", response_class=HTMLResponse)
async def knowledge_graph_page(request: Request):
    redirect = _require_login(request)
    if redirect:
        return redirect
    # Load full graph for initial render (admin view)
    import asyncio
    store = get_metadata_store()
    entities, relationships = await asyncio.to_thread(_load_lightrag_graph)
    apps = await store.list_datasets()
    return templates.TemplateResponse(request, "knowledge_graph.html", {
        "entities": entities, "relationships": relationships, "datasets": apps,
    })


@router.get("/api/knowledge-graph/filtered")
async def knowledge_graph_filtered(groups: str = "", app_id: int = 0):
    """Return filtered graph data by persona ACL and/or dataset."""
    from fastapi.responses import JSONResponse
    import asyncio

    user_groups = [g.strip() for g in groups.split(",") if g.strip()] if groups else ["ALL"]

    entities, relationships = await asyncio.to_thread(_load_lightrag_graph)

    no_persona_filter = "ALL" in user_groups or not groups
    no_app_filter = app_id == 0

    if no_persona_filter and no_app_filter:
        return JSONResponse({"entities": entities, "relationships": relationships})

    # Compute allowed entity sets for each active filter
    allowed = None

    if not no_persona_filter:
        from src.knowledge.graph_rag import _get_acl_allowed_entities
        acl_allowed = await _get_acl_allowed_entities(user_groups)
        if acl_allowed is not None:
            allowed = acl_allowed

    if not no_app_filter:
        from src.knowledge.graph_rag import _get_dataset_allowed_entities
        app_allowed = await asyncio.to_thread(_get_dataset_allowed_entities, app_id)
        if app_allowed is not None:
            if allowed is not None:
                allowed = allowed & app_allowed  # intersection
            else:
                allowed = app_allowed

    if allowed is None:
        return JSONResponse({"entities": entities, "relationships": relationships})

    filtered_entities = [e for e in entities if e["name"] in allowed]
    filtered_names = {e["name"] for e in filtered_entities}
    filtered_rels = [r for r in relationships
                     if r["source"] in filtered_names and r["target"] in filtered_names]

    return JSONResponse({"entities": filtered_entities, "relationships": filtered_rels})


def _load_lightrag_graph():
    """Load LightRAG graph data from GraphML file (runs in thread)."""
    import re as re_mod
    from pathlib import Path

    entities = []
    relationships = []
    junk = {"entity_name", "source_entity", "target_entity", "entity_type"}

    try:
        graphml = Path("data/lightrag/graph_chunk_entity_relation.graphml")
        if not graphml.exists():
            return entities, relationships

        content = graphml.read_text()

        # Parse nodes: <node id="NAME">...<data key="d1">TYPE</data>...</node>
        for match in re_mod.finditer(
            r'<node id="([^"]+)"[^>]*>.*?<data key="d1">(.*?)</data>',
            content, re_mod.DOTALL
        ):
            name = match.group(1)
            etype = match.group(2).strip()
            if name.lower() not in junk and etype not in ("UNKNOWN", "entity_type"):
                entities.append({"name": name, "type": etype})

        # Parse edges: <edge source="SRC" target="TGT">...<data key="d8">DESC</data>...</edge>
        for match in re_mod.finditer(
            r'<edge source="([^"]+)" target="([^"]+)"[^>]*>(.*?)</edge>',
            content, re_mod.DOTALL
        ):
            src = match.group(1)
            tgt = match.group(2)
            desc_match = re_mod.search(r'<data key="d8">(.*?)</data>', match.group(3), re_mod.DOTALL)
            desc = ""
            if desc_match:
                desc = desc_match.group(1).split("&lt;SEP&gt;")[0].split("<SEP>")[0].strip()[:120]
            if src.lower() not in junk and tgt.lower() not in junk:
                relationships.append({"source": src, "target": tgt, "label": desc})
    except Exception:
        pass
    return entities, relationships


@router.post("/api/knowledge-graph/merge/{proposal_id}/approve")
async def approve_merge(proposal_id: int):
    store = get_metadata_store()
    await store.approve_merge(proposal_id)
    return HTMLResponse('<td colspan="5" class="status-ok">Merged</td>')


@router.post("/api/knowledge-graph/merge/{proposal_id}/reject")
async def reject_merge(proposal_id: int):
    store = get_metadata_store()
    await store.reject_merge(proposal_id)
    return HTMLResponse('<td colspan="5">Rejected</td>')


@router.get("/api/knowledge-graph/graph-data")
async def knowledge_graph_data(query: str = "", entity_type: str = "", entity_id: int = 0):
    """Return graph data in Cytoscape.js format."""
    from fastapi.responses import JSONResponse
    store = get_metadata_store()

    TYPE_COLORS = {
        "person": "#2563eb", "organization": "#16a34a", "policy": "#dc2626",
        "project": "#ea580c", "system": "#9333ea", "date": "#6b7280",
        "location": "#0891b2", "document_section": "#ca8a04", "unknown": "#9ca3af",
    }

    nodes = {}
    edges = []

    if entity_id:
        # Show focused entity + its neighbors
        details = await store.get_entity_details(entity_id)
        if details["entity"]:
            ent = details["entity"]
            nodes[ent["id"]] = {"name": ent["name"], "type": ent["type"], "mentions": len(details["mentions"])}
            for r in details["relationships"]:
                # Add related entity as node
                related_entities = await store.search_entities(r["related_entity"], entity_type=r["entity_type"] or None)
                if related_entities:
                    rel_ent = related_entities[0]
                    rel_details = await store.get_entity_details(rel_ent.id)
                    nodes[rel_ent.id] = {"name": rel_ent.name, "type": rel_ent.entity_type, "mentions": len(rel_details["mentions"])}
                    if r["direction"] == "outgoing":
                        edges.append({"source": ent["id"], "target": rel_ent.id, "label": r["relationship_type"]})
                    else:
                        edges.append({"source": rel_ent.id, "target": ent["id"], "label": r["relationship_type"]})
    elif query:
        entities = await store.search_entities(query, entity_type=entity_type or None)
        for ent in entities[:20]:
            details = await store.get_entity_details(ent.id)
            nodes[ent.id] = {"name": ent.name, "type": ent.entity_type, "mentions": len(details["mentions"])}
            for r in details["relationships"]:
                related = await store.search_entities(r["related_entity"], entity_type=r["entity_type"] or None)
                if related:
                    rel = related[0]
                    rel_d = await store.get_entity_details(rel.id)
                    nodes[rel.id] = {"name": rel.name, "type": rel.entity_type, "mentions": len(rel_d["mentions"])}
                    if r["direction"] == "outgoing":
                        edges.append({"source": ent.id, "target": rel.id, "label": r["relationship_type"]})
                    else:
                        edges.append({"source": rel.id, "target": ent.id, "label": r["relationship_type"]})
    else:
        # Show all entities
        entities = await store.list_entities(entity_type=entity_type or None, limit=100)
        for ent in entities:
            details = await store.get_entity_details(ent.id)
            nodes[ent.id] = {"name": ent.name, "type": ent.entity_type, "mentions": len(details["mentions"])}
            for r in details["relationships"]:
                related = await store.search_entities(r["related_entity"], entity_type=r["entity_type"] or None)
                if related:
                    rel = related[0]
                    nodes[rel.id] = {"name": rel.name, "type": rel.entity_type, "mentions": 0}
                    if r["direction"] == "outgoing":
                        edges.append({"source": ent.id, "target": rel.id, "label": r["relationship_type"]})
                    else:
                        edges.append({"source": rel.id, "target": ent.id, "label": r["relationship_type"]})

    # Build Cytoscape elements
    elements = []
    for nid, data in nodes.items():
        size = max(20, min(60, 20 + data["mentions"] * 5))
        elements.append({
            "data": {
                "id": str(nid), "entityId": nid, "label": data["name"],
                "color": TYPE_COLORS.get(data["type"], "#9ca3af"), "size": size,
            }
        })

    seen_edges = set()
    for e in edges:
        key = (e["source"], e["target"], e["label"])
        if key not in seen_edges:
            seen_edges.add(key)
            elements.append({
                "data": {"source": str(e["source"]), "target": str(e["target"]), "label": e["label"]}
            })

    return JSONResponse({"elements": elements})


@router.get("/api/knowledge-graph/detail-json")
async def knowledge_graph_detail_json(entity_id: int = 0):
    from fastapi.responses import JSONResponse
    store = get_metadata_store()
    details = await store.get_entity_details(entity_id)
    return JSONResponse(details)


@router.post("/api/knowledge-graph/search")
async def search_knowledge_graph_api(query: str = Form(""), entity_type: str = Form("")):
    store = get_metadata_store()
    entities = await store.search_entities(query, entity_type=entity_type or None)
    rows = ""
    for e in entities:
        rows += f'<tr><td>{e.name}</td><td>{e.entity_type}</td><td>{e.first_seen_doc_id}</td><td><button hx-post="/admin/api/knowledge-graph/details" hx-vals=\'{{"entity_id": "{e.id}"}}\' hx-target="#kg-results" hx-swap="innerHTML">View</button></td></tr>'
    if not rows:
        return HTMLResponse(f'<p>No entities found matching "{query}".</p>')
    return HTMLResponse(f'<table><thead><tr><th>Name</th><th>Type</th><th>First Seen In</th><th>Actions</th></tr></thead><tbody>{rows}</tbody></table>')


@router.post("/api/knowledge-graph/details")
async def knowledge_graph_details(entity_id: int = Form(0)):
    store = get_metadata_store()
    details = await store.get_entity_details(entity_id)
    if not details["entity"]:
        return HTMLResponse("<p>Entity not found.</p>")
    entity = details["entity"]
    html = f'<div class="result-card"><h2>{entity["name"]} <span class="result-meta">({entity["type"]})</span></h2>'
    html += f'<h3>Mentioned in ({len(details["mentions"])} occurrences)</h3>'
    if details["mentions"]:
        html += '<table><thead><tr><th>Document</th><th>Chunk</th><th>Context</th></tr></thead><tbody>'
        for m in details["mentions"]:
            html += f'<tr><td>{m["doc_id"][:12]}...</td><td>{m["chunk_index"]}</td><td>{m["context_snippet"][:100]}</td></tr>'
        html += '</tbody></table>'
    html += f'<h3>Relationships ({len(details["relationships"])})</h3>'
    if details["relationships"]:
        html += '<table><thead><tr><th>Related Entity</th><th>Type</th><th>Relationship</th><th>Direction</th><th>Source Doc</th></tr></thead><tbody>'
        for r in details["relationships"]:
            html += f'<tr><td>{r["related_entity"]}</td><td>{r["entity_type"]}</td><td>{r["relationship_type"]}</td><td>{r["direction"]}</td><td>{r["doc_id"][:12]}...</td></tr>'
        html += '</tbody></table>'
    else:
        html += '<p>No relationships found.</p>'
    html += '<br><button onclick="window.location.href=\'/admin/knowledge-graph\'">Back to list</button></div>'
    return HTMLResponse(html)


@router.get("/settings")
async def settings_page(request: Request):
    redirect = _require_login(request)
    if redirect:
        return redirect
    return RedirectResponse(url="/admin/settings/security", status_code=307)


@router.get("/settings/{section}", response_class=HTMLResponse)
async def settings_section_page(request: Request, section: str):
    redirect = _require_login(request)
    if redirect:
        return redirect
    if section not in ("security", "models", "retrieval", "system", "maintenance"):
        raise HTTPException(status_code=404)
    return templates.TemplateResponse(request, f"settings_{section}.html",
                                      {"settings": settings, "active": section})


# Form field -> caster. Membership mirrors the persisted settings dict.
_SETTINGS_FIELDS = {
    "admin_username": str, "admin_password": str, "api_keys": str,
    "vllm_base_url": str, "vllm_model_name": str, "vllm_api_key": str,
    "embedding_mode": str, "embedding_api_url": str, "embedding_model_name": str,
    "mcp_port": int, "mcp_alt_port": int,
    "entity_merge_auto_threshold": float, "entity_merge_review_threshold": float,
    "max_parallel_ingestion": int, "llm_concurrency": int,
    "llm_max_context": int, "llm_max_output_tokens": int, "metadata_max_doc_length": int,
    "metadata_extraction_enabled": bool, "feedback_enabled": bool,
    "prf_enabled": bool, "strategy_memory_enabled": bool,
    "feedback_similarity_threshold": float,
}
# String fields that must NOT be cleared when submitted blank (creds/urls). vllm_api_key may be blanked.
_SETTINGS_KEEP_IF_BLANK = {
    "admin_username", "admin_password", "api_keys",
    "vllm_base_url", "vllm_model_name",
    "embedding_mode", "embedding_api_url", "embedding_model_name",
}


def _apply_settings_update(form) -> dict:
    """Partial update of the live `settings` object from a submitted form: only
    fields PRESENT in the form are touched (so a per-section save never clobbers
    another section). Returns the full persist dict. ``form`` is a Starlette
    FormData (has .getlist) or a plain dict (tests). Booleans use the last value
    (sections post a hidden 'false' + checkbox 'true', so an unchecked box still
    posts 'false')."""
    def last(name):
        return form.getlist(name)[-1] if hasattr(form, "getlist") else form[name]

    for name, caster in _SETTINGS_FIELDS.items():
        if name not in form:
            continue
        raw = last(name)
        if caster is bool:
            val = str(raw).strip().lower() in ("true", "1", "on", "yes")
        else:
            s = str(raw).strip()
            if s == "" and name in _SETTINGS_KEEP_IF_BLANK:
                continue
            if s == "" and caster is not str:
                continue                       # blank numeric -> keep current (avoid caster("") error)
            val = caster(s)                    # str("") == "" allowed (e.g. clearing vllm_api_key)
        setattr(settings, name, val)

    return {name: getattr(settings, name) for name in _SETTINGS_FIELDS}


@router.post("/api/settings")
async def save_settings(request: Request):
    """Persist a partial settings update (only the submitted section's fields)."""
    persist = _apply_settings_update(await request.form())
    Path("data/settings.json").write_text(json.dumps(persist, indent=2) + "\n")
    return HTMLResponse('<div class="status-ok">Settings saved successfully.</div>')


@router.post("/api/settings/list-llm-models")
async def list_llm_models(vllm_base_url: str = Form(""), vllm_api_key: str = Form("")):
    url = vllm_base_url or settings.vllm_base_url
    key = vllm_api_key or settings.vllm_api_key
    try:
        import requests as req
        headers = {"Authorization": f"Bearer {key}"} if key else {}
        resp = req.get(f'{url}/models', headers=headers, timeout=10, verify=settings.ssl_verify)
        model_ids = [m['id'] for m in resp.json().get('data', [])]
        if not model_ids:
            return HTMLResponse('<select name="vllm_model_name" id="vllm_model_name"><option value="">No models found</option></select>')
        options = "".join(f'<option value="{m}">{m}</option>' for m in model_ids)
        return HTMLResponse(f'<select name="vllm_model_name" id="vllm_model_name">{options}</select>')
    except Exception as e:
        return HTMLResponse(f'<select name="vllm_model_name" id="vllm_model_name"><option value="">Error: {e}</option></select>')


@router.post("/api/settings/list-embedding-models")
async def list_embedding_models(embedding_api_url: str = Form("")):
    url = embedding_api_url or settings.embedding_api_url
    try:
        import requests as req
        resp = req.get(f'{url}/models', timeout=10, verify=settings.ssl_verify)
        model_ids = [m['id'] for m in resp.json().get('data', [])]
        if not model_ids:
            return HTMLResponse('<select name="embedding_model_name" id="embedding_model_name"><option value="">No models found</option></select>')
        options = "".join(f'<option value="{m}">{m}</option>' for m in model_ids)
        return HTMLResponse(f'<select name="embedding_model_name" id="embedding_model_name">{options}</select>')
    except Exception as e:
        return HTMLResponse(f'<select name="embedding_model_name" id="embedding_model_name"><option value="">Error: {e}</option></select>')


@router.post("/api/settings/reconcile")
async def run_reconciliation():
    import asyncio
    from src.knowledge.reconciler import reconcile_entities, get_reconciliation_status
    status = get_reconciliation_status()
    if status.running:
        return HTMLResponse('<span style="color:#2563eb;">Reconciliation already running...</span>')
    # Run in background so UI can poll for progress
    asyncio.create_task(reconcile_entities(get_metadata_store()))
    return HTMLResponse('<span style="color:#2563eb;">Starting reconciliation...</span>')


@router.post("/api/settings/reconcile-stop")
async def stop_reconciliation_endpoint():
    from src.knowledge.reconciler import stop_reconciliation, get_reconciliation_status
    status = get_reconciliation_status()
    if status.running:
        stop_reconciliation()
        return HTMLResponse('<span style="color:#b45309;">Stopping after current pair...</span>')
    return HTMLResponse('<span>Not running.</span>')


@router.get("/api/settings/reconcile-status")
async def reconcile_status():
    from src.knowledge.reconciler import get_reconciliation_status
    status = get_reconciliation_status()

    # Terminal states — no further polling needed
    if status.done and status.stop_requested:
        return HTMLResponse(
            f'<span style="color:#b45309;">Stopped. Auto-merged: {status.auto_merged}, '
            f'Proposed: {status.proposed}, '
            f'LLM-checked: {status.scanned}, Skipped: {status.skipped}</span>'
        )

    if status.done:
        return HTMLResponse(
            f'<span class="status-ok">Done. Auto-merged: {status.auto_merged}, '
            f'Proposed for review: {status.proposed}, '
            f'LLM-checked: {status.scanned}, Skipped: {status.skipped}</span>'
        )

    if not status.running:
        return HTMLResponse('<span style="color:#666;">Idle</span>')

    # Active — keep polling
    pair_display = status.current_pair
    if len(pair_display) > 60:
        pair_display = pair_display[:57] + "..."

    stopping = ' <span style="color:#b45309;">(stopping...)</span>' if status.stop_requested else ''

    return HTMLResponse(
        f'<div hx-get="/admin/api/settings/reconcile-status" hx-trigger="every 2s" hx-swap="outerHTML">'
        f'<span style="color:#2563eb; font-weight:600;">{status.progress_pct}%</span> '
        f'({status.scanned + status.skipped + status.auto_merged}/{status.total_pairs} pairs) '
        f'&mdash; merged: {status.auto_merged}, proposed: {status.proposed}, checked: {status.scanned}{stopping}<br>'
        f'<span style="font-size:0.85rem; color:#666;">Comparing: {pair_display}</span>'
        f'</div>'
    )


@router.post("/api/settings/test-llm")
async def test_llm_connection(vllm_base_url: str = Form(""), vllm_api_key: str = Form("")):
    url = vllm_base_url or settings.vllm_base_url
    key = vllm_api_key or settings.vllm_api_key
    try:
        import requests as req
        headers = {"Authorization": f"Bearer {key}"} if key else {}
        resp = req.get(f'{url}/models', headers=headers, timeout=10, verify=settings.ssl_verify)
        model_ids = [m['id'] for m in resp.json().get('data', [])[:3]]
        return HTMLResponse(f'<span class="status-ok">Connected to {url}. Models: {", ".join(model_ids)}</span>')
    except Exception as e:
        return HTMLResponse(f'<span class="status-err">Failed connecting to {url}: {e}</span>')


@router.post("/api/settings/test-embedding")
async def test_embedding_connection(
    embedding_mode: str = Form(""),
    embedding_api_url: str = Form(""),
    embedding_model_name: str = Form(""),
):
    mode = embedding_mode or settings.embedding_mode
    url = embedding_api_url or settings.embedding_api_url
    model_name = embedding_model_name or settings.embedding_model_name
    try:
        if mode == "api":
            import requests as req
            resp = req.post(f'{url}/embeddings', json={"model": model_name, "input": ["test"]}, timeout=10, verify=settings.ssl_verify)
            dim = len(resp.json()['data'][0]['embedding'])
            return HTMLResponse(f'<span class="status-ok">Connected to {url}. Model: {model_name}, Dimension: {dim}</span>')
        else:
            from sentence_transformers import SentenceTransformer
            model = SentenceTransformer(model_name, device="cpu", trust_remote_code=True)
            dim = model.get_embedding_dimension() if hasattr(model, 'get_embedding_dimension') else model.get_sentence_embedding_dimension()
            return HTMLResponse(f'<span class="status-ok">Loaded locally. Model: {model_name}, Dimension: {dim}</span>')
    except Exception as e:
        return HTMLResponse(f'<span class="status-err">Failed: {e}</span>')


@router.get("/api/settings/query-metrics")
async def query_metrics_dashboard():
    """Return query performance metrics summary."""
    from src.db.models import QueryMetrics
    from sqlalchemy import select, func
    store = get_metadata_store()
    try:
        async with store.session_factory() as session:
            # Recent queries
            result = await session.execute(
                select(QueryMetrics).order_by(QueryMetrics.created_at.desc()).limit(50)
            )
            rows = list(result.scalars().all())

            if not rows:
                return HTMLResponse("<p>No query metrics yet. Run some queries in the playground to collect data.</p>")

            # Summary stats
            total = len(rows)
            non_cache = [r for r in rows if not r.cache_hit]
            avg_precision = sum(r.map_precision for r in non_cache) / len(non_cache) if non_cache else 0
            avg_time = sum(r.total_time_seconds for r in non_cache) / len(non_cache) if non_cache else 0
            avg_docs_read = sum(r.docs_map_read for r in non_cache) / len(non_cache) if non_cache else 0
            avg_docs_cited = sum(r.docs_cited for r in non_cache) / len(non_cache) if non_cache else 0
            cache_hits = sum(1 for r in rows if r.cache_hit)

            summary = f"""<div style="display:grid; grid-template-columns:repeat(5,1fr); gap:1rem; margin-bottom:1rem;">
                <div><strong>{total}</strong><br><span style="font-size:0.8rem; color:#6b7280;">Total Queries</span></div>
                <div><strong>{avg_precision:.0%}</strong><br><span style="font-size:0.8rem; color:#6b7280;">Avg MAP Precision</span></div>
                <div><strong>{avg_time:.1f}s</strong><br><span style="font-size:0.8rem; color:#6b7280;">Avg Query Time</span></div>
                <div><strong>{avg_docs_read:.0f}</strong><br><span style="font-size:0.8rem; color:#6b7280;">Avg Docs MAP'd</span></div>
                <div><strong>{cache_hits}</strong><br><span style="font-size:0.8rem; color:#6b7280;">Cache Hits</span></div>
            </div>"""

            # Feedback stats
            try:
                from src.db.models import QueryFeedback
                fb_result = await session.execute(select(QueryFeedback))
                fb_rows = list(fb_result.scalars().all())
                fb_total = len(fb_rows)
                fb_cited = sum(1 for r in fb_rows if r.was_cited)
                fb_relevant = sum(1 for r in fb_rows if r.was_in_map_reduce and not r.was_cited)
                fb_irrelevant = sum(1 for r in fb_rows if not r.was_in_map_reduce)
                fb_unique_queries = len({r.query_hash for r in fb_rows})
                fb_unique_docs = len({r.doc_id for r in fb_rows})

                if fb_total > 0:
                    summary += f"""<div style="display:grid; grid-template-columns:repeat(5,1fr); gap:1rem; margin-bottom:1rem; padding-top:0.5rem; border-top:1px solid #e5e7eb;">
                        <div><strong>{fb_total}</strong><br><span style="font-size:0.8rem; color:#6b7280;">Feedback Records</span></div>
                        <div><strong>{fb_unique_queries}</strong><br><span style="font-size:0.8rem; color:#6b7280;">Unique Queries</span></div>
                        <div><strong>{fb_unique_docs}</strong><br><span style="font-size:0.8rem; color:#6b7280;">Unique Docs Seen</span></div>
                        <div><strong>{fb_cited}</strong><br><span style="font-size:0.8rem; color:#6b7280;">Cited Signals</span></div>
                        <div><strong>{fb_irrelevant}</strong><br><span style="font-size:0.8rem; color:#6b7280;">Irrelevant Signals</span></div>
                    </div>"""
            except Exception:
                pass

            # Strategy memory stats
            try:
                from src.db.models import StrategyMemory
                from collections import Counter
                sm_result = await session.execute(select(StrategyMemory))
                sm_rows = list(sm_result.scalars().all())
                if sm_rows:
                    patterns = len({r.query_pattern for r in sm_rows})
                    strategies = Counter(r.query_type or r.strategy_used for r in sm_rows)
                    top_strategy = strategies.most_common(1)[0] if strategies else ("—", 0)
                    summary += f"""<div style="display:grid; grid-template-columns:repeat(3,1fr); gap:1rem; margin-bottom:1rem; padding-top:0.5rem; border-top:1px solid #e5e7eb;">
                        <div><strong>{len(sm_rows)}</strong><br><span style="font-size:0.8rem; color:#6b7280;">Strategy Records</span></div>
                        <div><strong>{patterns}</strong><br><span style="font-size:0.8rem; color:#6b7280;">Unique Patterns</span></div>
                        <div><strong>{top_strategy[0]} ({top_strategy[1]}x)</strong><br><span style="font-size:0.8rem; color:#6b7280;">Most Used Strategy</span></div>
                    </div>"""
            except Exception:
                pass

        # Recent queries table
        table_rows = ""
        for r in rows[:20]:
            precision = f"{r.map_precision:.0%}" if r.docs_map_read > 0 else "—"
            strategy = r.query_type or r.strategy_used or "—"
            table_rows += f"""<tr>
                <td style="max-width:300px; overflow:hidden; text-overflow:ellipsis; white-space:nowrap;">{r.query_text[:80]}</td>
                <td>{strategy}</td>
                <td>{'Cache' if r.cache_hit else f'{r.docs_map_read}→{r.docs_relevant}→{r.docs_cited}'}</td>
                <td>{precision}</td>
                <td>{r.total_time_seconds}s</td>
                <td>{r.created_at.strftime('%m-%d %H:%M') if r.created_at else ''}</td>
            </tr>"""

        table = f"""<table style="font-size:0.85rem;">
            <thead><tr><th>Query</th><th>Strategy</th><th>Docs (MAP→Rel→Cited)</th><th>Precision</th><th>Time</th><th>When</th></tr></thead>
            <tbody>{table_rows}</tbody>
        </table>"""

        return HTMLResponse(summary + table)
    except Exception as e:
        return HTMLResponse(f"<p>Error loading metrics: {e}</p>")


@router.post("/api/settings/purge-cache")
async def purge_query_cache():
    from src.retrieval.query_cache import cache_purge
    count = cache_purge()
    return HTMLResponse(f'<span class="status-ok">Purged {count} cached results.</span>')


@router.get("/api/settings/cache-stats")
async def cache_stats():
    from src.retrieval.query_cache import cache_stats
    stats = cache_stats()
    return HTMLResponse(f'{stats["entries"]} cached results')


@router.post("/api/settings/purge-knowledge-graph")
async def purge_knowledge_graph():
    """Delete all LightRAG knowledge graph data and reinitialize."""
    # Refuse while ingestion is in flight: a running pipeline holds the old
    # in-memory LightRAG instance and would re-flush its state into the
    # freshly-recreated directory, silently restoring the data we just deleted.
    if ingest_queue.has_active_jobs():
        return HTMLResponse(
            '<span style="color:#c00;">Cannot purge while ingestion is in progress — '
            'an in-flight job would re-write deleted graph data. '
            'Wait for ingestion to finish, then retry.</span>'
        )
    import shutil
    lightrag_dir = Path("data/lightrag")
    if lightrag_dir.exists():
        shutil.rmtree(str(lightrag_dir))
        lightrag_dir.mkdir(exist_ok=True)
    # Reset the LightRAG singleton so it reinitializes on next use
    from src.knowledge import graph_rag
    graph_rag._rag_instance = None
    graph_rag._initialized = False
    return HTMLResponse('<span class="status-ok">Knowledge graph purged. It will rebuild as documents are re-ingested.</span>')


@router.post("/api/settings/purge-orphan-chunks")
async def purge_orphan_chunks():
    """Delete LanceDB chunks whose doc_id has no metadata document row.

    Orphans arise when an ingestion is hard-killed (e.g. container stop) after
    chunks are written but before the metadata commit; the graceful failure
    path cleans these up automatically, but a SIGKILL cannot.
    """
    from src.admin.orphans import plan_orphan_purge

    # Cheap guard first: avoid reading the stores while ingestion is running.
    if ingest_queue.has_active_jobs():
        return HTMLResponse(
            '<span style="color:#c00;">Cannot purge orphans while ingestion is in progress. '
            'Wait for it to finish, then retry.</span>'
        )

    store = get_metadata_store()
    docs = await store.list_documents()
    meta_ids = {d.doc_id for d in docs}
    vector_store = get_vector_store()
    try:
        tbl = vector_store.table.to_pandas()
    except Exception as e:
        return HTMLResponse(f'<span style="color:#c00;">Could not read vector store: {e}</span>')
    if "doc_id" not in tbl.columns or len(tbl) == 0:
        return HTMLResponse('<span class="status-ok">No chunks present; nothing to purge.</span>')

    status, orphan_ids = plan_orphan_purge(meta_ids, set(tbl["doc_id"].unique()), ingestion_active=False)
    if status == "refused_empty_metadata":
        return HTMLResponse(
            '<span style="color:#c00;">Refusing to purge: metadata has 0 documents, so every chunk '
            'would be treated as an orphan. If you intend to wipe everything, use a full reset instead.</span>'
        )
    if not orphan_ids:
        return HTMLResponse('<span class="status-ok">No orphaned chunks found.</span>')

    deleted = 0
    for did in sorted(orphan_ids):
        deleted += int((tbl["doc_id"] == did).sum())
        vector_store.delete_by_doc_id(did)
    return HTMLResponse(
        f'<span class="status-ok">Purged {deleted} orphaned chunk(s) across {len(orphan_ids)} '
        f'document(s) that had no metadata record.</span>'
    )


@router.post("/api/settings/purge-feedback")
async def purge_feedback():
    """Clear all feedback and metrics data to start fresh."""
    from src.db.models import QueryFeedback, QueryMetrics
    from sqlalchemy import delete
    store = get_metadata_store()
    async with store.session_factory() as session:
        await session.execute(delete(QueryFeedback))
        await session.execute(delete(QueryMetrics))
        await session.commit()
    return HTMLResponse('<span class="status-ok">Feedback and metrics data cleared.</span>')


@router.post("/api/settings/backfill-metadata")
async def backfill_metadata():
    """Re-extract metadata for all documents missing metadata_tags."""
    import asyncio

    store = get_metadata_store()
    docs = await store.list_documents()
    def _needs_metadata(doc):
        meta = getattr(doc, 'metadata_tags', None)
        if not meta:
            return True
        if isinstance(meta, dict) and not meta.get("summary"):
            return True
        return False
    missing = [d for d in docs if _needs_metadata(d)]

    if not missing:
        return HTMLResponse('<span class="status-ok">All documents already have metadata.</span>')

    async def _backfill():
        from src.ingestion.metadata_extractor import extract_metadata
        from src.ingestion.embedder import embed_texts
        vs = get_vector_store()
        from src.retrieval.models import ChunkMetadata

        for i, doc in enumerate(missing):
            try:
                # Reconstruct text from large chunks
                chunks = vs.get_chunks_by_doc(doc.doc_id, limit=200, tier="large")
                if not chunks:
                    chunks = vs.get_chunks_by_doc(doc.doc_id, limit=200)
                if not chunks:
                    continue
                text = "\n\n".join(c.text for c in chunks)

                metadata = await asyncio.to_thread(extract_metadata, text, doc.filename)
                summary = metadata.get("summary", "")

                await store.update_document(doc.doc_id, summary=summary, metadata_tags=metadata)

                # Embed and store summary vector
                if summary:
                    doc_context = f"Document: {doc.filename} (type: {doc.doc_type}, category: {doc.category})\nSummary: {summary}"
                    vectors = await asyncio.to_thread(embed_texts, [doc_context])
                    if vectors:
                        meta = ChunkMetadata(
                            doc_id=doc.doc_id, filename=doc.filename, doc_type=doc.doc_type,
                            chunk_index=-1, start_char=0, acl_groups=doc.acl_groups,
                            category=doc.category, chunk_size_tier="summary",
                        )
                        await asyncio.to_thread(vs.upsert, texts=[doc_context], vectors=vectors, metadatas=[meta])
            except Exception as e:
                logging.getLogger(__name__).warning(f"Backfill failed for {doc.filename}: {e}")

    asyncio.create_task(_backfill())
    return HTMLResponse(f'<span style="color:#2563eb;">Backfilling metadata for {len(missing)} documents in background...</span>')


# ============================================================
# Backup & Restore
# ============================================================

_backup_status = {"state": "idle", "message": ""}  # idle | running | done | error


@router.post("/api/backup/create")
async def create_backup():
    """Create a tar.gz backup of all data + .env config."""
    import asyncio

    if _backup_status["state"] == "running":
        return HTMLResponse('<span style="color:#f59e0b;">Backup already in progress.</span>')

    _backup_status["state"] = "running"
    _backup_status["message"] = "Starting..."

    async def _run_backup():
        import tarfile, time as _time

        backup_dir = Path("backups")
        backup_dir.mkdir(exist_ok=True)
        timestamp = _time.strftime("%Y%m%d-%H%M%S")
        backup_name = f"sauron-backup-{timestamp}.tar.gz"
        backup_path = backup_dir / backup_name

        try:
            data_dir = Path("data")

            # Skip temp/unnecessary files
            skip_dirs = {"_transactions"}
            skip_names = {".DS_Store"}
            skip_suffixes = {".wal", ".shm", "-journal", ".tmp"}

            def should_skip(p):
                if p.name in skip_names:
                    return True
                if any(p.name.endswith(s) for s in skip_suffixes):
                    return True
                if any(part in skip_dirs for part in p.parts):
                    return True
                return False

            # Count files for progress
            all_files = [f for f in data_dir.rglob("*") if not should_skip(f)] if data_dir.exists() else []
            total = len(all_files) + 1  # +1 for .env
            _backup_status["message"] = f"Compressing 0/{total} files..."

            with tarfile.open(str(backup_path), "w:gz") as tar:
                if data_dir.exists():
                    count = 0
                    for f in all_files:
                        tar.add(str(f), arcname=str(f))
                        count += 1
                        if count % 100 == 0:
                            _backup_status["message"] = f"Compressing {count}/{total} files..."

                env_file = Path(".env")
                if env_file.exists():
                    tar.add(str(env_file), arcname=".env")

            size_mb = backup_path.stat().st_size / (1024 * 1024)
            _backup_status["state"] = "done"
            _backup_status["message"] = f"Backup created: {backup_name} ({size_mb:.1f} MB)"
        except Exception as e:
            _backup_status["state"] = "error"
            _backup_status["message"] = f"Backup failed: {e}"

    asyncio.create_task(asyncio.to_thread(lambda: asyncio.run(_run_backup())) if False else _run_backup())
    return HTMLResponse('<span style="color:#2563eb;">Backup started...</span>')


@router.get("/api/backup/status")
async def backup_status():
    """Return current backup status as HTML."""
    state = _backup_status["state"]
    msg = _backup_status["message"]
    if state == "running":
        return HTMLResponse(f'<span style="color:#2563eb; font-weight:600;">{msg}</span>')
    elif state == "done":
        _backup_status["state"] = "idle"
        return HTMLResponse(f'<span class="status-ok">{msg}</span>')
    elif state == "error":
        _backup_status["state"] = "idle"
        return HTMLResponse(f'<span class="status-err">{msg}</span>')
    return HTMLResponse("")


@router.get("/api/backup/list")
async def list_backups():
    """List available backups."""
    backup_dir = Path("backups")
    if not backup_dir.exists():
        return HTMLResponse("<p>No backups yet.</p>")

    backups = sorted(backup_dir.glob("sauron-backup-*.tar.gz"), reverse=True)
    if not backups:
        return HTMLResponse("<p>No backups yet.</p>")

    rows = ""
    for b in backups:
        size_mb = b.stat().st_size / (1024 * 1024)
        name = b.name
        rows += f"""<tr>
            <td>{name}</td>
            <td>{size_mb:.1f} MB</td>
            <td>
                <a href="/admin/api/backup/download/{name}" class="small" style="text-decoration:none;">Download</a>
                <button class="small" hx-post="/admin/api/backup/delete/{name}" hx-target="#backup-list" hx-swap="innerHTML" hx-confirm="Delete {name}?" style="margin-left:0.25rem;">Delete</button>
            </td>
        </tr>"""

    return HTMLResponse(f"""<table>
        <thead><tr><th>Backup</th><th>Size</th><th>Actions</th></tr></thead>
        <tbody>{rows}</tbody>
    </table>""")


@router.get("/api/backup/download/{filename}")
async def download_backup(filename: str):
    """Download a backup file."""
    from fastapi.responses import FileResponse
    import re

    # Sanitize filename
    if not re.match(r'^sauron-backup-[\d-]+\.tar\.gz$', filename):
        return HTMLResponse('<span class="status-err">Invalid filename.</span>', status_code=400)

    backup_path = Path("backups") / filename
    if not backup_path.exists():
        return HTMLResponse('<span class="status-err">Backup not found.</span>', status_code=404)

    return FileResponse(
        str(backup_path),
        media_type="application/gzip",
        filename=filename,
    )


@router.post("/api/backup/delete/{filename}")
async def delete_backup(filename: str):
    """Delete a backup file."""
    import re

    if not re.match(r'^sauron-backup-[\d-]+\.tar\.gz$', filename):
        return HTMLResponse('<span class="status-err">Invalid filename.</span>', status_code=400)

    backup_path = Path("backups") / filename
    if backup_path.exists():
        backup_path.unlink()

    # Return updated list
    return await list_backups()


@router.post("/api/backup/restore")
async def restore_backup(backup_file: UploadFile = File(...)):
    """Restore from an uploaded backup tar.gz file."""
    import tarfile, shutil, io

    try:
        contents = await backup_file.read()
        tar_bytes = io.BytesIO(contents)

        with tarfile.open(fileobj=tar_bytes, mode="r:gz") as tar:
            # Validate: must contain data/ directory
            members = tar.getnames()
            if not any(m.startswith("data/") or m == "data" for m in members):
                return HTMLResponse('<span class="status-err">Invalid backup: no data/ directory found.</span>')

            # Safety: reject paths that escape the current directory
            for m in members:
                if m.startswith("/") or ".." in m:
                    return HTMLResponse(f'<span class="status-err">Invalid backup: unsafe path {m}</span>')

            # Back up current data before overwriting
            data_dir = Path("data")
            data_bak = Path("data.pre-restore")
            if data_dir.exists():
                if data_bak.exists():
                    shutil.rmtree(str(data_bak))
                shutil.copytree(str(data_dir), str(data_bak))

            # Extract
            tar.extractall(".")

        size_mb = len(contents) / (1024 * 1024)
        return HTMLResponse(
            f'<span class="status-ok">Restored from {backup_file.filename} ({size_mb:.1f} MB). '
            f'Previous data saved to data.pre-restore/. Restart the server to apply.</span>'
        )
    except Exception as e:
        return HTMLResponse(f'<span class="status-err">Restore failed: {e}</span>')
