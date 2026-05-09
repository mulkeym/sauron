import json
import tempfile
from pathlib import Path
from typing import List
from fastapi import APIRouter, Request, Form, UploadFile, File
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from src.api.routes_ingest import get_metadata_store, get_vector_store, get_schema_registry
from src.config import settings
from src.ingestion.queue import ingest_queue

router = APIRouter(prefix="/admin", tags=["admin"])
TEMPLATE_DIR = Path(__file__).parent / "templates"
templates = Jinja2Templates(directory=str(TEMPLATE_DIR))

@router.get("/", response_class=HTMLResponse)
async def dashboard(request: Request):
    store = get_metadata_store()
    docs = await store.list_documents()
    categories = await store.list_categories()
    proposals = await store.list_proposals(status="pending")
    return templates.TemplateResponse(request, "dashboard.html", {"doc_count": len(docs), "category_count": len(categories), "pending_proposals": len(proposals)})

@router.get("/documents", response_class=HTMLResponse)
async def documents_page(request: Request):
    store = get_metadata_store()
    docs = await store.list_documents()
    return templates.TemplateResponse(request, "documents.html", {"documents": docs})

@router.get("/categories", response_class=HTMLResponse)
async def categories_page(request: Request):
    store = get_metadata_store()
    categories = await store.list_categories()
    return templates.TemplateResponse(request, "categories.html", {"categories": categories})

@router.get("/proposals", response_class=HTMLResponse)
async def proposals_page(request: Request):
    store = get_metadata_store()
    proposals = await store.list_proposals(status="pending")
    return templates.TemplateResponse(request, "proposals.html", {"proposals": proposals})

@router.get("/audit", response_class=HTMLResponse)
async def audit_page(request: Request):
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
    return templates.TemplateResponse(request, "audit.html", {"entries": entries})

@router.post("/api/proposals/{proposal_id}/approve")
async def approve_proposal(proposal_id: int):
    store = get_metadata_store()
    await store.approve_proposal(proposal_id, approved_by="admin")
    return HTMLResponse("<tr><td colspan='6'>Approved</td></tr>")

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
    return HTMLResponse(f'<span class="status-ok">Category "{name}" created. Reload to see it in the table.</span>')


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


@router.delete("/api/documents/{doc_id}")
async def delete_document(doc_id: str):
    store = get_metadata_store()
    # Remove from metadata DB
    await store.delete_document(doc_id)
    # Remove entity mentions and relationships for this doc
    await store.delete_entities_for_doc(doc_id)
    # Remove vector chunks from Qdrant
    try:
        vector_store = get_vector_store()
        vector_store.delete_by_doc_id(doc_id)
    except Exception:
        pass  # Qdrant may not be running in test
    return HTMLResponse("")


@router.get("/playground", response_class=HTMLResponse)
async def playground_page(request: Request):
    return templates.TemplateResponse(request, "playground.html", {})


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
        )

        # Build trace timeline
        step_labels = {
            "classify": "Classify Query",
            "retrieve": "Retrieve Documents",
            "evaluate": "Evaluate Context",
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
    auto_categorize: str = Form(""),
):
    groups = [g.strip() for g in acl_groups.split(",") if g.strip()]
    do_auto_cat = auto_categorize == "true"

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
            category=category, auto_categorize=do_auto_cat,
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
    jobs = ingest_queue.list_jobs()
    return templates.TemplateResponse(request, "queue.html", {"jobs": jobs})


@router.get("/api/queue/status")
async def queue_status():
    jobs = ingest_queue.list_jobs()
    if not jobs:
        return HTMLResponse('<p>No ingestion jobs.</p>')

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

        rows += f"""<tr>
            <td>{job.filename}</td><td>{status}</td><td>{job.progress}</td>
            <td>{job.uploaded_by}</td><td>{job.category or '-'}</td>
            <td>{job.chunk_count if job.step == 'complete' else '-'}</td>
            <td>{elapsed}</td>
        </tr>{error_row}"""

    return HTMLResponse(f"""<table>
        <thead><tr><th>File</th><th>Status</th><th>Progress</th><th>Uploaded By</th><th>Category</th><th>Chunks</th><th>Time</th></tr></thead>
        <tbody>{rows}</tbody>
    </table>""")


@router.get("/knowledge-graph", response_class=HTMLResponse)
async def knowledge_graph_page(request: Request):
    store = get_metadata_store()
    entities = await store.list_entities(limit=50)
    return templates.TemplateResponse(request, "knowledge_graph.html", {"entities": entities})


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


@router.get("/settings", response_class=HTMLResponse)
async def settings_page(request: Request):
    return templates.TemplateResponse(request, "settings.html", {"settings": settings})


@router.post("/api/settings")
async def save_settings(
    vllm_base_url: str = Form(""),
    vllm_model_name: str = Form(""),
    embedding_mode: str = Form(""),
    embedding_api_url: str = Form(""),
    embedding_model_name: str = Form(""),
    qdrant_host: str = Form(""),
    qdrant_port: int = Form(6333),
    qdrant_collection_name: str = Form(""),
    mcp_port: int = Form(8090),
):
    # Update in-memory settings
    if vllm_base_url:
        settings.vllm_base_url = vllm_base_url
    if vllm_model_name:
        settings.vllm_model_name = vllm_model_name
    if embedding_mode:
        settings.embedding_mode = embedding_mode
    if embedding_api_url:
        settings.embedding_api_url = embedding_api_url
    if embedding_model_name:
        settings.embedding_model_name = embedding_model_name
    if qdrant_host:
        settings.qdrant_host = qdrant_host
    settings.qdrant_port = qdrant_port
    if qdrant_collection_name:
        settings.qdrant_collection_name = qdrant_collection_name
    settings.mcp_port = mcp_port

    # Persist to .env file
    env_path = Path(".env")
    env_lines = {}
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            if "=" in line and not line.strip().startswith("#"):
                key, _, val = line.partition("=")
                env_lines[key.strip()] = val.strip()

    env_lines["VLLM_BASE_URL"] = settings.vllm_base_url
    env_lines["VLLM_MODEL_NAME"] = settings.vllm_model_name
    env_lines["EMBEDDING_MODE"] = settings.embedding_mode
    env_lines["EMBEDDING_API_URL"] = settings.embedding_api_url
    env_lines["EMBEDDING_MODEL_NAME"] = settings.embedding_model_name
    env_lines["QDRANT_HOST"] = settings.qdrant_host
    env_lines["QDRANT_PORT"] = str(settings.qdrant_port)
    env_lines["QDRANT_COLLECTION_NAME"] = settings.qdrant_collection_name
    env_lines["MCP_PORT"] = str(settings.mcp_port)

    env_path.write_text("\n".join(f"{k}={v}" for k, v in env_lines.items()) + "\n")

    return HTMLResponse('<div class="status-ok">Settings saved successfully.</div>')


@router.post("/api/settings/list-llm-models")
async def list_llm_models(vllm_base_url: str = Form("")):
    url = vllm_base_url or settings.vllm_base_url
    try:
        from openai import OpenAI
        client = OpenAI(base_url=url, api_key="not-needed")
        models = client.models.list()
        model_ids = [m.id for m in models.data]
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
        from openai import OpenAI
        client = OpenAI(base_url=url, api_key="not-needed")
        models = client.models.list()
        model_ids = [m.id for m in models.data]
        if not model_ids:
            return HTMLResponse('<select name="embedding_model_name" id="embedding_model_name"><option value="">No models found</option></select>')
        options = "".join(f'<option value="{m}">{m}</option>' for m in model_ids)
        return HTMLResponse(f'<select name="embedding_model_name" id="embedding_model_name">{options}</select>')
    except Exception as e:
        return HTMLResponse(f'<select name="embedding_model_name" id="embedding_model_name"><option value="">Error: {e}</option></select>')


@router.post("/api/settings/test-llm")
async def test_llm_connection(vllm_base_url: str = Form("")):
    url = vllm_base_url or settings.vllm_base_url
    try:
        from openai import OpenAI
        client = OpenAI(base_url=url, api_key="not-needed")
        models = client.models.list()
        model_ids = [m.id for m in models.data]
        return HTMLResponse(f'<span class="status-ok">Connected to {url}. Models: {", ".join(model_ids[:3])}</span>')
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
            from openai import OpenAI
            client = OpenAI(base_url=url, api_key="not-needed")
            result = client.embeddings.create(model=model_name, input=["test"])
            dim = len(result.data[0].embedding)
            return HTMLResponse(f'<span class="status-ok">Connected to {url}. Model: {model_name}, Dimension: {dim}</span>')
        else:
            from sentence_transformers import SentenceTransformer
            model = SentenceTransformer(model_name, device=settings.embedding_device)
            dim = model.get_sentence_embedding_dimension()
            return HTMLResponse(f'<span class="status-ok">Loaded locally. Model: {model_name}, Dimension: {dim}</span>')
    except Exception as e:
        return HTMLResponse(f'<span class="status-err">Failed: {e}</span>')
