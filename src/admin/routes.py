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
):
    if not name.strip():
        return HTMLResponse('<span class="status-err">Category name is required.</span>')
    store = get_metadata_store()
    existing = await store.get_category(name.strip())
    if existing:
        return HTMLResponse(f'<span class="status-err">Category "{name}" already exists.</span>')
    groups = [g.strip() for g in acl_groups.split(",") if g.strip()]
    keywords = [k.strip() for k in routing_keywords.split(",") if k.strip()]
    await store.add_category(name=name.strip(), description=description.strip(), acl_groups=groups, routing_keywords=keywords)
    return HTMLResponse(f'<span class="status-ok">Category "{name}" created. Reload to see it in the table.</span>')


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
    await store.delete_document(doc_id)
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
        from src.generation.rag_chain import agent_query
        result = await agent_query(
            question=question,
            user_groups=user_groups,
            vector_store=get_vector_store(),
            schema_registry=get_schema_registry(),
        )

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
        <div class="result-card">
            <div class="result-meta">Groups: {', '.join(user_groups)}</div>
            <div class="result-answer">{result.answer}</div>
            <h3 style="margin-bottom:0.5rem; font-size:0.95rem;">Citations ({len(result.citations)})</h3>
            {citations_html or '<p>No citations.</p>'}
        </div>""")
    except Exception as e:
        return HTMLResponse(f'<div class="status-err">Error: {e}</div>')


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
