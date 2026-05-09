import json
from pathlib import Path
from fastapi import APIRouter, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from src.api.routes_ingest import get_metadata_store
from src.config import settings

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

@router.delete("/api/documents/{doc_id}")
async def delete_document(doc_id: str):
    store = get_metadata_store()
    await store.delete_document(doc_id)
    return HTMLResponse("")


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
