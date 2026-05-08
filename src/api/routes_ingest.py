import json
import tempfile
from pathlib import Path
from fastapi import APIRouter, Depends, File, Form, UploadFile
from src.api.models import DocumentInfo, IngestResponse
from src.auth.dependencies import require_auth
from src.auth.models import UserContext
from src.db.metadata import MetadataStore
from src.ingestion.pipeline import ingest_document
from src.retrieval.vector_store import VectorStore

router = APIRouter(prefix="/api/v1", tags=["ingestion"])
_vector_store = None
_metadata_store = None

def get_vector_store():
    global _vector_store
    if _vector_store is None:
        _vector_store = VectorStore()
    return _vector_store

def get_metadata_store():
    global _metadata_store
    if _metadata_store is None:
        _metadata_store = MetadataStore()
    return _metadata_store

from src.db.schema_registry import SchemaRegistry

_schema_registry = None

def get_schema_registry():
    global _schema_registry
    if _schema_registry is None:
        _schema_registry = SchemaRegistry()
    return _schema_registry

@router.post("/ingest", response_model=IngestResponse)
async def ingest_file(file: UploadFile = File(...), acl_groups: str = Form(default="[]"), category: str = Form(default=""), user: UserContext = Depends(require_auth)):
    groups = json.loads(acl_groups)
    suffix = Path(file.filename).suffix
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = Path(tmp.name)
    try:
        result = await ingest_document(file_path=tmp_path, acl_groups=groups, uploaded_by=user.username, vector_store=get_vector_store(), metadata_store=get_metadata_store(), category=category)
    finally:
        tmp_path.unlink(missing_ok=True)
    return IngestResponse(doc_id=result.doc_id, filename=result.filename, doc_type=result.doc_type, chunk_count=result.chunk_count)

@router.get("/documents", response_model=list[DocumentInfo])
async def list_documents(user: UserContext = Depends(require_auth)):
    store = get_metadata_store()
    docs = await store.list_documents(user_groups=user.groups)
    return [DocumentInfo(doc_id=d.doc_id, filename=d.filename, doc_type=d.doc_type, category=d.category, acl_groups=d.acl_groups, chunk_count=d.chunk_count) for d in docs]
