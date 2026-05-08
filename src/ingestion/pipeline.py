import uuid
from dataclasses import dataclass
from pathlib import Path

from src.ingestion.parser import parse_document
from src.ingestion.chunker import chunk_text
from src.ingestion.embedder import embed_texts
from src.retrieval.models import ChunkMetadata
from src.retrieval.vector_store import VectorStore
from src.db.metadata import MetadataStore


@dataclass
class IngestResult:
    doc_id: str
    filename: str
    doc_type: str
    chunk_count: int


async def ingest_document(
    file_path,
    acl_groups,
    uploaded_by,
    vector_store,
    metadata_store,
    category="",
    chunk_size=512,
    chunk_overlap=50,
):
    doc_id = str(uuid.uuid4())
    parsed = parse_document(file_path)
    chunks = chunk_text(parsed.text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    texts = [c.text for c in chunks]
    metadatas = [
        ChunkMetadata(
            doc_id=doc_id,
            filename=parsed.filename,
            doc_type=parsed.doc_type,
            chunk_index=c.index,
            start_char=c.start_char,
            acl_groups=acl_groups,
            category=category,
        )
        for c in chunks
    ]
    vectors = embed_texts(texts) if texts else []
    if vectors:
        vector_store.upsert(texts=texts, vectors=vectors, metadatas=metadatas)
    await metadata_store.add_document(
        doc_id=doc_id,
        filename=parsed.filename,
        doc_type=parsed.doc_type,
        acl_groups=acl_groups,
        chunk_count=len(chunks),
        uploaded_by=uploaded_by,
        category=category,
    )
    return IngestResult(
        doc_id=doc_id,
        filename=parsed.filename,
        doc_type=parsed.doc_type,
        chunk_count=len(chunks),
    )
