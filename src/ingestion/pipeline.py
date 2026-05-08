import uuid
from dataclasses import dataclass
from pathlib import Path

from src.ingestion.parser import parse_document
from src.ingestion.chunker import chunk_text
from src.ingestion.embedder import embed_texts
from src.retrieval.models import ChunkMetadata
from src.retrieval.vector_store import VectorStore
from src.db.metadata import MetadataStore
from src.knowledge.categorizer import categorize_document


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
    auto_categorize=False,
):
    doc_id = str(uuid.uuid4())
    parsed = parse_document(file_path)

    if not category and auto_categorize:
        cat_result = categorize_document(
            filename=parsed.filename,
            doc_type=parsed.doc_type,
            text_preview=parsed.text[:500],
            metadata_store=metadata_store,
        )
        if cat_result.is_new:
            await metadata_store.add_proposal(
                proposed_name=cat_result.category,
                proposed_description=cat_result.description,
                proposed_acl_groups=cat_result.suggested_acl_groups,
                proposed_keywords=cat_result.suggested_keywords,
                proposed_by="auto-categorizer",
            )
            category = "uncategorized"
        else:
            category = cat_result.category
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
