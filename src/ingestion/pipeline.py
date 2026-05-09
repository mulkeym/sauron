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
from src.knowledge.extractor import extract_entities


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
    # Ensure category exists in categories table
    if category and category != "uncategorized":
        existing = await metadata_store.get_category(category)
        if not existing:
            await metadata_store.add_category(
                name=category, description="", acl_groups=acl_groups, routing_keywords=[],
            )
    # Extract entities and relationships from each chunk
    for chunk in chunks:
        extraction = extract_entities(chunk.text)
        entity_id_map = {}
        for ent in extraction.entities:
            eid = await metadata_store.add_entity(name=ent["name"], entity_type=ent["type"], first_seen_doc_id=doc_id)
            entity_id_map[ent["name"]] = eid
            await metadata_store.add_mention(entity_id=eid, doc_id=doc_id, chunk_index=chunk.index, context_snippet=chunk.text[:200])
        for rel in extraction.relationships:
            source_id = entity_id_map.get(rel["source"])
            if source_id is None:
                continue
            target_id = entity_id_map.get(rel["target"])
            if target_id is None:
                target_id = await metadata_store.add_entity(name=rel["target"], entity_type="unknown", first_seen_doc_id=doc_id)
            await metadata_store.add_relationship(source_entity_id=source_id, target_entity_id=target_id, relationship_type=rel.get("type", "related_to"), doc_id=doc_id, context_snippet=chunk.text[:100])
        for section in extraction.sections:
            await metadata_store.add_entity(name=section["name"], entity_type="document_section", first_seen_doc_id=doc_id)
    return IngestResult(
        doc_id=doc_id,
        filename=parsed.filename,
        doc_type=parsed.doc_type,
        chunk_count=len(chunks),
    )
