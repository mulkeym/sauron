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
    chunk_size=1024,
    chunk_overlap=100,
    auto_categorize=False,
    original_filename="",
):
    doc_id = str(uuid.uuid4())
    parsed = parse_document(Path(file_path))
    if original_filename:
        parsed.filename = original_filename

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
    # Inherit default ACL from category if none provided
    if not acl_groups and category and category != "uncategorized":
        cat_record = await metadata_store.get_category(category)
        if cat_record and cat_record.acl_groups:
            acl_groups = cat_record.acl_groups

    # Generate LLM document summary for contextual enrichment
    from src.generation.llm_client import generate as llm_generate
    import logging
    doc_summary = ""
    try:
        doc_summary = llm_generate(
            system_prompt="Summarize ALL items in this document in 2-4 sentences. List EVERY company, contract, or award mentioned — do not omit any. Include names, amounts, and dates.",
            user_prompt=parsed.text[:6000],
            temperature=0.0, max_tokens=1024,
        )
        logging.getLogger(__name__).info(f"Document summary: {doc_summary[:100]}")
    except Exception as e:
        logging.getLogger(__name__).warning(f"Summary generation failed: {e}")

    # Multi-pass indexing: store chunks at multiple granularities
    CHUNK_TIERS = [
        ("small", 1024, 100),
        ("medium", 2048, 200),
        ("large", 4096, 400),
        ("xlarge", 8192, 800),
    ]
    doc_context = f"Document: {parsed.filename} (type: {parsed.doc_type}, category: {category})"
    if doc_summary:
        doc_context += f"\nSummary: {doc_summary}"
    total_chunks = 0

    for tier_name, tier_size, tier_overlap in CHUNK_TIERS:
        tier_chunks = chunk_text(parsed.text, chunk_size=tier_size, chunk_overlap=tier_overlap)
        texts = [f"{doc_context}\n\n{c.text}" for c in tier_chunks]
        metadatas = [
            ChunkMetadata(
                doc_id=doc_id,
                filename=parsed.filename,
                doc_type=parsed.doc_type,
                chunk_index=c.index,
                start_char=c.start_char,
                acl_groups=acl_groups,
                category=category,
                chunk_size_tier=tier_name,
            )
            for c in tier_chunks
        ]
        vectors = embed_texts(texts) if texts else []
        if vectors:
            vector_store.upsert(texts=texts, vectors=vectors, metadatas=metadatas)
        if tier_name == "medium":
            total_chunks = len(tier_chunks)  # report medium tier count
            chunks = tier_chunks  # use medium tier for entity extraction
    await metadata_store.add_document(
        doc_id=doc_id,
        filename=parsed.filename,
        doc_type=parsed.doc_type,
        acl_groups=acl_groups,
        chunk_count=total_chunks,
        uploaded_by=uploaded_by,
        category=category,
    )
    # Structured handling for spreadsheets: clean sheets -> DuckDB + row narratives.
    # Shared helper (also called by the async queue worker) so both ingestion
    # paths get the structured path; fail-open inside the helper.
    from src.ingestion.tabular_ingest import maybe_ingest_spreadsheet
    await maybe_ingest_spreadsheet(
        file_path, doc_id, parsed.filename, parsed.doc_type,
        acl_groups, category, vector_store, metadata_store,
    )
    # Ensure category exists in categories table
    if category and category != "uncategorized":
        existing = await metadata_store.get_category(category)
        if not existing:
            await metadata_store.add_category(
                name=category, description="", acl_groups=acl_groups, routing_keywords=[],
            )
    # Build knowledge graph via LightRAG
    from src.knowledge.graph_rag import insert_document as lightrag_insert
    await lightrag_insert(parsed.text, doc_id=doc_id, filename=parsed.filename)
    return IngestResult(
        doc_id=doc_id,
        filename=parsed.filename,
        doc_type=parsed.doc_type,
        chunk_count=total_chunks,
    )
