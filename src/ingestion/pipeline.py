import logging
import uuid
from dataclasses import dataclass
from pathlib import Path

from src.ingestion.parser import parse_document
from src.ingestion.chunker import chunk_text
from src.ingestion.embedder import embed_texts
from src.ingestion.tabular_ingest import ingest_structured_sheets, ingest_grids, SPREADSHEET_DOC_TYPES
from src.ingestion.tabular_chunker import sheets_needing_text, build_tier_chunks
from src.ingestion.pdf_extract import extract_pdf
from src.retrieval.models import ChunkMetadata
from src.retrieval.vector_store import VectorStore
from src.db.metadata import MetadataStore
from src.knowledge.categorizer import categorize_document

logger = logging.getLogger(__name__)


def _is_structured_pdf(doc_type: str) -> bool:
    return doc_type == "pdf"


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
    dataset_id=None,
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
    chunks = []  # medium-tier chunks, retained for the return/entity count

    is_spreadsheet = parsed.doc_type in SPREADSHEET_DOC_TYPES
    is_pdf = _is_structured_pdf(parsed.doc_type)
    is_docx = parsed.doc_type == "docx"
    text_sheets = None
    enriched_prose = None
    from src.config import settings as _settings

    if is_spreadsheet:
        # Structured: clean sheets -> DuckDB + schema + row narratives; messy
        # sheets -> deterministic region narratives. Returns which clean sheets
        # fully succeeded so we can de-dup their full text below.
        grids, classifications, ingested = await ingest_structured_sheets(
            file_path, doc_id, parsed.filename, parsed.doc_type,
            acl_groups, category, vector_store, metadata_store,
            dataset_id=dataset_id,
        )
        text_sheets = sheets_needing_text(grids, classifications, ingested)
        if _settings.figure_extraction_enabled and Path(file_path).suffix.lower() in (".xlsx", ".xlsm"):
            try:
                from src.ingestion.figure_extract import enrich_text_with_figures
                fig_text, fig_grids = enrich_text_with_figures(Path(file_path), "")
                if fig_grids:
                    await ingest_grids(
                        fig_grids, doc_id, parsed.filename, parsed.doc_type,
                        acl_groups, category, vector_store, metadata_store,
                        dataset_id=dataset_id,
                    )
                if fig_text.strip():
                    enriched_prose = ((parsed.text or "") + "\n\n" + fig_text).strip()
            except Exception as fig_err:
                logger.warning(f"Spreadsheet figure extract failed: {fig_err}")
    elif is_pdf:
        try:
            extracted = extract_pdf(Path(file_path))
            if _settings.figure_extraction_enabled:
                from src.ingestion.figure_extract import enrich_pdf_with_figures
                extracted = enrich_pdf_with_figures(Path(file_path), extracted)
            await ingest_grids(
                extracted.table_grids, doc_id, parsed.filename, parsed.doc_type,
                acl_groups, category, vector_store, metadata_store,
                dataset_id=dataset_id,
            )
            enriched_prose = "\n\n".join(b.text for b in extracted.prose_blocks)
            logger.info(f"PDF structured extract [{parsed.filename}]: "
                        f"{len(extracted.table_grids)} table(s), method={extracted.method}, "
                        f"prose_blocks={len(extracted.prose_blocks)}")
        except Exception as e:
            logger.warning(f"PDF structured extract failed for {parsed.filename}, "
                           f"falling back to flat text: {e}")
            is_pdf = False   # fall back to parsed.text chunking below
    elif is_docx and _settings.figure_extraction_enabled:
        try:
            from src.ingestion.figure_extract import enrich_text_with_figures
            enriched_prose, fig_grids = enrich_text_with_figures(
                Path(file_path), parsed.text or "",
            )
            if fig_grids:
                await ingest_grids(
                    fig_grids, doc_id, parsed.filename, parsed.doc_type,
                    acl_groups, category, vector_store, metadata_store,
                    dataset_id=dataset_id,
                )
        except Exception as e:
            logger.warning(f"Word figure extract failed for {parsed.filename}: {e}")
            enriched_prose = parsed.text

    if enriched_prose and len(enriched_prose) > len(parsed.text or "") + 200:
        try:
            doc_summary = llm_generate(
                system_prompt=(
                    "Summarize ALL items in this document in 2-4 sentences. "
                    "Include names, amounts, hostnames, IPs, and dates when present."
                ),
                user_prompt=enriched_prose[:6000],
                temperature=0.0, max_tokens=1024,
            )
            if doc_summary:
                doc_context = (
                    f"Document: {parsed.filename} (type: {parsed.doc_type}, "
                    f"category: {category})\nSummary: {doc_summary}"
                )
        except Exception as se:
            logger.warning(f"Enriched summary generation failed: {se}")

    kg_source_text = enriched_prose if enriched_prose else parsed.text
    chunk_source = kg_source_text

    for tier_name, tier_size, tier_overlap in CHUNK_TIERS:
        if is_spreadsheet:
            # Structure-aware, row-atomic chunks for messy + failed-clean sheets
            # only. Clean sheets already in the structured store contribute none.
            tier_chunks = build_tier_chunks(text_sheets, chunk_size=tier_size)
            if enriched_prose and "## Embedded figures" in enriched_prose:
                from src.ingestion.chunker import Chunk
                fig_only = enriched_prose.split("## Embedded figures", 1)[-1].strip()
                if fig_only:
                    extra = chunk_text(fig_only, chunk_size=tier_size, chunk_overlap=tier_overlap)
                    base_i = len(tier_chunks)
                    for j, c in enumerate(extra):
                        tier_chunks.append(
                            Chunk(text=c.text, index=base_i + j, start_char=c.start_char)
                        )
        elif is_pdf or is_docx or enriched_prose:
            tier_chunks = chunk_text(chunk_source or "", chunk_size=tier_size, chunk_overlap=tier_overlap)
        else:
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
    # (Spreadsheet structured ingest + de-dup happened in the chunking loop above.)
    # Ensure category exists in categories table
    if category and category != "uncategorized":
        existing = await metadata_store.get_category(category)
        if not existing:
            await metadata_store.add_category(
                name=category, description="", acl_groups=acl_groups, routing_keywords=[],
            )
    # Knowledge graph: full text for PDF/DOCX; figure-only for Excel with images;
    # skip pure spreadsheet cell dumps.
    spreadsheet_figure_kg = ""
    if is_spreadsheet and enriched_prose and "## Embedded figures" in enriched_prose:
        spreadsheet_figure_kg = enriched_prose.split("## Embedded figures", 1)[-1].strip()
    if not is_spreadsheet or spreadsheet_figure_kg:
        from src.knowledge.graph_rag import insert_document as lightrag_insert
        await lightrag_insert(
            spreadsheet_figure_kg or kg_source_text or parsed.text,
            doc_id=doc_id,
            filename=parsed.filename,
        )
    return IngestResult(
        doc_id=doc_id,
        filename=parsed.filename,
        doc_type=parsed.doc_type,
        chunk_count=total_chunks,
    )
