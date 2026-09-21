"""Index already-extracted content without opening the source document."""
import logging

from src.ingestion.tabular_chunker import sheets_needing_text
from src.ingestion.tabular_ingest import ingest_grids, SPREADSHEET_DOC_TYPES

logger = logging.getLogger(__name__)


async def index_prepared(prepared, doc_id, acl_groups, category, vector_store,
                         metadata_store, dataset_id=0):
    parsed = prepared.parsed
    for warning in prepared.warnings:
        logger.warning("%s: %s", parsed.filename, warning)
    text_sheets = None
    enriched_prose = None
    figure_records = []

    async def store_grids(grids):
        return await ingest_grids(
            grids, doc_id, parsed.filename, parsed.doc_type,
            acl_groups, category, vector_store, metadata_store, dataset_id=dataset_id,
        )

    if parsed.doc_type in SPREADSHEET_DOC_TYPES:
        grids = prepared.spreadsheet_grids
        classifications, ingested = await store_grids(grids)
        text_sheets = sheets_needing_text(grids, classifications, ingested)
        if prepared.office:
            if prepared.office.table_grids:
                await store_grids(prepared.office.table_grids)
            figure_records = prepared.office.figures
            if prepared.office.enriched_text.strip():
                enriched_prose = ((parsed.text or "") + "\n\n## Embedded figures\n\n"
                                  + prepared.office.enriched_text).strip()
    elif prepared.pdf is not None:
        figure_records = prepared.pdf.figure_records
        try:
            await store_grids(prepared.pdf.table_grids)
            enriched_prose = "\n\n".join(b.text for b in prepared.pdf.prose_blocks)
        except Exception as exc:
            logger.warning("PDF table indexing failed for %s; using flat text: %s", parsed.filename, exc)
    elif prepared.office is not None:
        enriched_prose = prepared.office.enriched_text
        figure_records = prepared.office.figures
        if prepared.office.table_grids:
            await store_grids(prepared.office.table_grids)
    if prepared.figure_staging:
        from src.figures.storage import FigureStore
        figures = await FigureStore().publish_async(doc_id, figure_records, prepared.figure_staging)
        await metadata_store.put_figures(doc_id, figures)
    return text_sheets, enriched_prose, figure_records


def figure_index_entries(records):
    """Keep all labels on large Visio pages searchable with the same PNG citation."""
    from src.ingestion.chunker import chunk_text
    for record in records:
        if record.source_text or record.source == 'emf_render':
            header = f"Figure: {record.figure_id}\nCaption: {record.caption}\n"
            # Every search chunk carries its own evidence type, including chunks
            # after the first one in a long OCR or model-generated description.
            for label, text in (
                ('Source information', record.source_text or record.description),
                ('OCR text (recognition may contain errors)', record.ocr_text),
                ('Visual model interpretation (not source-verified connector facts)', record.vision_description),
            ):
                for chunk in chunk_text(text, chunk_size=1400, chunk_overlap=150) if text.strip() else []:
                    yield record, header + label + ':\n' + chunk.text
            continue
        if record.source != 'visio_page_render':
            yield record, record.retrieval_text()
            continue
        header = f"Figure: {record.figure_id}\nVisio page: {record.caption}\nSource page ID: {record.source_page_id}\n"
        for chunk in chunk_text(record.description, chunk_size=1400, chunk_overlap=150):
            yield record, header + chunk.text
