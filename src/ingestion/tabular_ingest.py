"""Wire clean spreadsheet sheets into the structured store + row narratives.

Called from the ingestion pipeline for spreadsheet documents. For each CLEAN
sheet: load rows into DuckDB, profile the columns (one LLM call), register +
persist the schema, and embed deterministic per-row narratives (the raw rows
stay authoritative in DuckDB). Fail-open per sheet — one bad sheet never aborts
the others or the surrounding document ingestion.
"""
import logging
from pathlib import Path

from src.ingestion.tabular import (
    read_sheets, classify_sheet, SheetGrid, SheetClassification,
)
from src.ingestion.tabular_chunker import find_table_region, messy_region_narratives
from src.ingestion.tabular_store import (
    connect_tabular, load_sheet_to_duckdb, schema_from_sheet,
)
from src.ingestion.table_profiler import profile_table, build_row_narratives
from src.ingestion.embedder import embed_texts
from src.retrieval.models import ChunkMetadata

logger = logging.getLogger(__name__)


def _ingest_one_clean_sheet(con, grid, cls, doc_id, acl_groups, schema_registry,
                            generate_fn):
    """Structured-ingest ONE already-classified clean sheet's DuckDB rows + build
    its enriched, registered schema. Returns ``(col_names, profile, data_rows,
    schema)``; the async caller persists the schema and embeds narratives via
    ``_save_and_embed_clean``. Raises on any failure (caller decides fail-open).
    This is the exact per-sheet logic previously inlined in
    ``ingest_spreadsheet_tables``."""
    _, col_names = load_sheet_to_duckdb(con, doc_id, grid.sheet_name, cls, grid)
    data_rows = grid.rows[cls.header_row_index + 1:]
    profile = profile_table(grid.sheet_name, col_names, cls.column_dtypes,
                            data_rows[:5], generate_fn=generate_fn)
    schema = schema_from_sheet(doc_id, grid.sheet_name, cls, grid, acl_groups=acl_groups)
    for col in schema.columns:
        if col.name in profile.column_descriptions:
            col.description = profile.column_descriptions[col.name]
    if profile.table_description:
        schema.description = profile.table_description
    schema_registry.register(schema)
    return col_names, profile, data_rows, schema


async def _save_and_embed_clean(metadata_store, vector_store, schema, col_names, profile,
                                data_rows, doc_id, filename, doc_type, acl_groups, category,
                                chunk_index):
    """Persist a clean sheet's schema and embed its per-row narratives. Returns
    the next chunk_index."""
    await metadata_store.save_schema(schema)
    narratives = [n for n in build_row_narratives(
        col_names, profile, data_rows, context=profile.table_description) if n.strip()]
    if narratives:
        vectors = embed_texts(narratives)
        metadatas = []
        for _ in narratives:
            metadatas.append(ChunkMetadata(
                doc_id=doc_id, filename=filename, doc_type=doc_type,
                chunk_index=chunk_index, start_char=0, acl_groups=acl_groups,
                category=category, chunk_size_tier="table_row",
            ))
            chunk_index += 1
        vector_store.upsert(texts=narratives, vectors=vectors, metadatas=metadatas)
    return chunk_index


async def ingest_spreadsheet_tables(file_path, doc_id, filename, doc_type, acl_groups,
                                    category, vector_store, metadata_store,
                                    schema_registry=None, generate_fn=None) -> int:
    """Backward-compatible wrapper: structured-ingest clean sheets only and
    return the count fully processed. New callers should use
    ``ingest_structured_sheets`` (which also returns grids/classifications and
    handles messy region narratives)."""
    _, _, ingested = await ingest_structured_sheets(
        file_path, doc_id, filename, doc_type, acl_groups, category,
        vector_store, metadata_store, schema_registry=schema_registry,
        generate_fn=generate_fn,
    )
    return len(ingested)


async def ingest_structured_sheets(file_path, doc_id, filename, doc_type, acl_groups,
                                   category, vector_store, metadata_store,
                                   schema_registry=None, generate_fn=None):
    """Read a spreadsheet's sheets once, structured-ingest clean sheets, and embed
    deterministic region narratives for messy sheets.

    Returns ``(grids, classifications, ingested_names)`` where ``ingested_names``
    is the set of CLEAN sheet names whose DuckDB rows + schema + per-row
    narratives ALL succeeded. The caller uses that set (via
    ``tabular_chunker.sheets_needing_text``) to decide which sheets still need
    full-text chunks. Fully fail-open: a read failure returns ([], [], set());
    a per-sheet failure is logged and the sheet is simply absent from
    ``ingested_names`` (so it falls back to text chunks).
    """
    if schema_registry is None:
        from src.api.routes_ingest import get_schema_registry
        schema_registry = get_schema_registry()

    try:
        grids = read_sheets(Path(file_path))
    except Exception as e:
        logger.warning(f"Tabular ingest: could not read sheets from {filename}: {e}")
        return [], [], set()

    classifications = [classify_sheet(g) for g in grids]
    for g, c in zip(grids, classifications):
        logger.info(
            f"Tabular ingest [{filename}]: sheet '{g.sheet_name}' classified {c.route.upper()} "
            f"({len(g.rows)} rows, header_row={c.header_row_index})")
    ingested: set[str] = set()
    chunk_index = 0
    con = None
    try:
        con = connect_tabular(read_only=False)
        for grid, cls in zip(grids, classifications):
            if cls.route == "clean":
                try:
                    col_names, profile, data_rows, schema = _ingest_one_clean_sheet(
                        con, grid, cls, doc_id, acl_groups, schema_registry, generate_fn)
                    chunk_index = await _save_and_embed_clean(
                        metadata_store, vector_store, schema, col_names, profile,
                        data_rows, doc_id, filename, doc_type, acl_groups, category,
                        chunk_index)
                    ingested.add(grid.sheet_name)
                    logger.info(
                        f"Tabular ingest [{filename}]: sheet '{grid.sheet_name}' CLEAN -> "
                        f"structured ({len(data_rows)} data rows to DuckDB + schema + "
                        f"narratives); full-text chunks suppressed")
                except Exception as e:
                    logger.warning(
                        f"Tabular ingest: failed on clean sheet '{grid.sheet_name}' "
                        f"of {filename}: {e}")
                    continue
            else:  # messy: deterministic region narratives (no LLM, no DuckDB)
                try:
                    region = find_table_region(grid.rows)
                    if region is None:
                        logger.info(
                            f"Tabular ingest [{filename}]: sheet '{grid.sheet_name}' MESSY -> "
                            f"no table-like region; full-text chunks only")
                        continue
                    narratives = messy_region_narratives(grid, region)
                    if not narratives:
                        logger.info(
                            f"Tabular ingest [{filename}]: sheet '{grid.sheet_name}' MESSY -> "
                            f"region {region} produced no narratives; full-text chunks only")
                        continue
                    logger.info(
                        f"Tabular ingest [{filename}]: sheet '{grid.sheet_name}' MESSY -> "
                        f"region rows {region}, {len(narratives)} region narrative(s) + "
                        f"full-text chunks")
                    vectors = embed_texts(narratives)
                    metadatas = [ChunkMetadata(
                        doc_id=doc_id, filename=filename, doc_type=doc_type,
                        chunk_index=chunk_index + i, start_char=0, acl_groups=acl_groups,
                        category=category, chunk_size_tier="table_row",
                    ) for i in range(len(narratives))]
                    chunk_index += len(narratives)
                    vector_store.upsert(texts=narratives, vectors=vectors, metadatas=metadatas)
                except Exception as e:
                    logger.warning(
                        f"Tabular ingest: region narratives failed on messy sheet "
                        f"'{grid.sheet_name}' of {filename}: {e}")
                    continue
    finally:
        if con is not None:
            con.close()

    logger.info(f"Tabular ingest: structured {len(ingested)} clean sheet(s) from {filename}")
    return grids, classifications, ingested


async def populate_schema_registry(metadata_store, schema_registry) -> int:
    """Load every persisted TableSchema into the in-memory registry. Returns count."""
    schemas = await metadata_store.load_all_schemas()
    for schema in schemas:
        schema_registry.register(schema)
    return len(schemas)


async def populate_hint_store(metadata_store, hint_store) -> int:
    """Load all persisted SchemaHints into the in-memory HintStore. Returns count."""
    hints = await metadata_store.load_all_hints()
    for h in hints:
        hint_store.register(h)
    return len(hints)


SPREADSHEET_DOC_TYPES = ("xlsx", "xls", "csv", "tsv")


async def cleanup_spreadsheet_tables(doc_id, metadata_store, schema_registry=None) -> int:
    """Drop a document's DuckDB tables and delete/unregister its schemas.

    Symmetric with the spreadsheet branch in the ingestion entry points — called
    from EVERY document-delete path, because the standard delete (metadata +
    chunks + KG) does NOT otherwise
    clean the structured store, leaving orphaned DuckDB tables and stale schemas
    (which would reload into the registry on restart). Fail-open. Returns the
    number of DuckDB tables dropped. A no-op for non-spreadsheet docs (no tables
    match the prefix).
    """
    if schema_registry is None:
        from src.api.routes_ingest import get_schema_registry
        schema_registry = get_schema_registry()

    from src.ingestion.tabular_store import connect_tabular, duckdb_table_name
    prefix = duckdb_table_name(doc_id, "")  # "doc_<safe_doc_id>_"

    # Remove this doc's schemas from the persistent store AND the live registry.
    try:
        for sc in await metadata_store.load_all_schemas():
            if sc.table.startswith(prefix):
                await metadata_store.delete_schema(sc.database, sc.table)
                schema_registry.remove(sc.database, sc.table)
    except Exception as e:
        logger.warning(f"Schema cleanup failed for doc {doc_id}: {e}")

    # Drop this doc's DuckDB tables.
    dropped = 0
    try:
        con = connect_tabular(read_only=False)
        try:
            tables = [r[0] for r in con.execute(
                "SELECT table_name FROM information_schema.tables").fetchall()]
            for t in tables:
                if t.startswith(prefix):
                    con.execute(f'DROP TABLE IF EXISTS "{t}"')
                    dropped += 1
        finally:
            con.close()
    except Exception as e:
        logger.warning(f"DuckDB table cleanup failed for doc {doc_id}: {e}")

    return dropped


async def purge_orphan_schemas(metadata_store, schema_registry=None) -> int:
    """Drop DuckDB tables and delete/unregister schemas whose doc_id has no live
    document in the metadata store (e.g. leftover test fixtures). Only operates on
    the ``doc_`` table namespace. Fail-open. Returns the number of orphan schemas
    removed."""
    if schema_registry is None:
        from src.api.routes_ingest import get_schema_registry
        schema_registry = get_schema_registry()

    from src.ingestion.tabular_store import connect_tabular, duckdb_table_name

    try:
        live_docs = await metadata_store.list_documents()
        live_prefixes = [duckdb_table_name(d.doc_id, "") for d in live_docs]
    except Exception as e:
        logger.warning(f"Orphan schema purge aborted (could not list live docs): {e}")
        return 0

    def _is_orphan(table_name: str) -> bool:
        # Only our namespace; orphan if no live doc prefix matches.
        return table_name.startswith("doc_") and not any(
            table_name.startswith(p) for p in live_prefixes)

    removed = 0
    try:
        for sc in await metadata_store.load_all_schemas():
            if _is_orphan(sc.table):
                await metadata_store.delete_schema(sc.database, sc.table)
                schema_registry.remove(sc.database, sc.table)
                removed += 1
    except Exception as e:
        logger.warning(f"Orphan schema purge failed: {e}")

    try:
        con = connect_tabular(read_only=False)
        try:
            tables = [r[0] for r in con.execute(
                "SELECT table_name FROM information_schema.tables").fetchall()]
            for t in tables:
                if _is_orphan(t):
                    con.execute(f'DROP TABLE IF EXISTS "{t}"')
        finally:
            con.close()
    except Exception as e:
        logger.warning(f"Orphan DuckDB table purge failed: {e}")

    logger.info(f"Orphan schema purge removed {removed} schema(s)")
    return removed
