import asyncio
from src.db.metadata import MetadataStore
from src.db.schema_registry import SchemaRegistry


async def get_document_resource(doc_id: str, user_groups: list[str], metadata_store: MetadataStore) -> dict:
    doc = await metadata_store.get_document(doc_id)
    if doc is None:
        return {"error": "Document not found"}
    if "ALL" not in user_groups and not any(g in doc.acl_groups for g in user_groups):
        return {"error": "Access denied for this document"}
    return {
        "doc_id": doc.doc_id,
        "filename": doc.filename,
        "doc_type": doc.doc_type,
        "category": doc.category,
        "acl_groups": doc.acl_groups,
        "chunk_count": doc.chunk_count,
    }


def get_category_resource(category_name: str, user_groups: list[str], metadata_store: MetadataStore) -> dict:
    filter_groups = None if "ALL" in user_groups else user_groups
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None
    if loop and loop.is_running():
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor() as pool:
            docs = pool.submit(
                asyncio.run, metadata_store.list_documents(user_groups=filter_groups)
            ).result()
    else:
        docs = asyncio.run(metadata_store.list_documents(user_groups=filter_groups))
    category_docs = [d for d in docs if d.category == category_name]
    return {
        "name": category_name,
        "doc_count": len(category_docs),
        "documents": [
            {"doc_id": d.doc_id, "filename": d.filename, "doc_type": d.doc_type}
            for d in category_docs
        ],
    }


def get_schema_resource(database_name: str, user_groups: list[str], schema_registry: SchemaRegistry) -> dict:
    schemas = schema_registry.list_for_user(user_groups)
    db_schemas = [s for s in schemas if s.database == database_name]
    return {
        "database": database_name,
        "tables": [
            {
                "table": s.table,
                "description": s.description,
                "columns": [
                    {"name": c.name, "dtype": c.dtype, "description": c.description}
                    for c in s.columns
                ],
            }
            for s in db_schemas
        ],
    }
