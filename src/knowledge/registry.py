import asyncio
from src.db.metadata import MetadataStore
from src.db.schema_registry import SchemaRegistry


class KnowledgeRegistry:
    def __init__(self, metadata_store: MetadataStore, schema_registry: SchemaRegistry):
        self._metadata_store = metadata_store
        self._schema_registry = schema_registry

    def _get_categories(self):
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None
        if loop and loop.is_running():
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as pool:
                return pool.submit(asyncio.run, self._metadata_store.list_categories()).result()
        return asyncio.run(self._metadata_store.list_categories())

    def suggest_sources(self, query, user_groups):
        categories = self._get_categories()
        query_lower = query.lower()
        suggestions = []
        for cat in categories:
            if not any(g in cat.acl_groups for g in user_groups):
                continue
            for keyword in cat.routing_keywords:
                if keyword.lower() in query_lower:
                    suggestions.append(cat.name)
                    break
        return suggestions

    def get_all_sources(self, user_groups):
        sources = []
        categories = self._get_categories()
        for cat in categories:
            if any(g in cat.acl_groups for g in user_groups):
                sources.append({"name": cat.name, "type": "document_category", "description": cat.description, "routing_keywords": cat.routing_keywords})
        for schema in self._schema_registry.list_for_user(user_groups):
            sources.append({"name": f"{schema.database}.{schema.table}", "type": "database", "description": schema.description})
        return sources
