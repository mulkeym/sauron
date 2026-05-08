import pytest
from unittest.mock import AsyncMock, MagicMock
from src.knowledge.registry import KnowledgeRegistry


@pytest.fixture
def mock_deps():
    metadata_store = AsyncMock()
    from src.db.schema_registry import SchemaRegistry
    return metadata_store, SchemaRegistry()


def test_get_routing_suggestion(mock_deps):
    metadata_store, schema_registry = mock_deps
    cat = MagicMock()
    cat.name = "finance_policies"
    cat.routing_keywords = ["expense", "budget", "revenue"]
    cat.acl_groups = ["finance"]
    metadata_store.list_categories.return_value = [cat]
    registry = KnowledgeRegistry(metadata_store=metadata_store, schema_registry=schema_registry)
    suggestions = registry.suggest_sources("What is our expense policy?", user_groups=["finance"])
    assert "finance_policies" in suggestions


def test_routing_no_match(mock_deps):
    metadata_store, schema_registry = mock_deps
    cat = MagicMock()
    cat.name = "finance_policies"
    cat.routing_keywords = ["expense", "budget"]
    cat.acl_groups = ["finance"]
    metadata_store.list_categories.return_value = [cat]
    registry = KnowledgeRegistry(metadata_store=metadata_store, schema_registry=schema_registry)
    suggestions = registry.suggest_sources("Tell me about server maintenance", user_groups=["finance"])
    assert isinstance(suggestions, list)


def test_routing_respects_acl(mock_deps):
    metadata_store, schema_registry = mock_deps
    cat_finance = MagicMock()
    cat_finance.name = "finance_policies"
    cat_finance.routing_keywords = ["expense"]
    cat_finance.acl_groups = ["finance"]
    cat_it = MagicMock()
    cat_it.name = "it_runbooks"
    cat_it.routing_keywords = ["server"]
    cat_it.acl_groups = ["it_support"]
    metadata_store.list_categories.return_value = [cat_finance, cat_it]
    registry = KnowledgeRegistry(metadata_store=metadata_store, schema_registry=schema_registry)
    suggestions = registry.suggest_sources("expense policy", user_groups=["it_support"])
    assert "finance_policies" not in suggestions


def test_get_all_sources(mock_deps):
    metadata_store, schema_registry = mock_deps
    cat = MagicMock()
    cat.name = "finance"
    cat.description = "Finance docs"
    cat.acl_groups = ["finance"]
    cat.routing_keywords = ["budget"]
    metadata_store.list_categories.return_value = [cat]
    from src.db.schema_registry import TableSchema
    schema_registry.register(TableSchema(database="fin_db", table="budget", columns=[], description="Budget", acl_groups=["finance"]))
    registry = KnowledgeRegistry(metadata_store=metadata_store, schema_registry=schema_registry)
    sources = registry.get_all_sources(user_groups=["finance"])
    assert len(sources) == 2
