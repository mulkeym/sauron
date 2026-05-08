import pytest
from unittest.mock import MagicMock, AsyncMock
from src.mcp.resources import get_document_resource, get_category_resource, get_schema_resource
from src.db.schema_registry import SchemaRegistry, TableSchema, ColumnSchema


@pytest.mark.asyncio
async def test_get_document_resource():
    mock_metadata = AsyncMock()
    mock_doc = MagicMock()
    mock_doc.doc_id = "doc-1"
    mock_doc.filename = "policy.pdf"
    mock_doc.doc_type = "pdf"
    mock_doc.category = "finance_policies"
    mock_doc.acl_groups = ["finance"]
    mock_doc.chunk_count = 5
    mock_metadata.get_document.return_value = mock_doc
    result = await get_document_resource("doc-1", user_groups=["finance"], metadata_store=mock_metadata)
    assert result["doc_id"] == "doc-1"
    assert result["filename"] == "policy.pdf"


@pytest.mark.asyncio
async def test_get_document_resource_access_denied():
    mock_metadata = AsyncMock()
    mock_doc = MagicMock()
    mock_doc.acl_groups = ["finance"]
    mock_metadata.get_document.return_value = mock_doc
    result = await get_document_resource("doc-1", user_groups=["it_support"], metadata_store=mock_metadata)
    assert "error" in result
    assert "access" in result["error"].lower() or "denied" in result["error"].lower()


def test_get_category_resource():
    mock_metadata = MagicMock()
    mock_doc1 = MagicMock()
    mock_doc1.category = "finance_policies"
    mock_doc1.doc_id = "d1"
    mock_doc1.filename = "a.pdf"
    mock_doc1.doc_type = "pdf"
    mock_doc1.acl_groups = ["finance"]
    mock_metadata.list_documents = AsyncMock(return_value=[mock_doc1])
    result = get_category_resource("finance_policies", user_groups=["finance"], metadata_store=mock_metadata)
    assert result["name"] == "finance_policies"
    assert len(result["documents"]) == 1


def test_get_schema_resource():
    registry = SchemaRegistry()
    registry.register(
        TableSchema(
            database="finance_db",
            table="budget",
            columns=[ColumnSchema(name="amount", dtype="numeric", description="USD")],
            description="Budget data",
            acl_groups=["finance"],
        )
    )
    result = get_schema_resource("finance_db", user_groups=["finance"], schema_registry=registry)
    assert len(result["tables"]) == 1
    assert result["tables"][0]["table"] == "budget"
