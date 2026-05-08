import pytest
from unittest.mock import patch, AsyncMock, MagicMock
from src.knowledge.categorizer import categorize_document, CategorizationResult


@pytest.fixture
def mock_store():
    store = AsyncMock()
    cat1 = MagicMock()
    cat1.name = "finance_policies"
    cat1.description = "Finance policy documents"
    cat1.routing_keywords = ["expense", "budget", "revenue"]
    store.list_categories.return_value = [cat1]
    return store


def test_categorize_matches_existing(mock_store):
    with patch("src.knowledge.categorizer.generate", return_value='{"category": "finance_policies", "confidence": 0.95, "is_new": false}'):
        result = categorize_document(filename="expense_policy.pdf", doc_type="pdf", text_preview="Expenses over $500...", metadata_store=mock_store)
    assert result.category == "finance_policies"
    assert result.is_new is False


def test_categorize_proposes_new(mock_store):
    with patch("src.knowledge.categorizer.generate", return_value='{"category": "legal_compliance", "confidence": 0.85, "is_new": true, "description": "Legal docs", "suggested_acl_groups": ["legal"], "suggested_keywords": ["contract"]}'):
        result = categorize_document(filename="contract.pdf", doc_type="pdf", text_preview="This agreement...", metadata_store=mock_store)
    assert result.category == "legal_compliance"
    assert result.is_new is True


def test_categorize_fallback_on_bad_json(mock_store):
    with patch("src.knowledge.categorizer.generate", return_value="not json"):
        result = categorize_document(filename="x.txt", doc_type="txt", text_preview="stuff", metadata_store=mock_store)
    assert result.category == "uncategorized"
