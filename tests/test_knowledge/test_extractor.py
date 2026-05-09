import pytest
from unittest.mock import patch
from src.knowledge.extractor import extract_entities, ExtractionResult


def test_extract_entities_from_text():
    mock_response = '{"entities": [{"name": "Mike", "type": "person"}, {"name": "Policy 4.2", "type": "policy"}], "relationships": [{"source": "Policy 4.2", "target": "expense reporting", "type": "governs"}], "sections": []}'
    with patch("src.knowledge.extractor.generate", return_value=mock_response):
        result = extract_entities("Policy 4.2 governs expense reporting. Mike reviewed it.")
    assert len(result.entities) == 2
    assert len(result.relationships) == 1


def test_extract_entities_with_sections():
    mock_response = '{"entities": [{"name": "TOEE 26", "type": "project"}], "relationships": [], "sections": [{"name": "Section 4.2: Expense Reporting", "parent": null}]}'
    with patch("src.knowledge.extractor.generate", return_value=mock_response):
        result = extract_entities("Section 4.2: Expense Reporting...")
    assert len(result.sections) == 1


def test_extract_entities_bad_json_returns_empty():
    with patch("src.knowledge.extractor.generate", return_value="I can't extract anything"):
        result = extract_entities("random text")
    assert result.entities == []


def test_extract_entities_empty_text():
    result = extract_entities("")
    assert result.entities == []
