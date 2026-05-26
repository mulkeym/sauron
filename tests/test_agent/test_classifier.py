"""Tests for the table-aware query classifier."""
from src.agent.state import QueryType
from src.agent import classifier
from src.agent.classifier import format_available_tables, classify_query
from src.db.schema_registry import TableSchema, ColumnSchema


def _schema(table="doc_x_pay", desc="GS pay by grade and step", acl=None):
    return TableSchema(database="spreadsheets", table=table,
                       columns=[ColumnSchema("grade", "VARCHAR", "")],
                       description=desc, acl_groups=acl or ["ALL"])


def test_format_available_tables():
    assert format_available_tables([_schema()]) == "- doc_x_pay: GS pay by grade and step"
    assert format_available_tables([]) == ""


def test_classify_injects_tables_and_can_pick_analytical(monkeypatch):
    captured = {}
    def fake_generate(system_prompt, user_prompt, **kwargs):
        captured["system"] = system_prompt
        return '{"query_type": "analytical", "sub_tasks": []}'
    monkeypatch.setattr(classifier, "generate", fake_generate)

    out = classify_query({"question": "pay for GS-12 step 5"},
                         available_tables="- doc_x_pay: GS pay by grade and step")
    assert "doc_x_pay" in captured["system"]            # tables injected into the prompt
    assert "Available structured tables" in captured["system"]
    assert out["query_type"] == QueryType.ANALYTICAL


def test_classify_omits_tables_section_when_none(monkeypatch):
    captured = {}
    def fake_generate(system_prompt, user_prompt, **kwargs):
        captured["system"] = system_prompt
        return '{"query_type": "lookup", "sub_tasks": []}'
    monkeypatch.setattr(classifier, "generate", fake_generate)

    classify_query({"question": "who is John?"})
    assert "Available structured tables" not in captured["system"]
