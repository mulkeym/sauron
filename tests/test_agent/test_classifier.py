"""Tests for the table-aware query classifier."""
from unittest.mock import patch
from src.agent.state import QueryType, AgentState
from src.agent import classifier
from src.agent.classifier import format_available_tables, classify_query
from src.db.schema_registry import TableSchema, ColumnSchema
from src.agent.classifier import _classify_node_factory
from src.db.schema_registry import SchemaRegistry


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


def test_node_factory_passes_acl_filtered_tables(monkeypatch):
    captured = {}
    def fake_generate(system_prompt, user_prompt, **kwargs):
        captured["system"] = system_prompt
        return '{"query_type": "analytical", "sub_tasks": []}'
    monkeypatch.setattr(classifier, "generate", fake_generate)

    reg = SchemaRegistry()
    reg.register(_schema(table="doc_pay", desc="pay", acl=["ALL"]))
    reg.register(_schema(table="doc_secret", desc="secret", acl=["admins"]))

    node = _classify_node_factory(reg)
    out = node({"question": "pay?", "user_groups": ["ALL"]})

    assert "doc_pay" in captured["system"]          # visible to ALL
    assert "doc_secret" not in captured["system"]   # ACL-filtered out
    assert out["query_type"] == QueryType.ANALYTICAL


def test_node_factory_with_no_registry(monkeypatch):
    captured = {}
    def fake_generate(system_prompt, user_prompt, **kwargs):
        captured["system"] = system_prompt
        return '{"query_type": "lookup", "sub_tasks": []}'
    monkeypatch.setattr(classifier, "generate", fake_generate)

    node = _classify_node_factory(None)
    node({"question": "x", "user_groups": ["ALL"]})
    assert "Available structured tables" not in captured["system"]


def test_classify_lookup():
    with patch("src.agent.classifier.generate", return_value='{"query_type": "lookup", "sub_tasks": ["Find policy 4.2 content"]}'):
        state = AgentState(question="What does policy 4.2 say?", user_groups=["finance"])
        result = classify_query(state)
    assert result["query_type"] == QueryType.LOOKUP


def test_classify_sweep():
    with patch("src.agent.classifier.generate", return_value='{"query_type": "sweep", "sub_tasks": ["Find all questions by Mike in meetings"]}'):
        state = AgentState(question="What questions did Mike ask in all meetings the last 30 days?", user_groups=["engineering"])
        result = classify_query(state)
    assert result["query_type"] == QueryType.SWEEP


def test_classify_analytical():
    with patch("src.agent.classifier.generate", return_value='{"query_type": "analytical", "sub_tasks": ["Query Q3 revenue from database"]}'):
        state = AgentState(question="What was our Q3 revenue?", user_groups=["finance"])
        result = classify_query(state)
    assert result["query_type"] == QueryType.ANALYTICAL


def test_classify_cross_reference():
    with patch("src.agent.classifier.generate", return_value='{"query_type": "cross_reference", "sub_tasks": ["Get Q3 spending from database", "Find expense policy in docs"]}'):
        state = AgentState(question="Does our Q3 spending comply with policy 4.2?", user_groups=["finance"])
        result = classify_query(state)
    assert result["query_type"] == QueryType.CROSS_REFERENCE
    assert len(result["sub_tasks"]) == 2


def test_classify_temporal():
    with patch("src.agent.classifier.generate", return_value='{"query_type": "temporal", "sub_tasks": ["Find docs changed in last month"]}'):
        state = AgentState(question="What policies changed last month?", user_groups=["finance"])
        result = classify_query(state)
    assert result["query_type"] == QueryType.TEMPORAL


def test_classify_fallback_on_bad_json():
    with patch("src.agent.classifier.generate", return_value="I'm not sure how to classify this"):
        state = AgentState(question="Tell me something", user_groups=["finance"])
        result = classify_query(state)
    assert result["query_type"] == QueryType.LOOKUP
