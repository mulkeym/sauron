import pytest
from unittest.mock import patch
from src.agent.classifier import classify_query
from src.agent.state import AgentState, QueryType

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
