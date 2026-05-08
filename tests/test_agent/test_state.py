import pytest
from src.agent.state import AgentState, QueryType

def test_agent_state_defaults():
    state = AgentState(question="What is policy 4.2?", user_groups=["finance"])
    assert state["question"] == "What is policy 4.2?"
    assert state["user_groups"] == ["finance"]

def test_query_type_enum():
    assert QueryType.LOOKUP == "lookup"
    assert QueryType.SWEEP == "sweep"
    assert QueryType.ANALYTICAL == "analytical"
    assert QueryType.CROSS_REFERENCE == "cross_reference"
    assert QueryType.TEMPORAL == "temporal"
