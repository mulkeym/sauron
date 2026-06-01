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

def test_agent_state_accepts_structured_trace():
    from src.agent.state import AgentState
    st: AgentState = {"question": "q", "structured_trace": {"status": "ran", "query_type": "sweep"}}
    assert st["structured_trace"]["status"] == "ran"
    assert "structured_trace" in AgentState.__annotations__


def test_agent_state_declares_progress_channel():
    # `progress` MUST be a declared field so LangGraph keeps it as a state channel;
    # otherwise the injected reporter is silently dropped and classify sub-steps
    # never reach the async status.
    from src.agent.state import AgentState
    assert "progress" in AgentState.__annotations__
