import pytest
from unittest.mock import patch
from src.agent.evaluator import evaluate_context
from src.agent.state import AgentState, QueryType
from src.retrieval.models import RetrievedChunk, ChunkMetadata


def _make_chunk(text, score=0.9):
    return RetrievedChunk(
        text=text,
        score=score,
        metadata=ChunkMetadata(
            doc_id="d1",
            filename="test.pdf",
            doc_type="pdf",
            chunk_index=0,
            start_char=0,
            acl_groups=["finance"],
        ),
    )


def test_sufficient_context():
    with patch("src.agent.evaluator.generate", return_value='{"sufficient": true, "reason": "Context contains the answer"}'):
        state = AgentState(
            question="What is policy 4.2?",
            user_groups=["finance"],
            query_type=QueryType.LOOKUP,
            retrieved_chunks=[_make_chunk("Policy 4.2: Expenses over $500 need approval")],
            retrieval_attempts=1,
            needs_reretrieval=False,
        )
        result = evaluate_context(state)
    assert result["needs_reretrieval"] is False


def test_insufficient_context():
    with patch("src.agent.evaluator.generate", return_value='{"sufficient": false, "reason": "No relevant information found"}'):
        state = AgentState(
            question="What is policy 4.2?",
            user_groups=["finance"],
            query_type=QueryType.LOOKUP,
            retrieved_chunks=[_make_chunk("This is about server maintenance", score=0.3)],
            retrieval_attempts=1,
            needs_reretrieval=False,
        )
        result = evaluate_context(state)
    assert result["needs_reretrieval"] is True


def test_max_retrieval_attempts_stops_loop():
    state = AgentState(
        question="Something",
        user_groups=["finance"],
        query_type=QueryType.LOOKUP,
        retrieved_chunks=[],
        retrieval_attempts=3,
        needs_reretrieval=False,
    )
    result = evaluate_context(state)
    assert result["needs_reretrieval"] is False


def test_empty_chunks_needs_reretrieval():
    state = AgentState(
        question="Something",
        user_groups=["finance"],
        query_type=QueryType.LOOKUP,
        retrieved_chunks=[],
        retrieval_attempts=1,
        needs_reretrieval=False,
    )
    result = evaluate_context(state)
    assert result["needs_reretrieval"] is True
