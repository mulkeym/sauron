import pytest
from unittest.mock import patch, MagicMock, AsyncMock
from src.generation.rag_chain import RAGResponse
from src.retrieval.models import Citation


def _mock_rag_response(answer="Test answer", citations=None):
    return RAGResponse(
        answer=answer,
        citations=citations
        or [
            Citation(
                doc_id="d1",
                filename="test.pdf",
                doc_type="pdf",
                chunk_index=0,
                page=1,
                snippet="test snippet",
                relevance=0.9,
            )
        ],
    )


@pytest.mark.asyncio
async def test_ask_quick():
    from src.mcp.tools_high import ask

    with patch(
        "src.mcp.tools_high.agent_query",
        new_callable=AsyncMock,
        return_value=_mock_rag_response(),
    ):
        result = await ask(
            question="What is policy 4.2?",
            user_groups=["finance"],
            vector_store=MagicMock(),
            schema_registry=MagicMock(),
            depth="quick",
        )
    assert result["answer"] == "Test answer"
    assert len(result["citations"]) == 1


@pytest.mark.asyncio
async def test_ask_with_context():
    from src.mcp.tools_high import ask

    with patch(
        "src.mcp.tools_high.agent_query",
        new_callable=AsyncMock,
        return_value=_mock_rag_response(),
    ) as mock_agent:
        result = await ask(
            question="PTO policy?",
            user_groups=["hr"],
            vector_store=MagicMock(),
            schema_registry=MagicMock(),
            context="Onboarding new employee in CA",
        )
    # Verify context was included in the question passed to agent_query
    call_args = mock_agent.call_args
    assert "Onboarding" in call_args.kwargs.get("question", "") or "Onboarding" in str(call_args)


@pytest.mark.asyncio
async def test_summarize_topic():
    from src.mcp.tools_high import summarize_topic

    with patch(
        "src.mcp.tools_high.agent_query",
        new_callable=AsyncMock,
        return_value=_mock_rag_response("Summary of Q3 results"),
    ):
        result = await summarize_topic(
            topic="Q3 financial results",
            user_groups=["finance"],
            vector_store=MagicMock(),
            schema_registry=MagicMock(),
        )
    assert result["summary"] == "Summary of Q3 results"


@pytest.mark.asyncio
async def test_compare():
    from src.mcp.tools_high import compare

    with patch(
        "src.mcp.tools_high.agent_query",
        new_callable=AsyncMock,
        return_value=_mock_rag_response("Policy A requires X, Policy B requires Y"),
    ):
        result = await compare(
            item_a="Policy A",
            item_b="Policy B",
            user_groups=["finance"],
            vector_store=MagicMock(),
            schema_registry=MagicMock(),
        )
    assert "comparison" in result
