import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from src.generation.rag_chain import agent_query_streamed, agent_query, RAGResponse
from src.retrieval.models import Citation


@pytest.mark.asyncio
async def test_streamed_fires_callback_per_node_on_cache_miss():
    seen = []
    miss = MagicMock(accepted=False, query_vector=None)
    resp = RAGResponse(answer="A", citations=[])

    async def fake_run_streamed(*args, **kwargs):
        cb = kwargs["step_callback"]
        for node in ("classify", "retrieve", "synthesize"):
            cb(node)
        return resp

    with patch("src.generation.rag_chain.judged_cache_lookup", new_callable=AsyncMock, return_value=miss):
        with patch("src.agent.graph.run_agent_streamed", side_effect=fake_run_streamed):
            out = await agent_query_streamed(
                "q", ["finance"], MagicMock(), MagicMock(), None,
                step_callback=seen.append,
            )
    assert out.answer == "A"
    assert seen == ["cache_check", "classify", "retrieve", "synthesize"]


@pytest.mark.asyncio
async def test_streamed_returns_cached_without_running_graph():
    cached = {"answer": "cached!", "citations": [], "cached_query": "old q"}
    decision = MagicMock(accepted=True, cached=cached, query_vector=None)
    with patch("src.generation.rag_chain.judged_cache_lookup", new_callable=AsyncMock, return_value=decision):
        with patch("src.agent.graph.run_agent_streamed", new_callable=AsyncMock) as run:
            out = await agent_query_streamed("q", ["finance"], MagicMock(), MagicMock(), None)
    assert out.cached is True
    assert out.answer == "cached!"
    run.assert_not_awaited()


@pytest.mark.asyncio
async def test_agent_query_delegates_with_no_callback():
    miss = MagicMock(accepted=False, query_vector=None)
    resp = RAGResponse(answer="Z", citations=[])
    with patch("src.generation.rag_chain.judged_cache_lookup", new_callable=AsyncMock, return_value=miss):
        with patch("src.agent.graph.run_agent_streamed", new_callable=AsyncMock, return_value=resp) as run:
            out = await agent_query("q", ["finance"], MagicMock(), MagicMock(), None)
    assert out.answer == "Z"
    assert run.await_args.kwargs["step_callback"] is None
