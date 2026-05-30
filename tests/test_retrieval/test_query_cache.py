"""Shared cache-decision helper."""
import pytest

from src.retrieval import query_cache as qc


@pytest.mark.asyncio
async def test_accepted_hit(monkeypatch):
    monkeypatch.setattr(qc, "embed_query", lambda q: [0.1, 0.2])
    monkeypatch.setattr(qc, "cache_lookup",
        lambda v, g, **k: {"answer": "A", "cached_query": "old q", "citations": [], "cached_at": 0})
    async def _judge(original_query, new_query, cached_answer):
        return {"applicable": True, "confidence": 0.9, "reason": "same"}
    monkeypatch.setattr(qc, "cache_judge", _judge)

    d = await qc.judged_cache_lookup("new q", ["ALL"])
    assert d.hit is True and d.accepted is True
    assert d.cached["answer"] == "A"
    assert d.judgment["applicable"] is True
    assert d.query_vector == [0.1, 0.2]


@pytest.mark.asyncio
async def test_judge_rejects(monkeypatch):
    monkeypatch.setattr(qc, "embed_query", lambda q: [0.0])
    monkeypatch.setattr(qc, "cache_lookup",
        lambda v, g, **k: {"answer": "A", "cached_query": "old", "citations": [], "cached_at": 0})
    async def _judge(**k):
        return {"applicable": False, "confidence": 0.1, "reason": "different"}
    monkeypatch.setattr(qc, "cache_judge", _judge)

    d = await qc.judged_cache_lookup("q", ["ALL"])
    assert d.hit is True and d.accepted is False


@pytest.mark.asyncio
async def test_miss_does_not_judge(monkeypatch):
    monkeypatch.setattr(qc, "embed_query", lambda q: [0.0])
    monkeypatch.setattr(qc, "cache_lookup", lambda v, g, **k: None)
    called = {"judge": False}
    async def _judge(**k):
        called["judge"] = True
        return {"applicable": True}
    monkeypatch.setattr(qc, "cache_judge", _judge)

    d = await qc.judged_cache_lookup("q", ["ALL"])
    assert d.hit is False and d.accepted is False
    assert d.query_vector == [0.0]
    assert called["judge"] is False


@pytest.mark.asyncio
async def test_skip_cache_skips_lookup(monkeypatch):
    monkeypatch.setattr(qc, "embed_query", lambda q: [0.5])
    called = {"lookup": False}
    def _lookup(v, g, **k):
        called["lookup"] = True
        return {"answer": "A", "cached_query": "x", "citations": [], "cached_at": 0}
    monkeypatch.setattr(qc, "cache_lookup", _lookup)

    d = await qc.judged_cache_lookup("q", ["ALL"], skip_cache=True)
    assert d.hit is False and d.accepted is False
    assert d.query_vector == [0.5]      # still embedded for a later cache_store
    assert called["lookup"] is False


@pytest.mark.asyncio
async def test_embed_failure_is_fail_open(monkeypatch):
    def _boom(q):
        raise RuntimeError("embed down")
    monkeypatch.setattr(qc, "embed_query", _boom)
    d = await qc.judged_cache_lookup("q", ["ALL"])
    assert d.query_vector is None and d.hit is False and d.accepted is False


@pytest.mark.asyncio
async def test_judge_failure_fails_open_to_accept(monkeypatch):
    # cache_judge already returns applicable=True on its own internal error;
    # the helper must honor that (serve the cache when the judge is down).
    monkeypatch.setattr(qc, "embed_query", lambda q: [0.0])
    monkeypatch.setattr(qc, "cache_lookup",
        lambda v, g, **k: {"answer": "A", "cached_query": "old", "citations": [], "cached_at": 0})
    async def _judge(**k):
        return {"applicable": True, "confidence": 0.5, "reason": "Judge unavailable, using cache"}
    monkeypatch.setattr(qc, "cache_judge", _judge)

    d = await qc.judged_cache_lookup("q", ["ALL"])
    assert d.accepted is True
