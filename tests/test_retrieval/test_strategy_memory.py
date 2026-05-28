import sys
import types
import pytest


class _Rec:
    def __init__(self, qt, discovered, relevant, cited, t=1.0):
        self.query_type = qt
        self.strategy_used = qt
        self.docs_discovered = discovered
        self.docs_relevant = relevant
        self.docs_cited = cited
        self.total_time_seconds = t


@pytest.mark.asyncio
async def test_best_strategy_prefers_cited_over_precision(monkeypatch):
    from src.retrieval import strategy_memory as sm

    records = [
        _Rec("lookup", discovered=1, relevant=1, cited=1),
        _Rec("lookup", discovered=1, relevant=1, cited=1),
        _Rec("lookup", discovered=1, relevant=1, cited=1),
        _Rec("sweep", discovered=10, relevant=8, cited=8),
        _Rec("sweep", discovered=10, relevant=8, cited=8),
        _Rec("sweep", discovered=10, relevant=8, cited=8),
    ]

    class FakeSession:
        async def __aenter__(self): return self
        async def __aexit__(self, *a): return False
        async def execute(self, q):
            class R:
                def scalars(self_):
                    class S:
                        def all(self__): return records
                    return S()
            return R()

    class FakeStore:
        def session_factory(self): return FakeSession()

    # Inject a lightweight fake for src.api.routes_ingest so the heavy
    # import chain (lancedb, etc.) is never triggered.
    fake_routes = types.ModuleType("src.api.routes_ingest")
    fake_routes.get_metadata_store = lambda: FakeStore()
    monkeypatch.setitem(sys.modules, "src.api.routes_ingest", fake_routes)

    monkeypatch.setattr(sm.settings, "strategy_memory_enabled", True)

    best = await sm.get_best_strategy("how many contracts did the army award")
    assert best["strategy"] == "sweep"        # cited-weighted winner
    assert best["count"] == 3
    assert best["margin"] > 0
