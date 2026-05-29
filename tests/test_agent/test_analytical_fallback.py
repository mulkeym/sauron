"""retrieve_analytical: zero-row SQL must fall back to structured narratives,
then to map-reduce only when structured gates out."""
import pytest

import src.agent.strategies.analytical as analytical
import src.agent.strategies.structured as structured
import src.agent.strategies.map_reduce as map_reduce
from src.agent.strategies.structured import StructuredLookupTrace


class _Schema:
    def __init__(self, table="doc_pay"):
        self.table = table


class _Registry:
    def list_for_user(self, groups):
        return [_Schema()]


def _state():
    return {"question": "pay range for an officer?", "user_groups": ["ALL"],
            "retrieval_attempts": 0}


@pytest.fixture(autouse=True)
def _stub_hints(monkeypatch):
    async def _no_hints(schemas, hint_store, metadata_store):
        return {}
    monkeypatch.setattr(structured, "resolve_hints_for_schemas", _no_hints)
    monkeypatch.setattr("src.api.routes_ingest.get_hint_store", lambda: object(), raising=False)
    monkeypatch.setattr("src.api.routes_ingest.get_metadata_store", lambda: object(), raising=False)


@pytest.mark.asyncio
async def test_rows_returned_unchanged(monkeypatch):
    trace = StructuredLookupTrace(query_type="analytical", status="ran", row_count=2,
                                  rows=[{"a": 1}, {"a": 2}])
    monkeypatch.setattr(structured, "run_structured_lookup", lambda *a, **k: trace)
    called = {"structured": False}
    async def _should_not_run(*a, **k):
        called["structured"] = True
        return {}
    monkeypatch.setattr(structured, "retrieve_structured", _should_not_run)

    out = await analytical.retrieve_analytical(_state(), vector_store=object(), schema_registry=_Registry())
    assert out["sql_results"] == [{"a": 1}, {"a": 2}]
    assert called["structured"] is False


@pytest.mark.asyncio
async def test_zero_rows_falls_back_to_structured(monkeypatch):
    trace = StructuredLookupTrace(query_type="analytical", status="ran", row_count=0, rows=[])
    monkeypatch.setattr(structured, "run_structured_lookup", lambda *a, **k: trace)
    async def _structured(state, vector_store, schema_registry):
        return {"sql_results": [], "retrieved_chunks": [{"id": "row1"}],
                "structured_trace": {"query_type": "sweep"}}
    monkeypatch.setattr(structured, "retrieve_structured", _structured)

    out = await analytical.retrieve_analytical(_state(), vector_store=object(), schema_registry=_Registry())
    assert out["retrieved_chunks"] == [{"id": "row1"}]
    assert out["structured_trace"]["fell_back"] is True          # analytical trace preserved + flagged
    assert out["structured_trace"]["query_type"] == "analytical"


@pytest.mark.asyncio
async def test_zero_rows_then_structured_empty_falls_back_to_map_reduce(monkeypatch):
    trace = StructuredLookupTrace(query_type="analytical", status="ran", row_count=0, rows=[])
    monkeypatch.setattr(structured, "run_structured_lookup", lambda *a, **k: trace)
    async def _structured(state, vector_store, schema_registry):
        return {}  # table not relevant / gated out
    monkeypatch.setattr(structured, "retrieve_structured", _structured)
    async def _map_reduce(state, vector_store):
        return {"retrieved_chunks": [{"id": "mr"}]}
    monkeypatch.setattr(map_reduce, "retrieve_map_reduce", _map_reduce)

    out = await analytical.retrieve_analytical(_state(), vector_store=object(), schema_registry=_Registry())
    assert out["retrieved_chunks"] == [{"id": "mr"}]
    assert out["structured_trace"]["fell_back"] is True


@pytest.mark.asyncio
async def test_hard_error_still_falls_back_to_map_reduce(monkeypatch):
    trace = StructuredLookupTrace(query_type="analytical", status="error", error="bad sql")
    monkeypatch.setattr(structured, "run_structured_lookup", lambda *a, **k: trace)
    async def _map_reduce(state, vector_store):
        return {"retrieved_chunks": [{"id": "mr"}]}
    monkeypatch.setattr(map_reduce, "retrieve_map_reduce", _map_reduce)

    out = await analytical.retrieve_analytical(_state(), vector_store=object(), schema_registry=_Registry())
    assert out["retrieved_chunks"] == [{"id": "mr"}]
    assert out["structured_trace"]["status"] == "error"
