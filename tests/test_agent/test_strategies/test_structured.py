"""Tests for shared structured retrieval (SQL core + gate + retrieve_structured)."""
import duckdb
import pytest

from src.ingestion.tabular import SheetGrid, SheetClassification
from src.ingestion.tabular_store import load_sheet_to_duckdb, schema_from_sheet
from src.agent.strategies import structured
from src.agent.strategies.structured import structured_sql_rows


def _pay_schema_and_db(tmp_path, monkeypatch):
    from src.config import settings
    monkeypatch.setattr(settings, "tabular_duckdb_path", str(tmp_path / "t.duckdb"))
    grid = SheetGrid("Pay", [["grade", "step", "salary"]] + [[f"GS-{g}", 5, 80000 + g] for g in range(10, 14)])
    cls = SheetClassification("Pay", "clean", 0, ["text", "number", "number"], "clean table")
    from src.ingestion.tabular_store import connect_tabular
    con = connect_tabular(read_only=False)
    load_sheet_to_duckdb(con, "doc1", "Pay", cls, grid)
    con.close()
    return schema_from_sheet("doc1", "Pay", cls, grid, acl_groups=["ALL"])


def test_structured_sql_rows_generates_and_runs(tmp_path, monkeypatch):
    schema = _pay_schema_and_db(tmp_path, monkeypatch)
    monkeypatch.setattr(structured, "generate",
                        lambda **kw: f'SELECT grade, salary FROM "{schema.table}" WHERE step = 5 ORDER BY salary')
    rows = structured_sql_rows("engineer pay", [schema])
    assert rows[0] == {"grade": "GS-10", "salary": 80010.0}
    assert len(rows) == 4


def test_structured_sql_rows_raises_on_disallowed_table(tmp_path, monkeypatch):
    schema = _pay_schema_and_db(tmp_path, monkeypatch)
    monkeypatch.setattr(structured, "generate", lambda **kw: 'SELECT * FROM "doc_other_secret"')
    with pytest.raises(Exception):
        structured_sql_rows("x", [schema])  # allowlist rejects -> raises (caller falls back)


from src.agent.strategies.structured import tables_relevant_to
from src.db.schema_registry import TableSchema, ColumnSchema


def _schema(table, desc):
    return TableSchema(database="spreadsheets", table=table,
                       columns=[ColumnSchema("grade", "DOUBLE", "")],
                       description=desc, acl_groups=["ALL"])


def test_gate_keeps_only_matching_tables():
    pay = _schema("doc_pay", "GS pay rates by grade and locality")
    weather = _schema("doc_weather", "daily weather observations")
    # Deterministic fake embeddings: question vector == pay-text vector; weather orthogonal.
    eq = lambda q: [1.0, 0.0]
    et = lambda texts: [[1.0, 0.0] if "pay" in t.lower() else [0.0, 1.0] for t in texts]
    out = tables_relevant_to("what is the pay for grade 12", [pay, weather],
                             threshold=0.5, embed_query_fn=eq, embed_texts_fn=et)
    assert [s.table for s in out] == ["doc_pay"]


def test_gate_empty_when_no_schemas():
    assert tables_relevant_to("anything", []) == []


from unittest.mock import MagicMock
from src.agent.strategies.structured import retrieve_structured
from src.retrieval.models import RetrievedChunk, ChunkMetadata


def _narr_chunk():
    return RetrievedChunk(text="GS pay grade=GS-12: salary is 86415", score=0.9,
                          metadata=ChunkMetadata(doc_id="d", filename="f", doc_type="xlsx",
                          chunk_index=0, start_char=0, acl_groups=["ALL"], chunk_size_tier="table_row"))


def _reg(schemas):
    r = MagicMock()
    r.list_for_user.return_value = schemas
    return r


@pytest.mark.asyncio
async def test_retrieve_structured_returns_sql_and_narratives(monkeypatch):
    schema = _schema("doc_pay", "GS pay rates")
    monkeypatch.setattr(structured, "tables_relevant_to", lambda q, s, **k: [schema])
    monkeypatch.setattr(structured, "structured_sql_rows", lambda q, s: [{"salary": 86415.0}])
    monkeypatch.setattr(structured, "embed_query", lambda q: [0.1], raising=False)
    vs = MagicMock()
    vs.search.return_value = [_narr_chunk()]
    out = await retrieve_structured({"question": "pay?", "user_groups": ["ALL"]},
                                    vector_store=vs, schema_registry=_reg([schema]))
    assert out["sql_results"] == [{"salary": 86415.0}]
    assert len(out["retrieved_chunks"]) == 1
    # narratives searched on the table_row tier
    assert vs.search.call_args.kwargs.get("tier") == "table_row"


@pytest.mark.asyncio
async def test_retrieve_structured_empty_when_gate_misses(monkeypatch):
    monkeypatch.setattr(structured, "tables_relevant_to", lambda q, s, **k: [])
    out = await retrieve_structured({"question": "weather?", "user_groups": ["ALL"]},
                                    vector_store=MagicMock(), schema_registry=_reg([_schema("doc_pay", "pay")]))
    assert out == {}


@pytest.mark.asyncio
async def test_retrieve_structured_fail_open_on_gate_error(monkeypatch):
    """If the embedding gate raises (service down), retrieve_structured returns {} (RAG-only)."""
    def boom(q, s, **k):
        raise RuntimeError("embedding service down")
    monkeypatch.setattr(structured, "tables_relevant_to", boom)
    out = await retrieve_structured(
        {"question": "pay?", "user_groups": ["ALL"]},
        vector_store=MagicMock(),
        schema_registry=_reg([_schema("doc_pay", "pay")]),
    )
    assert out == {}


@pytest.mark.asyncio
async def test_retrieve_structured_fail_open_on_sql_error(monkeypatch):
    schema = _schema("doc_pay", "GS pay rates")
    monkeypatch.setattr(structured, "tables_relevant_to", lambda q, s, **k: [schema])
    def boom(q, s):
        raise RuntimeError("sql gen down")
    monkeypatch.setattr(structured, "structured_sql_rows", boom)
    monkeypatch.setattr(structured, "embed_query", lambda q: [0.1], raising=False)
    vs = MagicMock(); vs.search.return_value = [_narr_chunk()]
    out = await retrieve_structured({"question": "pay?", "user_groups": ["ALL"]},
                                    vector_store=vs, schema_registry=_reg([schema]))
    assert out["sql_results"] == []                 # SQL failed -> empty, no raise
    assert len(out["retrieved_chunks"]) == 1        # narratives still returned
