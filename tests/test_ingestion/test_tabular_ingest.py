"""Integration tests for the spreadsheet structured-ingest orchestrator."""
import json
from unittest.mock import MagicMock

import openpyxl
import pytest

from src.db.metadata import MetadataStore
from src.db.schema_registry import SchemaRegistry
from src.ingestion.tabular_store import connect_tabular, duckdb_table_name


def _write_xlsx(path, sheets):
    wb = openpyxl.Workbook()
    wb.remove(wb.active)
    for name, rows in sheets.items():
        ws = wb.create_sheet(title=name)
        for row in rows:
            ws.append(row)
    wb.save(path)


def _fake_profile_generate(system_prompt, user_prompt, **kwargs):
    return json.dumps({
        "column_descriptions": {"grade": "Pay grade", "step": "Step", "salary": "Annual salary"},
        "key_columns": ["grade", "step"],
        "measure_columns": ["salary"],
        "table_description": "GS pay by grade and step",
    })


@pytest.mark.asyncio
async def test_ingest_clean_sheet_stores_rows_schema_and_narratives(tmp_path, monkeypatch):
    from src.config import settings
    monkeypatch.setattr(settings, "tabular_duckdb_path", str(tmp_path / "t.duckdb"))

    # embed_texts is patched where the orchestrator looks it up
    import src.ingestion.tabular_ingest as ti
    monkeypatch.setattr(ti, "embed_texts", lambda texts: [[0.1, 0.2, 0.3] for _ in texts])

    xlsx = tmp_path / "pay.xlsx"
    _write_xlsx(xlsx, {
        "Pay": [["grade", "step", "salary"]] + [[f"GS-{g}", 5, 80000 + g] for g in range(10, 14)],
    })

    vector_store = MagicMock()
    store = MetadataStore(database_url=f"sqlite+aiosqlite:///{tmp_path}/m.db")
    await store.init()
    registry = SchemaRegistry()

    from src.ingestion.tabular_ingest import ingest_spreadsheet_tables
    n = await ingest_spreadsheet_tables(
        str(xlsx), "doc1", "pay.xlsx", "xlsx", ["ALL"], "",
        vector_store, store, schema_registry=registry, generate_fn=_fake_profile_generate,
    )

    assert n == 1
    # narratives embedded + upserted, one per data row, marked as table_row
    vector_store.upsert.assert_called_once()
    kwargs = vector_store.upsert.call_args.kwargs
    assert len(kwargs["texts"]) == 4
    assert all("Pay grade=" in t for t in kwargs["texts"])
    assert all(m.chunk_size_tier == "table_row" for m in kwargs["metadatas"])
    # schema registered in-memory AND persisted
    assert registry.list_for_user(["ALL"])
    persisted = await store.load_all_schemas()
    assert persisted[0].table == duckdb_table_name("doc1", "Pay")
    # rows landed in DuckDB
    con = connect_tabular(read_only=True)
    cnt = con.execute(f'SELECT COUNT(*) FROM "{duckdb_table_name("doc1", "Pay")}"').fetchone()[0]
    con.close()
    assert cnt == 4


@pytest.mark.asyncio
async def test_messy_only_workbook_stores_nothing(tmp_path, monkeypatch):
    from src.config import settings
    monkeypatch.setattr(settings, "tabular_duckdb_path", str(tmp_path / "t.duckdb"))
    import src.ingestion.tabular_ingest as ti
    monkeypatch.setattr(ti, "embed_texts", lambda texts: [[0.1] for _ in texts])

    xlsx = tmp_path / "notes.xlsx"
    _write_xlsx(xlsx, {"Readme": [["This is a narrative note."], ["Updated 2024."]]})

    vector_store = MagicMock()
    store = MetadataStore(database_url=f"sqlite+aiosqlite:///{tmp_path}/m.db")
    await store.init()
    registry = SchemaRegistry()

    from src.ingestion.tabular_ingest import ingest_spreadsheet_tables
    n = await ingest_spreadsheet_tables(
        str(xlsx), "doc2", "notes.xlsx", "xlsx", ["ALL"], "",
        vector_store, store, schema_registry=registry, generate_fn=_fake_profile_generate,
    )
    assert n == 0
    vector_store.upsert.assert_not_called()
    assert await store.load_all_schemas() == []


@pytest.mark.asyncio
async def test_per_sheet_failure_is_contained(tmp_path, monkeypatch):
    from src.config import settings
    monkeypatch.setattr(settings, "tabular_duckdb_path", str(tmp_path / "t.duckdb"))
    import src.ingestion.tabular_ingest as ti
    # embedding blows up -> the clean sheet fails, but the call must not raise
    monkeypatch.setattr(ti, "embed_texts", lambda texts: (_ for _ in ()).throw(RuntimeError("embed down")))

    xlsx = tmp_path / "pay.xlsx"
    _write_xlsx(xlsx, {
        "Pay": [["grade", "step", "salary"]] + [[f"GS-{g}", 5, 80000 + g] for g in range(10, 14)],
    })

    vector_store = MagicMock()
    store = MetadataStore(database_url=f"sqlite+aiosqlite:///{tmp_path}/m.db")
    await store.init()
    registry = SchemaRegistry()

    from src.ingestion.tabular_ingest import ingest_spreadsheet_tables
    n = await ingest_spreadsheet_tables(  # must NOT raise
        str(xlsx), "doc3", "pay.xlsx", "xlsx", ["ALL"], "",
        vector_store, store, schema_registry=registry, generate_fn=_fake_profile_generate,
    )
    assert n == 0  # the sheet failed at the embed step, so it was not counted
    vector_store.upsert.assert_not_called()
