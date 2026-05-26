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
    # The sheet wrote DuckDB rows + a schema before embedding failed; that
    # partial state is acceptable (valid for SQL, idempotent on re-ingest).
    # `n` counts only FULLY-processed sheets, so it is 0 and no narratives upsert.
    assert n == 0  # the sheet failed at the embed step, so it was not counted
    vector_store.upsert.assert_not_called()


@pytest.mark.asyncio
async def test_ingest_document_invokes_tabular_branch_for_spreadsheet(tmp_path, monkeypatch):
    import src.ingestion.pipeline as pipe
    import src.ingestion.tabular_ingest as ti
    import src.generation.llm_client as llm
    import src.knowledge.graph_rag as kg

    # Patch heavy collaborators so we exercise only the branch logic.
    monkeypatch.setattr(pipe, "embed_texts", lambda texts: [[0.1] for _ in texts])
    monkeypatch.setattr(llm, "generate", lambda **kw: "summary")

    async def _noop_insert(*a, **k):
        return None
    monkeypatch.setattr(kg, "insert_document", _noop_insert)

    calls = []

    async def _fake_orch(*args, **kwargs):
        calls.append(args)
        return 1
    monkeypatch.setattr(ti, "ingest_spreadsheet_tables", _fake_orch)

    store = MetadataStore(database_url=f"sqlite+aiosqlite:///{tmp_path}/m.db")
    await store.init()
    vector_store = MagicMock()

    xlsx = tmp_path / "pay.xlsx"
    _write_xlsx(xlsx, {"Pay": [["grade", "salary"], ["GS-12", 86415], ["GS-13", 102000], ["GS-14", 120000]]})
    await pipe.ingest_document(str(xlsx), ["ALL"], "tester", vector_store, store)
    assert len(calls) == 1  # orchestrator invoked for the spreadsheet

    txt = tmp_path / "note.txt"
    txt.write_text("just some text\n", encoding="utf-8")
    await pipe.ingest_document(str(txt), ["ALL"], "tester", vector_store, store)
    assert len(calls) == 1  # NOT invoked again for a non-spreadsheet


@pytest.mark.asyncio
async def test_populate_schema_registry_loads_persisted(tmp_path):
    from src.db.schema_registry import TableSchema, ColumnSchema
    from src.ingestion.tabular_ingest import populate_schema_registry

    store = MetadataStore(database_url=f"sqlite+aiosqlite:///{tmp_path}/m.db")
    await store.init()
    await store.save_schema(TableSchema(
        database="spreadsheets", table="doc_x_pay",
        columns=[ColumnSchema(name="grade", dtype="VARCHAR", description="Pay grade")],
        description="GS pay", acl_groups=["ALL"],
    ))

    registry = SchemaRegistry()
    await populate_schema_registry(store, registry)

    loaded = registry.list_for_user(["ALL"])
    assert len(loaded) == 1
    assert loaded[0].table == "doc_x_pay"


@pytest.mark.asyncio
async def test_maybe_ingest_skips_non_spreadsheet(monkeypatch):
    import src.ingestion.tabular_ingest as ti
    calls = []
    async def spy(*a, **k):
        calls.append(a)
        return 1
    monkeypatch.setattr(ti, "ingest_spreadsheet_tables", spy)
    n = await ti.maybe_ingest_spreadsheet("/x", "d", "f.md", "markdown", ["ALL"], "", MagicMock(), MagicMock())
    assert n == 0
    assert calls == []


@pytest.mark.asyncio
async def test_maybe_ingest_runs_for_spreadsheet(monkeypatch):
    import src.ingestion.tabular_ingest as ti
    calls = []
    async def spy(*a, **k):
        calls.append(a)
        return 2
    monkeypatch.setattr(ti, "ingest_spreadsheet_tables", spy)
    n = await ti.maybe_ingest_spreadsheet("/x", "d", "f.xlsx", "xlsx", ["ALL"], "", MagicMock(), MagicMock())
    assert n == 2
    assert len(calls) == 1


@pytest.mark.asyncio
async def test_maybe_ingest_is_fail_open(monkeypatch):
    import src.ingestion.tabular_ingest as ti
    async def boom(*a, **k):
        raise RuntimeError("structured ingest blew up")
    monkeypatch.setattr(ti, "ingest_spreadsheet_tables", boom)
    n = await ti.maybe_ingest_spreadsheet("/x", "d", "f.xlsx", "xlsx", ["ALL"], "", MagicMock(), MagicMock())
    assert n == 0


def test_schema_registry_remove():
    from src.db.schema_registry import SchemaRegistry, TableSchema, ColumnSchema
    reg = SchemaRegistry()
    s = TableSchema(database="spreadsheets", table="doc_x",
                    columns=[ColumnSchema("a", "DOUBLE", "")], description="", acl_groups=["ALL"])
    reg.register(s)
    assert reg.get_schema("spreadsheets", "doc_x") is not None
    reg.remove("spreadsheets", "doc_x")
    assert reg.get_schema("spreadsheets", "doc_x") is None
    reg.remove("spreadsheets", "doc_x")  # idempotent: no error on missing key


@pytest.mark.asyncio
async def test_cleanup_spreadsheet_tables_drops_tables_and_schemas(tmp_path, monkeypatch):
    from src.config import settings
    monkeypatch.setattr(settings, "tabular_duckdb_path", str(tmp_path / "t.duckdb"))
    from src.ingestion.tabular import SheetGrid, SheetClassification
    from src.ingestion.tabular_store import load_sheet_to_duckdb, schema_from_sheet
    from src.ingestion.tabular_ingest import cleanup_spreadsheet_tables

    grid = SheetGrid("Pay", [["grade", "salary"], ["GS-12", 86415], ["GS-13", 102000], ["GS-14", 120000]])
    cls = SheetClassification("Pay", "clean", 0, ["text", "number"], "clean table")
    con = connect_tabular(read_only=False)
    table, _ = load_sheet_to_duckdb(con, "doc1", "Pay", cls, grid)
    con.close()

    store = MetadataStore(database_url=f"sqlite+aiosqlite:///{tmp_path}/m.db")
    await store.init()
    schema = schema_from_sheet("doc1", "Pay", cls, grid, acl_groups=["ALL"])
    await store.save_schema(schema)
    reg = SchemaRegistry()
    reg.register(schema)
    assert reg.get_schema(schema.database, table) is not None  # precondition

    dropped = await cleanup_spreadsheet_tables("doc1", store, schema_registry=reg)

    assert dropped == 1
    assert await store.load_all_schemas() == []               # persisted schema gone
    assert reg.get_schema(schema.database, table) is None      # live registry cleared
    con = connect_tabular(read_only=True)
    remaining = con.execute(
        "SELECT count(*) FROM information_schema.tables WHERE table_name = ?", [table]).fetchone()[0]
    con.close()
    assert remaining == 0                                      # DuckDB table dropped


@pytest.mark.asyncio
async def test_cleanup_spreadsheet_tables_noop_for_unknown_doc(tmp_path, monkeypatch):
    from src.config import settings
    monkeypatch.setattr(settings, "tabular_duckdb_path", str(tmp_path / "t.duckdb"))
    from src.ingestion.tabular_ingest import cleanup_spreadsheet_tables
    store = MetadataStore(database_url=f"sqlite+aiosqlite:///{tmp_path}/m.db")
    await store.init()
    n = await cleanup_spreadsheet_tables("nonexistent-doc", store, schema_registry=SchemaRegistry())
    assert n == 0
