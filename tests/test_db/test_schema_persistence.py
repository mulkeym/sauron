"""Round-trip persistence for registered table schemas."""
import pytest

from src.db.metadata import MetadataStore
from src.db.schema_registry import TableSchema, ColumnSchema


def _schema(table="doc_x_pay"):
    return TableSchema(
        database="spreadsheets",
        table=table,
        columns=[
            ColumnSchema(name="grade", dtype="VARCHAR", description="Pay grade"),
            ColumnSchema(name="salary", dtype="DOUBLE", description="Annual salary"),
        ],
        description="GS pay table",
        acl_groups=["ALL"],
    )


@pytest.mark.asyncio
async def test_save_and_load_schema_round_trip(tmp_path):
    store = MetadataStore(database_url=f"sqlite+aiosqlite:///{tmp_path}/m.db")
    await store.init()
    await store.save_schema(_schema())

    loaded = await store.load_all_schemas()
    assert len(loaded) == 1
    s = loaded[0]
    assert (s.database, s.table) == ("spreadsheets", "doc_x_pay")
    assert [(c.name, c.dtype, c.description) for c in s.columns] == [
        ("grade", "VARCHAR", "Pay grade"), ("salary", "DOUBLE", "Annual salary")]
    assert s.acl_groups == ["ALL"]


@pytest.mark.asyncio
async def test_save_is_idempotent_on_same_table(tmp_path):
    store = MetadataStore(database_url=f"sqlite+aiosqlite:///{tmp_path}/m.db")
    await store.init()
    await store.save_schema(_schema())
    await store.save_schema(_schema())  # re-save same db.table => no duplicate
    assert len(await store.load_all_schemas()) == 1


@pytest.mark.asyncio
async def test_delete_schema(tmp_path):
    store = MetadataStore(database_url=f"sqlite+aiosqlite:///{tmp_path}/m.db")
    await store.init()
    await store.save_schema(_schema("doc_a"))
    await store.save_schema(_schema("doc_b"))
    await store.delete_schema("spreadsheets", "doc_a")
    remaining = [s.table for s in await store.load_all_schemas()]
    assert remaining == ["doc_b"]
