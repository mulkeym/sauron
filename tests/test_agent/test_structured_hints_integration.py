import pytest
from types import SimpleNamespace
from src.db.hint_store import HintStore, SchemaHint
from src.db.schema_registry import TableSchema, ColumnSchema
from src.agent.strategies.structured import resolve_hints_for_schemas


class _MS:
    def __init__(self, docs):
        self._docs = docs

    async def list_documents(self, user_groups=None):
        return self._docs


def _schema(table):
    return TableSchema(database="spreadsheets", table=table,
                       columns=[ColumnSchema("locname", "VARCHAR")],
                       description="d", acl_groups=["executives"])


@pytest.mark.asyncio
async def test_resolve_hints_for_schemas_maps_table_to_doc_scope():
    from src.ingestion.tabular_store import duckdb_table_name
    doc = SimpleNamespace(doc_id="abc", category="OPM", dataset_id=0)
    table = duckdb_table_name("abc", "all_gs")
    store = HintStore()
    store.register(SchemaHint(scope_type="category", scope_value="OPM",
                              hint_type="value_glossary", target_column="locname",
                              payload={"TU": "Tampa"}))
    out = await resolve_hints_for_schemas([_schema(table)], store, _MS([doc]))
    assert out[table].column_glossaries == {"locname": {"TU": "Tampa"}}


@pytest.mark.asyncio
async def test_resolve_hints_for_schemas_empty_when_no_owning_doc():
    store = HintStore()
    out = await resolve_hints_for_schemas([_schema("doc_ghost_all_gs")], store, _MS([]))
    assert out == {}


def test_text_to_sql_prompt_directs_glossary_use_and_no_refusal():
    # Guards the SQL-generation direction: the model must be told to map a named
    # place to its CODE via the `CODE (meaning)` glossary, honor Notes, and never
    # refuse (the failure mode where it returned a prose apology instead of SQL).
    from src.agent.strategies.structured import TEXT_TO_SQL_PROMPT
    p = TEXT_TO_SQL_PROMPT.lower()
    assert "code (meaning)" in p
    assert "notes" in p
    assert "never refuse" in p
    assert "closest applicable code" in p
