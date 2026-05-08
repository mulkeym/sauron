import pytest
from src.db.schema_registry import SchemaRegistry, TableSchema, ColumnSchema


def test_register_and_get_schema():
    registry = SchemaRegistry()
    schema = TableSchema(
        database="finance_db",
        table="quarterly_results",
        columns=[
            ColumnSchema(name="quarter", dtype="varchar", description="Q1-Q4"),
            ColumnSchema(name="revenue", dtype="numeric", description="Revenue in USD"),
            ColumnSchema(name="year", dtype="integer", description="Fiscal year"),
        ],
        description="Quarterly financial results",
        acl_groups=["finance", "executives"],
    )
    registry.register(schema)
    result = registry.get_schema("finance_db", "quarterly_results")
    assert result is not None
    assert result.table == "quarterly_results"
    assert len(result.columns) == 3


def test_get_nonexistent_schema():
    registry = SchemaRegistry()
    assert registry.get_schema("nope", "nope") is None


def test_list_schemas_for_user():
    registry = SchemaRegistry()
    registry.register(TableSchema(database="finance_db", table="budget", columns=[], description="Budget", acl_groups=["finance"]))
    registry.register(TableSchema(database="it_db", table="servers", columns=[], description="Servers", acl_groups=["it_support"]))
    assert len(registry.list_for_user(["finance"])) == 1
    assert len(registry.list_for_user(["finance", "it_support"])) == 2


def test_schema_to_prompt_string():
    registry = SchemaRegistry()
    schema = TableSchema(
        database="finance_db",
        table="quarterly_results",
        columns=[
            ColumnSchema(name="quarter", dtype="varchar", description="Q1-Q4"),
            ColumnSchema(name="revenue", dtype="numeric", description="Revenue in USD"),
        ],
        description="Quarterly results",
        acl_groups=["finance"],
    )
    registry.register(schema)
    prompt = registry.schemas_to_prompt(["finance"])
    assert "quarterly_results" in prompt
    assert "revenue" in prompt
    assert "numeric" in prompt
