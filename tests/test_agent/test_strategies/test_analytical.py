import pytest
from unittest.mock import patch, AsyncMock, MagicMock
from src.agent.strategies.analytical import retrieve_analytical
from src.agent.state import AgentState, QueryType
from src.db.schema_registry import SchemaRegistry, TableSchema, ColumnSchema


@pytest.fixture
def registry():
    r = SchemaRegistry()
    r.register(TableSchema(
        database="finance_db",
        table="quarterly_results",
        columns=[
            ColumnSchema(name="quarter", dtype="varchar", description="Q1-Q4"),
            ColumnSchema(name="revenue", dtype="numeric", description="Revenue in USD"),
            ColumnSchema(name="year", dtype="integer", description="Fiscal year"),
        ],
        description="Quarterly financial results",
        acl_groups=["finance"],
    ))
    return r


@pytest.mark.asyncio
async def test_analytical_generates_and_executes_sql(registry):
    mock_vector_store = MagicMock()
    with patch("src.agent.strategies.analytical.generate", return_value="SELECT revenue FROM quarterly_results WHERE quarter = 'Q3' AND year = 2026"):
        with patch("src.agent.strategies.analytical.execute_sql", new_callable=AsyncMock, return_value=[{"revenue": 1500000}]):
            state = AgentState(
                question="What was Q3 2026 revenue?",
                user_groups=["finance"],
                query_type=QueryType.ANALYTICAL,
                retrieved_chunks=[],
                sql_results=[],
                retrieval_attempts=0,
            )
            result = await retrieve_analytical(state, vector_store=mock_vector_store, schema_registry=registry)
    assert len(result["sql_results"]) == 1
    assert result["sql_results"][0]["revenue"] == 1500000


@pytest.mark.asyncio
async def test_analytical_no_schemas_available():
    mock_vector_store = MagicMock()
    mock_vector_store.search = MagicMock(return_value=[])
    empty_registry = SchemaRegistry()
    state = AgentState(
        question="What was Q3 revenue?",
        user_groups=["engineering"],
        query_type=QueryType.ANALYTICAL,
        retrieved_chunks=[],
        sql_results=[],
        retrieval_attempts=0,
    )
    result = await retrieve_analytical(state, vector_store=mock_vector_store, schema_registry=empty_registry)
    assert result["sql_results"] == []
    assert "retrieved_chunks" in result
