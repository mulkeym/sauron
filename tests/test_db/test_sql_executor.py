import pytest
import pytest_asyncio
import aiosqlite


@pytest_asyncio.fixture
async def test_db(tmp_path):
    db_path = tmp_path / "test.db"
    async with aiosqlite.connect(str(db_path)) as db:
        await db.execute("CREATE TABLE quarterly_results (quarter TEXT, revenue REAL, year INTEGER)")
        await db.execute("INSERT INTO quarterly_results VALUES ('Q3', 1500000, 2026)")
        await db.execute("INSERT INTO quarterly_results VALUES ('Q2', 1200000, 2026)")
        await db.commit()
    return str(db_path)


@pytest.mark.asyncio
async def test_execute_select(test_db):
    from src.db.sql_executor import execute_sql
    result = await execute_sql(
        database_url=f"sqlite+aiosqlite:///{test_db}",
        sql="SELECT quarter, revenue FROM quarterly_results WHERE quarter = 'Q3'",
    )
    assert len(result) == 1
    assert result[0]["quarter"] == "Q3"
    assert result[0]["revenue"] == 1500000


@pytest.mark.asyncio
async def test_execute_rejects_write_operations(test_db):
    from src.db.sql_executor import execute_sql
    with pytest.raises(ValueError, match="Only SELECT"):
        await execute_sql(
            database_url=f"sqlite+aiosqlite:///{test_db}",
            sql="DROP TABLE quarterly_results",
        )


@pytest.mark.asyncio
async def test_execute_rejects_multiple_statements(test_db):
    from src.db.sql_executor import execute_sql
    with pytest.raises(ValueError, match="single SELECT"):
        await execute_sql(
            database_url=f"sqlite+aiosqlite:///{test_db}",
            sql="SELECT 1; DROP TABLE quarterly_results",
        )
