import pytest
from unittest.mock import MagicMock, patch, AsyncMock
import json
import pandas as pd
import server
import asyncio

# Mock the database pool and connection
@pytest.fixture
def mock_db_pool():
    with patch('server._db_pool', new_callable=AsyncMock) as mock_pool:
        # asyncpg Pool.acquire is not a coroutine, it returns an async context manager
        mock_pool.acquire = MagicMock()
        mock_conn = AsyncMock()
        mock_pool.acquire.return_value.__aenter__.return_value = mock_conn
        mock_pool.acquire.return_value.__aexit__.return_value = None
        
        yield mock_pool, mock_conn

@pytest.mark.asyncio
async def test_list_tables(mock_db_pool):
    mock_pool, mock_conn = mock_db_pool
    
    # Setup mock return value
    # asyncpg returns a list of Record objects (dict-like)
    mock_conn.fetch.return_value = [{'table_name': 'table1'}, {'table_name': 'table2'}]
    
    # Run the tool
    result = await server.list_network_tables()
    data = json.loads(result)
    
    assert "tables" in data
    assert data["tables"] == ['table1', 'table2']
    mock_pool.acquire.assert_called_once()

@pytest.mark.asyncio
async def test_describe_network_table():
    with patch('server.fetch_as_dataframe', new_callable=AsyncMock) as mock_fetch:
        # Mock returns for columns and sample data
        cols_df = pd.DataFrame({'column_name': ['id', 'val'], 'data_type': ['int', 'text']})
        sample_df = pd.DataFrame({'id': [1], 'val': ['x']})
        
        # side_effect allows different returns for sequential calls
        mock_fetch.side_effect = [cols_df, sample_df]
        
        result = await server.describe_network_table("valid_table")
        data = json.loads(result)
        
        assert "columns" in data
        assert "sample_data_toon" in data
        assert "id | val" in data["sample_data_toon"]
        assert "1 | x" in data["sample_data_toon"]

@pytest.mark.asyncio
async def test_query_data_validation():
    # Test valid queries
    # We patch fetch_as_dataframe to avoid DB calls
    with patch('server.fetch_as_dataframe', new_callable=AsyncMock) as mock_fetch:
        mock_fetch.return_value = pd.DataFrame()
        
        assert "Error" not in await server.query_data("SELECT * FROM table")
        assert "Error" not in await server.query_data("WITH cte AS (SELECT 1) SELECT * FROM cte")
        assert "Error" not in await server.query_data("EXPLAIN SELECT * FROM table")
        
    # Test invalid queries
    assert "Error: Only SELECT" in await server.query_data("INSERT INTO table VALUES (1)")
    assert "Error: Only SELECT" in await server.query_data("DROP TABLE table")
    assert "Error: Only SELECT" in await server.query_data("UPDATE table SET x=1")

@pytest.mark.asyncio
async def test_query_data_execution():
    with patch('server.fetch_as_dataframe', new_callable=AsyncMock) as mock_fetch:
        # Mock pandas return
        df = pd.DataFrame({'col1': [1, 2], 'col2': ['a', 'b']})
        mock_fetch.return_value = df
        
        result = await server.query_data("SELECT * FROM table")
        
        # Verify TOON format
        lines = result.strip().split('\n')
        assert len(lines) == 3
        assert lines[0] == "col1 | col2"
        assert "1 | a" in lines[1]
        assert "2 | b" in lines[2]
        mock_fetch.assert_called_once()

def test_get_mlab_documentation():
    # Mock the global cache
    server._cached_docs = {
        "ndt": {"description": "test ndt"},
        "traceroute": {"description": "test traceroute"}
    }
    
    # Test getting full docs
    all_docs = json.loads(server.get_mlab_documentation())
    assert "ndt" in all_docs
    
    # Test specific topic
    ndt_doc = json.loads(server.get_mlab_documentation("ndt"))
    assert ndt_doc["description"] == "test ndt"
    
    # Test missing topic
    assert server.get_mlab_documentation("missing") == "null"
    
    # Test case-insensitivity
    ndt_doc_upper = json.loads(server.get_mlab_documentation("NDT"))
    assert ndt_doc_upper["description"] == "test ndt"

@pytest.mark.asyncio
async def test_analyze_metrics():
    with patch('server.fetch_as_dataframe', new_callable=AsyncMock) as mock_fetch:
        df = pd.DataFrame({'val': [1, 2, 3, 4, 5, 100]})
        mock_fetch.return_value = df
        
        result = await server.analyze_metrics("SELECT * FROM t", "val")
        stats = json.loads(result)
        
        assert stats['count'] == 6
        assert stats['mean'] > 0

@pytest.mark.asyncio
async def test_analyze_change_points_from_sql():
    with patch('server.fetch_as_dataframe', new_callable=AsyncMock) as mock_fetch:
        with patch('server.vwcd') as mock_vwcd:
            # Setup mock data
            df = pd.DataFrame({'metric': [10, 10, 10, 20, 20, 20]})
            mock_fetch.return_value = df
            
            # Setup vwcd mock
            mock_vwcd.vwcd.return_value = ([3], None, None, 0.1)
            mock_vwcd.get_segments.return_value = [{"start": 0, "end": 3, "mean": 10}, {"start": 3, "end": 6, "mean": 20}]
            
            # Test valid execution
            result = await server.analyze_change_points_from_sql("SELECT * FROM t", "metric")
            data = json.loads(result)
            
            assert len(data["change_points"]) == 1
            assert data["change_points"][0] == 3
            assert len(data["segments"]) == 2
            mock_fetch.assert_called_once()
            
            # Note: server.vwcd.vwcd is called in an executor, but we mocked the module 'server.vwcd'
            # The server code uses: vwcd.vwcd(X). 
            # We are patching 'server.vwcd', so it should capture it.
            # However, since we use run_in_executor, we might need to be careful.
            # But run_in_executor simply calls the function.
            
    # Test invalid query
    assert "Error: Only SELECT" in await server.analyze_change_points_from_sql("DELETE FROM t", "metric")
    
    # Test missing column
    with patch('server.fetch_as_dataframe', new_callable=AsyncMock) as mock_fetch:
        mock_fetch.return_value = pd.DataFrame({'metric': [10]})
        assert "Metric column" in await server.analyze_change_points_from_sql("SELECT * FROM t", "wrong_col")