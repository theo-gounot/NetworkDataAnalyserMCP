import pytest
from unittest.mock import MagicMock, patch
import json
import pandas as pd
import server

# Mock the database pool and connection
@pytest.fixture
def mock_db_pool():
    with patch('server._db_pool') as mock_pool:
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        
        # Setup context manager for cursor
        mock_cursor.__enter__.return_value = mock_cursor
        mock_cursor.__exit__.return_value = None
        
        mock_conn.cursor.return_value = mock_cursor
        mock_pool.getconn.return_value = mock_conn
        
        yield mock_pool, mock_conn, mock_cursor

def test_list_tables(mock_db_pool):
    mock_pool, mock_conn, mock_cursor = mock_db_pool
    
    # Setup mock return value
    mock_cursor.fetchall.return_value = [('table1',), ('table2',)]
    
    # Run the tool
    result = server.list_tables()
    data = json.loads(result)
    
    assert "tables" in data
    assert data["tables"] == ['table1', 'table2']
    mock_pool.getconn.assert_called_once()
    mock_pool.putconn.assert_called_once_with(mock_conn)

@patch('server.get_db_connection')
@patch('pandas.read_sql')
def test_query_data_validation(mock_read_sql, mock_get_db):
    # Mock the connection context manager
    mock_conn = MagicMock()
    mock_get_db.return_value.__enter__.return_value = mock_conn
    mock_read_sql.return_value = pd.DataFrame()

    # Test valid queries
    assert "Error" not in server.query_data("SELECT * FROM table")
    assert "Error" not in server.query_data("WITH cte AS (SELECT 1) SELECT * FROM cte")
    assert "Error" not in server.query_data("EXPLAIN SELECT * FROM table")
    
    # Test invalid queries
    assert "Error: Only SELECT" in server.query_data("INSERT INTO table VALUES (1)")
    assert "Error: Only SELECT" in server.query_data("DROP TABLE table")
    assert "Error: Only SELECT" in server.query_data("UPDATE table SET x=1")

@patch('pandas.read_sql')
def test_query_data_execution(mock_read_sql, mock_db_pool):
    mock_pool, mock_conn, _ = mock_db_pool
    
    # Mock pandas return
    df = pd.DataFrame({'col1': [1, 2], 'col2': ['a', 'b']})
    mock_read_sql.return_value = df
    
    result = server.query_data("SELECT * FROM table")
    data = json.loads(result)
    
    assert len(data) == 2
    assert data[0]['col1'] == 1
    mock_read_sql.assert_called_once()

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

def test_analyze_metrics(mock_db_pool):
    with patch('pandas.read_sql') as mock_read_sql:
        df = pd.DataFrame({'val': [1, 2, 3, 4, 5, 100]})
        mock_read_sql.return_value = df
        
        result = server.analyze_metrics("SELECT * FROM t", "val")
        stats = json.loads(result)
        
        assert stats['count'] == 6
        assert stats['mean'] > 0

@patch('server.vwcd')
@patch('pandas.read_sql')
def test_analyze_change_points_from_sql(mock_read_sql, mock_vwcd, mock_db_pool):
    mock_pool, mock_conn, _ = mock_db_pool
    
    # Setup mock data
    df = pd.DataFrame({'metric': [10, 10, 10, 20, 20, 20]})
    mock_read_sql.return_value = df
    
    # Setup vwcd mock
    mock_vwcd.vwcd.return_value = ([3], None, None, 0.1)
    mock_vwcd.get_segments.return_value = [{"start": 0, "end": 3, "mean": 10}, {"start": 3, "end": 6, "mean": 20}]
    
    # Test valid execution
    result = server.analyze_change_points_from_sql("SELECT * FROM t", "metric")
    data = json.loads(result)
    
    assert len(data["change_points"]) == 1
    assert data["change_points"][0] == 3
    assert len(data["segments"]) == 2
    mock_read_sql.assert_called_once()
    mock_vwcd.vwcd.assert_called_once()
    
    # Test invalid query
    assert "Error: Only SELECT" in server.analyze_change_points_from_sql("DELETE FROM t", "metric")
    
    # Test missing column
    assert "Metric column" in server.analyze_change_points_from_sql("SELECT * FROM t", "wrong_col")
    
    # Test empty data
    mock_read_sql.return_value = pd.DataFrame()
    assert "No data found" in server.analyze_change_points_from_sql("SELECT * FROM t", "metric")