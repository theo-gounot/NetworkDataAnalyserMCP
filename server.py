import os
import json
import pandas as pd
import numpy as np
import psycopg2
from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP
import vwcd

# Load environment variables
load_dotenv()

# Configuration
DB_HOST = os.getenv("DB_HOST")
DB_PORT = os.getenv("DB_PORT")
DB_NAME = os.getenv("DB_NAME")
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
MCP_HOST = os.getenv("MCP_HOST", "0.0.0.0")
MCP_PORT = int(os.getenv("MCP_PORT", 8000))

# Initialize FastMCP
mcp = FastMCP(name="Data-Analyzer-MCP", host=MCP_HOST, port=MCP_PORT)

# --- Database Helper ---
def get_db_connection():
    """Establish a connection to the PostgreSQL database."""
    return psycopg2.connect(
        host=DB_HOST,
        port=DB_PORT,
        database=DB_NAME,
        user=DB_USER,
        password=DB_PASSWORD
    )

# --- Tools ---

@mcp.tool()
def list_tables() -> str:
    """List all available tables in the public schema of the database."""
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("""
            SELECT table_name 
            FROM information_schema.tables 
            WHERE table_schema = 'public'
        """)
        tables = [row[0] for row in cur.fetchall()]
        cur.close()
        conn.close()
        return json.dumps({"tables": tables})
    except Exception as e:
        return f"Error listing tables: {str(e)}"

@mcp.tool()
def describe_table(table_name: str) -> str:
    """Get the column names, types, and a few sample rows for a specific table."""
    try:
        conn = get_db_connection()
        # Get column info
        query_cols = f"""
            SELECT column_name, data_type 
            FROM information_schema.columns 
            WHERE table_name = %s
        """
        df_cols = pd.read_sql(query_cols, conn, params=(table_name,))
        
        # Get sample rows
        query_sample = f"SELECT * FROM {table_name} LIMIT 3"
        df_sample = pd.read_sql(query_sample, conn)
        
        conn.close()
        
        return json.dumps({
            "columns": df_cols.to_dict(orient="records"),
            "sample_data": df_sample.to_dict(orient="records")
        }, default=str)
    except Exception as e:
        return f"Error describing table {table_name}: {str(e)}"

@mcp.tool()
def query_data(sql_query: str) -> str:
    """
    Execute a read-only SQL query on the database.
    Use this to fetch specific datasets for analysis.
    """
    if not sql_query.strip().lower().startswith("select"):
        return "Error: Only SELECT queries are allowed."
    
    try:
        conn = get_db_connection()
        df = pd.read_sql(sql_query, conn)
        conn.close()
        return df.to_json(orient="records", date_format="iso")
    except Exception as e:
        return f"Error executing query: {str(e)}"

@mcp.tool()
def analyze_metrics(sql_query: str, metric_column: str, groupby_column: str = None) -> str:
    """
    Fetch data via SQL and perform statistical analysis on a specific metric.
    Returns mean, median, std dev, and percentiles (p5, p95).
    """
    try:
        conn = get_db_connection()
        df = pd.read_sql(sql_query, conn)
        conn.close()
        
        if df.empty:
            return "No data found."
        
        if metric_column not in df.columns:
            return f"Metric column '{metric_column}' not found."
        
        # Ensure metric is numeric
        df[metric_column] = pd.to_numeric(df[metric_column], errors='coerce')
        
        if groupby_column and groupby_column in df.columns:
            stats = df.groupby(groupby_column)[metric_column].agg(
                ['count', 'mean', 'median', 'std', 'min', 'max']
            ).reset_index()
            # Add percentiles
            stats['p05'] = df.groupby(groupby_column)[metric_column].quantile(0.05).values
            stats['p95'] = df.groupby(groupby_column)[metric_column].quantile(0.95).values
        else:
            stats = df[metric_column].describe(percentiles=[.05, .5, .95]).to_dict()
            
        return json.dumps(stats.to_dict() if isinstance(stats, pd.DataFrame) else stats)
    except Exception as e:
        return f"Error analyzing metrics: {str(e)}"

@mcp.tool()
def detect_change_points(data: list[float]) -> str:
    """
    Detect change points in a time series using the VWCD algorithm.
    Returns a list of segments with their start/end indices and statistical description (mean, std).
    """
    try:
        if not data:
            return "Error: Empty data provided."
        
        # Convert to numpy array
        X = np.array(data)
        
        # Run VWCD
        CP, M0, S0, elapsed = vwcd.vwcd(X)
        
        # Format results by calculating stats for each segment defined by CP
        segments = []
        change_points = [-1] + CP + [len(X)-1]
        
        for i in range(len(change_points) - 1):
            s = change_points[i] + 1
            e = change_points[i+1]
            if s > e: continue 
            
            segment_data = X[s:e+1]
            mean_val = float(np.mean(segment_data))
            std_val = float(np.std(segment_data, ddof=1)) if len(segment_data) > 1 else 0.0
            
            segments.append({
                "segment_index": i,
                "start_index": int(s),
                "end_index": int(e),
                "length": int(len(segment_data)),
                "mean": mean_val,
                "std": std_val
            })
            
        return json.dumps({
            "change_points": [int(cp) for cp in CP],
            "segments": segments,
            "elapsed_time_ms": elapsed * 1000
        })

    except Exception as e:
        return f"Error detecting change points: {str(e)}"

@mcp.tool()
def get_mlab_documentation(topic: str = None) -> str:
    """
    Get documentation about M-Lab tools (NDT, Traceroute).
    """
    try:
        with open("documentation.json", "r") as f:
            doc = json.load(f)
        if topic and topic.lower() in doc:
            return json.dumps(doc[topic.lower()])
        return json.dumps(doc)
    except Exception as e:
        return f"Error reading documentation: {str(e)}"

# --- Prompts ---

@mcp.prompt()
def investigate_latency(raspberry_id: str = None):
    """Template for investigating high latency."""
    scope = f"for the Raspberry Pi '{raspberry_id}'" if raspberry_id else "across all devices"
    return f"""I need to investigate high latency {scope}. 
Please follow these steps:
1. List tables to find NDT/Traceroute data.
2. Query average and p95 RTT for the last 24h.
3. Use Traceroute data to identify bottleneck hops if RTT is high.
"""

@mcp.prompt()
def network_stability_report(days: int = 7):
    """Template for a stability report."""
    return f"""Please generate a network stability report for the last {days} days.
1. Identify all active Raspberry Pi units.
2. Calculate avg download/upload and packet loss.
3. Identify frequency of RTT spikes.
4. Compare performance across units.
"""

@mcp.prompt()
def help():
    """Basic help and discovery of the database."""
    return """I can help you analyze network data. 
Start by listing the tables to see what data we have:
- Use `list_tables` to see the schema.
- Use `describe_table` to inspect specific metrics.
- Ask me to 'Investigate latency' for a deep dive.
"""

@mcp.prompt()
def check_device_status(raspberry_id: str):
    """Check if a device is reporting data recently."""
    return f"""Please check the status of {raspberry_id}.
1. Query the most recent timestamp for this device in the NDT or Traceroute tables.
2. If it's older than 1 hour, warn that it might be offline.
"""

if __name__ == "__main__":
    # Run via SSE
    mcp.run(transport="sse")
