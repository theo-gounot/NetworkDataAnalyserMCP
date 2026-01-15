import os
import json
import logging
import pandas as pd
import numpy as np
import asyncpg
from contextlib import asynccontextmanager
from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP
import vwcd

# --- Logging Configuration ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("Data-Analyzer-MCP")

# --- Environment & Configuration ---
load_dotenv()

DB_HOST = os.getenv("DB_HOST")
DB_PORT = os.getenv("DB_PORT")
DB_NAME = os.getenv("DB_NAME")
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
MCP_HOST = os.getenv("MCP_HOST", "0.0.0.0")
MCP_PORT = int(os.getenv("MCP_PORT", 8000))
DB_TIMEOUT = int(os.getenv("DB_TIMEOUT", 300))

# --- Global State ---
_db_pool = None
_cached_docs = {}

@asynccontextmanager
async def lifespan(context):
    """Initialize database pool and cache documentation on startup."""
    global _db_pool, _cached_docs
    
    # Init DB Pool
    try:
        if not _db_pool:
            _db_pool = await asyncpg.create_pool(
                host=DB_HOST,
                port=DB_PORT,
                database=DB_NAME,
                user=DB_USER,
                password=DB_PASSWORD,
                min_size=1,
                max_size=20,
                command_timeout=DB_TIMEOUT
            )
            logger.info("Database connection pool initialized.")
    except Exception as e:
        logger.error(f"Failed to initialize database pool: {e}")

    # Load Documentation
    try:
        if not _cached_docs:
            if os.path.exists("documentation.json"):
                with open("documentation.json", "r") as f:
                    _cached_docs = json.load(f)
                logger.info("Documentation loaded into cache.")
            else:
                logger.warning("documentation.json not found.")
    except Exception as e:
        logger.error(f"Failed to load documentation: {e}")
        
    yield
    
    # Cleanup
    if _db_pool:
        await _db_pool.close()
        _db_pool = None
        logger.info("Database connection pool closed.")

# Initialize FastMCP with lifespan
mcp = FastMCP(name="Data-Analyzer-MCP", host=MCP_HOST, port=MCP_PORT, lifespan=lifespan)

# --- Database Helper ---
@asynccontextmanager
async def get_db_connection():
    """Yields a connection from the pool and ensures it's returned."""
    if _db_pool is None:
        raise Exception("Database pool is not initialized.")
    
    async with _db_pool.acquire() as conn:
        yield conn

async def fetch_as_dataframe(query: str, *args) -> pd.DataFrame:
    """Helper to fetch data using asyncpg and convert to DataFrame."""
    async with get_db_connection() as conn:
        records = await conn.fetch(query, *args)
        if not records:
            return pd.DataFrame()
        return pd.DataFrame([dict(r) for r in records])

def to_toon(df: pd.DataFrame) -> str:
    """Converts a DataFrame to Token Oriented Object Notation (TOON) - a compact pipe-separated format."""
    if df.empty:
        return ""
    
    # Header: Column names
    header = " | ".join(df.columns)
    
    # Rows: Values
    # We use a custom format to keep it compact: 
    # - ISO format for dates
    # - Minimal precision for floats to save tokens
    def format_val(v):
        if pd.isna(v): return ""
        if isinstance(v, (float, np.floating)): return f"{v:.4g}"
        return str(v)

    rows = []
    for _, row in df.iterrows():
        rows.append(" | ".join(row.map(format_val)))
    
    return f"{header}\n" + "\n".join(rows)

# --- Tools ---

@mcp.tool(description="List all available tables in the public schema of the database.")
async def list_network_tables() -> str:
    """Start here. Discover the available database tables (public schema) to understand what network telemetry data is accessible."""
    logger.info("Tool called: list_tables")
    try:
        query = """
            SELECT table_name 
            FROM information_schema.tables 
            WHERE table_schema = 'public'
        """
        async with get_db_connection() as conn:
            records = await conn.fetch(query)
            tables = [r['table_name'] for r in records]
        return json.dumps({"tables": tables})
    except Exception as e:
        logger.error(f"Error in list_tables: {e}")
        return f"Error listing tables: {str(e)}"

@mcp.tool(description="Get the column names, types, and a few sample rows for a specific table.")
async def describe_network_table(table_name: str) -> str:
    """Inspect the schema of a specific table. Returns column names and sample rows. **ALWAYS** run this before writing a custom SQL query to ensure your column names are correct."""
    logger.info(f"Tool called: describe_table (table={table_name})")
    try:
        # Validate table_name to prevent injection (simple alphanumeric check recommended)
        if not table_name.replace("_","").isalnum():
             return "Error: Invalid table name."

        # Get column info
        query_cols = """
            SELECT column_name, data_type 
            FROM information_schema.columns 
            WHERE table_name = $1
        """
        df_cols = await fetch_as_dataframe(query_cols, table_name)
        
        # Get sample rows
        query_sample = f"SELECT * FROM {table_name} LIMIT 3"
        df_sample = await fetch_as_dataframe(query_sample)
        
        return json.dumps({
            "columns": df_cols.to_dict(orient="records") if not df_cols.empty else [],
            "sample_data_toon": to_toon(df_sample)
        }, default=str)
    except Exception as e:
        logger.error(f"Error in describe_table: {e}")
        return f"Error describing table {table_name}: {str(e)}"

@mcp.tool(description="Execute a read-only SQL query (SELECT/WITH) to retrieve data. Use this for custom filtering and joins.")
async def query_data(sql_query: str) -> str:
    """
    Execute custom read-only SQL (SELECT/WITH) for complex aggregations or filtering not covered by other tools. 
    Returns data in TOON format (Pipe-separated) for token efficiency.
    """
    logger.info(f"Tool called: query_data")
    
    cleaned_query = sql_query.strip().lower()
    valid_starts = ("select", "with", "explain")
    
    if not any(cleaned_query.startswith(prefix) for prefix in valid_starts):
        logger.warning(f"Blocked invalid query: {sql_query[:50]}...")
        return "Error: Only SELECT, WITH, and EXPLAIN queries are allowed."
    
    try:
        df = await fetch_as_dataframe(sql_query)
        if df.empty:
            return "No data found."
        return to_toon(df)
    except Exception as e:
        logger.error(f"Error in query_data: {e}")
        return f"Error executing query: {str(e)}"

@mcp.tool(description="Calculate statistical metrics (mean, median, p95) for a numeric column from a SQL query.")
async def analyze_metrics(sql_query: str, metric_column: str, groupby_column: str = None) -> str:
    """
    Fetch data via SQL and perform statistical analysis on a specific metric.
    Returns mean, median, std dev, and percentiles (p5, p95).
    """
    logger.info(f"Tool called: analyze_metrics (metric={metric_column})")
    try:
        df = await fetch_as_dataframe(sql_query)
        
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
        logger.error(f"Error in analyze_metrics: {e}")
        return f"Error analyzing metrics: {str(e)}"

@mcp.tool(description="Detect structural changes (changepoints) in a time-series column using the VWCD algorithm.")
async def analyze_change_points_from_sql(sql_query: str, metric_column: str) -> str:
    """
    Execute a SQL query and detect change points on a specific metric column using the VWCD algorithm.
    This avoids fetching large datasets to the client.
    """
    logger.info(f"Tool called: analyze_change_points_from_sql (metric={metric_column})")
    
    # Basic query validation
    cleaned_query = sql_query.strip().lower()
    valid_starts = ("select", "with", "explain")
    if not any(cleaned_query.startswith(prefix) for prefix in valid_starts):
        return "Error: Only SELECT, WITH, and EXPLAIN queries are allowed."

    try:
        df = await fetch_as_dataframe(sql_query)
        
        if df.empty:
            return "No data found."
        
        if metric_column not in df.columns:
            return f"Metric column '{metric_column}' not found. Available columns: {list(df.columns)}"
        
        # Ensure metric is numeric and drop NaNs
        data = pd.to_numeric(df[metric_column], errors='coerce').dropna().tolist()
        
        if not data:
            return "Error: No valid numeric data found in the specified column."

        # Convert to numpy array for VWCD
        X = np.array(data)
        
        # Run VWCD
        # Note: VWCD is CPU bound and synchronous. In a high-load async server, 
        # this should be run in a separate thread/process to avoid blocking the event loop.
        # However, for simplicity here we call it directly, or we could use loop.run_in_executor
        import asyncio
        loop = asyncio.get_running_loop()
        CP, _, _, elapsed = await loop.run_in_executor(None, vwcd.vwcd, X)
        
        # Get segments with statistical description
        segments = vwcd.get_segments(X, CP)
            
        return json.dumps({
            "change_points": [int(cp) for cp in CP],
            "segments": segments,
            "elapsed_time_ms": elapsed * 1000
        })

    except Exception as e:
        logger.error(f"Error in analyze_change_points_from_sql: {e}")
        return f"Error analyzing change points from SQL: {str(e)}"

@mcp.tool(description="Get reference documentation for M-Lab network metrics (NDT, Traceroute).")
def get_mlab_documentation(topic: str = None) -> str:
    """
    Get documentation about M-Lab tools (NDT, Traceroute).
    Uses cached documentation.
    """
    logger.info(f"Tool called: get_mlab_documentation (topic={topic})")
    try:
        # Use cached docs
        doc = _cached_docs
        if not doc:
            return "Error: Documentation not loaded."
            
        if topic:
            return json.dumps(doc.get(topic.lower()))
        return json.dumps(doc)
    except Exception as e:
        logger.error(f"Error reading documentation: {e}")
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
    logger.info("Starting Data Analyzer MCP Server...")
    mcp.run(transport="sse")
