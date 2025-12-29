# Data Analyzer MCP Server

An MCP server designed for the LAND-UFRJ lab to analyze network data from M-Lab's NDT and Traceroute tools collected via Raspberry Pis. It provides database access, statistical analysis, and dynamic documentation.

## Features

- **Dynamic Database Discovery**: Automatically detects tables and schemas.
- **Statistical Analysis**: Calculates mean, median, p95, and more for network metrics.
- **Context-Aware Prompts**: Includes templates for investigating latency and generating stability reports.
- **Documentation**: Built-in reference for M-Lab metrics (NDT, Traceroute).

## Prerequisites

- Python 3.10+
- PostgreSQL Database (with NDT/Traceroute data)

## Installation

1. Clone the repository.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Configure the environment:
   Copy `.env.example` to `.env` and fill in your database credentials.
   ```bash
   cp .env.example .env
   ```

## Configuration

Edit `.env` to set your Database and Server configuration:

```ini
DB_HOST=your_db_host
DB_PORT=5432
DB_NAME=your_db_name
DB_USER=your_db_user
DB_PASSWORD=your_db_password

MCP_HOST=0.0.0.0
MCP_PORT=8000
```

## Usage

Run the script directly if configured:

```bash
python server.py
```
