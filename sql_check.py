#!/usr/bin/env python3
"""
Standalone SQL checker for Agentic Q&A dataset.

- Loads CSVs from --data-dir into an in-memory DuckDB with the same schemas used by the project
- Lets you run arbitrary SQL via --sql / --sql-file or a predefined sample via --sample
- Prints results (or 'No data found'), shows row count and execution time

Usage examples:
- python3 sql_check.py --sample
- python3 sql_check.py --sql "SELECT COUNT(*) FROM FACT_SALES"
- python3 sql_check.py --sql-file my_query.sql
- python3 sql_check.py --data-dir ./data --sql "..."
"""

import argparse
import time
from pathlib import Path
from typing import Optional

import duckdb
import pandas as pd


SAMPLE_SQL = (
    """
    SELECT fs.year, fs.month, SUM(fs.contracts) AS contracts
    FROM FACT_SALES fs
    JOIN DIM_MODEL dm ON dm.model_id = fs.model_id
    JOIN DIM_COUNTRY dc ON dc.country_code = fs.country_code
    GROUP BY fs.year, fs.month
    ORDER BY fs.month
    LIMIT 10
    """
    .strip()
)

# Additional ready-made samples
SAMPLES = {
    # User's original question: Monthly RAV4 HEV sales in Germany in 2024
    "rav4_germany_2024": (
        """
        SELECT fs.year, fs.month, SUM(fs.contracts) AS contracts
        FROM FACT_SALES fs
        JOIN DIM_MODEL dm ON dm.model_id = fs.model_id
        JOIN DIM_COUNTRY dc ON dc.country_code = fs.country_code
        WHERE dm.model_name = 'RAV4'
          AND dm.powertrain = 'HEV'
          AND dc.country = 'Germany'
          AND fs.year = 2024
        GROUP BY fs.year, fs.month
        ORDER BY fs.month
        LIMIT 10000
        """.strip()
    ),
    # Compare Toyota vs Lexus SUV sales in Western Europe in 2024
    "toyota_vs_lexus_suv_western_europe_2024": (
        """
        SELECT dm.brand, SUM(fs.contracts) AS contracts
        FROM FACT_SALES fs
        JOIN DIM_MODEL dm ON dm.model_id = fs.model_id
        JOIN DIM_COUNTRY dc ON dc.country_code = fs.country_code
        WHERE dm.segment = 'SUV'
          AND dm.brand IN ('Toyota','Lexus')
          AND dc.region = 'Western Europe'
          AND fs.year = 2024
        GROUP BY dm.brand
        ORDER BY contracts DESC
        """.strip()
    ),
    # Monthly sales by country in 2024 (overview)
    "monthly_sales_by_country_2024": (
        """
        SELECT dc.country, fs.month, SUM(fs.contracts) AS contracts
        FROM FACT_SALES fs
        JOIN DIM_COUNTRY dc ON dc.country_code = fs.country_code
        WHERE fs.year = 2024
        GROUP BY dc.country, fs.month
        ORDER BY dc.country, fs.month
        LIMIT 200
        """.strip()
    ),
}


def create_tables(conn: duckdb.DuckDBPyConnection) -> None:
    """Create tables with the same schema as the main project.

    Parameters
    ----------
    conn : duckdb.DuckDBPyConnection
        An active DuckDB database connection.

    """
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS DIM_MODEL (
            model_id INTEGER PRIMARY KEY,
            model_name VARCHAR NOT NULL,
            brand VARCHAR NOT NULL,
            segment VARCHAR NOT NULL,
            powertrain VARCHAR NOT NULL
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS DIM_COUNTRY (
            country_code VARCHAR PRIMARY KEY,
            country VARCHAR NOT NULL,
            region VARCHAR NOT NULL
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS DIM_ORDERTYPE (
            ordertype_id INTEGER PRIMARY KEY,
            ordertype_name VARCHAR NOT NULL,
            description VARCHAR NOT NULL
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS FACT_SALES (
            model_id INTEGER NOT NULL,
            country_code VARCHAR NOT NULL,
            year INTEGER NOT NULL,
            month INTEGER NOT NULL,
            contracts INTEGER NOT NULL,
            PRIMARY KEY (model_id, country_code, year, month)
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS FACT_SALES_ORDERTYPE (
            model_id INTEGER NOT NULL,
            country_code VARCHAR NOT NULL,
            year INTEGER NOT NULL,
            month INTEGER NOT NULL,
            contracts INTEGER NOT NULL,
            ordertype_id INTEGER NOT NULL,
            PRIMARY KEY (model_id, country_code, year, month, ordertype_id)
        )
        """
    )


def load_csvs(conn: duckdb.DuckDBPyConnection, data_dir: Path) -> None:
    """Load CSVs from a directory into the corresponding DuckDB tables.

    This mirrors the project's loading approach (via pandas -> register -> insert)
    to minimize any differences in type inference.

    Important: Ensure columns are in the exact table order before INSERT to avoid
    position-based misalignment (e.g., DIM_COUNTRY CSV has 'country' first).

    Parameters
    ----------
    conn : duckdb.DuckDBPyConnection
        An active DuckDB database connection.
    data_dir : Path
        The path to the directory containing the CSV data files.

    """
    mapping = [
        ("DIM_MODEL.csv", "DIM_MODEL"),
        ("DIM_COUNTRY.csv", "DIM_COUNTRY"),
        ("DIM_ORDERTYPE.csv", "DIM_ORDERTYPE"),
        ("FACT_SALES.csv", "FACT_SALES"),
        ("FACT_SALES_ORDERTYPE.csv", "FACT_SALES_ORDERTYPE"),
    ]

    # Expected column order per table (must match CREATE TABLE definitions)
    schema_order = {
        "DIM_MODEL": ["model_id", "model_name", "brand", "segment", "powertrain"],
        "DIM_COUNTRY": ["country_code", "country", "region"],
        "DIM_ORDERTYPE": ["ordertype_id", "ordertype_name", "description"],
        "FACT_SALES": ["model_id", "country_code", "year", "month", "contracts"],
        "FACT_SALES_ORDERTYPE": ["model_id", "country_code", "year", "month", "contracts", "ordertype_id"],
    }

    for filename, table in mapping:
        path = data_dir / filename
        if not path.exists():
            print(f"[WARN] Missing CSV: {path}")
            continue

        df = pd.read_csv(path)
        # Reorder columns to match table schema if all expected cols are present
        expected = schema_order[table]
        missing = [c for c in expected if c not in df.columns]
        if missing:
            print(f"[ERROR] {table}: missing expected columns in CSV: {missing}")
            continue
        df = df[expected]

        # Clear and load
        conn.execute(f"DELETE FROM {table}")
        conn.register("temp_df", df)
        # Explicit column list for safety
        cols_csv = ", ".join(expected)
        conn.execute(f"INSERT INTO {table} ({cols_csv}) SELECT {cols_csv} FROM temp_df")
        conn.unregister("temp_df")
        print(f"[INFO] Loaded {len(df):,} rows into {table}")


def run_sql(
    conn: duckdb.DuckDBPyConnection,
    sql: str,
    limit: Optional[int] = None,
    explain: bool = False,
    explain_analyze: bool = False,
) -> None:
    """Execute a SQL query and print the results to the console.

    Optionally includes EXPLAIN plans and limits the number of rows displayed.

    Parameters
    ----------
    conn : duckdb.DuckDBPyConnection
        An active DuckDB database connection.
    sql : str
        The SQL query string to execute.
    limit : Optional[int], optional
        The maximum number of rows to display in the output.
    explain : bool, optional
        If True, prints the EXPLAIN plan for the query.
    explain_analyze : bool, optional
        If True, prints the EXPLAIN ANALYZE plan for the query.

    """
    sql_to_run = sql.strip()
    if not sql_to_run:
        print("[ERROR] Empty SQL provided")
        return

    # Print explain plan if requested
    if explain or explain_analyze:
        try:
            explain_kw = "EXPLAIN ANALYZE" if explain_analyze else "EXPLAIN"
            plan_df = conn.execute(f"{explain_kw} {sql_to_run}").df()
            print("\n🧩 Explain plan:" + (" (ANALYZE)" if explain_analyze else ""))
            print(plan_df.to_string(index=False))
        except Exception as e:
            print(f"[WARN] Explain failed: {e}")

    start = time.perf_counter()
    try:
        # Use DuckDB's .df() to get column names easily
        df = conn.execute(sql_to_run).df()
    except Exception as e:
        print(f"❌ Query failed: {e}")
        return
    elapsed_ms = int((time.perf_counter() - start) * 1000)

    print("\n💾 SQL Query:")
    print(sql_to_run)

    print("\n📊 Results:")
    if df.empty:
        print("No data found")
    else:
        # Respect optional limit for display
        if limit is not None:
            to_show = df.head(limit)
        else:
            to_show = df
        # Render without index
        print(to_show.to_string(index=False))
        if limit is not None and len(df) > limit:
            print(f"... and {len(df) - limit} more rows")

    print(f"\n⏱️ Executed in {elapsed_ms} ms, returned {len(df)} rows")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the script.

    Returns
    -------
    argparse.Namespace
        An object containing the parsed command-line arguments.

    """
    p = argparse.ArgumentParser(description="Standalone SQL checker for Agentic Q&A dataset")
    p.add_argument(
        "--data-dir",
        type=Path,
        default=Path("./data"),
        help="Directory containing CSV files (default: ./data)",
    )
    g = p.add_mutually_exclusive_group(required=False)
    g.add_argument("--sql", type=str, help="Run this SQL string")
    g.add_argument("--sql-file", type=Path, help="Path to a .sql file to run")
    g.add_argument("--sample", action="store_true", help="Run the predefined sample SQL")
    p.add_argument("--sample-name", type=str, choices=sorted(list(SAMPLES.keys())), help="Which predefined sample to run")
    p.add_argument("--explain", action="store_true", help="Print EXPLAIN plan before running the query")
    p.add_argument("--explain-analyze", action="store_true", help="Print EXPLAIN ANALYZE plan before running the query")
    p.add_argument("--limit", type=int, default=10000, help="Max rows to display (default: 10000)")
    return p.parse_args()


def main() -> None:
    """Run the standalone SQL checker.

    This script initializes an in-memory DuckDB, loads data from CSV files,
    and executes a user-provided SQL query, displaying the results and
    performance metrics.
    """
    args = parse_args()

    data_dir = args.data_dir
    if not data_dir.exists():
        print(f"[ERROR] Data directory not found: {data_dir}")
        return

    # Connect to in-memory DuckDB and prepare schema + data
    conn = duckdb.connect(":memory:")
    create_tables(conn)
    load_csvs(conn, data_dir)

    # Decide which SQL to run
    sql: Optional[str] = None
    if args.sample or args.sample_name:
        if args.sample_name:
            sql = SAMPLES[args.sample_name]
            print(f"[INFO] Running sample SQL: {args.sample_name}")
        else:
            sql = SAMPLE_SQL
            print("[INFO] Running default sample SQL (aggregate by year/month)")
    elif args.sql_file:
        try:
            sql = args.sql_file.read_text(encoding="utf-8")
        except Exception as e:
            print(f"[ERROR] Failed to read SQL file: {e}")
            return
    elif args.sql:
        sql = args.sql
    else:
        # Interactive one-liner input (no multiline editor)
        try:
            sql = input("Enter SQL to run: ").strip()
        except KeyboardInterrupt:
            print("\n[INFO] Aborted")
            return

    run_sql(
        conn,
        sql,
        limit=args.limit,
        explain=args.explain,
        explain_analyze=args.explain_analyze,
    )


if __name__ == "__main__":
    main()
