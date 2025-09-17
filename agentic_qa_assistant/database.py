"""Database loader and connection management for DuckDB."""

import duckdb
import pandas as pd
from pathlib import Path
from typing import Optional, List, Tuple, Dict, Any
import logging

logger = logging.getLogger(__name__)

class DatabaseManager:
    """Manages DuckDB connection and data loading."""
    
    def __init__(self, db_path: str = ":memory:"):
        """Initialize the DatabaseManager and connect to DuckDB.

        Parameters
        ----------
        db_path : str, optional
            Path to the DuckDB database file. Defaults to ":memory:" for an
            in-memory database.

        """
        self.db_path = db_path
        self.conn = duckdb.connect(db_path)
        
    def create_tables(self):
        """Create all required tables in the database.

        This method defines the schema for dimension and fact tables.
        """
        
        # Create dimension tables
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS DIM_MODEL (
                model_id INTEGER PRIMARY KEY,
                model_name VARCHAR NOT NULL,
                brand VARCHAR NOT NULL,
                segment VARCHAR NOT NULL,
                powertrain VARCHAR NOT NULL
            )
        """)
        
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS DIM_COUNTRY (
                country_code VARCHAR PRIMARY KEY,
                country VARCHAR NOT NULL,
                region VARCHAR NOT NULL
            )
        """)
        
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS DIM_ORDERTYPE (
                ordertype_id INTEGER PRIMARY KEY,
                ordertype_name VARCHAR NOT NULL,
                description VARCHAR NOT NULL
            )
        """)
        
        # Create fact tables (without foreign key constraints for simplicity)
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS FACT_SALES (
                model_id INTEGER NOT NULL,
                country_code VARCHAR NOT NULL,
                year INTEGER NOT NULL,
                month INTEGER NOT NULL,
                contracts INTEGER NOT NULL,
                PRIMARY KEY (model_id, country_code, year, month)
            )
        """)
        
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS FACT_SALES_ORDERTYPE (
                model_id INTEGER NOT NULL,
                country_code VARCHAR NOT NULL,
                year INTEGER NOT NULL,
                month INTEGER NOT NULL,
                contracts INTEGER NOT NULL,
                ordertype_id INTEGER NOT NULL,
                PRIMARY KEY (model_id, country_code, year, month, ordertype_id)
            )
        """)
        
        logger.info("Created all database tables")
        
    def load_csv_data(self, data_dir: Path):
        """Load data from CSV files into the corresponding database tables.

        Parameters
        ----------
        data_dir : Path
            The directory containing the CSV data files.

        """
        # Load dimension tables first to satisfy foreign key constraints
        csv_files = [
            ('DIM_MODEL.csv', 'DIM_MODEL'),
            ('DIM_COUNTRY.csv', 'DIM_COUNTRY'),
            ('DIM_ORDERTYPE.csv', 'DIM_ORDERTYPE'),
            ('FACT_SALES.csv', 'FACT_SALES'),
            ('FACT_SALES_ORDERTYPE.csv', 'FACT_SALES_ORDERTYPE')
        ]

        # Ensure DataFrame columns match table schemas exactly (avoid position-based misalignment)
        schema_order = {
            'DIM_MODEL': ['model_id', 'model_name', 'brand', 'segment', 'powertrain'],
            'DIM_COUNTRY': ['country_code', 'country', 'region'],
            'DIM_ORDERTYPE': ['ordertype_id', 'ordertype_name', 'description'],
            'FACT_SALES': ['model_id', 'country_code', 'year', 'month', 'contracts'],
            'FACT_SALES_ORDERTYPE': ['model_id', 'country_code', 'year', 'month', 'contracts', 'ordertype_id'],
        }
        
        for csv_file, table_name in csv_files:
            csv_path = data_dir / csv_file
            if csv_path.exists():
                # Load CSV into pandas first for validation
                df = pd.read_csv(csv_path)

                # Reorder columns to match table schema exactly
                expected_cols = schema_order[table_name]
                missing = [c for c in expected_cols if c not in df.columns]
                if missing:
                    logger.error(f"{table_name}: missing expected columns in CSV: {missing}")
                    continue
                df = df[expected_cols]
                
                # Validate data types and constraints
                if table_name in ['FACT_SALES', 'FACT_SALES_ORDERTYPE']:
                    # Validate fact table constraints
                    assert df['year'].min() >= 2000, f"Invalid year in {table_name}"
                    assert df['month'].min() >= 1 and df['month'].max() <= 12, f"Invalid month in {table_name}"
                    assert df['contracts'].min() >= 0, f"Negative contracts in {table_name}"
                
                # Clear existing data and load new data using explicit column list
                self.conn.execute(f"DELETE FROM {table_name}")
                self.conn.register("temp_df", df)
                cols_csv = ", ".join(expected_cols)
                self.conn.execute(f"INSERT INTO {table_name} ({cols_csv}) SELECT {cols_csv} FROM temp_df")
                self.conn.unregister("temp_df")
                
                logger.info(f"Loaded {len(df)} rows into {table_name}")
            else:
                logger.warning(f"CSV file not found: {csv_path}")
                
    def initialize_database(self, data_dir: Path):
        """Create tables, load data, and verify integrity.

        Parameters
        ----------
        data_dir : Path
            The directory containing the CSV data files.

        """
        self.create_tables()
        self.load_csv_data(data_dir)
        
        # Verify data integrity
        self._verify_data_integrity()
        
    def _verify_data_integrity(self):
        """Perform and log data integrity and referential checks."""
        
        # Quick table row counts
        try:
            dim_model_count = self.conn.execute("SELECT COUNT(*) FROM DIM_MODEL").fetchone()[0]
            dim_country_count = self.conn.execute("SELECT COUNT(*) FROM DIM_COUNTRY").fetchone()[0]
            dim_ordertype_count = self.conn.execute("SELECT COUNT(*) FROM DIM_ORDERTYPE").fetchone()[0]
            fact_sales_count = self.conn.execute("SELECT COUNT(*) FROM FACT_SALES").fetchone()[0]
            fact_sales_ot_count = self.conn.execute("SELECT COUNT(*) FROM FACT_SALES_ORDERTYPE").fetchone()[0]
            logger.info("Table row counts: DIM_MODEL=%s, DIM_COUNTRY=%s, DIM_ORDERTYPE=%s, FACT_SALES=%s, FACT_SALES_ORDERTYPE=%s",
                        dim_model_count, dim_country_count, dim_ordertype_count, fact_sales_count, fact_sales_ot_count)
        except Exception as e:
            logger.warning(f"Failed to fetch table row counts: {e}")
        
        # Check referential integrity (model_id)
        orphaned_sales = self.conn.execute("""
            SELECT COUNT(*) 
            FROM FACT_SALES fs 
            LEFT JOIN DIM_MODEL dm ON fs.model_id = dm.model_id
            WHERE dm.model_id IS NULL
        """).fetchone()[0]
        
        if orphaned_sales > 0:
            logger.warning(f"Found {orphaned_sales} sales records with invalid model_id")
            
        # Check referential integrity (country_code)
        orphaned_countries = self.conn.execute("""
            SELECT COUNT(*) 
            FROM FACT_SALES fs 
            LEFT JOIN DIM_COUNTRY dc ON fs.country_code = dc.country_code
            WHERE dc.country_code IS NULL
        """).fetchone()[0]
        
        if orphaned_countries > 0:
            logger.warning(f"Found {orphaned_countries} sales records with invalid country_code")
        else:
            logger.info("All referential integrity checks passed")
            
        # Log data summary
        total_sales = self.conn.execute("SELECT SUM(contracts) FROM FACT_SALES").fetchone()[0]
        total_models = self.conn.execute("SELECT COUNT(*) FROM DIM_MODEL").fetchone()[0]
        total_countries = self.conn.execute("SELECT COUNT(*) FROM DIM_COUNTRY").fetchone()[0]
        years = self.conn.execute("SELECT MIN(year), MAX(year) FROM FACT_SALES").fetchone()
        distinct_months = self.conn.execute("SELECT COUNT(DISTINCT month) FROM FACT_SALES").fetchone()[0]
        
        logger.info("Database initialized successfully:")
        logger.info(f"  - Total contracts: {total_sales:,}")
        logger.info(f"  - Models: {total_models}")
        logger.info(f"  - Countries: {total_countries}")
        logger.info(f"  - Years: {years[0]} to {years[1]}")
        logger.info(f"  - Distinct months present: {distinct_months}")
        
    def get_schema_info(self) -> dict:
        """Retrieve schema information for all managed tables.

        Returns
        -------
        dict
            A dictionary where keys are table names and values are lists
            of column information dictionaries.

        """
        tables = ['DIM_MODEL', 'DIM_COUNTRY', 'DIM_ORDERTYPE', 'FACT_SALES', 'FACT_SALES_ORDERTYPE']
        schema_info = {}
        
        for table in tables:
            columns = self.conn.execute(f"PRAGMA table_info('{table}')").fetchall()
            schema_info[table] = [
                {'name': col[1], 'type': col[2], 'notnull': bool(col[3]), 'pk': bool(col[5])}
                for col in columns
            ]
            
        return schema_info
        
    def execute_query(self, sql: str, parameters: Optional[dict] = None, timeout: int = 2) -> List[Tuple]:
        """Execute a SQL query with parameter binding.

        Note: The timeout parameter is not natively supported by DuckDB's Python
        API and is kept for interface compatibility.

        Parameters
        ----------
        sql : str
            The SQL query to execute.
        parameters : Optional[dict], optional
            A dictionary of parameters for query binding.
        timeout : int, optional
            Query timeout in seconds (currently ignored).

        Returns
        -------
        List[Tuple]
            A list of tuples representing the query results.

        """
        try:
            # Set query timeout (DuckDB doesn't have built-in timeout, so we'll implement it)
            if parameters:
                result = self.conn.execute(sql, parameters).fetchall()
            else:
                result = self.conn.execute(sql).fetchall()
                
            return result
            
        except Exception as e:
            logger.error(f"Query execution failed: {e}")
            raise
            
    def get_explain_plan(self, sql: str, analyze: bool = False) -> str:
        """Get the EXPLAIN or EXPLAIN ANALYZE plan for a SQL query.

        Parameters
        ----------
        sql : str
            The SQL query to explain.
        analyze : bool, optional
            If True, runs EXPLAIN ANALYZE to include execution metrics.

        Returns
        -------
        str
            A string representation of the query plan.

        """
        try:
            prefix = "EXPLAIN ANALYZE" if analyze else "EXPLAIN"
            rows = self.conn.execute(f"{prefix} {sql}").fetchall()
            # Join rows into a readable string
            return "\n".join(["\t".join(str(col) for col in row) for row in rows])
        except Exception as e:
            logger.error(f"Failed to get explain plan: {e}")
            return f"Failed to get explain plan: {e}"
            
    def close(self):
        """Close the database connection."""
        if self.conn:
            self.conn.close()
            
    def get_integrity_report(self) -> Dict[str, Any]:
        """Compute and return a data integrity report.

        Returns
        -------
        Dict[str, Any]
            A dictionary containing various data integrity metrics, such as
            row counts, orphan checks, and data summaries.

        """
        report = {}
        try:
            report['row_counts'] = {
                'DIM_MODEL': self.conn.execute("SELECT COUNT(*) FROM DIM_MODEL").fetchone()[0],
                'DIM_COUNTRY': self.conn.execute("SELECT COUNT(*) FROM DIM_COUNTRY").fetchone()[0],
                'DIM_ORDERTYPE': self.conn.execute("SELECT COUNT(*) FROM DIM_ORDERTYPE").fetchone()[0],
                'FACT_SALES': self.conn.execute("SELECT COUNT(*) FROM FACT_SALES").fetchone()[0],
                'FACT_SALES_ORDERTYPE': self.conn.execute("SELECT COUNT(*) FROM FACT_SALES_ORDERTYPE").fetchone()[0],
            }
        except Exception as e:
            report['row_counts_error'] = str(e)
        try:
            report['orphaned_model_sales'] = self.conn.execute(
                """
                SELECT COUNT(*) 
                FROM FACT_SALES fs 
                LEFT JOIN DIM_MODEL dm ON fs.model_id = dm.model_id
                WHERE dm.model_id IS NULL
                """
            ).fetchone()[0]
            report['orphaned_country_sales'] = self.conn.execute(
                """
                SELECT COUNT(*) 
                FROM FACT_SALES fs 
                LEFT JOIN DIM_COUNTRY dc ON fs.country_code = dc.country_code
                WHERE dc.country_code IS NULL
                """
            ).fetchone()[0]
        except Exception as e:
            report['orphan_checks_error'] = str(e)
        try:
            total_contracts = self.conn.execute("SELECT SUM(contracts) FROM FACT_SALES").fetchone()[0]
            min_year, max_year = self.conn.execute("SELECT MIN(year), MAX(year) FROM FACT_SALES").fetchone()
            distinct_months = self.conn.execute("SELECT COUNT(DISTINCT month) FROM FACT_SALES").fetchone()[0]
            report['summary'] = {
                'total_contracts': int(total_contracts) if total_contracts is not None else 0,
                'models': self.conn.execute("SELECT COUNT(*) FROM DIM_MODEL").fetchone()[0],
                'countries': self.conn.execute("SELECT COUNT(*) FROM DIM_COUNTRY").fetchone()[0],
                'year_range': [min_year, max_year],
                'distinct_months': distinct_months,
            }
        except Exception as e:
            report['summary_error'] = str(e)
        return report

    def get_integrity_report_text(self) -> str:
        """Generate a human-readable text summary of the integrity report.

        Returns
        -------
        str
            A formatted string summarizing the data integrity report.

        """
        r = self.get_integrity_report()
        lines = ["Data Integrity Report:"]
        rc = r.get('row_counts', {})
        lines.append(f"- Row counts: DIM_MODEL={rc.get('DIM_MODEL')}, DIM_COUNTRY={rc.get('DIM_COUNTRY')}, DIM_ORDERTYPE={rc.get('DIM_ORDERTYPE')}, FACT_SALES={rc.get('FACT_SALES')}, FACT_SALES_ORDERTYPE={rc.get('FACT_SALES_ORDERTYPE')}")
        if 'row_counts_error' in r:
            lines.append(f"- Row counts error: {r['row_counts_error']}")
        if 'orphaned_model_sales' in r:
            lines.append(f"- Orphaned FACT_SALES by model_id: {r['orphaned_model_sales']}")
        if 'orphaned_country_sales' in r:
            lines.append(f"- Orphaned FACT_SALES by country_code: {r['orphaned_country_sales']}")
        if 'orphan_checks_error' in r:
            lines.append(f"- Orphan checks error: {r['orphan_checks_error']}")
        s = r.get('summary') or {}
        if s:
            lines.append(f"- Total contracts: {s.get('total_contracts')}")
            yr = s.get('year_range') or [None, None]
            lines.append(f"- Year range: {yr[0]} to {yr[1]}")
            lines.append(f"- Distinct months present: {s.get('distinct_months')}")
            lines.append(f"- Models: {s.get('models')}, Countries: {s.get('countries')}")
        if 'summary_error' in r:
            lines.append(f"- Summary error: {r['summary_error']}")
        return "\n".join(lines)
            
    def __enter__(self):
        """Context manager entry."""
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit, ensuring the connection is closed."""
        self.close()


def create_database(data_dir: Path, db_path: str = ":memory:") -> DatabaseManager:
    """Create and initialize a new database instance from CSV data.

    Parameters
    ----------
    data_dir : Path
        The directory containing the CSV data files.
    db_path : str, optional
        Path for the DuckDB database file. Defaults to ":memory:".

    Returns
    -------
    DatabaseManager
        An initialized DatabaseManager instance with data loaded.

    """
    db = DatabaseManager(db_path)
    db.initialize_database(data_dir)
    return db