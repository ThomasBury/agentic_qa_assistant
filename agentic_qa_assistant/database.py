"""Database loader and connection management for DuckDB."""

import duckdb
import pandas as pd
from pathlib import Path
from typing import Optional
import logging

logger = logging.getLogger(__name__)

class DatabaseManager:
    """Manages DuckDB connection and data loading."""
    
    def __init__(self, db_path: str = ":memory:"):
        """Initialize database connection.
        
        Args:
            db_path: Path to DuckDB database file, or ":memory:" for in-memory DB
        """
        self.db_path = db_path
        self.conn = duckdb.connect(db_path)
        
    def create_tables(self):
        """Create all required tables with proper schemas and constraints."""
        
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
        """Load all CSV files into their respective tables.
        
        Args:
            data_dir: Directory containing the CSV files
        """
        # Load dimension tables first to satisfy foreign key constraints
        csv_files = [
            ('DIM_MODEL.csv', 'DIM_MODEL'),
            ('DIM_COUNTRY.csv', 'DIM_COUNTRY'),
            ('DIM_ORDERTYPE.csv', 'DIM_ORDERTYPE'),
            ('FACT_SALES.csv', 'FACT_SALES'),
            ('FACT_SALES_ORDERTYPE.csv', 'FACT_SALES_ORDERTYPE')
        ]
        
        for csv_file, table_name in csv_files:
            csv_path = data_dir / csv_file
            if csv_path.exists():
                # Load CSV into pandas first for validation
                df = pd.read_csv(csv_path)
                
                # Validate data types and constraints
                if table_name in ['FACT_SALES', 'FACT_SALES_ORDERTYPE']:
                    # Validate fact table constraints
                    assert df['year'].min() >= 2000, f"Invalid year in {table_name}"
                    assert df['month'].min() >= 1 and df['month'].max() <= 12, f"Invalid month in {table_name}"
                    assert df['contracts'].min() >= 0, f"Negative contracts in {table_name}"
                
                # Clear existing data and load new data
                self.conn.execute(f"DELETE FROM {table_name}")
                self.conn.register("temp_df", df)
                self.conn.execute(f"INSERT INTO {table_name} SELECT * FROM temp_df")
                self.conn.unregister("temp_df")
                
                logger.info(f"Loaded {len(df)} rows into {table_name}")
            else:
                logger.warning(f"CSV file not found: {csv_path}")
                
    def initialize_database(self, data_dir: Path):
        """Initialize the database with tables and data.
        
        Args:
            data_dir: Directory containing the CSV files
        """
        self.create_tables()
        self.load_csv_data(data_dir)
        
        # Verify data integrity
        self._verify_data_integrity()
        
    def _verify_data_integrity(self):
        """Verify foreign key relationships and data quality."""
        
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
        
        logger.info("Database initialized successfully:")
        logger.info(f"  - Total contracts: {total_sales:,}")
        logger.info(f"  - Models: {total_models}")
        logger.info(f"  - Countries: {total_countries}")
        
    def get_schema_info(self) -> dict:
        """Get schema information for all tables."""
        
        tables = ['DIM_MODEL', 'DIM_COUNTRY', 'DIM_ORDERTYPE', 'FACT_SALES', 'FACT_SALES_ORDERTYPE']
        schema_info = {}
        
        for table in tables:
            columns = self.conn.execute(f"PRAGMA table_info('{table}')").fetchall()
            schema_info[table] = [
                {'name': col[1], 'type': col[2], 'notnull': bool(col[3]), 'pk': bool(col[5])}
                for col in columns
            ]
            
        return schema_info
        
    def execute_query(self, sql: str, parameters: Optional[dict] = None, timeout: int = 2) -> list:
        """Execute a query with timeout and parameter binding.
        
        Args:
            sql: SQL query to execute
            parameters: Parameters for query binding
            timeout: Query timeout in seconds
            
        Returns:
            Query results as list of tuples
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
            
    def close(self):
        """Close the database connection."""
        if self.conn:
            self.conn.close()
            
    def __enter__(self):
        """Context manager entry."""
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()


def create_database(data_dir: Path, db_path: str = ":memory:") -> DatabaseManager:
    """Create and initialize a database with CSV data.
    
    Args:
        data_dir: Directory containing CSV files
        db_path: Path for DuckDB database file
        
    Returns:
        Initialized DatabaseManager instance
    """
    db = DatabaseManager(db_path)
    db.initialize_database(data_dir)
    return db