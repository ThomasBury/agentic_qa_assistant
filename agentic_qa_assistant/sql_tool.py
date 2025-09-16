"""SQL Tool with safety validation and query generation."""

import time
import re
import logging
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum

import sqlglot
from sqlglot import parse_one, exp
from pydantic import BaseModel
from openai import OpenAI

from .database import DatabaseManager

logger = logging.getLogger(__name__)


class QueryValidationError(Exception):
    """Raised when SQL query validation fails."""
    pass


@dataclass
class SqlResult:
    """Result of SQL execution."""
    rows: List[Tuple]
    sql_used: str
    execution_ms: int
    row_count: int
    columns: List[str]


class SqlValidator:
    """Validates SQL queries for safety and compliance."""
    
    # Allowlisted tables and columns
    ALLOWED_TABLES = {
        'DIM_MODEL': {'model_id', 'model_name', 'brand', 'segment', 'powertrain'},
        'DIM_COUNTRY': {'country_code', 'country', 'region'},
        'DIM_ORDERTYPE': {'ordertype_id', 'ordertype_name', 'description'},
        'FACT_SALES': {'model_id', 'country_code', 'year', 'month', 'contracts'},
        'FACT_SALES_ORDERTYPE': {'model_id', 'country_code', 'year', 'month', 'contracts', 'ordertype_id'}
    }
    
    # Denied SQL statements and functions
    DENIED_STATEMENTS = {
        'CREATE', 'INSERT', 'UPDATE', 'DELETE', 'DROP', 'ALTER', 'TRUNCATE',
        'EXEC', 'EXECUTE', 'CALL', 'DECLARE', 'SET'
    }
    
    DENIED_FUNCTIONS = {
        'LOAD', 'COPY', 'EXPORT', 'IMPORT', 'PRAGMA'
    }
    
    def __init__(self):
        self.max_limit = 10000  # Maximum rows allowed
        
    def validate_sql(self, sql: str) -> str:
        """Validate SQL query for safety and compliance.
        
        Args:
            sql: SQL query to validate
            
        Returns:
            Cleaned and validated SQL query
            
        Raises:
            QueryValidationError: If query violates safety rules
        """
        try:
            # Clean the SQL
            sql = sql.strip().rstrip(';')
            
            # Parse the SQL
            parsed = parse_one(sql, dialect='duckdb')
            
            # Check if it's a SELECT statement
            if not isinstance(parsed, exp.Select):
                raise QueryValidationError("Only SELECT statements are allowed")
                
            # Validate tables and columns
            self._validate_tables_and_columns(parsed)
            
            # Check for denied functions
            self._check_denied_functions(parsed)
            
            # Ensure LIMIT clause exists for safety
            sql = self._ensure_limit_clause(sql, parsed)
            
            logger.info(f"SQL validation passed: {sql[:100]}...")
            return sql
            
        except sqlglot.ParseError as e:
            raise QueryValidationError(f"SQL parsing error: {e}")
        except Exception as e:
            raise QueryValidationError(f"SQL validation error: {e}")
            
    def _validate_tables_and_columns(self, parsed: exp.Expression):
        """Validate that only allowed tables and columns are referenced."""
        
        # Find all table references
        for table in parsed.find_all(exp.Table):
            table_name = table.name.upper()
            if table_name not in self.ALLOWED_TABLES:
                raise QueryValidationError(f"Table '{table_name}' not allowed")
                
        # Find all column references
        for column in parsed.find_all(exp.Column):
            if column.table:
                table_name = column.table.upper()
                column_name = column.name.lower()
                
                if table_name in self.ALLOWED_TABLES:
                    allowed_columns = self.ALLOWED_TABLES[table_name]
                    if column_name not in allowed_columns:
                        raise QueryValidationError(
                            f"Column '{column_name}' not allowed in table '{table_name}'"
                        )
                        
    def _check_denied_functions(self, parsed: exp.Expression):
        """Check for denied functions and statements."""
        
        # Check for denied functions
        for func in parsed.find_all(exp.Anonymous):
            func_name = func.this.upper() if hasattr(func, 'this') else ''
            if func_name in self.DENIED_FUNCTIONS:
                raise QueryValidationError(f"Function '{func_name}' not allowed")
                
        # Additional check for subqueries with CTEs or complex operations
        for cte in parsed.find_all(exp.CTE):
            raise QueryValidationError("Common Table Expressions (CTEs) not allowed")
            
    def _ensure_limit_clause(self, sql: str, parsed: exp.Select) -> str:
        """Ensure query has a reasonable LIMIT clause."""
        
        # Check if LIMIT already exists
        if parsed.args.get('limit'):
            limit_value = parsed.args['limit']
            if hasattr(limit_value, 'expression'):
                try:
                    limit_num = int(str(limit_value.expression))
                    if limit_num > self.max_limit:
                        raise QueryValidationError(f"LIMIT {limit_num} exceeds maximum {self.max_limit}")
                except ValueError:
                    raise QueryValidationError("Invalid LIMIT value")
            return sql
        else:
            # Add LIMIT if not present
            return f"{sql} LIMIT {self.max_limit}"


class SqlGenerator:
    """Generates SQL queries from natural language questions."""
    
    def __init__(self, openai_client: OpenAI, db_manager: DatabaseManager, model_name: str = "gpt-5-mini"):
        self.client = openai_client
        self.db_manager = db_manager
        self.schema_info = db_manager.get_schema_info()
        self.model_name = model_name
        
    def generate_sql(self, question: str) -> str:
        """Generate SQL query from natural language question.
        
        Args:
            question: Natural language question
            
        Returns:
            Generated SQL query
        """
        # Create schema description for the prompt
        schema_desc = self._create_schema_description()
        
        system_prompt = f"""You are an expert SQL query generator. Generate ONLY a single SELECT statement.

Database Schema:
{schema_desc}

Rules:
1. Use ONLY the tables and columns shown above
2. Generate ONLY SELECT statements - no CREATE, INSERT, UPDATE, DELETE
3. Use proper JOINs to connect related tables
4. Use GROUP BY when aggregating data
5. Include ORDER BY for meaningful sorting
6. Return only the SQL query, no explanations
7. Do not include LIMIT clause - it will be added automatically

Examples:
Question: "Monthly RAV4 HEV sales in Germany in 2024"
SQL: SELECT fs.year, fs.month, SUM(fs.contracts) AS contracts FROM FACT_SALES fs JOIN DIM_MODEL dm ON dm.model_id = fs.model_id JOIN DIM_COUNTRY dc ON dc.country_code = fs.country_code WHERE dm.model_name = 'RAV4' AND dm.powertrain = 'HEV' AND dc.country = 'Germany' AND fs.year = 2024 GROUP BY fs.year, fs.month ORDER BY fs.month"""
        
        user_prompt = f"Generate SQL for: {question}"
        
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,  # Using cost-efficient model
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.1,
                max_tokens=500
            )
            
            sql = response.choices[0].message.content.strip()
            
            # Clean up the SQL (remove markdown formatting if present)
            if sql.startswith("```sql"):
                sql = sql[6:]
            if sql.endswith("```"):
                sql = sql[:-3]
                
            return sql.strip()
            
        except Exception as e:
            logger.error(f"SQL generation failed: {e}")
            raise QueryValidationError(f"Failed to generate SQL: {e}")
            
    def _create_schema_description(self) -> str:
        """Create a human-readable schema description for the prompt."""
        
        schema_lines = []
        
        for table_name, columns in self.schema_info.items():
            col_descriptions = []
            for col in columns:
                col_desc = f"{col['name']} ({col['type']})"
                if col['pk']:
                    col_desc += " PRIMARY KEY"
                col_descriptions.append(col_desc)
            
            schema_lines.append(f"{table_name}: {', '.join(col_descriptions)}")
            
        return "\n".join(schema_lines)


class SqlTool:
    """Main SQL tool that combines generation and validation."""
    
    def __init__(self, openai_client: OpenAI, db_manager: DatabaseManager, model_name: str = "gpt-5-mini"):
        self.generator = SqlGenerator(openai_client, db_manager, model_name=model_name)
        self.validator = SqlValidator()
        self.db_manager = db_manager
        
    def execute_question(self, question: str, timeout: int = 2) -> SqlResult:
        """Execute a natural language question as SQL.
        
        Args:
            question: Natural language question
            timeout: Query timeout in seconds
            
        Returns:
            SqlResult with query results and metadata
        """
        start_time = time.time()
        
        try:
            # Generate SQL from question
            raw_sql = self.generator.generate_sql(question)
            logger.info(f"Generated SQL: {raw_sql}")
            
            # Validate SQL for safety
            validated_sql = self.validator.validate_sql(raw_sql)
            
            # Execute the query
            execution_start = time.time()
            rows = self.db_manager.execute_query(validated_sql, timeout=timeout)
            execution_time = int((time.time() - execution_start) * 1000)
            
            # Extract column names (simplified approach)
            columns = self._extract_column_names(validated_sql)
            
            result = SqlResult(
                rows=rows,
                sql_used=validated_sql,
                execution_ms=execution_time,
                row_count=len(rows),
                columns=columns
            )
            
            logger.info(f"SQL execution completed: {len(rows)} rows in {execution_time}ms")
            return result
            
        except Exception as e:
            logger.error(f"SQL execution failed for question '{question}': {e}")
            raise
            
    def _extract_column_names(self, sql: str) -> List[str]:
        """Extract column names from SELECT clause (simplified)."""
        try:
            parsed = parse_one(sql, dialect='duckdb')
            columns = []
            
            for expression in parsed.expressions:
                if isinstance(expression, exp.Alias):
                    columns.append(expression.alias)
                elif isinstance(expression, exp.Column):
                    columns.append(expression.name)
                elif hasattr(expression, 'name'):
                    columns.append(str(expression))
                else:
                    columns.append(str(expression))
                    
            return columns
            
        except Exception as e:
            logger.warning(f"Failed to extract column names: {e}")
            return []