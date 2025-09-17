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
from .llm_utils import chat_params_for_model, is_reasoning_model

from .database import DatabaseManager

logger = logging.getLogger(__name__)


class QueryValidationError(Exception):
    """Raised when SQL query validation fails."""
    pass


@dataclass
class SqlResult:
    """Represents the result of a successful SQL tool execution.

    Attributes
    ----------
    rows : List[Tuple]
        The result rows from the query execution.
    sql_used : str
        The exact SQL query that was executed.
    execution_ms : int
        The query execution time in milliseconds.
    row_count : int
        The number of rows returned by the query.
    columns : List[str]
        The names of the columns in the result set.
    explain_plan : Optional[str]
        The EXPLAIN or EXPLAIN ANALYZE plan, if requested.
    """
    rows: List[Tuple]
    sql_used: str
    execution_ms: int
    row_count: int
    columns: List[str]
    explain_plan: Optional[str] = None


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
        """Validate a SQL query for safety and compliance.

        This method uses `sqlglot` to parse the SQL and check for disallowed
        statements, tables, columns, and functions. It also ensures a LIMIT
        clause is present.

        Parameters
        ----------
        sql : str
            The SQL query string to validate.

        Returns
        -------
        str
            The cleaned and validated SQL query, with a LIMIT clause added if needed.

        Raises
        ------
        QueryValidationError
            If the query violates any safety rules.
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
        """Validate that the parsed query only references allowed tables and columns.

        Parameters
        ----------
        parsed : sqlglot.exp.Expression
            The parsed SQL expression tree.
        """
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
        """Check for the use of denied functions and statements in the parsed query.

        Parameters
        ----------
        parsed : sqlglot.exp.Expression
            The parsed SQL expression tree.
        """
        # Check for denied functions
        for func in parsed.find_all(exp.Anonymous):
            func_name = func.this.upper() if hasattr(func, 'this') else ''
            if func_name in self.DENIED_FUNCTIONS:
                raise QueryValidationError(f"Function '{func_name}' not allowed")
                
        # Additional check for subqueries with CTEs or complex operations
        for cte in parsed.find_all(exp.CTE):
            raise QueryValidationError("Common Table Expressions (CTEs) not allowed")
            
    def _ensure_limit_clause(self, sql: str, parsed: exp.Select) -> str:
        """Ensure the query has a reasonable LIMIT clause to prevent excessive results.

        If no LIMIT is present, one is added. If an existing LIMIT is too high,
        an error is raised.

        Parameters
        ----------
        sql : str
            The original SQL string.
        parsed : sqlglot.exp.Select
            The parsed SELECT expression.

        Returns
        -------
        str
            The SQL string, potentially with an added or validated LIMIT clause.
        """
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
    
    def __init__(self, openai_client: OpenAI, db_manager: DatabaseManager, model_name: str = "gpt-5-nano", cost_tracker=None, fallback_model: str = "gpt-4o-mini"):
        """Initialize the SqlGenerator.

        Parameters
        ----------
        openai_client : OpenAI
            An initialized OpenAI client.
        db_manager : DatabaseManager
            The database manager instance to get schema information from.
        model_name : str, optional
            The name of the OpenAI model to use for SQL generation.
        cost_tracker : object, optional
            An instance of CostTracker to record token usage.
        """
        self.client = openai_client
        self.db_manager = db_manager
        self.schema_info = db_manager.get_schema_info()
        self.model_name = model_name
        self.cost_tracker = cost_tracker
        self.fallback_model = fallback_model
        
    def generate_sql(self, question: str) -> str:
        """Generate a SQL query from a natural language question using an LLM.

        Parameters
        ----------
        question : str
            The natural language question.

        Returns
        -------
        str
            The generated SQL query string.
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
                **chat_params_for_model(self.model_name, 500, temperature=0.1)
            )
            
            # Cost tracking
            try:
                if self.cost_tracker and getattr(response, 'usage', None):
                    u = response.usage
                    self.cost_tracker.add_chat_usage_detailed(
                        self.model_name,
                        prompt_tokens=getattr(u, 'prompt_tokens', 0),
                        completion_tokens=getattr(u, 'completion_tokens', 0),
                        total_tokens=getattr(u, 'total_tokens', 0),
                        cached_prompt_tokens=getattr(u, 'prompt_tokens_details', {}).get('cached_tokens', 0) if hasattr(u, 'prompt_tokens_details') else 0,
                        reasoning_tokens=getattr(u, 'completion_tokens_details', {}).get('reasoning_tokens', 0) if hasattr(u, 'completion_tokens_details') else 0,
                    )
            except Exception:
                pass
            
            sql = response.choices[0].message.content.strip()
            
            # Attempt robust extraction
            sql = self._extract_select_from_text(sql)
            # Guardrail: retry once with a faster non-reasoning model if not a SELECT
            if not sql or "select" not in sql.lower():
                logger.info("LLM did not produce a SELECT; retrying with fallback model %s", self.fallback_model)
                response2 = self.client.chat.completions.create(
                    model=self.fallback_model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    **chat_params_for_model(self.fallback_model, 300, temperature=0.1)
                )
                sql2 = response2.choices[0].message.content.strip()
                sql2 = self._extract_select_from_text(sql2)
                if not sql2 or "select" not in sql2.lower():
                    logger.info("Fallback model also failed to produce a SELECT; skipping SQL path.")
                    return ""
                return sql2
                
            return sql
            
        except Exception as e:
            logger.error(f"SQL generation failed: {e}")
            raise QueryValidationError(f"Failed to generate SQL: {e}")
            
    def _create_schema_description(self) -> str:
        """Create a human-readable schema description for the LLM prompt.

        Returns
        -------
        str
            A formatted string describing the database schema.
        """
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

    def _extract_select_from_text(self, text: str) -> str:
        """Best-effort extraction of the first SELECT statement from LLM text.

        - Strips code fences and leading labels (e.g., 'SQL:').
        - Returns empty string if no 'select' found.
        """
        if not text:
            return ""
        t = text.strip()
        # Remove triple backtick fences if present
        if t.startswith("```sql"):
            t = t[6:]
        if t.startswith("```"):
            t = t[3:]
        if t.endswith("```"):
            t = t[:-3]
        # Remove leading 'SQL:' label if present
        if t.lower().startswith("sql:"):
            t = t[4:].strip()
        # Find first occurrence of 'select'
        low = t.lower()
        idx = low.find("select")
        if idx == -1:
            return ""
        candidate = t[idx:].strip()
        # Cut off any trailing code fence accidentally kept
        fence_idx = candidate.find("```")
        if fence_idx != -1:
            candidate = candidate[:fence_idx].strip()
        return candidate


class SqlTool:
    """Main SQL tool that combines generation and validation."""
    
    def __init__(self, openai_client: OpenAI, db_manager: DatabaseManager, model_name: str = "gpt-5-nano", cost_tracker=None):
        """Initialize the SqlTool.

        Parameters
        ----------
        openai_client : OpenAI
            An initialized OpenAI client.
        db_manager : DatabaseManager
            The database manager instance for query execution.
        model_name : str, optional
            The name of the OpenAI model to use for SQL generation.
        cost_tracker : object, optional
            An instance of CostTracker to record token usage.
        """
        self.generator = SqlGenerator(openai_client, db_manager, model_name=model_name, cost_tracker=cost_tracker)
        self.validator = SqlValidator()
        self.db_manager = db_manager
        self.cost_tracker = cost_tracker
        
    def execute_question(self, question: str, timeout: int = 2, explain: bool = False, explain_analyze: bool = False) -> SqlResult:
        """Generate, validate, and execute a SQL query from a natural language question.

        Parameters
        ----------
        question : str
            The natural language question.
        timeout : int, optional
            Query timeout in seconds (currently ignored by DuckDB driver).
        explain : bool, optional
            If True, include the EXPLAIN plan in the result.
        explain_analyze : bool, optional
            If True, include the EXPLAIN ANALYZE plan in the result.

        Returns
        -------
        SqlResult
            An object containing the query results and metadata.
        """
        start_time = time.time()
        
        try:
            # Generate SQL from question
            raw_sql = self.generator.generate_sql(question)
            if not raw_sql.strip():
                logger.info("No SQL generated for this question; skipping SQL execution.")
                # Return an empty result to signal 'no SQL path' without raising
                result = SqlResult(
                    rows=[],
                    sql_used="",
                    execution_ms=0,
                    row_count=0,
                    columns=[],
                    explain_plan=None,
                )
                return result
            logger.info(f"Generated SQL: {raw_sql}")
            
            # Validate SQL for safety
            validated_sql = self.validator.validate_sql(raw_sql)
            
            # Optionally get explain plan
            plan_text = None
            if explain or explain_analyze:
                plan_text = self.db_manager.get_explain_plan(validated_sql, analyze=explain_analyze)
                logger.info("Explain plan obtained")
            
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
                columns=columns,
                explain_plan=plan_text,
            )
            
            logger.info(f"SQL execution completed: {len(rows)} rows in {execution_time}ms")
            return result
            
        except Exception as e:
            logger.error(f"SQL execution failed for question '{question}': {e}")
            raise
            
    def _extract_column_names(self, sql: str) -> List[str]:
        """Extract column names from a SELECT clause using `sqlglot`.

        This is a simplified approach that handles aliases and direct column names.

        Parameters
        ----------
        sql : str
            The SQL query string.

        Returns
        -------
        List[str]
            A list of extracted column names or their string representations.
        """
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