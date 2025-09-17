"""Hybrid orchestrator for combining SQL and RAG results."""

import time
import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed

from openai import OpenAI
from .llm_utils import chat_params_for_model, is_reasoning_model

from .sql_tool import SqlTool, SqlResult
from .rag_tool import RagTool, RagResult, Citation
from .router import RoutingDecision

logger = logging.getLogger(__name__)


@dataclass
class HybridTrace:
    """Trace information for a hybrid execution flow.

    Attributes
    ----------
    sql_trace : Optional[Dict[str, Any]]
        Trace information from the SQL tool execution.
    rag_trace : Optional[Dict[str, Any]]
        Trace information from the RAG tool execution.
    composition_ms : int
        Time taken in milliseconds to compose the final answer.
    total_ms : int
        Total time taken in milliseconds for the entire hybrid execution.
    errors : List[str]
        A list of error messages encountered during execution.
    """
    sql_trace: Optional[Dict[str, Any]] = None
    rag_trace: Optional[Dict[str, Any]] = None
    composition_ms: int = 0
    total_ms: int = 0
    errors: List[str] = None
    
    def __post_init__(self):
        if self.errors is None:
            self.errors = []


@dataclass
class HybridResult:
    """Combined result from the hybrid execution of SQL and RAG tools.

    Attributes
    ----------
    answer : str
        The final composed answer.
    sql_result : Optional[SqlResult]
        The result from the SQL tool, if executed.
    rag_result : Optional[RagResult]
        The result from the RAG tool, if executed.
    citations : List[Citation]
        A list of citations supporting the answer, primarily from the RAG tool.
    trace : HybridTrace
        The trace object containing performance and execution details.
    """
    answer: str
    sql_result: Optional[SqlResult] = None
    rag_result: Optional[RagResult] = None
    citations: List[Citation] = None
    trace: HybridTrace = None
    
    def __post_init__(self):
        if self.citations is None:
            self.citations = []
        if self.trace is None:
            self.trace = HybridTrace()


class AnswerComposer:
    """Composes a final answer from SQL and RAG results using an LLM."""
    
    def __init__(self, openai_client: OpenAI, model_name: str = "gpt-5-nano", composition_model: Optional[str] = None):
        """Initialize the AnswerComposer.

        Parameters
        ----------
        openai_client : OpenAI
            An instance of the OpenAI client.
        model_name : str, optional
            The name of the OpenAI model to use for answer composition.

        """
        self.model_name = model_name
        self.composition_model = composition_model
        self.client = openai_client
        
    def compose_hybrid_answer(self, question: str, sql_result: Optional[SqlResult], 
                            rag_result: Optional[RagResult]) -> tuple[str, int]:
        """Compose a hybrid answer from optional SQL and RAG results.

        If both results are available, an LLM is used to synthesize a coherent
        answer. If only one is available, it is formatted appropriately.

        Parameters
        ----------
        question : str
            The original user question.
        sql_result : Optional[SqlResult]
            The result from the SQL tool.
        rag_result : Optional[RagResult]
            The result from the RAG tool.

        Returns
        -------
        tuple[str, int]
            A tuple containing the composed answer string and the composition
            time in milliseconds.

        """
        start_time = time.time()
        
        # Handle cases where one or both results are missing
        if not sql_result and not rag_result:
            return "No results available to answer the question.", int((time.time() - start_time) * 1000)
            
        if not sql_result:
            return rag_result.answer, int((time.time() - start_time) * 1000)
            
        if not rag_result:
            return self._format_sql_only_answer(sql_result), int((time.time() - start_time) * 1000)
            
        # Both results available - compose hybrid answer
        try:
            composed_answer = self._compose_with_llm(question, sql_result, rag_result)
            composition_time = int((time.time() - start_time) * 1000)
            return composed_answer, composition_time
            
        except Exception as e:
            logger.error(f"LLM composition failed: {e}")
            # Fallback to simple concatenation
            fallback_answer = self._compose_fallback(sql_result, rag_result)
            composition_time = int((time.time() - start_time) * 1000)
            return fallback_answer, composition_time
            
    def _format_sql_only_answer(self, sql_result: SqlResult) -> str:
        """Format a SQL-only answer, including a markdown table of results.

        Parameters
        ----------
        sql_result : SqlResult
            The result from the SQL tool.

        Returns
        -------
        str
            A formatted string containing a summary, a markdown table of the
            results (if any), and the SQL query used.

        """
        if not sql_result.rows:
            return f"No data found. Query executed successfully but returned no results.\n\n**SQL used:**\n```sql\n{sql_result.sql_used}\n```"
            
        # Create a simple table format
        answer_parts = []
        
        # Add summary
        answer_parts.append(f"Found {sql_result.row_count} result(s):")
        
        # Add table if reasonable size
        if sql_result.row_count <= 20:
            answer_parts.append("")
            answer_parts.append(self._format_table(sql_result.rows, sql_result.columns))
        else:
            answer_parts.append(f"(Showing first 10 of {sql_result.row_count} results)")
            answer_parts.append("")
            answer_parts.append(self._format_table(sql_result.rows[:10], sql_result.columns))
            
        # Add SQL query
        answer_parts.append(f"\n**SQL used:**\n```sql\n{sql_result.sql_used}\n```")
        
        return "\n".join(answer_parts)
        
    def _format_table(self, rows: List[tuple], columns: List[str]) -> str:
        """Format a list of tuples into a markdown table.

        Parameters
        ----------
        rows : List[tuple]
            The data rows to format.
        columns : List[str]
            The header columns for the table.

        Returns
        -------
        str
            A string representing the data in a markdown table format.

        """
        if not rows:
            return "No data"
            
        # Use column names or create generic ones
        headers = columns if columns else [f"col_{i+1}" for i in range(len(rows[0]))]
        
        # Create markdown table
        table_lines = []
        table_lines.append("| " + " | ".join(str(h) for h in headers) + " |")
        table_lines.append("| " + " | ".join("---" for _ in headers) + " |")
        
        for row in rows:
            table_lines.append("| " + " | ".join(str(cell) for cell in row) + " |")
            
        return "\n".join(table_lines)
        
    def _compose_with_llm(self, question: str, sql_result: SqlResult, rag_result: RagResult) -> str:
        """Use an LLM to compose a coherent answer from SQL and RAG results.

        Parameters
        ----------
        question : str
            The original user question.
        sql_result : SqlResult
            The result from the SQL tool.
        rag_result : RagResult
            The result from the RAG tool.

        Returns
        -------
        str
            The synthesized answer from the language model.

        """
        # Format SQL data for the prompt
        sql_summary = f"SQL Results ({sql_result.row_count} rows):\n"
        if sql_result.rows:
            sql_summary += self._format_table(sql_result.rows[:10], sql_result.columns)
        else:
            sql_summary += "No data found"
            
        sql_summary += f"\n\nSQL Query Used:\n{sql_result.sql_used}"
        
        # Format RAG data  
        rag_summary = f"Document Information:\n{rag_result.answer}"
        
        system_prompt = """You are an expert assistant that combines quantitative data with policy/document information to provide comprehensive answers.

Guidelines:
1. Start with the most relevant information for the user's question
2. Present numerical data clearly (use tables when helpful)
3. Include policy/warranty information with proper citations
4. Show the SQL query used in a code block
5. Maintain all citations from the document analysis
6. Be concise but complete

Format:
- Use markdown for structure
- Present data in tables when appropriate  
- Keep citations in [Document p.X] format
- Include the SQL query in a ```sql code block```"""

        user_prompt = f"""Question: {question}

Quantitative Data:
{sql_summary}

Policy/Document Information:  
{rag_summary}

Please provide a comprehensive answer that combines both the quantitative data and policy information. Include the SQL query used and maintain all document citations."""

        try:
            # Choose a faster composition model when the primary model is a reasoning model
            model_to_use = self.composition_model or ("gpt-4o-mini" if is_reasoning_model(self.model_name) else self.model_name)
            response = self.client.chat.completions.create(
                model=model_to_use,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                **chat_params_for_model(model_to_use, 1200, temperature=0.1)
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            logger.error(f"LLM composition failed: {e}")
            raise
            
    def _compose_fallback(self, sql_result: SqlResult, rag_result: RagResult) -> str:
        """Create a simple fallback composition by concatenating results.

        This method is used if the LLM-based composition fails.

        Parameters
        ----------
        sql_result : SqlResult
            The result from the SQL tool.
        rag_result : RagResult
            The result from the RAG tool.

        Returns
        -------
        str
            A simple concatenation of the formatted SQL and RAG results.

        """
        parts = []
        
        # Add data section
        if sql_result and sql_result.rows:
            parts.append("## Data Analysis")
            parts.append(f"Found {sql_result.row_count} result(s):")
            parts.append("")
            parts.append(self._format_table(sql_result.rows[:10], sql_result.columns))
            parts.append("")
            parts.append(f"**SQL used:** ```sql\n{sql_result.sql_used}\n```")
            parts.append("")
            
        # Add document information
        if rag_result and rag_result.answer:
            parts.append("## Policy & Documentation")
            parts.append(rag_result.answer)
            
        return "\n".join(parts)


class HybridOrchestrator:
    """Orchestrates parallel execution of SQL and RAG tools for hybrid queries."""
    
    def __init__(self, sql_tool: SqlTool, rag_tool: RagTool, openai_client: OpenAI, model_name: str = "gpt-5-nano"):
        """Initialize the HybridOrchestrator.

        Parameters
        ----------
        sql_tool : SqlTool
            An instance of the SQL tool.
        rag_tool : RagTool
            An instance of the RAG tool.
        openai_client : OpenAI
            An instance of the OpenAI client.
        model_name : str, optional
            The name of the OpenAI model to use for answer composition.

        """
        self.sql_tool = sql_tool
        self.rag_tool = rag_tool
        self.composer = AnswerComposer(openai_client, model_name=model_name)
        
    def execute_hybrid(self, question: str, routing_decision: RoutingDecision) -> HybridResult:
        """Execute a hybrid query by running SQL and RAG tools in parallel.

        Parameters
        ----------
        question : str
            The user's question.
        routing_decision : RoutingDecision
            The decision object from the router.

        Returns
        -------
        HybridResult
            The combined and composed result from the tool executions.

        """
        start_time = time.time()
        logger.info(f"Executing hybrid query: {question[:50]}...")
        
        # Execute tools in parallel
        sql_result, rag_result, trace = self._execute_parallel(question)
        
        # Compose final answer
        composition_start = time.time()
        composed_answer, composition_ms = self.composer.compose_hybrid_answer(question, sql_result, rag_result)
        
        # Collect citations
        citations = []
        if rag_result and rag_result.citations:
            citations.extend(rag_result.citations)
            
        # Update trace
        trace.composition_ms = composition_ms
        trace.total_ms = int((time.time() - start_time) * 1000)
        
        logger.info(f"Hybrid execution completed in {trace.total_ms}ms")
        
        return HybridResult(
            answer=composed_answer,
            sql_result=sql_result,
            rag_result=rag_result,
            citations=citations,
            trace=trace
        )
        
    def _execute_parallel(self, question: str) -> tuple[Optional[SqlResult], Optional[RagResult], HybridTrace]:
        """Execute the SQL and RAG tools in parallel using a thread pool.

        Parameters
        ----------
        question : str
            The user's question.

        Returns
        -------
        tuple[Optional[SqlResult], Optional[RagResult], HybridTrace]
            A tuple containing the SQL result, RAG result, and a trace object
            with execution details and any errors.

        """
        sql_result = None
        rag_result = None
        trace = HybridTrace()
        
        # Execute in parallel using ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=2) as executor:
            # Submit both tasks
            future_to_tool = {}
            
            try:
                sql_future = executor.submit(self._execute_sql_safe, question)
                future_to_tool[sql_future] = 'sql'
                
                rag_future = executor.submit(self._execute_rag_safe, question)  
                future_to_tool[rag_future] = 'rag'
                
                # Collect results as they complete
                for future in as_completed(future_to_tool):
                    tool_name = future_to_tool[future]
                    
                    try:
                        result = future.result()
                        
                        if tool_name == 'sql':
                            sql_result = result
                            if result:
                                trace.sql_trace = {
                                    'execution_ms': result.execution_ms,
                                    'row_count': result.row_count,
                                    'sql_used': result.sql_used[:100] + '...' if len(result.sql_used) > 100 else result.sql_used
                                }
                        elif tool_name == 'rag':
                            rag_result = result
                            if result:
                                trace.rag_trace = {
                                    'retrieval_ms': result.retrieval_ms,
                                    'synthesis_ms': result.synthesis_ms,
                                    'chunks_retrieved': result.chunks_retrieved,
                                    'confidence_score': result.confidence_score
                                }
                                
                    except Exception as e:
                        error_msg = f"{tool_name.upper()} execution failed: {e}"
                        trace.errors.append(error_msg)
                        logger.warning(error_msg)
                        
            except Exception as e:
                trace.errors.append(f"Parallel execution setup failed: {e}")
                logger.error(f"Parallel execution setup failed: {e}")
                
        return sql_result, rag_result, trace
        
    def _execute_sql_safe(self, question: str) -> Optional[SqlResult]:
        """Execute the SQL tool with error handling to prevent crashes."""
        try:
            return self.sql_tool.execute_question(question)
        except Exception as e:
            logger.warning(f"SQL execution failed: {e}")
            return None
            
    def _execute_rag_safe(self, question: str) -> Optional[RagResult]:
        """Execute the RAG tool with error handling to prevent crashes."""
        try:
            return self.rag_tool.answer_question(question)
        except Exception as e:
            logger.warning(f"RAG execution failed: {e}")
            return None