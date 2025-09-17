"""Main CLI interface for the Agentic Q&A Assistant."""
import sys
import logging
from pathlib import Path
import traceback
from typing import Dict

import click
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.markdown import Markdown
from rich.progress import Progress, SpinnerColumn, TextColumn
from dotenv import load_dotenv
from openai import OpenAI, AuthenticationError

from .database import create_database
from .rag_pipeline import RagPipeline
from .sql_tool import SqlTool
from .rag_tool import RagTool
from .router import SmartRouter, ToolChoice
from .hybrid_orchestrator import HybridOrchestrator

# Load environment variables
load_dotenv()

console = Console()
logger = logging.getLogger(__name__)


class AgenticAssistant:
    """The main assistant class that orchestrates all components.

    This class initializes the database, RAG pipeline, and all tools,
    and handles the routing and execution of user questions.
    """

    def __init__(self, data_dir: Path, docs_dir: Path, openai_api_key: str, models: Dict[str, str]):
        """Initialize the assistant and all its components.

        Parameters
        ----------
        data_dir : Path
            The directory containing the CSV data files.
        docs_dir : Path
            The directory containing the PDF/text documents for RAG.
        openai_api_key : str
            The OpenAI API key.
        models : Dict[str, str]
            A dictionary mapping tool roles to model names.
        """
        self.models = models
        self.console = console
        self.openai_client = OpenAI(api_key=openai_api_key)
        
        # Initialize components
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
            transient=True
        ) as progress:
            
            # Load database
            task = progress.add_task("Loading database...", total=None)
            self.db = create_database(data_dir)
            progress.update(task, description="✓ Database loaded")
            
            # Initialize cost tracker
            from .cost_tracker import CostTracker
            self.cost_tracker = CostTracker()
            
            # Initialize RAG pipeline
            task = progress.add_task("Setting up RAG pipeline...", total=None)
            self.rag_pipeline = RagPipeline(self.openai_client, Path("./vector_index"), cost_tracker=self.cost_tracker)
            
            # Get all document files
            doc_files = []
            for pattern in ['*.pdf', '*.txt']:
                doc_files.extend(list(docs_dir.glob(pattern)))
            
            if doc_files:
                progress.update(task, description="Processing documents...")
                self.rag_pipeline.ingest_documents(doc_files)
            progress.update(task, description="✓ RAG pipeline ready")
            
            # Initialize tools
            task = progress.add_task("Initializing tools...", total=None)
            self.sql_tool = SqlTool(self.openai_client, self.db, model_name=self.models.get("sql", "gpt-5-nano"), cost_tracker=self.cost_tracker)
            # If a fast synthesis model is provided, pass it; otherwise default to gpt-4o-mini when primary is reasoning
            synthesis_model = self.models.get("rag_synthesis")
            self.rag_tool = RagTool(
                self.openai_client,
                self.rag_pipeline,
                model_name=self.models.get("rag", "gpt-5-nano"),
                cost_tracker=self.cost_tracker,
                synthesis_model=synthesis_model
            )
            self.router = SmartRouter(self.openai_client, model_name=self.models.get("router", "gpt-5-nano"))
            self.hybrid_orchestrator = HybridOrchestrator(
                self.sql_tool,
                self.rag_tool,
                self.openai_client,
                model_name=self.models.get("composition", "gpt-5-nano")
            )
            progress.update(task, description="✓ All tools ready")
            
        # Show corpus stats
        stats = self.rag_pipeline.get_corpus_stats()
        console.print(f"📊 Corpus: {stats['total_chunks']} chunks from {len(stats['sources'])} documents")
        
    def answer_question(self, question: str, show_trace: bool = False) -> None:
        """Route and answer a user's question, displaying the result.

        Parameters
        ----------
        question : str
            The user's question.
        show_trace : bool, optional
            Whether to display detailed execution trace information.
        """
        console.print(f"\n❓ [bold]Question:[/bold] {question}")
        
        try:
            # Route the question
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console,
                transient=True
            ) as progress:
                task = progress.add_task("Routing question...", total=None)
                routing_decision = self.router.route(question)
                progress.update(task, description=f"✓ Routed to {routing_decision.decision.value}")
                
            # Execute based on routing decision
            if routing_decision.decision == ToolChoice.SQL:
                self._execute_sql(question, show_trace)
            elif routing_decision.decision == ToolChoice.RAG:
                self._execute_rag(question, show_trace)
            else:  # HYBRID
                self._execute_hybrid(question, routing_decision, show_trace)
                
            # Show routing trace if requested
            if show_trace:
                self._show_routing_trace(routing_decision)
                
        except Exception as e:
            console.print(f"❌ [bold red]Error:[/bold red] {e}")
            if show_trace:
                console.print(f"[dim]{traceback.format_exc()}[/dim]")
                
    def _execute_sql(self, question: str, show_trace: bool) -> None:
        """Execute a question routed to the SQL tool and display the results.

        Parameters
        ----------
        question : str
            The user's question.
        show_trace : bool
            Whether to show detailed trace information.
        """
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
            transient=True
        ) as progress:
            task = progress.add_task("Generating and executing SQL...", total=None)
            
            result = self.sql_tool.execute_question(question)
            progress.update(task, description="✓ SQL executed")
            
        # Display results
        console.print("\n📊 [bold]SQL Results:[/bold]")
        
        if result.rows:
            # Create table
            table = Table(show_header=True, header_style="bold magenta")
            
            # Add columns
            for col in result.columns:
                table.add_column(str(col))
                
            # Add rows (limit to first 20 for display)
            for row in result.rows[:20]:
                table.add_row(*[str(cell) for cell in row])
                
            console.print(table)
            
            if len(result.rows) > 20:
                console.print(f"[dim]... and {len(result.rows) - 20} more rows[/dim]")
        else:
            console.print("[yellow]No data found[/yellow]")
            
        # Show SQL used
        console.print("\n💾 [bold]SQL Query:[/bold]")
        console.print(Panel(result.sql_used, expand=False, border_style="blue"))

        # Token usage summary if available
        if hasattr(self, 'cost_tracker') and self.cost_tracker:
            console.print("\n🔢 [bold]Token Usage:[/bold]")
            console.print(Panel(self.cost_tracker.summary_text(), border_style="yellow"))
        
        # Show execution stats
        console.print(f"⏱️ Executed in {result.execution_ms}ms, returned {result.row_count} rows")
        
    def _execute_rag(self, question: str, show_trace: bool) -> None:
        """Execute a question routed to the RAG tool and display the results.

        Parameters
        ----------
        question : str
            The user's question.
        show_trace : bool
            Whether to show detailed trace information.
        """
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
            transient=True
        ) as progress:
            task = progress.add_task("Searching documents and generating answer...", total=None)
            
            result = self.rag_tool.answer_question(question)
            progress.update(task, description="✓ Answer generated")
            
        # Display answer
        console.print("\n📚 [bold]Answer:[/bold]")
        console.print(Panel(Markdown(result.answer), border_style="green"))
        
        # Show citations
        if result.citations:
            console.print("\n📖 [bold]Citations:[/bold]")
            for i, citation in enumerate(result.citations, 1):
                console.print(f"[{i}] {citation.source} p.{citation.page} ({citation.doc_type})")
                console.print(f"    [dim]{citation.text_snippet}[/dim]")
                
        # Show retrieval stats
        console.print(f"⏱️ Retrieved {result.chunks_retrieved} chunks in {result.retrieval_ms}ms, synthesized in {result.synthesis_ms}ms")
        
    def _execute_hybrid(self, question: str, routing_decision, show_trace: bool) -> None:
        """Execute a question routed to the Hybrid orchestrator and display the results.

        Parameters
        ----------
        question : str
            The user's question.
        routing_decision : RoutingDecision
            The decision object from the router.
        show_trace : bool
            Whether to show detailed trace information.
        """
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
            transient=True
        ) as progress:
            task = progress.add_task("Executing SQL and RAG in parallel...", total=None)
            
            result = self.hybrid_orchestrator.execute_hybrid(question, routing_decision)
            progress.update(task, description="✓ Hybrid execution complete")
            
        # Display answer
        console.print("\n🔀 [bold]Hybrid Answer:[/bold]")
        console.print(Panel(Markdown(result.answer), border_style="cyan"))
        
        # Show token usage if available
        if hasattr(self, 'cost_tracker') and self.cost_tracker:
            console.print("\n🔢 [bold]Token Usage:[/bold]")
            console.print(Panel(self.cost_tracker.summary_text(), border_style="yellow"))
        
        # Show citations if available
        if result.citations:
            console.print("\n📖 [bold]Citations:[/bold]")
            for i, citation in enumerate(result.citations, 1):
                console.print(f"[{i}] {citation.source} p.{citation.page} ({citation.doc_type})")
                
        # Show execution details if trace enabled
        if show_trace:
            self._show_hybrid_trace(result.trace)
            
        console.print(f"⏱️ Total execution time: {result.trace.total_ms}ms")
        
    def _show_routing_trace(self, decision) -> None:
        """Display the detailed trace of a routing decision.

        Parameters
        ----------
        decision : RoutingDecision
            The routing decision object to display.

        """
        console.print("\n🧭 [bold]Routing Decision:[/bold]")
        console.print(f"Decision: {decision.decision.value}")
        console.print(f"Confidence: {decision.confidence:.2f}")
        console.print(f"Latency: {decision.latency_ms}ms")
        
        if decision.matched_keywords:
            console.print("Matched keywords:")
            for category, keywords in decision.matched_keywords.items():
                console.print(f"  {category}: {', '.join(keywords)}")
                
        if decision.reasons:
            console.print("Reasons:")
            for reason in decision.reasons:
                console.print(f"  • {reason}")
                
    def _show_hybrid_trace(self, trace) -> None:
        """Display the detailed trace of a hybrid execution.

        Parameters
        ----------
        trace : HybridTrace
            The hybrid trace object to display.

        """
        console.print("\n🔍 [bold]Execution Trace:[/bold]")
        
        if trace.sql_trace:
            console.print(f"SQL: {trace.sql_trace['execution_ms']}ms, {trace.sql_trace['row_count']} rows")
            
        if trace.rag_trace:
            console.print(f"RAG: {trace.rag_trace['retrieval_ms']}ms retrieval + {trace.rag_trace['synthesis_ms']}ms synthesis")
            console.print(f"     {trace.rag_trace['chunks_retrieved']} chunks, confidence: {trace.rag_trace.get('confidence_score', 'N/A')}")
            
        console.print(f"Composition: {trace.composition_ms}ms")
        
        if trace.errors:
            console.print("[bold red]Errors:[/bold red]")
            for error in trace.errors:
                console.print(f"  • {error}")


@click.group()
def cli():
    """Agentic Q&A Assistant - SQL + RAG + Owner's Manuals"""
    pass


@cli.command()
@click.option('--data-dir', type=click.Path(exists=True, path_type=Path), 
              default='./data', help='Directory containing CSV files')
@click.option('--docs-dir', type=click.Path(exists=True, path_type=Path),
              default='./docs', help='Directory containing PDF/text documents')
@click.option('--trace/--no-trace', default=False, 
              help='Show detailed execution trace')
@click.option('--integrity-report', type=click.Path(path_type=Path), default=None, help='Write a data integrity report to this file before starting chat')
@click.option('--token-report', type=click.Path(path_type=Path), default=None, help='Write token usage JSON to this file when exiting chat (Ctrl+C)')
@click.option('--openai-key', envvar='OPENAI_API_KEY', 
              help='OpenAI API key (or set OPENAI_API_KEY env var)')
@click.option('--model-sql', default='gpt-5-nano', help='Model for SQL generation.')
@click.option('--model-rag', default='gpt-4o-mini', help='Model for RAG synthesis.')
@click.option('--model-composition', default='gpt-4o-mini', help='Model for hybrid answer composition.')
def chat(data_dir: Path, docs_dir: Path, trace: bool, integrity_report: Path, token_report: Path, openai_key: str, model_sql: str, model_rag: str, model_composition: str):
    """Interactive chat mode."""
    
    if not openai_key:
        console.print("❌ [bold red]Error:[/bold red] OpenAI API key required. Set OPENAI_API_KEY environment variable or use --openai-key")
        sys.exit(1)

    model_config = {
        "sql": model_sql,
        "rag": model_rag,
        "composition": model_composition,
        "router": "gpt-5-nano"  # Keep router fixed to a fast model
    }
    # Initialize assistant
    try:
        console.print("🤖 Initializing assistant with models...")
        console.print(f"   - SQL: [cyan]{model_sql}[/cyan]")
        console.print(f"   - RAG: [cyan]{model_rag}[/cyan]")
        console.print(f"   - Composition: [cyan]{model_composition}[/cyan]")
        assistant = AgenticAssistant(
            data_dir, docs_dir, openai_key, models=model_config
        )
        # Optional integrity report
        if integrity_report:
            try:
                report_text = assistant.db.get_integrity_report_text()
                integrity_report.write_text(report_text, encoding='utf-8')
                console.print(f"🧪 Wrote integrity report to {integrity_report}")
            except Exception as e:
                console.print(f"[yellow]Warning:[/yellow] Failed to write integrity report: {e}")
    except AuthenticationError:
        console.print("❌ [bold red]Authentication Error:[/bold red] Invalid OpenAI API key or insufficient permissions.")
        console.print("   Please check your OPENAI_API_KEY and ensure it has the correct scopes for the model.")
        sys.exit(1)
    except Exception as e:
        console.print(f"❌ [bold red]Initialization failed:[/bold red] {e}")
        sys.exit(1)
        
    console.print("\n🤖 [bold green]Agentic Q&A Assistant ready![/bold green]")
    console.print("Type your questions or 'quit' to exit. Use '/trace' to toggle trace mode.")
    
    show_trace = trace
    
    while True:
        try:
            question = console.input("\n❓ [bold]Ask me anything:[/bold] ").strip()
            
            if not question:
                continue
                
            if question.lower() in ['quit', 'exit', 'q']:
                break
                
            if question == '/trace':
                show_trace = not show_trace
                console.print(f"🔍 Trace mode {'enabled' if show_trace else 'disabled'}")
                continue
                
            assistant.answer_question(question, show_trace)
            
        except KeyboardInterrupt:
            console.print("\n👋 Goodbye!")
            # Write token report on exit if requested
            if token_report:
                try:
                    token_report.write_text(assistant.cost_tracker.summary_json(), encoding='utf-8')
                    console.print(f"🧮 Wrote token usage JSON to {token_report}")
                except Exception as e:
                    console.print(f"[yellow]Warning:[/yellow] Failed to write token report: {e}")
            break
        except Exception as e:
            console.print(f"❌ [bold red]Error:[/bold red] {e}")


@cli.command()
@click.argument('question')
@click.option('--data-dir', type=click.Path(exists=True, path_type=Path), 
              default='./data', help='Directory containing CSV files')
@click.option('--docs-dir', type=click.Path(exists=True, path_type=Path),
              default='./docs', help='Directory containing PDF/text documents')  
@click.option('--trace/--no-trace', default=False,
              help='Show detailed execution trace')
@click.option('--explain', is_flag=True, default=False, help='Print SQL EXPLAIN plan before execution')
@click.option('--explain-analyze', is_flag=True, default=False, help='Print SQL EXPLAIN ANALYZE plan before execution')
@click.option('--integrity-report', type=click.Path(path_type=Path), default=None, help='Write a data integrity report to this file before execution')
@click.option('--token-report', type=click.Path(path_type=Path), default=None, help='Write token usage JSON to this file after execution')
@click.option('--openai-key', envvar='OPENAI_API_KEY',
              help='OpenAI API key (or set OPENAI_API_KEY env var)')
@click.option('--model-sql', default='gpt-5-nano', help='Model for SQL generation.')
@click.option('--model-rag', default='gpt-4o-mini', help='Model for RAG synthesis.')
@click.option('--model-composition', default='gpt-4o-mini', help='Model for hybrid answer composition.')
def ask(question: str, data_dir: Path, docs_dir: Path, trace: bool, explain: bool, explain_analyze: bool, integrity_report: Path, token_report: Path, openai_key: str, model_sql: str, model_rag: str, model_composition: str):
    """Ask a single question."""
    
    if not openai_key:
        console.print("❌ [bold red]Error:[/bold red] OpenAI API key required. Set OPENAI_API_KEY environment variable or use --openai-key")
        sys.exit(1)

    model_config = {
        "sql": model_sql,
        "rag": model_rag,
        "composition": model_composition,
        "router": "gpt-5-nano"  # Keep router fixed to a fast model
    }
    # Initialize assistant
    try:
        console.print("🤖 Initializing assistant with models...")
        console.print(f"   - SQL: [cyan]{model_sql}[/cyan]")
        console.print(f"   - RAG: [cyan]{model_rag}[/cyan]")
        console.print(f"   - Composition: [cyan]{model_composition}[/cyan]")
        assistant = AgenticAssistant(data_dir, docs_dir, openai_key, models=model_config)
        # Optional integrity report
        if integrity_report:
            try:
                report_text = assistant.db.get_integrity_report_text()
                integrity_report.write_text(report_text, encoding='utf-8')
                console.print(f"🧪 Wrote integrity report to {integrity_report}")
            except Exception as e:
                console.print(f"[yellow]Warning:[/yellow] Failed to write integrity report: {e}")
        # Temporarily override SQL execution to include explain flags via orchestrator path
        # We use the hybrid orchestrator path to keep behavior consistent
        if explain or explain_analyze:
            # Route question to SQL-only to include explain plan in SQL tool
            result = assistant.sql_tool.execute_question(question, explain=explain, explain_analyze=explain_analyze)
            # Display results like _execute_sql
            console.print("\n📊 [bold]SQL Results:[/bold]")
            if result.rows:
                table = Table(show_header=True, header_style="bold magenta")
                for col in result.columns:
                    table.add_column(str(col))
                for row in result.rows[:20]:
                    table.add_row(*[str(cell) for cell in row])
                console.print(table)
                if len(result.rows) > 20:
                    console.print(f"[dim]... and {len(result.rows) - 20} more rows[/dim]")
            else:
                console.print("[yellow]No data found[/yellow]")
            console.print("\n💾 [bold]SQL Query:[/bold]")
            console.print(Panel(result.sql_used, expand=False, border_style="blue"))
            if result.explain_plan:
                console.print("\n🧩 [bold]Explain Plan:[/bold]")
                console.print(Panel(result.explain_plan, expand=False, border_style="magenta"))
            console.print(f"⏱️ Executed in {result.execution_ms}ms, returned {result.row_count} rows")
            if token_report:
                try:
                    token_report.write_text(assistant.cost_tracker.summary_json(), encoding='utf-8')
                    console.print(f"🧮 Wrote token usage JSON to {token_report}")
                except Exception as e:
                    console.print(f"[yellow]Warning:[/yellow] Failed to write token report: {e}")
        else:
            assistant.answer_question(question, trace)
            if token_report:
                try:
                    token_report.write_text(assistant.cost_tracker.summary_json(), encoding='utf-8')
                    console.print(f"🧮 Wrote token usage JSON to {token_report}")
                except Exception as e:
                    console.print(f"[yellow]Warning:[/yellow] Failed to write token report: {e}")
    except AuthenticationError:
        console.print("❌ [bold red]Authentication Error:[/bold red] Invalid OpenAI API key or insufficient permissions.")
        console.print("   Please check your OPENAI_API_KEY and ensure it has the correct scopes for the model.")
        sys.exit(1)
    except Exception as e:
        console.print(f"❌ [bold red]Error:[/bold red] {e}")
        if trace:
            console.print(f"[dim]{traceback.format_exc()}[/dim]")
        sys.exit(1)


@cli.command()
@click.option('--data-dir', type=click.Path(exists=True, path_type=Path), 
              default='./data', help='Directory containing CSV files')
@click.option('--docs-dir', type=click.Path(exists=True, path_type=Path),
              default='./docs', help='Directory containing PDF/text documents')
@click.option('--explain', is_flag=True, default=False, help='Print SQL EXPLAIN plan before SQL execution')
@click.option('--explain-analyze', is_flag=True, default=False, help='Print SQL EXPLAIN ANALYZE plan before SQL execution')
@click.option('--integrity-report', type=click.Path(path_type=Path), default=None, help='Write a data integrity report to this file before running')
@click.option('--token-report', type=click.Path(path_type=Path), default=None, help='Write token usage JSON to this file after running')
@click.option('--openai-key', envvar='OPENAI_API_KEY',
              help='OpenAI API key (or set OPENAI_API_KEY env var)')
@click.option('--model-sql', default='gpt-5-nano', help='Model for SQL generation.')
@click.option('--model-rag', default='gpt-4o-mini', help='Model for RAG synthesis.')
@click.option('--model-composition', default='gpt-4o-mini', help='Model for hybrid answer composition.')
def demo(data_dir: Path, docs_dir: Path, explain: bool, explain_analyze: bool, integrity_report: Path, token_report: Path, openai_key: str, model_sql: str, model_rag: str, model_composition: str):
    """Run the 4 demo questions from the PRD."""
    
    if not openai_key:
        console.print("❌ [bold red]Error:[/bold red] OpenAI API key required. Set OPENAI_API_KEY environment variable or use --openai-key")
        sys.exit(1)

    model_config = {
        "sql": model_sql,
        "rag": model_rag,
        "composition": model_composition,
        "router": "gpt-5-nano"  # Keep router fixed to a fast model
    }
    demo_questions = [
        "Monthly RAV4 HEV sales in Germany in 2024",
        "What is the standard Toyota warranty for Europe?", 
        "Where is the tire repair kit located for the UX?",
        "Compare Toyota vs Lexus SUV sales in Western Europe in 2024 and summarize key warranty differences"
    ]
    
    console.print("🎯 [bold]Running Demo Questions[/bold]")
    
    try:
        console.print("🤖 Initializing assistant with models...")
        console.print(f"   - SQL: [cyan]{model_sql}[/cyan]")
        console.print(f"   - RAG: [cyan]{model_rag}[/cyan]")
        console.print(f"   - Composition: [cyan]{model_composition}[/cyan]")
        assistant = AgenticAssistant(data_dir, docs_dir, openai_key, models=model_config)

        # Optional: write data integrity report
        if integrity_report:
            try:
                report_text = assistant.db.get_integrity_report_text()
                integrity_report.write_text(report_text, encoding='utf-8')
                console.print(f"🧪 Wrote integrity report to {integrity_report}")
            except Exception as e:
                console.print(f"[yellow]Warning:[/yellow] Failed to write integrity report: {e}")

        for i, question in enumerate(demo_questions, 1):
            console.print(f"\n{'='*60}")
            console.print(f"[bold]Demo Question {i}/4[/bold]")
            if explain or explain_analyze:
                # When explain flags are set, and router picks SQL, include plan in output
                routing_decision = assistant.router.route(question)
                if routing_decision.decision == ToolChoice.SQL:
                    result = assistant.sql_tool.execute_question(question, explain=explain, explain_analyze=explain_analyze)
                    console.print("\n📊 [bold]SQL Results:[/bold]")
                    if result.rows:
                        table = Table(show_header=True, header_style="bold magenta")
                        for col in result.columns:
                            table.add_column(str(col))
                        for row in result.rows[:20]:
                            table.add_row(*[str(cell) for cell in row])
                        console.print(table)
                        if len(result.rows) > 20:
                            console.print(f"[dim]... and {len(result.rows) - 20} more rows[/dim]")
                    else:
                        console.print("[yellow]No data found[/yellow]")
                    console.print("\n💾 [bold]SQL Query:[/bold]")
                    console.print(Panel(result.sql_used, expand=False, border_style="blue"))
                    if result.explain_plan:
                        console.print("\n🧩 [bold]Explain Plan:[/bold]")
                        console.print(Panel(result.explain_plan, expand=False, border_style="magenta"))
                    console.print(f"⏱️ Executed in {result.execution_ms}ms, returned {result.row_count} rows")
                else:
                    assistant.answer_question(question, show_trace=True)
            else:
                assistant.answer_question(question, show_trace=True)

        console.print(f"\n{'='*60}")
        # Optional token report
        if token_report:
            try:
                token_report.write_text(assistant.cost_tracker.summary_json(), encoding='utf-8')
                console.print(f"🧮 Wrote token usage JSON to {token_report}")
            except Exception as e:
                console.print(f"[yellow]Warning:[/yellow] Failed to write token report: {e}")
        console.print("✅ [bold green]All demo questions completed![/bold green]")

    except AuthenticationError:
        console.print("❌ [bold red]Authentication Error:[/bold red] Invalid OpenAI API key or insufficient permissions.")
        console.print("   Please check your OPENAI_API_KEY and ensure it has the correct scopes for the model.")
        sys.exit(1)
    except Exception as e:
        console.print(f"❌ [bold red]Demo failed:[/bold red] {e}")
        sys.exit(1)


def main():
    """Main entry point for the CLI application."""
    # Set up basic logging
    logging.basicConfig(
        level=logging.WARNING,  # Keep it quiet by default
        format='%(name)s - %(levelname)s - %(message)s'
    )
    
    # Run CLI
    cli()


if __name__ == '__main__':
    main()
