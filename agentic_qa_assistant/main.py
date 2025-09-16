"""Main CLI interface for the Agentic Q&A Assistant."""

import os
import sys
import logging
from pathlib import Path
from typing import Optional
import traceback

import click
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.markdown import Markdown
from rich.text import Text
from rich.progress import Progress, SpinnerColumn, TextColumn
from dotenv import load_dotenv
from openai import OpenAI

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
    """Main assistant orchestrating all components."""
    
    def __init__(self, data_dir: Path, docs_dir: Path, openai_api_key: str, model: str):
        """Initialize the assistant with all components.
        
        Args:
            data_dir: Directory containing CSV files
            docs_dir: Directory containing PDF/text documents
            openai_api_key: OpenAI API key
            model: The OpenAI model to use for generation
        """
        self.model = model
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
            
            # Initialize RAG pipeline
            task = progress.add_task("Setting up RAG pipeline...", total=None)
            self.rag_pipeline = RagPipeline(self.openai_client, Path("./vector_index"))
            
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
            self.sql_tool = SqlTool(self.openai_client, self.db, model_name=self.model)
            self.rag_tool = RagTool(self.openai_client, self.rag_pipeline, model_name=self.model)
            self.router = SmartRouter(self.openai_client, model_name=self.model)
            self.hybrid_orchestrator = HybridOrchestrator(self.sql_tool, self.rag_tool, self.openai_client, model_name=self.model)
            progress.update(task, description="✓ All tools ready")
            
        # Show corpus stats
        stats = self.rag_pipeline.get_corpus_stats()
        console.print(f"📊 Corpus: {stats['total_chunks']} chunks from {len(stats['sources'])} documents")
        
    def answer_question(self, question: str, show_trace: bool = False) -> None:
        """Answer a user question using the appropriate tool(s).
        
        Args:
            question: User question
            show_trace: Whether to show detailed execution trace
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
        """Execute SQL-only question."""
        
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
        
        # Show execution stats
        console.print(f"⏱️ Executed in {result.execution_ms}ms, returned {result.row_count} rows")
        
    def _execute_rag(self, question: str, show_trace: bool) -> None:
        """Execute RAG-only question."""
        
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
        """Execute hybrid question."""
        
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
        """Show routing decision trace."""
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
        """Show hybrid execution trace."""
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
@click.option('--openai-key', envvar='OPENAI_API_KEY', 
              help='OpenAI API key (or set OPENAI_API_KEY env var)')
@click.option('--model', default='gpt-5-mini', help='OpenAI model to use for generation.')
def chat(data_dir: Path, docs_dir: Path, trace: bool, openai_key: str, model: str):
    """Interactive chat mode."""
    
    if not openai_key:
        console.print("❌ [bold red]Error:[/bold red] OpenAI API key required. Set OPENAI_API_KEY environment variable or use --openai-key")
        sys.exit(1)
        
    # Initialize assistant
    try:
        console.print(f"🤖 Initializing assistant with model [bold cyan]{model}[/bold cyan]...")
        assistant = AgenticAssistant(data_dir, docs_dir, openai_key, model=model)
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
@click.option('--openai-key', envvar='OPENAI_API_KEY',
              help='OpenAI API key (or set OPENAI_API_KEY env var)')
@click.option('--model', default='gpt-5-mini', help='OpenAI model to use for generation.')
def ask(question: str, data_dir: Path, docs_dir: Path, trace: bool, openai_key: str, model: str):
    """Ask a single question."""
    
    if not openai_key:
        console.print("❌ [bold red]Error:[/bold red] OpenAI API key required. Set OPENAI_API_KEY environment variable or use --openai-key")
        sys.exit(1)
        
    # Initialize assistant
    try:
        console.print(f"🤖 Initializing assistant with model [bold cyan]{model}[/bold cyan]...")
        assistant = AgenticAssistant(data_dir, docs_dir, openai_key, model=model)
        assistant.answer_question(question, trace)
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
@click.option('--openai-key', envvar='OPENAI_API_KEY',
              help='OpenAI API key (or set OPENAI_API_KEY env var)')
@click.option('--model', default='gpt-5-mini', help='OpenAI model to use for generation.')
def demo(data_dir: Path, docs_dir: Path, openai_key: str, model: str):
    """Run the 4 demo questions from the PRD."""
    
    if not openai_key:
        console.print("❌ [bold red]Error:[/bold red] OpenAI API key required. Set OPENAI_API_KEY environment variable or use --openai-key")
        sys.exit(1)
        
    demo_questions = [
        "Monthly RAV4 HEV sales in Germany in 2024",
        "What is the standard Toyota warranty for Europe?", 
        "Where is the tire repair kit located for the UX?",
        "Compare Toyota vs Lexus SUV sales in Western Europe in 2024 and summarize key warranty differences"
    ]
    
    console.print("🎯 [bold]Running Demo Questions[/bold]")
    
    try:
        console.print(f"🤖 Initializing assistant with model [bold cyan]{model}[/bold cyan]...")
        assistant = AgenticAssistant(data_dir, docs_dir, openai_key, model=model)
        
        for i, question in enumerate(demo_questions, 1):
            console.print(f"\n{'='*60}")
            console.print(f"[bold]Demo Question {i}/4[/bold]")
            assistant.answer_question(question, show_trace=True)
            
        console.print(f"\n{'='*60}")
        console.print("✅ [bold green]All demo questions completed![/bold green]")
        
    except Exception as e:
        console.print(f"❌ [bold red]Demo failed:[/bold red] {e}")
        sys.exit(1)


def main():
    """Main entry point."""
    # Set up basic logging
    logging.basicConfig(
        level=logging.WARNING,  # Keep it quiet by default
        format='%(name)s - %(levelname)s - %(message)s'
    )
    
    # Run CLI
    cli()


if __name__ == '__main__':
    main()
