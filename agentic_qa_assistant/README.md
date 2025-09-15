# Agentic Q&A Assistant

**A production-ready AI assistant combining SQL analytics, RAG document search, and owner's manual knowledge**

## Overview

This project implements the Agentic Q&A Assistant as specified in the PRD, providing intelligent routing between SQL queries for sales data analysis and RAG-based retrieval for warranty policies, contracts, and owner's manuals.

## Features

### 🎯 **Intelligent Routing**
- **Rule-based routing** with keyword matching for fast, deterministic decisions
- **LLM fallback** for ambiguous cases using GPT-4o-mini
- **Hybrid mode** for questions requiring both SQL and RAG

### 📊 **SQL Tool**
- **Safe SQL generation** with AST validation using sqlglot
- **Read-only queries** with table/column allowlists
- **Parameterization** and timeout controls
- **Automatic LIMIT enforcement** (max 10k rows)

### 📚 **RAG Pipeline**
- **PDF and text processing** with semantic chunking (400-800 tokens, 100 overlap)
- **FAISS vector index** with text-embedding-3-small
- **Metadata-rich chunks** (source, page, doc_type, brand, year)
- **Citation-aware answers** with [document p.X] format

### 🔄 **Hybrid Orchestration**
- **Parallel execution** of SQL and RAG tools
- **Intelligent composition** of numeric data with policy information
- **Comprehensive tracing** with execution metrics

### 🖥️ **Rich CLI Interface**
- **Interactive chat mode** with trace toggle
- **Single question mode** for scripting
- **Demo mode** with 4 PRD questions
- **Beautiful terminal output** with Rich formatting

## Quick Start

### Prerequisites

- Python 3.12+
- OpenAI API key
- uv package manager

### Installation

```bash
# Clone and navigate to project
cd agentic_qa_assistant

# Install dependencies with uv
uv sync

# Set up environment
echo "OPENAI_API_KEY=your_key_here" > .env
```

### Run Demo Questions

```bash
# Run all 4 demo questions from the PRD
uv run agentic-qa demo

# Or use the module directly
uv run python -m agentic_qa_assistant.main demo
```

### Interactive Chat

```bash
# Start interactive mode
uv run agentic-qa chat

# With trace mode enabled
uv run agentic-qa chat --trace
```

### Single Questions

```bash
# Ask a single question
uv run agentic-qa ask "Monthly RAV4 HEV sales in Germany in 2024"

# With detailed trace
uv run agentic-qa ask --trace "What is the Toyota warranty coverage?"
```

## Demo Questions

The system handles these PRD demo questions:

1. **SQL**: "Monthly RAV4 HEV sales in Germany in 2024"
2. **RAG**: "What is the standard Toyota warranty for Europe?"
3. **RAG**: "Where is the tire repair kit located for the UX?"
4. **Hybrid**: "Compare Toyota vs Lexus SUV sales in Western Europe in 2024 and summarize key warranty differences"

## Architecture

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│    CLI      │────│  Assistant  │────│   Router    │
└─────────────┘    └─────────────┘    └─────────────┘
                           │                   │
                           ▼                   ▼
                   ┌─────────────┐    ┌─────────────┐
                   │ SQL Tool    │    │  RAG Tool   │
                   └─────────────┘    └─────────────┘
                           │                   │
                           ▼                   ▼
                   ┌─────────────┐    ┌─────────────┐
                   │   DuckDB    │    │ FAISS Index │
                   │ (CSV Data)  │    │ (Documents) │
                   └─────────────┘    └─────────────┘
```

### Core Components

- **Router**: Intelligent tool selection (SQL/RAG/Hybrid)
- **SQL Tool**: Safe query generation and execution
- **RAG Tool**: Document retrieval with citations
- **Hybrid Orchestrator**: Parallel execution and composition
- **Database**: DuckDB with sales/model data
- **Vector Index**: FAISS with document chunks

## Data Sources

### Structured Data (SQL)
- **DIM_MODEL**: Vehicle models with brand/segment/powertrain
- **DIM_COUNTRY**: Countries with regions
- **DIM_ORDERTYPE**: Order types (Private/Fleet/Demo)
- **FACT_SALES**: Monthly sales contracts by model/country
- **FACT_SALES_ORDERTYPE**: Sales by order type

### Documents (RAG)
- **Contracts**: Toyota & Lexus 2023 contracts
- **Warranty**: Policy appendix with coverage details
- **Manuals**: Toyota UX 2023 owner's manual sample

## Known Facts (for validation)

- Toyota EU warranty: **5 years / 100,000 km**
- Lexus EU warranty: **4 years / 100,000 km**
- Corrosion coverage: **12 years unlimited km** (both brands)
- UX tire repair kit: **Rear cargo area, under floor panel, left compartment**

## Testing

```bash
# Run comprehensive test suite
uv run python tests/test_components.py
```

Tests cover:
- ✅ Database schema and operations
- ✅ Document processing and chunking
- ✅ SQL validation and safety
- ✅ Router decision logic
- ✅ Integration with demo questions
- ✅ Golden SQL query validation

## Performance

**Latency Targets (from PRD):**
- SQL queries: < 2.5s P50
- RAG/Hybrid: < 3.5-5s P50

**Cost Optimization:**
- Embeddings corpus < 10MB
- Single-shot LLM calls per request
- Cost-efficient models (GPT-4o-mini, text-embedding-3-small)

## Safety & Security

### SQL Safety
- ✅ Read-only database access
- ✅ AST validation with sqlglot
- ✅ Table/column allowlists
- ✅ Query timeouts (2s)
- ✅ Row limits (10k max)
- ✅ Parameterized queries

### RAG Grounding
- ✅ Citation requirements
- ✅ Source attribution
- ✅ "Insufficient evidence" fallbacks
- ✅ Confidence scoring

## Project Structure

```
agentic_qa_assistant/
├── agentic_qa_assistant/
│   ├── main.py              # CLI interface
│   ├── database.py          # DuckDB management
│   ├── sql_tool.py          # SQL generation & validation
│   ├── rag_pipeline.py      # Document processing & search
│   ├── rag_tool.py          # RAG with citations
│   ├── router.py            # Intelligent routing
│   └── hybrid_orchestrator.py # Parallel execution
├── data/                    # CSV files
├── docs/                    # PDF/text documents
├── tests/                   # Test suite
└── vector_index/            # FAISS index (auto-created)
```

## Environment Variables

```bash
# Required
OPENAI_API_KEY=your_openai_api_key

# Optional
LOGFIRE_TOKEN=your_logfire_token  # For advanced tracing
```

## Development

### Adding New Documents
1. Place PDFs/text files in `docs/` directory
2. Restart the assistant (index rebuilds automatically)
3. Documents are automatically chunked and indexed

### Adding New Data
1. Update CSV files in `data/` directory
2. Ensure proper schema (see existing files)
3. Restart the assistant (database reloads)

### Extending SQL Schema
1. Update `SqlValidator.ALLOWED_TABLES` in `sql_tool.py`
2. Update table creation in `database.py`
3. Add corresponding CSV files

## Production Considerations

### Scalability
- **Vector store**: Migrate to pgvector or Qdrant for production
- **Database**: Move to PostgreSQL for multi-user access
- **Caching**: Add Redis for query result caching
- **API**: Wrap in FastAPI for web service deployment

### Monitoring
- **Tracing**: Logfire integration ready (PRD requirement)
- **Metrics**: Per-tool latency and success rates
- **Cost tracking**: Token usage monitoring
- **Quality**: RAG precision@k evaluation

### Legal & Compliance
- **Manual licensing**: Implement proper ToS compliance for owner's manuals
- **Data privacy**: No PII storage as designed
- **Audit logging**: All queries and responses logged

## Troubleshooting

### Common Issues

**"OpenAI API key required"**
```bash
export OPENAI_API_KEY="your-key-here"
# Or create .env file
```

**"No chunks created from documents"**
- Ensure PDFs are readable text (not scanned images)
- Check file permissions in `docs/` directory

**"SQL validation failed"**
- Only SELECT statements allowed
- Use tables: DIM_MODEL, DIM_COUNTRY, DIM_ORDERTYPE, FACT_SALES, FACT_SALES_ORDERTYPE
- Queries auto-limited to 10k rows

**Vector index errors**
- Delete `vector_index/` directory to rebuild
- Ensure sufficient disk space (index ~10MB)

## License

This project is created for technical assessment purposes. See individual data sources for their respective licenses.

---

**Built with:** Python 3.12, uv, PydanticAI, DuckDB, FAISS, OpenAI GPT-4o-mini, Rich CLI