# Changelog

## v0.2.0 - 2025-09-17

### Features
- llm: model-aware param mapping for Chat/Responses and temperature guards for reasoning models (gpt-5-*, o*)
- sql: robust SELECT extraction from LLM output and single retry with gpt-4o-mini when invalid SQL is generated
- rag/compose: default to fast model (gpt-4o-mini) for synthesis and hybrid composition when primary is a reasoning model
- cli: per-role model flags (--model-sql, --model-rag, --model-composition) and multi-LLM wiring

### Tests
- llm utils: unit tests for token/temperature/chat param mapping

### Docs
- README, TECHNICAL_DESIGN, PRODUCTION_ROADMAP updated for multi-LLM model roles, parameter guards, and new flags
