# POC → MVP Production Roadmap (Cost-Effective, Modular, Maintainable)

This roadmap outlines the sequenced work to de-risk data sourcing, production-grade infrastructure, and deliver a secure, cost-optimized, and provider-agnostic AI service.

## 1) Data Acquisition & Licensing

**Goal:** Responsibly source product owner manuals with clear licensing, reproducibility, and a mechanism for data freshness.

- **Sourcing Strategy (Priority Order)**
  - **Vendor/Partner APIs:** Primary method. Implement OAuth2/signed requests with strict rate-limiting compliance. Store full license metadata per document.
  - **Data Partnerships:** Formalize with Data Processing Agreements (DPAs) and SLAs. Prefer push-based ingestion via authenticated partner webhooks or signed URLs.
  - **Licensed Scraping (Last Resort):** Only for explicitly permitted sources. Version-controlled crawl rules (e.g., Scrapy/Playwright) respecting `robots.txt`, `noai` tags. Implement polite crawling with IP rotation, ETag/Last-Modified headers, and short time windows.
- **Freshness & Validation Pipeline**
  - **Change Detection:** Pre-flight checks using HEAD requests for ETag/Last-Modified headers. Fallback to checksum comparison of normalized text content.
  - **Immutable Storage:** Write all raw artifacts to versioned, S3-compatible object storage. Metadata must include source URL, license terms, crawl config version, timestamp, and content hash.
  - **Validation & Quarantine:** Validate MIME type and size. Monitor text extraction health (page count, language detection). Deduplicate content using SimHash/MinHash. Quarantine failed items for review.
  - **Orchestration:** Scheduled periodic crawls via Cloud Scheduler/Airflow. Event-driven ingestion for partner webhooks. All jobs must implement exponential backoff with jitter.
- **Governance**
  - Maintain a manifest of per-document license terms and allowed uses. Automate enforcement of exclusion lists.
  - Full audit logging for all data ingress and access (identity, timestamp, action, document ID).

## 2) Production Data Infrastructure

**Goal:** Migrate to a cloud-agnostic stack based on PostgreSQL for operational data and scalable object storage, with automated schema management.

- **SQL Database (PostgreSQL)**
  - Use `pgvector` for dense vector embeddings and built-in GIN indexes with `tsvector` for full-text search.
  - **Core Tables:** `products`, `manuals`, `file_artifacts`, `sections`, `chunks`, `embeddings`, `vendors`, `licenses`, `crawl_jobs`.
  - **Indexing Strategy:** B-Tree on all Foreign Keys and common filters (e.g., `is_public`). GIN on `tsvector` columns. HNSW on `pgvector` columns. Composite indexes for high-impact query patterns (e.g., `(product_id, is_public, uploaded_at)`).
  - **Migrations:** Alembic for schema migration management. Auto-generate migrations from SQLAlchemy ORM models. Implement a pre-commit CI hook to prevent model-database schema drift.
- **Schema Documentation (Auto-Generated)**
  - Use SQLAlchemy 2.x ORM as the source of truth. Auto-generate API documentation with MkDocs + mkdocstrings.
  - Generate Entity-Relationship Diagrams (ERDs) in CI using SchemaSpy or dbdocs, publishing a static site artifact with each release.
- **Object Storage**
  - S3-compatible (MinIO for local dev; S3/GCS for production) with versioning and lifecycle policies.
  - **Bucket Structure:**
    - `raw/`: Original, unprocessed documents.
    - `processed/`: Extracted and normalized text.
    - `derived/`: Generated chunks, embeddings.
    - `logs/`: Ingestion and processing audit logs.
  - Enable server-side encryption (SSE-S3/AES-256) by default. Access controlled via strict, least-privilege IAM roles.

## 3) Core AI Service Architecture

**Goal:** Build a containerized, modular API with a provider-agnostic LLM layer and a robust, evaluable multi-stage RAG pipeline.

- **Framework & Runtime**
  - **API Server:** FastAPI served via Uvicorn inside a multi-stage Docker container.
  - **Project Structure:**
    - `api/`: Routers and Pydantic schemas.
    - `core/`: RAG pipeline, chunking, and retrieval logic.
    - `providers/`: Abstracted clients for LLM (LiteLLM), VectorDB, and storage.
    - `data/`: SQLAlchemy models and Alembic migrations.
  - **Orchestration:** LangGraph for defining complex, deterministic multi-step workflows (e.g., `ingest → chunk → embed → index`, `query → retrieve → rerank → synthesize`).
- **LLM Abstraction & Prompt Management**
  - **Provider Agnosticism:** Mandate LiteLLM for all Chat and Embedding calls. Unifies configuration, retries, timeouts, and most importantly, **cost tracking** across providers (OpenAI, Anthropic, etc.).
  - Enforce model-aware parameter mapping: reasoning models (`gpt-5-*`, `o*`) → `max_completion_tokens` (Chat) or `max_output_tokens` (Responses) and omit `temperature`; others → `max_tokens` and include `temperature` if desired.
  - Per-role model configuration (SQL generation, RAG synthesis, hybrid composition) with fast defaults for synthesis/composition (e.g., `gpt-4o-mini`) to reduce latency.
  - **Prompt Management:** For complex prompts (e.g., for SQL generation or response synthesis), adopt a structured version-controlled approach. **Suggestion:** Evaluate `microsoft/poml` (Prompt OML) to define prompts as code, separate from application logic, enabling versioning, testing, and easy iteration.
- **Multi-Stage RAG Pipeline**
  - **a) Embeddings:** Host `BAAI/bge-small-en-v1.5` (or `bge-m3` for multilingual) as a separate internal service using Text Embeddings Inference (TEI) for maximum performance. Implement batch processing, L2 normalization, and evaluate INT8 quantization.
  - **b) Vector DB & Hybrid Search:**
    - **Primary:** Qdrant for production. Excellent performance for HNSW-based dense vector search, built-in sparse (BM25) scoring, and rich payload filtering.
    - **Alternative (Simplified):** Rely on PostgreSQL (`pgvector` + `tsvector`) if managing fewer components is a higher priority than peak retrieval performance.
  - **c) Text-to-SQL:**
    - Use a specialized, smaller model (e.g., `SQLCoder-7B/2B`) via vLLM + LiteLLM for high accuracy and low cost on known schema queries.
    - Provide the DB schema DDL as context within the prompt. Cache successful query templates by a hash of the prompt and schema version.
- **Quality Enhancements**
  - **Chunking:** Semantic chunking (e.g., with `langchain-text-splitters`) with token limits. Preserve structural metadata (heading hierarchy, list context).
  - **Optional Reranking:** Integrate `bge-reranker-base` as a configurable step to refine top-k results when the latency budget allows.
  - **Citations:** Return precise source references (document ID, page number, section header). Enforce strict limits on the number of retrieved contexts (`top_k`) and context token length.

## 4) Configuration & Security

**Goal:** Eliminate hardcoded values and ensure a secure, secrets-aware configuration system.

- **Externalized Configuration**
  - `config.yaml` for default settings. Environment-specific overrides via environment variables, validated by `pydantic-settings`.
  - Secrets (API keys, DB URLs) must be injected at runtime from a secure vault (e.g., HashiCorp Vault, AWS Secrets Manager, GCP Secret Manager) and never stored in version control or config files.
- **Security Hardening**
  - **Least Privilege:** Apply strict IAM roles and network policies. Restrict egress traffic from containers.
  - **Container Security:** Use pinned, minimal base images. Run as a non-root user. Implement read-only filesystems where possible. Scan for vulnerabilities in CI (e.g., with Grype, Trivy).
  - **API Security:** Implement authentication (JWT/API Keys) and rate limiting on public endpoints. Use a WAF/CDN if internet-exposed.
  - **Prompt Injection Mitigation:**
    - **Input Sanitization:** Treat all user input as untrusted. Implement allow-lists for input characters and length limits where feasible.
    - **Intent Classification:** For critical endpoints, use a preliminary LLM call to classify user intent as valid or potentially malicious before processing the main prompt.
    - **Contextualization:** Clearly separate instructions, context, and user input within prompts using unambiguous delimiters (e.g., `### SYSTEM:`, `### USER:`).
    - **Zero Trust in Output:** Never execute raw LLM-generated code (e.g., SQL) without a sanitization step. Use parameterized queries.

## 5) Cost & Performance Optimization

**Goal:** Aggressively minimize LLM and infrastructure spend while maintaining strict latency SLOs.

- **LLM Cost Controls**
  - **Caching:** Implement LiteLLM's caching layer (Redis) for frequent prompts and common retrieved contexts.
  - **Context Management:** Use tight chunk sizes, limit `top_k` (≤ 6–8), and compress citation formatting.
  - **Efficient Embeddings:** The choice of a local, small embedding model (`BGE-small`) is the primary cost saver. Batch embedding jobs.
  - **Intelligent Routing:** Use LiteLLM to route requests to the most cost-effective provider that meets the latency/quality requirement for a given task.
  - **Parameter Mapping Guards:** Reasoning models (`gpt-5-*`, `o*`) require `max_completion_tokens`/`max_output_tokens` and ignore non-default `temperature`. Implement guards to avoid 400s.
  - **Multi-LLM Strategy:** Use smaller, faster models (e.g., `gpt-4o-mini`) for RAG synthesis and hybrid composition when primary is a reasoning model.
- **Infrastructure Cost Controls**
  - **Autoscaling:** Scale API workers and embedding inference pods independently based on QoS metrics. Scale batch workers to zero when idle.
  - **Spot Instances:** Use spot/preemptible VMs for all batch processing jobs (embedding, indexing). Design jobs to be **idempotent** and checkpoint progress.
  - **Data Lifecycle:** Move `raw/` artifacts to cold storage after 30 days. Automatically purge `derived/` artifacts (which can be recomputed) after a shorter period.
  - **Collocation:** Deploy compute in the same cloud region and availability zone as the database and vector store to minimize latency and data egress costs.
- **Latency Optimizations**
  - Keep connections to databases and model servers warm (connection pooling, HTTP keep-alive).
  - Apply reranking selectively based on query complexity or low confidence scores from the initial retrieval.

## 6) Observability & MLOps

**Goal:** Achieve full visibility into system health, API performance, RAG quality, and LLM costs.

- **Unified Observability**
  - Integrate **Logfire** (or LangSmith/LangFuse) as the primary observability platform. Instrument FastAPI, LiteLLM, and custom RAG steps to emit structured logs, traces, and metrics.
  - **Trace Attributes:** Capture `prompt_hash`, `model`, `input_size`, `latency`, `token_usage`, `estimated_cost`, and `retrieved_document_ids` on all spans.
  - **Dashboards:** Monitor p95 latency, error rate, retrieval hit rate, and cost-per-request.
- **Monitoring & Alerts**
  - Alert on error rate bursts, anomalous token spending, and degradation in retrieval quality (e.g., spike in `null` result rate).
  - Ensure PII redaction in logs. Sample traces in high-volume production environments.
- **Continuous Evaluation**
  - Maintain a **golden dataset** of question/ground-truth answer pairs. Run automated weekly evaluations to track metrics like Exact Match, BLEU/ROUGE, and citation precision/recall.
  - Use feature flags to canary new models or pipeline changes. Implement automatic rollback based on evaluation regression.

---

### Implementation Phases

- **Phase 0: Foundation (2-3 wks)**
  - Repo setup, CI/CD, Dockerization, Base FastAPI app with LiteLLM integration, Configuration & Secrets management.
- **Phase 1: Core Pipelines (4-5 wks)**
  - Ingestion & validation pipeline, PostgreSQL schema + Alembic migrations, Qdrant deployment, Embedding service (TEI), Basic RAG retrieval endpoint.
- **Phase 2: Enhanced Search & Observability (3-4 wks)**
  - Hybrid search integration, Optional reranker, Text-to-SQL endpoint, Full Logfire instrumentation, Dashboards & alerting.
- **Phase 3: Optimization & Validation (2-3 wks)**
  - Cost tuning (caching, batching), Golden set creation & automated eval, Final security review, Documentation finalization.

### References

- **FastAPI:** <https://fastapi.tiangolo.com/>
- **LangGraph:** <https://langchain-ai.github.io/langgraph/>
- **LiteLLM:** <https://docs.litellm.ai/docs/>
- **Prompt Management (poml):** <https://github.com/microsoft/poml>
- **Qdrant:** <https://qdrant.tech/documentation/>
- **pgvector:** <https://github.com/pgvector/pgvector>
- **Text Embeddings Inference (TEI):** <https://huggingface.co/docs/text-embeddings-inference/index>
- **Logfire:** <https://docs.pydantic.dev/logfire/>