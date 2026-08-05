# src/config.py
from __future__ import annotations
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # LLM
    vllm_base_url: str = "https://api.openai.com/v1"
    vllm_model_name: str = "gpt-4.1-mini"
    vllm_api_key: str = ""  # OpenAI API key (or leave empty for local endpoints)
    vllm_request_timeout: int = 300  # seconds - increase for thinking/local models
    ssl_verify: bool = True  # set to False for self-signed certs

    # Embeddings
    embedding_mode: str = "local"  # "local" (sentence-transformers, no server needed) or "api" (external endpoint)
    embedding_api_url: str = "http://localhost:8000/v1"  # OpenAI-compatible /v1/embeddings endpoint (only used when mode=api)
    embedding_model_name: str = "nomic-ai/nomic-embed-text-v1"  # local default; set to API model name when mode=api
    embedding_batch_size: int = 64  # batch size for local embedding (CPU)
    embedding_dimension: int = 0  # auto-detect from first embedding call if 0

    # LanceDB
    lancedb_path: str = "data/lancedb"
    lancedb_table_name: str = "chunks"

    # Tabular store (structured spreadsheet querying)
    tabular_duckdb_path: str = "data/tabular.duckdb"

    # Admin UI
    admin_username: str = "admin"
    admin_password: str = "password123"

    # Auth
    jwt_secret_key: str = "change-me-in-production"
    jwt_algorithm: str = "HS256"
    jwt_expiration_minutes: int = 480
    api_keys: str = "dev-key-1"  # comma-separated

    # LDAP
    ldap_enabled: bool = False

    # Metadata DB
    database_url: str = "sqlite+aiosqlite:///./data/metadata.db"

    # Database Registry (for text-to-SQL)
    registered_databases: str = ""  # comma-separated list of "name=url" pairs

    # MCP Server (native Streamable HTTP, mounted in the API process)
    mcp_server_name: str = "sauron"
    mcp_enabled: bool = True
    mcp_path: str = "/mcp"
    mcp_stateless_http: bool = True
    # When OpenWebUI forwarding is enabled, it signs user identity with this
    # shared secret and sends it in X-OpenWebUI-User-Jwt. Keep the secret out of
    # data/settings.json; configure it as an environment/Kubernetes secret.
    mcp_openwebui_jwt_secret: str = ""
    mcp_openwebui_groups_header: str = "X-Sauron-User-Groups"
    # "ALL" is Sauron's superuser ACL. Never grant it from a forwarded group
    # name unless an operator explicitly opts in.
    mcp_openwebui_allow_all_group: bool = False
    # Deprecated compatibility fields for older settings.json/.env files. The
    # production MCP endpoint no longer listens on separate ports.
    mcp_port: int = 8090
    mcp_alt_port: int = 8091
    # Concurrency
    max_parallel_ingestion: int = 3  # concurrent file ingestion jobs
    max_parallel_async_query: int = 3  # concurrent async query worker slots
    async_query_ttl_seconds: int = 3600  # how long finished async jobs are retained
    max_async_query_jobs: int = 100  # cap on tracked async jobs (reject new submits past this)
    async_query_timeout_seconds: int = 600  # per-job ceiling so a wedged query can't hold a slot forever
    llm_concurrency: int = 4  # concurrent LLM calls (map-reduce, entity extraction)

    # Knowledge graph (LightRAG) extract tuning
    # Smaller chunks = more extract LLM calls (slower, sometimes better for tiny models).
    # 1200 tokens ≈ half the calls of the old 500 default for large PDFs.
    kg_chunk_token_size: int = 1200
    kg_chunk_overlap_token_size: int = 100
    # 0 = adaptive timeout from estimated chunk count; else fixed seconds per attempt
    kg_extract_timeout_seconds: int = 0
    kg_extract_timeout_max_seconds: int = 3600  # cap for adaptive (1 hour)
    kg_extract_timeout_min_seconds: int = 900   # floor for adaptive (15 min)
    kg_extract_sec_per_chunk: float = 22.0      # wall-time estimate per chunk / concurrency
    kg_extract_max_retries: int = 1  # full retries are expensive; prefer one long attempt

    # Metadata extraction
    metadata_extraction_enabled: bool = True  # disable to skip metadata step
    metadata_max_doc_length: int = 200000  # chars sent to LLM for extraction

    # LLM context limits (adjust based on your model's context window)
    llm_max_context: int = 200000  # max chars sent to LLM for synthesis (~50K tokens for 256K context models)
    llm_max_output_tokens: int = 32768  # max tokens the LLM can generate in a response
    llm_seed: int = 0  # fixed seed for deterministic LLM sampling (classification stability)
    map_doc_char_budget: int = 80000  # max chars per MAP extraction call (~20K tokens); tighter than llm_max_context so one oversized doc can't run to the request timeout

    # Structured/SQL consolidation + repair loop
    sql_result_budget_chars: int = 130000  # serialized SQL-result size (chars) that counts as "too large"; ~65% of llm_max_context, kept under the synthesizer cap. Effective budget is min(this, 0.65*llm_max_context).
    sql_wide_table_cell_threshold: int = 5000  # rows*cols above which the pre-flight gate steers text-to-SQL away from SELECT *
    sql_repair_max_retries: int = 2  # retries after the first generation (so max 3 generations total)
    sql_relevance_judge_enabled: bool = True  # on a flagged result, ask the LLM why it's unhelpful and feed that into the retry
    sql_thinking_on_wide_table: bool = True  # enable model reasoning for SQL generation when the wide-table gate fires (off elsewhere for speed)
    sql_thinking_max_tokens: int = 4096  # max_tokens for a thinking SQL-generation call (reasoning + SQL needs more than the default 2048)
    # Multi-turn table router: narrow candidate tables before rendering the
    # (value-dumping) text-to-SQL schema prompt, so it stays within the model
    # context as the corpus grows. Without this a broad question over a large
    # corpus sends every table's schema+values and overflows the context.
    sql_table_routing_enabled: bool = True  # run the LLM table-routing turn before text-to-SQL
    sql_table_routing_max_selected: int = 8  # max tables fed into one text-to-SQL prompt after routing
    sql_table_routing_catalog_budget_chars: int = 120000  # max size of the compact routing catalog; above this, embedding-rank candidates down to fit (~30K tokens)
    sql_schema_prompt_budget_chars: int = 600000  # hard cap on the text-to-SQL schema prompt (~150K tokens, safely under a 256K-token context); over this, value dumps are dropped, then tables truncated. Safety net behind the router.

    # Relevance feedback
    feedback_enabled: bool = True
    feedback_similarity_threshold: float = 0.85
    feedback_boost_cited: float = 0.3
    feedback_boost_relevant: float = 0.2
    feedback_penalty_irrelevant: float = 0.1
    feedback_decay_days: int = 90

    # Pseudo-relevance feedback
    prf_enabled: bool = True
    prf_top_k: int = 5  # number of top results to extract terms from
    prf_max_terms: int = 10  # max terms to append to expanded query

    # Strategy memory
    strategy_memory_enabled: bool = True
    strategy_memory_min_runs: int = 3   # min recorded runs before memory may override routing
    strategy_memory_margin: float = 0.15  # min normalized composite margin to override

    # Final-N reranking
    rerank_final_enabled: bool = True
    rerank_final_top_n: int = 50  # cap on chunks the final CrossEncoder pass scores/leads
    rerank_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    # Chunking
    chunk_size: int = 1024
    chunk_overlap: int = 100

    # Entity Reconciliation
    entity_merge_auto_threshold: float = 0.9  # auto-merge above this confidence
    entity_merge_review_threshold: float = 0.7  # propose for review above this

    # SharePoint Integration (future)
    sharepoint_tenant_id: str = ""
    sharepoint_client_id: str = ""
    sharepoint_client_secret: str = ""
    sharepoint_site_url: str = ""  # e.g., "https://contoso.sharepoint.com/sites/docs"
    sharepoint_sync_interval: int = 3600  # seconds between delta syncs
    sharepoint_doc_library: str = "Shared Documents"  # default library to crawl

    # Audit
    audit_log_path: str = "data/audit.jsonl"

    # Embedded figure / image extraction (PDF vision + OCR)
    figure_extraction_enabled: bool = True
    figure_max_per_doc: int = 20
    figure_min_width: int = 80          # skip logos / icons smaller than this
    figure_min_height: int = 80
    figure_min_area: int = 12000        # width*height threshold
    figure_ocr_first: bool = True
    figure_vision_timeout_seconds: int = 90
    figure_vision_max_tokens: int = 2048
    figure_page_render_dpi_scale: float = 1.5  # pypdfium2 render scale (~108 dpi at 1.5)
    figure_render_text_sparse_pages: bool = True  # full-page render when page text is sparse
    figure_sparse_text_chars: int = 40  # below this, treat page as image-heavy

    @property
    def api_key_list(self) -> list[str]:
        return [k.strip() for k in self.api_keys.split(",") if k.strip()]

    @property
    def database_registry(self) -> dict[str, str]:
        if not self.registered_databases:
            return {}
        pairs = [p.strip() for p in self.registered_databases.split(",") if "=" in p]
        return {k.strip(): v.strip() for p in pairs for k, v in [p.split("=", 1)]}

    # extra="ignore": tolerate leftover/deprecated env vars (e.g. a removed
    # EMBEDDING_DEVICE still present in a deployed .env) instead of crashing startup.
    model_config = {"env_file": ".env", "env_file_encoding": "utf-8", "extra": "ignore"}


def _load_persisted_settings(s: Settings) -> Settings:
    """Apply settings saved via the admin UI (data/settings.json).

    These override docker-compose defaults but NOT explicit host env vars.
    The file lives on the mounted volume so it survives container restarts.
    """
    import json
    from pathlib import Path

    path = Path("data/settings.json")
    if not path.exists():
        return s

    try:
        saved = json.loads(path.read_text())
    except Exception:
        return s

    for key, value in saved.items():
        if hasattr(s, key):
            setattr(s, key, value)
    return s


settings = _load_persisted_settings(Settings())
