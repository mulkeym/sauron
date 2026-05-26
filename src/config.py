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
    embedding_device: str = "cpu"  # "cpu", "cuda", "cuda:0", or "multi-gpu" (only used when mode=local)
    embedding_batch_size: int = 64  # batch size for local embedding
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

    # MCP Server
    mcp_server_name: str = "sauron"
    mcp_port: int = 8090
    mcp_alt_port: int = 8091  # Alternative HTTP port for network access

    # Concurrency
    max_parallel_ingestion: int = 3  # concurrent file ingestion jobs
    llm_concurrency: int = 4  # concurrent LLM calls (map-reduce, entity extraction)

    # Metadata extraction
    metadata_extraction_enabled: bool = True  # disable to skip metadata step
    metadata_max_doc_length: int = 200000  # chars sent to LLM for extraction

    # LLM context limits (adjust based on your model's context window)
    llm_max_context: int = 200000  # max chars sent to LLM for synthesis (~50K tokens for 256K context models)
    llm_max_output_tokens: int = 32768  # max tokens the LLM can generate in a response
    map_doc_char_budget: int = 80000  # max chars per MAP extraction call (~20K tokens); tighter than llm_max_context so one oversized doc can't run to the request timeout

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

    @property
    def api_key_list(self) -> list[str]:
        return [k.strip() for k in self.api_keys.split(",") if k.strip()]

    @property
    def database_registry(self) -> dict[str, str]:
        if not self.registered_databases:
            return {}
        pairs = [p.strip() for p in self.registered_databases.split(",") if "=" in p]
        return {k.strip(): v.strip() for p in pairs for k, v in [p.split("=", 1)]}

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


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
