# src/config.py
from __future__ import annotations
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # LLM
    vllm_base_url: str = "http://localhost:8000/v1"
    vllm_model_name: str = "google/gemma-4-31b-it"
    vllm_request_timeout: int = 300  # seconds - increase for thinking models

    # Embeddings
    embedding_mode: str = "api"  # "api" (external OpenAI-compat endpoint) or "local" (sentence-transformers)
    embedding_api_url: str = "http://localhost:8000/v1"  # OpenAI-compatible /v1/embeddings endpoint
    embedding_model_name: str = "intfloat/e5-large-v2"
    embedding_device: str = "cpu"  # only used when embedding_mode=local
    embedding_dimension: int = 0  # auto-detect from first embedding call if 0

    # Qdrant
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333
    qdrant_collection_name: str = "documents"

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
    mcp_server_name: str = "rag-knowledge-service"
    mcp_port: int = 8090
    mcp_alt_port: int = 8091  # Alternative HTTP port for network access

    # Chunking
    chunk_size: int = 1024
    chunk_overlap: int = 100

    # Entity Reconciliation
    entity_merge_auto_threshold: float = 0.9  # auto-merge above this confidence
    entity_merge_review_threshold: float = 0.7  # propose for review above this

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


settings = Settings()
