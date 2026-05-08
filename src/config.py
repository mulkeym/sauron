# src/config.py
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # LLM
    vllm_base_url: str = "http://localhost:8000/v1"
    vllm_model_name: str = "google/gemma-4-31b-it"

    # Embeddings
    embedding_model_name: str = "intfloat/e5-large-v2"
    embedding_device: str = "cpu"

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
