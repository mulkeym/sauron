# src/config.py
from __future__ import annotations
from pydantic_settings import BaseSettings
from pydantic import Field, field_validator
from typing import Literal


class Settings(BaseSettings):
    @field_validator("*")
    @classmethod
    def nonnegative_numbers(cls, value, info):
        if isinstance(value, (int, float)) and not isinstance(value, bool) and value < 0:
            raise ValueError("Numeric settings must be nonnegative")
        import math
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError("Numeric settings must be finite")
        positive = {"llm_concurrency", "max_parallel_ingestion", "max_parallel_async_query",
                    "embedding_batch_size", "vllm_request_timeout", "llm_max_context",
                    "llm_max_output_tokens", "chunk_size", "max_async_query_jobs"}
        if info.field_name in positive and value == 0:
            raise ValueError("This setting must be greater than zero")
        return value

    # LLM
    vllm_base_url: str = "https://api.openai.com/v1"
    vllm_model_name: str = "gpt-4.1-mini"
    vllm_api_key: str = ""  # OpenAI API key (or leave empty for local endpoints)
    vllm_request_timeout: int = 300  # seconds - increase for thinking/local models
    ssl_verify: bool = True  # set to False for self-signed certs

    llm_answer_temperature: float = Field(
        default=0.1, ge=0.0, le=2.0, title="Final answer temperature",
        description="Sampling temperature for final answers, repairs and previews (0–2). Higher values increase output variation. For Gemma 4 thinking, start at 1.0 and evaluate answer accuracy. Models with fixed sampling omit this parameter.")
    llm_answer_thinking: Literal["default", "enabled", "disabled"] = Field(
        default="default", title="Final answer reasoning (thinking)",
        description="Thinking preference for final answers and answer previews. Provider default omits the control. Reasoning shares the output token budget and can add latency/cost. Classification, ingestion and the separate SQL option are unchanged.")
    llm_reasoning_adapter: Literal["auto", "vllm_template"] = Field(
        default="auto", title="Reasoning request adapter",
        description="Auto verifies OpenRouter model metadata. Select vLLM template only when your server and chat template support enable_thinking; this cannot override server restrictions.")

    # Embeddings
    embedding_mode: Literal["local", "api"] = "local"  # local model or external endpoint
    embedding_api_url: str = "http://localhost:8000/v1"  # OpenAI-compatible /v1/embeddings endpoint (only used when mode=api)
    embedding_model_name: str = "nomic-ai/nomic-embed-text-v1"  # local default; set to API model name when mode=api
    embedding_batch_size: int = Field(default=4, ge=1, le=512, title="Local embedding batch size",
        description="Maximum passages per local model batch. Applies to all chunk tiers; larger batches use more memory. Changes apply on the next request.")
    embedding_cpu_threads: int = Field(default=0, ge=0, le=256, title="Embedding CPU threads",
        description="CPU threads per model operation. 0 detects available CPUs, respecting container quota and affinity. Explicit values are capped to available CPUs. Changes recycle the worker on the next request.")
    embedding_cpu_interop_threads: int = Field(default=1, ge=1, le=256, title="Embedding inter-op threads",
        description="Parallel model operations. Start at 1; increasing this can oversubscribe CPUs. Capped to available CPUs; changes recycle the worker.")
    embedding_worker_memory_mb: int = Field(default=4096, ge=128, title="Embedding memory budget (MiB)",
        description="Worker memory budget with additional container headroom protection. A limit breach stops the worker, not the API. Lower batch size if requests exceed this budget.")
    embedding_worker_timeout_seconds: int = Field(default=1200, ge=1, title="Embedding request timeout (seconds)",
        description="Time limit per embedding request, including model loading on a cold worker. Queue waiting does not count.")
    embedding_worker_idle_seconds: int = Field(default=300, ge=1, title="Embedding idle timeout (seconds)",
        description="Keep the model loaded between requests for this long, then free its memory. The next request starts a new worker.")
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
    api_keys: str = ""  # comma-separated; no implicit shared development key

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
    # shared secret and sends it in X-OpenWebUI-User-Jwt. Operators may set it
    # in the environment or the protected, owner-readable admin settings file.
    mcp_openwebui_jwt_secret: str = ""
    mcp_openwebui_trust_headers: bool = Field(
        default=True,
        title="Trust OpenWebUI user headers",
        description="Accept a forwarded username without a JWT on MCP and chat-completion requests. "
        "A valid Sauron API key is still required; its holder can assert usernames and groups. "
        "Keep the connector key private to your trusted OpenWebUI backend.",
    )
    mcp_openwebui_username_header: str = Field(
        default="X-Sauron-Username",
        title="OpenWebUI username header",
        description="Header carrying the username when trusted user headers are enabled. "
        "In OpenWebUI, set its value to {{USER_NAME}} or {{USER_EMAIL}}.",
        min_length=1,
        pattern=r"^[!#$%&'*+.^_`|~0-9A-Za-z-]+$",
    )
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
    # Native file parsers run serially in disposable child processes. These
    # limits are separate from the concurrency of indexing/LLM work above.
    extraction_work_dir: str = "data/extraction"
    extraction_timeout_seconds: int = Field(default=1200, ge=1)
    extraction_memory_mb: int = Field(default=4096, ge=128)
    extraction_max_result_mb: int = Field(default=32, ge=1)
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
    kg_llm_max_output_tokens: int = Field(
        default=4096, ge=256, le=32768, title="Knowledge graph output token limit",
        description="Maximum generated tokens per graph model call. Truncated chunks are marked incomplete and can be retried.")
    kg_llm_timeout_seconds: int = Field(
        default=180, ge=10, le=600, title="Knowledge graph model deadline (seconds)",
        description="Total deadline per graph model call, including adapter retries. Failed chunks do not cancel other chunks; the whole-document budget still applies.")
    kg_llm_disable_thinking: bool = Field(
        default=True, title="Disable thinking for knowledge graph",
        description="Use the configured reasoning adapter to disable thinking for graph extraction and summaries, independently of final answers. Select vLLM template for compatible local servers; unsupported providers log a warning.")
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
    query_cache_mode: Literal["off", "exact", "semantic"] = "off"
    query_cache_ttl_seconds: int = Field(default=3600, ge=1, le=604800)
    query_cache_min_confidence: float = Field(default=0.95, ge=0, le=1)
    answer_domain_instructions: str = Field(default="", max_length=20000)
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
    strategy_memory_enabled: bool = False
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

    # Staged technical-document rollout; metadata capture is independent of selection.
    revision_selection_enabled: bool = Field(default=True, description="Select established newest applicable editions after ACL/dataset checks. Ambiguous editions remain available.")
    technical_structure_enabled: bool = Field(default=False, description="Use section-aware technical chunks for new ingestions; existing sources require reprocessing.")
    procedure_retrieval_enabled: bool = Field(default=False, description="Enable bounded procedure evidence retrieval and automatic/explicit procedure routing.")
    troubleshooting_retrieval_enabled: bool = Field(default=False, description="Enable troubleshooting evidence retrieval after procedure evaluation.")
    technical_section_max_chars: int = Field(default=16000, ge=1000, le=64000)
    technical_followup_top_k: int = Field(default=12, ge=1, le=30)

    # Visio conversion remains in the disposable ingestion worker.
    visio_enabled: bool = Field(default=True, description="Ingest modern .vsdx source text and rendered diagram pages.")
    visio_repair_text_layout: bool = Field(default=True, description="Restore saved Visio text-box wrapping and alignment, including styled runs and translation-only groups; unsupported layouts remain native.")
    visio_text_min_scale: float = Field(default=0.5, ge=0.25, le=1.0, description="Smallest preview font scale used to fit Visio labels into their saved boxes; a 4-point floor also applies. Source text is unchanged.")
    visio_max_pages: int = Field(default=100, ge=1, le=1000)
    visio_max_shapes: int = Field(default=20000, ge=1, le=100000, description="Maximum shapes per Visio page.")
    visio_max_unpacked_mb: int = Field(default=256, ge=1, le=1024)
    visio_converter_max_mb: int = Field(default=64, ge=1, le=256, description="Maximum converter output bytes, in MiB.")
    visio_timeout_seconds: int = Field(default=120, ge=1, le=900, description="Timeout for each Visio converter process; the overall extraction timeout also applies.")
    visio_render_max_edge: int = Field(default=4096, ge=256, le=8192, description="Longest edge of the full Visio PNG; figure pixel and storage limits also apply.")

    # Native metafile conversion shares the disposable extraction worker.
    emf_enabled: bool = Field(default=True, description="Convert standalone and Visio-embedded EMF images to PNG using Inkscape.")
    emf_max_input_mb: int = Field(default=32, ge=1, le=256)
    emf_max_conversions_per_doc: int = Field(default=256, ge=1, le=1000, description="Maximum distinct embedded EMFs per document; repeated images reuse a conversion.")
    emf_timeout_seconds: int = Field(default=120, ge=1, le=900, description="Total native EMF conversion time per document, within the extraction worker timeout.")
    emf_render_max_edge: int = Field(default=4096, ge=256, le=8192, description="Standalone EMF PNG resolution; figure pixel limits also apply.")
    emf_embedded_max_edge: int = Field(default=1024, ge=256, le=4096, description="Resolution of embedded EMF icons before composing a Visio page.")
    emf_ocr_timeout_seconds: int = Field(default=30, ge=1, le=300)
    emf_vision_enabled: bool = Field(default=True, description="Analyze rendered EMF diagrams with the configured vision-capable LLM; OCR and images survive model failures.")

    # Original bytes are retained only when a private directory is explicitly configured.
    source_originals_dir: str = Field(default="", title="Private original-document storage", description="Persistent directory for exact uploaded files. Blank disables retention. Changes apply to new uploads; existing documents need their exact originals backfilled.")
    source_originals_max_mb: int = Field(default=2048, ge=1, le=102400, description="Maximum size in MiB for retaining one original document.")
    source_download_jwt_secret: str = Field(default="", title="Original-download signing secret", description="Dedicated secret of at least 32 characters, shared with OpenWebUI original-download settings. Separate from MCP authentication. Leave blank when editing to keep the saved value.")
    source_download_webui_url: str = Field(default="", title="OpenWebUI public URL", description="Browser-facing OpenWebUI URL used in original-document links, including any reverse-proxy prefix.")

    # Embedded figure / image extraction (PDF vision + OCR)
    figure_store_enabled: bool = Field(default=True, description="Retain extracted PNGs for new ingestions. Existing documents require figure backfill.")
    figure_store_max_per_doc: int = Field(default=100, ge=1, le=1000)
    figure_store_max_doc_mb: int = Field(default=100, ge=1, le=1024)
    figure_max_pixels: int = Field(default=16000000, ge=10000, le=64000000)
    figure_full_max_mb: int = Field(default=12, ge=1, le=64)
    figure_preview_max_edge: int = Field(default=2000, ge=256, le=4096)
    figure_preview_max_mb: int = Field(default=3, ge=1, le=8)
    figure_mcp_max_mb: int = Field(default=8, ge=1, le=32, description="Maximum total base64 image bytes per MCP result.")
    figure_render_vector_pages: bool = Field(default=True, description="Render pages with substantial vector drawings; may include surrounding text and tables.")
    answer_images: Literal["auto", "requested", "off"] = Field(default="auto", description="Automatically attach cited diagrams, attach only when requested, or disable answer attachments.")
    answer_max_images: int = Field(default=2, ge=0, le=5)
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

    Persisted admin settings override environment defaults.
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

    try:
        return Settings.model_validate({**s.model_dump(), **saved})
    except Exception:
        import logging
        logging.getLogger(__name__).warning("Invalid persisted settings; using environment defaults")
        return s


settings = _load_persisted_settings(Settings())
