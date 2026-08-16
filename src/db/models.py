from datetime import datetime, timezone

from sqlalchemy import JSON, DateTime, Float, String, UniqueConstraint
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


class DocumentRecord(Base):
    __tablename__ = "documents"

    doc_id: Mapped[str] = mapped_column(String, primary_key=True)
    filename: Mapped[str] = mapped_column(String, nullable=False)
    doc_type: Mapped[str] = mapped_column(String, nullable=False)
    content_hash: Mapped[str] = mapped_column(String, default="")
    dataset_id: Mapped[int] = mapped_column(default=0)  # 0 = unassigned
    category: Mapped[str] = mapped_column(String, default="")
    acl_groups: Mapped[list] = mapped_column(JSON, default=list)
    chunk_count: Mapped[int] = mapped_column(default=0)
    source_url: Mapped[str] = mapped_column(String, default="")
    summary: Mapped[str] = mapped_column(String, default="")
    metadata_tags: Mapped[dict] = mapped_column(JSON, default=dict)
    uploaded_by: Mapped[str] = mapped_column(String, default="")
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
    )


class Dataset(Base):
    __tablename__ = "datasets"
    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String, unique=True, nullable=False)
    slug: Mapped[str] = mapped_column(String, unique=True, nullable=False)  # URL-friendly
    description: Mapped[str] = mapped_column(String, default="")
    owner: Mapped[str] = mapped_column(String, default="admin")
    default_acl_groups: Mapped[list] = mapped_column(JSON, default=list)
    allowed_categories: Mapped[list] = mapped_column(JSON, default=list)
    active: Mapped[bool] = mapped_column(default=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
    )


class WebConnector(Base):
    __tablename__ = "web_connectors"
    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String, nullable=False)
    base_url: Mapped[str] = mapped_column(String, nullable=False)
    dataset_id: Mapped[int] = mapped_column(default=0)
    acl_groups: Mapped[list] = mapped_column(JSON, default=list)
    category: Mapped[str] = mapped_column(String, default="")
    crawl_depth: Mapped[int] = mapped_column(default=1)  # how many links deep to follow
    url_pattern: Mapped[str] = mapped_column(String, default="")  # only follow matching URLs
    download_file_types: Mapped[list] = mapped_column(JSON, default=list)  # [".pdf", ".docx"]
    additional_urls: Mapped[list] = mapped_column(JSON, default=list)  # extra seed URLs
    max_pages: Mapped[int] = mapped_column(default=100)
    respect_robots: Mapped[bool] = mapped_column(default=True)
    active: Mapped[bool] = mapped_column(default=True)
    last_crawl: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=True)
    pages_found: Mapped[int] = mapped_column(default=0)
    pages_ingested: Mapped[int] = mapped_column(default=0)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
    )


class AclGroup(Base):
    __tablename__ = "acl_groups"
    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String, unique=True, nullable=False)  # internal name (e.g., "it_support")
    display_name: Mapped[str] = mapped_column(String, default="")  # human-friendly (e.g., "IT Support Team")
    ad_group_dn: Mapped[str] = mapped_column(String, default="")  # future: AD distinguished name (e.g., "CN=IT Support,OU=Groups,DC=corp,DC=com")
    description: Mapped[str] = mapped_column(String, default="")
    active: Mapped[bool] = mapped_column(default=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
    )


class Persona(Base):
    """Lab / playground test user with ACL group memberships.

    Used by the admin Playground and Knowledge Graph "act as" dropdowns.
    Not production auth — local simulation of AD group membership.
    """
    __tablename__ = "personas"
    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String, unique=True, nullable=False)  # stable id (e.g., "mike")
    display_name: Mapped[str] = mapped_column(String, default="")  # e.g., "Mike (Finance, Executives)"
    role: Mapped[str] = mapped_column(String, default="")  # e.g., "Finance Manager"
    groups: Mapped[list] = mapped_column(JSON, default=list)  # ACL group names
    active: Mapped[bool] = mapped_column(default=True)
    sort_order: Mapped[int] = mapped_column(default=0)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
    )


class Category(Base):
    __tablename__ = "categories"
    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String, unique=True, nullable=False)
    description: Mapped[str] = mapped_column(String, default="")
    acl_groups: Mapped[list] = mapped_column(JSON, default=list)
    routing_keywords: Mapped[list] = mapped_column(JSON, default=list)
    grs_number: Mapped[str] = mapped_column(String, default="")  # NARA GRS mapping (e.g., "3.1", "5.8")
    doc_count: Mapped[int] = mapped_column(default=0)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
    )


class CategoryProposal(Base):
    __tablename__ = "category_proposals"
    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    proposed_name: Mapped[str] = mapped_column(String, nullable=False)
    proposed_description: Mapped[str] = mapped_column(String, default="")
    proposed_acl_groups: Mapped[list] = mapped_column(JSON, default=list)
    proposed_keywords: Mapped[list] = mapped_column(JSON, default=list)
    proposed_grs: Mapped[str] = mapped_column(String, default="")
    proposed_by: Mapped[str] = mapped_column(String, default="")
    status: Mapped[str] = mapped_column(String, default="pending")
    reviewed_by: Mapped[str] = mapped_column(String, default="")
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
    )


class Entity(Base):
    __tablename__ = "entities"
    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String, nullable=False)
    entity_type: Mapped[str] = mapped_column(String, nullable=False)
    first_seen_doc_id: Mapped[str] = mapped_column(String, default="")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    __table_args__ = (UniqueConstraint("name", "entity_type", name="uq_entity_name_type"),)


class EntityMention(Base):
    __tablename__ = "entity_mentions"
    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    entity_id: Mapped[int] = mapped_column(nullable=False)
    doc_id: Mapped[str] = mapped_column(String, nullable=False)
    chunk_index: Mapped[int] = mapped_column(default=0)
    context_snippet: Mapped[str] = mapped_column(String, default="")


class EntityMergeProposal(Base):
    __tablename__ = "entity_merge_proposals"
    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    entity_a_id: Mapped[int] = mapped_column(nullable=False)
    entity_a_name: Mapped[str] = mapped_column(String, default="")
    entity_b_id: Mapped[int] = mapped_column(nullable=False)
    entity_b_name: Mapped[str] = mapped_column(String, default="")
    confidence: Mapped[float] = mapped_column(default=0.0)
    reason: Mapped[str] = mapped_column(String, default="")
    status: Mapped[str] = mapped_column(String, default="pending")  # pending, approved, rejected, auto_merged
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))


class Relationship(Base):
    __tablename__ = "relationships"
    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    source_entity_id: Mapped[int] = mapped_column(nullable=False)
    target_entity_id: Mapped[int] = mapped_column(nullable=False)
    relationship_type: Mapped[str] = mapped_column(String, nullable=False)
    doc_id: Mapped[str] = mapped_column(String, default="")
    context_snippet: Mapped[str] = mapped_column(String, default="")


class QueryMetrics(Base):
    __tablename__ = "query_metrics"
    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    query_text: Mapped[str] = mapped_column(String, nullable=False)
    query_type: Mapped[str] = mapped_column(String, default="")  # sweep, lookup, etc.
    strategy_used: Mapped[str] = mapped_column(String, default="")
    user_groups: Mapped[list] = mapped_column(JSON, default=list)
    docs_discovered: Mapped[int] = mapped_column(default=0)
    docs_map_read: Mapped[int] = mapped_column(default=0)
    docs_relevant: Mapped[int] = mapped_column(default=0)
    docs_cited: Mapped[int] = mapped_column(default=0)
    map_precision: Mapped[float] = mapped_column(default=0.0)
    total_time_seconds: Mapped[float] = mapped_column(default=0.0)
    retrieval_time: Mapped[float] = mapped_column(default=0.0)
    map_time: Mapped[float] = mapped_column(default=0.0)
    synthesis_time: Mapped[float] = mapped_column(default=0.0)
    context_chars: Mapped[int] = mapped_column(default=0)
    answer_length: Mapped[int] = mapped_column(default=0)
    feedback_boost_applied: Mapped[int] = mapped_column(default=0)
    prf_triggered: Mapped[bool] = mapped_column(default=False)
    cache_hit: Mapped[bool] = mapped_column(default=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
    )


class QueryActivity(Base):
    __tablename__ = "query_activity"
    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
    )
    source: Mapped[str] = mapped_column(String, nullable=False)
    tool: Mapped[str] = mapped_column(String, nullable=False)
    username: Mapped[str] = mapped_column(String, default="")
    user_groups: Mapped[list] = mapped_column(JSON, default=list)
    query_text: Mapped[str] = mapped_column(String, default="")
    strategy: Mapped[str] = mapped_column(String, default="")
    duration_seconds: Mapped[float] = mapped_column(default=0.0)
    status: Mapped[str] = mapped_column(String, default="ok")
    cache_hit: Mapped[bool] = mapped_column(default=False)
    error: Mapped[str] = mapped_column(String, default="")


class QueryFeedback(Base):
    __tablename__ = "query_feedback"
    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    query_hash: Mapped[str] = mapped_column(String, nullable=False, index=True)
    query_text: Mapped[str] = mapped_column(String, nullable=False)
    query_vector_blob: Mapped[bytes] = mapped_column(default=b"")
    query_type: Mapped[str] = mapped_column(String, default="")
    doc_id: Mapped[str] = mapped_column(String, nullable=False)
    filename: Mapped[str] = mapped_column(String, default="")
    relevance_score: Mapped[float] = mapped_column(default=0.0)
    was_cited: Mapped[bool] = mapped_column(default=False)
    was_in_map_reduce: Mapped[bool] = mapped_column(default=False)
    user_groups: Mapped[list] = mapped_column(JSON, default=list)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
    )


class StrategyMemory(Base):
    __tablename__ = "strategy_memory"
    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    query_pattern: Mapped[str] = mapped_column(String, nullable=False, index=True)
    query_type: Mapped[str] = mapped_column(String, default="")
    strategy_used: Mapped[str] = mapped_column(String, default="")
    docs_discovered: Mapped[int] = mapped_column(default=0)
    docs_relevant: Mapped[int] = mapped_column(default=0)
    docs_cited: Mapped[int] = mapped_column(default=0)
    answer_length: Mapped[int] = mapped_column(default=0)
    total_time_seconds: Mapped[float] = mapped_column(default=0.0)
    metadata_fields_useful: Mapped[list] = mapped_column(JSON, default=list)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
    )


class RegisteredSchema(Base):
    __tablename__ = "registered_schemas"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    database: Mapped[str] = mapped_column(String, nullable=False)
    table_name: Mapped[str] = mapped_column(String, nullable=False)
    columns: Mapped[list] = mapped_column(JSON, default=list)   # [{"name","dtype","description"}]
    description: Mapped[str] = mapped_column(String, default="")
    acl_groups: Mapped[list] = mapped_column(JSON, default=list)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
    )
    __table_args__ = (UniqueConstraint("database", "table_name", name="uq_registered_schema"),)


class SchemaHintRecord(Base):
    __tablename__ = "schema_hints"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    scope_type: Mapped[str] = mapped_column(String, nullable=False)   # "category" | "dataset"
    scope_value: Mapped[str] = mapped_column(String, nullable=False)
    hint_type: Mapped[str] = mapped_column(String, nullable=False)    # value_glossary | column_note | table_note
    target_column: Mapped[str] = mapped_column(String, default="")    # "" == applies to table (table_note)
    payload: Mapped[dict] = mapped_column(JSON, default=dict)
    provenance: Mapped[str] = mapped_column(String, default="curated")
    confidence: Mapped[float] = mapped_column(Float, default=1.0)
    created_by: Mapped[str] = mapped_column(String, default="")
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
    )


class ApiApplication(Base):
    """Trusted client application that calls the Sauron API (service identity).

    Distinct from Dataset (formerly "applications" workspaces). User ACL still
    comes from JWT groups; this is machine/app credentials only.
    """
    __tablename__ = "api_applications"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String, unique=True, nullable=False)  # slug e.g. sdwan-demo-chat
    display_name: Mapped[str] = mapped_column(String, default="")
    description: Mapped[str] = mapped_column(String, default="")
    active: Mapped[bool] = mapped_column(default=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
    )


class ApiKeyRecord(Base):
    """Hashed API key for an application. Full secret is never stored."""
    __tablename__ = "api_key_records"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    application_id: Mapped[int] = mapped_column(nullable=False, index=True)
    key_prefix: Mapped[str] = mapped_column(String, default="")  # first chars for UI, e.g. sk_ab12…
    key_hash: Mapped[str] = mapped_column(String, unique=True, nullable=False, index=True)
    label: Mapped[str] = mapped_column(String, default="")  # e.g. "prod", "rotate-2026-07"
    active: Mapped[bool] = mapped_column(default=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
    )
    last_used_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=True)
    revoked_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=True)
