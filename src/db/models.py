from datetime import datetime, timezone

from sqlalchemy import JSON, DateTime, String, UniqueConstraint
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


class DocumentRecord(Base):
    __tablename__ = "documents"

    doc_id: Mapped[str] = mapped_column(String, primary_key=True)
    filename: Mapped[str] = mapped_column(String, nullable=False)
    doc_type: Mapped[str] = mapped_column(String, nullable=False)
    content_hash: Mapped[str] = mapped_column(String, default="")  # SHA-256 of file content
    category: Mapped[str] = mapped_column(String, default="")
    acl_groups: Mapped[list] = mapped_column(JSON, default=list)
    chunk_count: Mapped[int] = mapped_column(default=0)
    uploaded_by: Mapped[str] = mapped_column(String, default="")
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
