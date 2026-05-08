from datetime import datetime, timezone

from sqlalchemy import JSON, DateTime, String
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


class DocumentRecord(Base):
    __tablename__ = "documents"

    doc_id: Mapped[str] = mapped_column(String, primary_key=True)
    filename: Mapped[str] = mapped_column(String, nullable=False)
    doc_type: Mapped[str] = mapped_column(String, nullable=False)
    category: Mapped[str] = mapped_column(String, default="")
    acl_groups: Mapped[list] = mapped_column(JSON, default=list)
    chunk_count: Mapped[int] = mapped_column(default=0)
    uploaded_by: Mapped[str] = mapped_column(String, default="")
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
    proposed_by: Mapped[str] = mapped_column(String, default="")
    status: Mapped[str] = mapped_column(String, default="pending")
    reviewed_by: Mapped[str] = mapped_column(String, default="")
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
    )
