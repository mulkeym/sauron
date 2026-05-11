from __future__ import annotations
from sqlalchemy import select, delete
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker

from sqlalchemy import update
from src.db.models import Base, DocumentRecord, Category, CategoryProposal, Entity, EntityMention, EntityMergeProposal, Relationship


class MetadataStore:
    def __init__(self, database_url: str | None = None):
        if database_url is None:
            from src.config import settings
            database_url = settings.database_url
        self.engine = create_async_engine(database_url)
        self.session_factory = async_sessionmaker(self.engine, expire_on_commit=False)

    async def init(self):
        async with self.engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

    async def add_document(
        self,
        doc_id,
        filename,
        doc_type,
        acl_groups,
        chunk_count,
        uploaded_by,
        category="",
    ):
        record = DocumentRecord(
            doc_id=doc_id,
            filename=filename,
            doc_type=doc_type,
            acl_groups=acl_groups,
            chunk_count=chunk_count,
            uploaded_by=uploaded_by,
            category=category,
        )
        async with self.session_factory() as session:
            session.add(record)
            await session.commit()
            await session.refresh(record)
        return record

    async def get_document(self, doc_id):
        async with self.session_factory() as session:
            return await session.get(DocumentRecord, doc_id)

    async def list_documents(self, user_groups=None):
        async with self.session_factory() as session:
            result = await session.execute(select(DocumentRecord))
            docs = list(result.scalars().all())
        if user_groups is not None:
            docs = [d for d in docs if any(g in d.acl_groups for g in user_groups)]
        return docs

    async def update_document_category(self, doc_id: str, category: str):
        async with self.session_factory() as session:
            await session.execute(
                update(DocumentRecord).where(DocumentRecord.doc_id == doc_id).values(category=category)
            )
            await session.commit()

    async def update_document(self, doc_id: str, **fields):
        async with self.session_factory() as session:
            await session.execute(
                update(DocumentRecord).where(DocumentRecord.doc_id == doc_id).values(**fields)
            )
            await session.commit()

    async def delete_document(self, doc_id):
        async with self.session_factory() as session:
            await session.execute(delete(DocumentRecord).where(DocumentRecord.doc_id == doc_id))
            # Also clean up entity mentions and relationships for this doc
            await session.execute(delete(EntityMention).where(EntityMention.doc_id == doc_id))
            await session.execute(delete(Relationship).where(Relationship.doc_id == doc_id))
            await session.commit()

    async def add_category(self, name, description, acl_groups, routing_keywords, grs_number=""):
        record = Category(name=name, description=description, acl_groups=acl_groups, routing_keywords=routing_keywords, grs_number=grs_number)
        async with self.session_factory() as session:
            session.add(record)
            await session.commit()
            await session.refresh(record)
        return record

    async def get_category(self, name):
        async with self.session_factory() as session:
            result = await session.execute(select(Category).where(Category.name == name))
            return result.scalar_one_or_none()

    async def list_categories(self):
        async with self.session_factory() as session:
            result = await session.execute(select(Category))
            return list(result.scalars().all())

    async def add_proposal(self, proposed_name, proposed_description, proposed_acl_groups, proposed_keywords, proposed_by):
        record = CategoryProposal(proposed_name=proposed_name, proposed_description=proposed_description, proposed_acl_groups=proposed_acl_groups, proposed_keywords=proposed_keywords, proposed_by=proposed_by)
        async with self.session_factory() as session:
            session.add(record)
            await session.commit()
            await session.refresh(record)
        return record

    async def list_proposals(self, status="pending"):
        async with self.session_factory() as session:
            result = await session.execute(select(CategoryProposal).where(CategoryProposal.status == status))
            return list(result.scalars().all())

    async def approve_proposal(self, proposal_id, approved_by):
        async with self.session_factory() as session:
            proposal = await session.get(CategoryProposal, proposal_id)
            if proposal:
                proposal.status = "approved"
                proposal.reviewed_by = approved_by
                cat = Category(name=proposal.proposed_name, description=proposal.proposed_description, acl_groups=proposal.proposed_acl_groups, routing_keywords=proposal.proposed_keywords)
                session.add(cat)
                await session.commit()

    async def reject_proposal(self, proposal_id, rejected_by):
        async with self.session_factory() as session:
            proposal = await session.get(CategoryProposal, proposal_id)
            if proposal:
                proposal.status = "rejected"
                proposal.reviewed_by = rejected_by
                await session.commit()

    async def add_entity(self, name, entity_type, first_seen_doc_id):
        async with self.session_factory() as session:
            result = await session.execute(select(Entity).where(Entity.name == name, Entity.entity_type == entity_type))
            existing = result.scalar_one_or_none()
            if existing:
                return existing.id
            entity = Entity(name=name, entity_type=entity_type, first_seen_doc_id=first_seen_doc_id)
            session.add(entity)
            await session.commit()
            await session.refresh(entity)
            return entity.id

    async def add_mention(self, entity_id, doc_id, chunk_index, context_snippet):
        async with self.session_factory() as session:
            mention = EntityMention(entity_id=entity_id, doc_id=doc_id, chunk_index=chunk_index, context_snippet=context_snippet[:200])
            session.add(mention)
            await session.commit()

    async def add_relationship(self, source_entity_id, target_entity_id, relationship_type, doc_id, context_snippet=""):
        async with self.session_factory() as session:
            rel = Relationship(source_entity_id=source_entity_id, target_entity_id=target_entity_id, relationship_type=relationship_type, doc_id=doc_id, context_snippet=context_snippet[:200])
            session.add(rel)
            await session.commit()

    async def search_entities(self, query, entity_type=None):
        async with self.session_factory() as session:
            stmt = select(Entity).where(Entity.name.ilike(f"%{query}%"))
            if entity_type:
                stmt = stmt.where(Entity.entity_type == entity_type)
            result = await session.execute(stmt)
            return list(result.scalars().all())

    async def get_entity_details(self, entity_id):
        async with self.session_factory() as session:
            entity = await session.get(Entity, entity_id)
            if not entity:
                return {"entity": None, "mentions": [], "relationships": []}
            mentions_result = await session.execute(select(EntityMention).where(EntityMention.entity_id == entity_id))
            mentions = [{"doc_id": m.doc_id, "chunk_index": m.chunk_index, "context_snippet": m.context_snippet} for m in mentions_result.scalars().all()]
            rels_as_source = await session.execute(select(Relationship).where(Relationship.source_entity_id == entity_id))
            rels_as_target = await session.execute(select(Relationship).where(Relationship.target_entity_id == entity_id))
            relationships = []
            for r in rels_as_source.scalars().all():
                target = await session.get(Entity, r.target_entity_id)
                relationships.append({"related_entity": target.name if target else "unknown", "entity_type": target.entity_type if target else "", "relationship_type": r.relationship_type, "direction": "outgoing", "doc_id": r.doc_id, "context": r.context_snippet})
            for r in rels_as_target.scalars().all():
                source = await session.get(Entity, r.source_entity_id)
                relationships.append({"related_entity": source.name if source else "unknown", "entity_type": source.entity_type if source else "", "relationship_type": r.relationship_type, "direction": "incoming", "doc_id": r.doc_id, "context": r.context_snippet})
            return {"entity": {"id": entity.id, "name": entity.name, "type": entity.entity_type}, "mentions": mentions, "relationships": relationships}

    async def list_entities(self, entity_type=None, limit=100):
        async with self.session_factory() as session:
            stmt = select(Entity)
            if entity_type:
                stmt = stmt.where(Entity.entity_type == entity_type)
            stmt = stmt.limit(limit)
            result = await session.execute(stmt)
            return list(result.scalars().all())

    async def delete_entities_for_doc(self, doc_id):
        async with self.session_factory() as session:
            await session.execute(delete(EntityMention).where(EntityMention.doc_id == doc_id))
            await session.execute(delete(Relationship).where(Relationship.doc_id == doc_id))
            await session.commit()
        # Clean up orphaned entities (no remaining mentions)
        await self._cleanup_orphaned_entities()

    async def add_merge_proposal(self, entity_a_id, entity_a_name, entity_b_id, entity_b_name, confidence, reason, status="pending"):
        async with self.session_factory() as session:
            proposal = EntityMergeProposal(
                entity_a_id=entity_a_id, entity_a_name=entity_a_name,
                entity_b_id=entity_b_id, entity_b_name=entity_b_name,
                confidence=confidence, reason=reason, status=status,
            )
            session.add(proposal)
            await session.commit()

    async def list_merge_proposals(self, status="pending"):
        async with self.session_factory() as session:
            result = await session.execute(select(EntityMergeProposal).where(EntityMergeProposal.status == status))
            return list(result.scalars().all())

    async def merge_entities(self, keep_id, remove_id):
        """Merge remove_id into keep_id: move all mentions and relationships, then delete."""
        async with self.session_factory() as session:
            # Update mentions
            await session.execute(update(EntityMention).where(EntityMention.entity_id == remove_id).values(entity_id=keep_id))
            # Update relationships
            await session.execute(update(Relationship).where(Relationship.source_entity_id == remove_id).values(source_entity_id=keep_id))
            await session.execute(update(Relationship).where(Relationship.target_entity_id == remove_id).values(target_entity_id=keep_id))
            # Delete the merged entity
            await session.execute(delete(Entity).where(Entity.id == remove_id))
            await session.commit()

    async def approve_merge(self, proposal_id):
        async with self.session_factory() as session:
            proposal = await session.get(EntityMergeProposal, proposal_id)
            if proposal:
                proposal.status = "approved"
                await session.commit()
        await self.merge_entities(proposal.entity_a_id, proposal.entity_b_id)

    async def reject_merge(self, proposal_id):
        async with self.session_factory() as session:
            proposal = await session.get(EntityMergeProposal, proposal_id)
            if proposal:
                proposal.status = "rejected"
                await session.commit()

    async def _cleanup_orphaned_entities(self):
        from sqlalchemy import func
        async with self.session_factory() as session:
            # Find entities with zero mentions
            subq = select(EntityMention.entity_id).distinct()
            result = await session.execute(
                select(Entity).where(Entity.id.notin_(subq))
            )
            orphans = list(result.scalars().all())
            if orphans:
                orphan_ids = [e.id for e in orphans]
                await session.execute(delete(Entity).where(Entity.id.in_(orphan_ids)))
                await session.commit()
