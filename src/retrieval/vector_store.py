from __future__ import annotations
import logging
import uuid

import lancedb
import pyarrow as pa

from src.config import settings
from src.retrieval.models import ChunkMetadata, RetrievedChunk

logger = logging.getLogger(__name__)


def _detect_vector_size() -> int:
    """Detect embedding dimension from config or by calling the embedding endpoint."""
    if settings.embedding_dimension > 0:
        return settings.embedding_dimension
    try:
        from src.ingestion.embedder import embed_texts
        vectors = embed_texts(["dimension probe"])
        dim = len(vectors[0])
        settings.embedding_dimension = dim
        return dim
    except Exception:
        return 4096  # safe default for large models


class VectorStore:
    _cross_encoder = None  # class-level cache for CrossEncoder model
    _cross_encoder_model = None  # class-level cache for the raw CrossEncoder

    def __init__(self):
        self.db = lancedb.connect(settings.lancedb_path)
        self.table_name = settings.lancedb_table_name
        self._table = None

    @classmethod
    def _get_cross_encoder(cls):
        """Lazy-load and cache the CrossEncoder reranker (loads model once)."""
        if cls._cross_encoder is None:
            from lancedb.rerankers import CrossEncoderReranker
            cls._cross_encoder = CrossEncoderReranker(column="text")
            logger.info("CrossEncoder model loaded and cached")
        return cls._cross_encoder

    @classmethod
    def _get_cross_encoder_model(cls):
        """Lazy-load and cache a raw sentence_transformers CrossEncoder for
        scoring explicit (query, text) pairs (the lancedb reranker only drives
        Lance query pipelines)."""
        # NOTE: cached for the process lifetime; changing settings.rerank_model needs a restart.
        if cls._cross_encoder_model is None:
            from sentence_transformers import CrossEncoder
            cls._cross_encoder_model = CrossEncoder(settings.rerank_model)
        return cls._cross_encoder_model

    @property
    def table(self):
        if self._table is None:
            self._table = self._ensure_table()
        return self._table

    def _build_schema(self) -> pa.Schema:
        dim = _detect_vector_size()
        return pa.schema([
            pa.field("id", pa.string()),
            pa.field("text", pa.string()),
            pa.field("vector", pa.list_(pa.float32(), dim)),
            pa.field("doc_id", pa.string()),
            pa.field("filename", pa.string()),
            pa.field("doc_type", pa.string()),
            pa.field("chunk_index", pa.int32()),
            pa.field("start_char", pa.int32()),
            pa.field("acl_groups", pa.list_(pa.string())),
            pa.field("category", pa.string()),
            pa.field("chunk_size_tier", pa.string()),
            pa.field("page", pa.int32()),
            pa.field("speaker", pa.string()),
            pa.field("utterance_type", pa.string()),
        ])

    def _ensure_table(self):
        try:
            table = self.db.open_table(self.table_name)
        except Exception:
            table = self.db.create_table(self.table_name, schema=self._build_schema())
            logger.info(f"Created LanceDB table '{self.table_name}'")
        self._ensure_indexes(table)
        return table

    def _ensure_indexes(self, table):
        """Create FTS and scalar indexes if not present."""
        try:
            existing = {idx["name"] if isinstance(idx, dict) else str(idx) for idx in table.list_indices()}
        except Exception:
            existing = set()

        try:
            if not any("fts" in name.lower() or "text" in name.lower() for name in existing):
                table.create_fts_index("text", replace=True)
                logger.info("Created FTS index on 'text'")
        except Exception as e:
            logger.warning(f"Could not create FTS index: {e}")

        for field, idx_type in [("doc_id", None), ("acl_groups", "LABEL_LIST")]:
            try:
                if not any(field in name for name in existing):
                    if idx_type:
                        table.create_scalar_index(field, index_type=idx_type)
                    else:
                        table.create_scalar_index(field)
                    logger.info(f"Created scalar index on '{field}'")
            except Exception as e:
                logger.debug(f"Index on '{field}': {e}")

    def _build_acl_filter(self, user_groups: list[str]) -> str | None:
        if "ALL" in user_groups:
            return None
        quoted = ", ".join(f"'{g}'" for g in user_groups)
        return f"array_has_any(acl_groups, make_array({quoted}))"

    def _build_filter(self, user_groups: list[str], tier: str | None = None, doc_ids: list[str] | None = None) -> str | None:
        """Build combined ACL + tier + doc_id filter."""
        parts = []
        acl = self._build_acl_filter(user_groups)
        if acl:
            parts.append(acl)
        if tier:
            parts.append(f"chunk_size_tier = '{tier}'")
        if doc_ids:
            quoted = ", ".join(f"'{d}'" for d in doc_ids)
            parts.append(f"doc_id IN ({quoted})")
        return " AND ".join(parts) if parts else None

    def _results_to_chunks(self, results: list[dict]) -> list[RetrievedChunk]:
        chunks = []
        for row in results:
            meta = ChunkMetadata(
                doc_id=row["doc_id"],
                filename=row["filename"],
                doc_type=row["doc_type"],
                chunk_index=row["chunk_index"],
                start_char=row.get("start_char", 0),
                acl_groups=row["acl_groups"],
                category=row.get("category", ""),
                page=row.get("page"),
                speaker=row.get("speaker"),
                utterance_type=row.get("utterance_type"),
            )
            # LanceDB returns _distance (lower=better) or _relevance_score
            score = row.get("_relevance_score", 0.0)
            if score == 0.0 and "_distance" in row:
                score = 1.0 / (1.0 + row["_distance"])
            chunks.append(RetrievedChunk(text=row["text"], score=score, metadata=meta))
        return chunks

    def upsert(self, texts: list[str], vectors: list[list[float]], metadatas: list[ChunkMetadata]) -> None:
        records = []
        for text, vector, meta in zip(texts, vectors, metadatas):
            record = meta.model_dump()
            record["id"] = str(uuid.uuid4())
            record["text"] = text
            record["vector"] = vector
            # Ensure nullable fields have defaults for PyArrow
            record.setdefault("page", None)
            record.setdefault("speaker", None)
            record.setdefault("utterance_type", None)
            records.append(record)
        self.table.add(records)

    def search(self, vector: list[float], user_groups: list[str], top_k: int = 10, tier: str | None = None, doc_ids: list[str] | None = None) -> list[RetrievedChunk]:
        """Semantic-only vector search."""
        query = self.table.search(vector).limit(top_k)
        combined = self._build_filter(user_groups, tier, doc_ids)
        if combined:
            query = query.where(combined)
        return self._results_to_chunks(query.to_list())

    def hybrid_search(self, vector: list[float], text_query: str, user_groups: list[str], top_k: int = 10, tier: str | None = None, doc_ids: list[str] | None = None) -> list[RetrievedChunk]:
        """Hybrid search: vector + BM25 FTS with RRF fusion."""
        from lancedb.rerankers import RRFReranker

        combined = self._build_filter(user_groups, tier, doc_ids)
        try:
            query = (
                self.table.search(query_type="hybrid")
                .vector(vector)
                .text(text_query)
                .rerank(reranker=RRFReranker())
                .limit(top_k)
            )
            if combined:
                query = query.where(combined, prefilter=True)
            return self._results_to_chunks(query.to_list())
        except Exception as e:
            logger.warning(f"Hybrid search failed, falling back to semantic: {e}")
            return self.search(vector, user_groups, top_k, tier, doc_ids)

    def hybrid_search_reranked(self, vector: list[float], text_query: str, user_groups: list[str], top_k: int = 10, tier: str | None = None, doc_ids: list[str] | None = None) -> list[RetrievedChunk]:
        """Hybrid search with CrossEncoder reranking for highest quality."""
        combined = self._build_filter(user_groups, tier, doc_ids)
        try:
            reranker = self._get_cross_encoder()
            query = (
                self.table.search(query_type="hybrid")
                .vector(vector)
                .text(text_query)
                .rerank(reranker=reranker)
                .limit(top_k)
            )
            if combined:
                query = query.where(combined, prefilter=True)
            return self._results_to_chunks(query.to_list())
        except Exception as e:
            logger.warning(f"Reranked search failed, falling back to hybrid: {e}")
            return self.hybrid_search(vector, text_query, user_groups, top_k, tier, doc_ids)

    def rerank_chunks(self, chunks, text_query, top_n, boosts=None):
        """Rerank the top_n highest-scoring non-synthetic chunks with a
        CrossEncoder so they lead (ordered by relevance), re-adding the
        doc-level feedback boost. Mutates chunk.score in place and returns
        the same list. Fail-open: any error leaves scores unchanged."""
        boosts = boosts or {}
        SYNTHETIC = {"map-reduce", "knowledge-graph", "metadata-context"}
        if not chunks or len(chunks) < 2:
            return chunks
        regular = [c for c in chunks if c.metadata.doc_id not in SYNTHETIC]
        if len(regular) < 2:
            return chunks
        try:
            model = self._get_cross_encoder_model()
        except Exception as e:
            logger.warning(f"Rerank model load failed, skipping rerank: {e}")
            return chunks

        regular.sort(key=lambda c: c.score, reverse=True)
        head = regular[:top_n]
        tail = regular[top_n:]
        try:
            raw = list(model.predict([(text_query, c.text) for c in head]))
        except Exception as e:
            logger.warning(f"Rerank predict failed, skipping rerank: {e}")
            return chunks

        lo, hi = min(raw), max(raw)
        span = (hi - lo) or 1.0
        tail_max = max((c.score for c in tail), default=0.0)
        base = tail_max + 0.01  # guarantee reranked head leads the tail
        for c, r in zip(head, raw):
            norm = (r - lo) / span  # [0, 1]
            c.score = base + norm + boosts.get(c.metadata.doc_id, 0.0)
        return chunks

    def get_chunks_by_doc(self, doc_id: str, limit: int = 200, tier: str | None = None) -> list[RetrievedChunk]:
        """Retrieve chunks for a document, optionally filtered by tier."""
        try:
            where = f"doc_id = '{doc_id}'"
            if tier:
                where += f" AND chunk_size_tier = '{tier}'"
            results = self.table.search().where(where).limit(limit).to_list()
            chunks = self._results_to_chunks(results)
            chunks.sort(key=lambda c: c.metadata.chunk_index)
            return chunks
        except Exception as e:
            logger.warning(f"get_chunks_by_doc failed: {e}")
            return []

    def expand_window(self, chunks: list[RetrievedChunk], window: int = 3) -> list[RetrievedChunk]:
        """Pull neighboring chunks from the same document and tier."""
        if not chunks:
            return chunks

        # Detect the dominant tier from input chunks to stay consistent
        tier_counts: dict[str, int] = {}
        for c in chunks:
            t = c.metadata.chunk_size_tier
            tier_counts[t] = tier_counts.get(t, 0) + 1
        dominant_tier = max(tier_counts, key=tier_counts.get) if tier_counts else "medium"

        # Collect doc_ids that need expansion
        doc_chunk_map: dict[str, set[int]] = {}
        existing = set()
        for c in chunks:
            key = (c.metadata.doc_id, c.metadata.chunk_index)
            existing.add(key)
            doc_chunk_map.setdefault(c.metadata.doc_id, set()).add(c.metadata.chunk_index)

        # Calculate needed neighbor indexes
        needed_by_doc: dict[str, set[int]] = {}
        for doc_id, indexes in doc_chunk_map.items():
            for idx in indexes:
                for offset in range(-window, window + 1):
                    neighbor = idx + offset
                    if neighbor >= 0 and (doc_id, neighbor) not in existing:
                        needed_by_doc.setdefault(doc_id, set()).add(neighbor)

        if not needed_by_doc:
            return chunks

        # Batch fetch per document
        new_chunks = []
        for doc_id, needed_indexes in needed_by_doc.items():
            idx_list = ", ".join(str(i) for i in needed_indexes)
            try:
                results = (
                    self.table.search()
                    .where(f"doc_id = '{doc_id}' AND chunk_index IN ({idx_list}) AND chunk_size_tier = '{dominant_tier}'")
                    .limit(len(needed_indexes))
                    .to_list()
                )
                for row in results:
                    meta = ChunkMetadata(
                        doc_id=row["doc_id"], filename=row["filename"],
                        doc_type=row["doc_type"], chunk_index=row["chunk_index"],
                        start_char=row.get("start_char", 0), acl_groups=row["acl_groups"],
                        category=row.get("category", ""), page=row.get("page"),
                        speaker=row.get("speaker"), utterance_type=row.get("utterance_type"),
                    )
                    new_chunks.append(RetrievedChunk(text=row["text"], score=0.4, metadata=meta))
            except Exception as e:
                logger.debug(f"Window expansion for {doc_id}: {e}")

        if new_chunks:
            logger.info(f"Window expansion: added {len(new_chunks)} neighbor chunks")

        all_chunks = chunks + new_chunks
        all_chunks.sort(key=lambda c: (c.metadata.doc_id, c.metadata.chunk_index))
        return all_chunks

    def delete_by_doc_id(self, doc_id: str) -> None:
        """Delete all chunks for a given document."""
        self.table.delete(f"doc_id = '{doc_id}'")

    def optimize(self) -> None:
        """Compact files and clean up old versions."""
        from datetime import timedelta
        self.table.compact_files()
        self.table.cleanup_old_versions(older_than=timedelta(days=1))
