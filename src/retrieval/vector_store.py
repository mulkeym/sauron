import logging
import uuid

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, FieldCondition, Filter, Fusion, MatchAny, MatchTextAny,
    MatchValue, PointStruct, Prefetch, TextIndexParams, TextIndexType,
    TokenizerType, VectorParams,
)

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
    def __init__(self):
        self.client = QdrantClient(host=settings.qdrant_host, port=settings.qdrant_port)
        self.collection = settings.qdrant_collection_name
        self._ensure_collection()

    def _ensure_collection(self):
        if not self.client.collection_exists(self.collection):
            vector_size = _detect_vector_size()
            self.client.create_collection(
                collection_name=self.collection,
                vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
            )
        # Ensure text index exists for hybrid search
        self._ensure_text_index()

    def _ensure_text_index(self):
        """Create payload indexes for hybrid search and window expansion."""
        try:
            info = self.client.get_collection(self.collection)
            existing_indexes = info.payload_schema or {}
            if "text" not in existing_indexes:
                self.client.create_payload_index(
                    collection_name=self.collection,
                    field_name="text",
                    field_schema=TextIndexParams(
                        type=TextIndexType.TEXT,
                        tokenizer=TokenizerType.MULTILINGUAL,
                        lowercase=True,
                        min_token_len=2,
                    ),
                )
                logger.info("Created full-text index on 'text' field")
            if "doc_id" not in existing_indexes:
                self.client.create_payload_index(
                    collection_name=self.collection,
                    field_name="doc_id",
                    field_schema="keyword",
                )
                logger.info("Created index on 'doc_id' field")
            if "chunk_index" not in existing_indexes:
                self.client.create_payload_index(
                    collection_name=self.collection,
                    field_name="chunk_index",
                    field_schema="integer",
                )
                logger.info("Created index on 'chunk_index' field")
        except Exception as e:
            logger.warning(f"Could not create indexes: {e}")

    def upsert(self, texts: list[str], vectors: list[list[float]], metadatas: list[ChunkMetadata]) -> None:
        points = []
        for text, vector, meta in zip(texts, vectors, metadatas):
            point_id = str(uuid.uuid4())
            payload = meta.model_dump()
            payload["text"] = text
            points.append(PointStruct(id=point_id, vector=vector, payload=payload))
        self.client.upsert(collection_name=self.collection, points=points)

    def _build_acl_filter(self, user_groups: list[str]) -> Filter | None:
        if "ALL" not in user_groups:
            return Filter(must=[FieldCondition(key="acl_groups", match=MatchAny(any=user_groups))])
        return None

    def _points_to_chunks(self, results) -> list[RetrievedChunk]:
        chunks = []
        for point in results.points:
            payload = dict(point.payload)
            text = payload.pop("text", "")
            chunks.append(RetrievedChunk(text=text, score=point.score, metadata=ChunkMetadata(**payload)))
        return chunks

    def search(self, vector: list[float], user_groups: list[str], top_k: int = 10) -> list[RetrievedChunk]:
        """Semantic-only vector search (original behavior)."""
        results = self.client.query_points(
            collection_name=self.collection,
            query=vector,
            query_filter=self._build_acl_filter(user_groups),
            limit=top_k,
            with_payload=True,
        )
        return self._points_to_chunks(results)

    def hybrid_search(self, vector: list[float], text_query: str, user_groups: list[str], top_k: int = 10) -> list[RetrievedChunk]:
        """Hybrid search: combines semantic vector similarity with keyword matching using Reciprocal Rank Fusion."""
        from qdrant_client.models import FusionQuery

        acl_filter = self._build_acl_filter(user_groups)

        # Build keyword filter
        keyword_conditions = [FieldCondition(key="text", match=MatchTextAny(text_any=text_query))]
        if acl_filter:
            keyword_conditions.extend(acl_filter.must)

        try:
            results = self.client.query_points(
                collection_name=self.collection,
                prefetch=[
                    # Semantic search branch
                    Prefetch(
                        query=vector,
                        limit=top_k * 3,
                        filter=acl_filter,
                    ),
                    # Keyword search branch — finds chunks containing query terms
                    Prefetch(
                        query=vector,
                        limit=top_k * 3,
                        filter=Filter(must=keyword_conditions),
                    ),
                ],
                query=FusionQuery(fusion=Fusion.RRF),
                limit=top_k,
                with_payload=True,
            )
            return self._points_to_chunks(results)
        except Exception as e:
            logger.warning(f"Hybrid search failed, falling back to semantic: {e}")
            return self.search(vector, user_groups, top_k)

    def expand_window(self, chunks: list[RetrievedChunk], window: int = 3) -> list[RetrievedChunk]:
        """For each retrieved chunk, also pull neighboring chunks from the same document.

        This solves the section-header problem: if we match a chunk with "ARMY",
        we also pull the next N chunks which contain the contractor details.
        """
        if not chunks:
            return chunks

        # Collect (doc_id, chunk_index) pairs to fetch — neighbors of each match
        needed = set()
        existing = {(c.metadata.doc_id, c.metadata.chunk_index) for c in chunks}
        for c in chunks:
            for offset in range(-window, window + 1):
                neighbor_idx = c.metadata.chunk_index + offset
                if neighbor_idx >= 0:
                    key = (c.metadata.doc_id, neighbor_idx)
                    if key not in existing:
                        needed.add(key)

        if not needed:
            return chunks

        # Fetch neighbor chunks from Qdrant by doc_id + chunk_index
        from qdrant_client.models import MatchValue
        new_chunks = []
        for doc_id, chunk_idx in needed:
            try:
                results = self.client.scroll(
                    collection_name=self.collection,
                    scroll_filter=Filter(must=[
                        FieldCondition(key="doc_id", match=MatchValue(value=doc_id)),
                        FieldCondition(key="chunk_index", match=MatchValue(value=chunk_idx)),
                    ]),
                    limit=1,
                    with_payload=True,
                )
                for point in results[0]:
                    payload = dict(point.payload)
                    text = payload.pop("text", "")
                    new_chunks.append(RetrievedChunk(
                        text=text, score=0.4,  # Lower score since these are window-expanded
                        metadata=ChunkMetadata(**payload),
                    ))
            except Exception:
                continue

        if new_chunks:
            logger.info(f"Window expansion: added {len(new_chunks)} neighbor chunks")

        # Sort all chunks by doc_id then chunk_index for coherent reading order
        all_chunks = chunks + new_chunks
        all_chunks.sort(key=lambda c: (c.metadata.doc_id, c.metadata.chunk_index))
        return all_chunks

    def delete_by_doc_id(self, doc_id: str) -> None:
        self.client.delete(
            collection_name=self.collection,
            points_selector=Filter(must=[FieldCondition(key="doc_id", match=MatchAny(any=[doc_id]))]),
        )
