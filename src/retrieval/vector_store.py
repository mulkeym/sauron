import uuid

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, FieldCondition, Filter, MatchAny, PointStruct, VectorParams

from src.config import settings
from src.retrieval.models import ChunkMetadata, RetrievedChunk


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

    def upsert(self, texts: list[str], vectors: list[list[float]], metadatas: list[ChunkMetadata]) -> None:
        points = []
        for text, vector, meta in zip(texts, vectors, metadatas):
            point_id = str(uuid.uuid4())
            payload = meta.model_dump()
            payload["text"] = text
            points.append(PointStruct(id=point_id, vector=vector, payload=payload))
        self.client.upsert(collection_name=self.collection, points=points)

    def search(self, vector: list[float], user_groups: list[str], top_k: int = 10) -> list[RetrievedChunk]:
        acl_filter = Filter(must=[FieldCondition(key="acl_groups", match=MatchAny(any=user_groups))])
        results = self.client.query_points(
            collection_name=self.collection,
            query=vector,
            query_filter=acl_filter,
            limit=top_k,
            with_payload=True,
        )
        chunks = []
        for point in results.points:
            payload = point.payload
            text = payload.pop("text", "")
            chunks.append(RetrievedChunk(text=text, score=point.score, metadata=ChunkMetadata(**payload)))
        return chunks

    def delete_by_doc_id(self, doc_id: str) -> None:
        self.client.delete(
            collection_name=self.collection,
            points_selector=Filter(must=[FieldCondition(key="doc_id", match=MatchAny(any=[doc_id]))]),
        )
