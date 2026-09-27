from src.retrieval.models import RetrievedChunk, ChunkMetadata


def _chunk(doc_id, idx, score):
    return RetrievedChunk(
        text="t", score=score,
        metadata=ChunkMetadata(doc_id=doc_id, filename="f", doc_type="text",
                               chunk_index=idx, start_char=0, acl_groups=["ALL"]),
    )


def test_apply_boosts_adds_and_resorts():
    from src.retrieval.feedback import apply_feedback_boosts_to_chunks
    chunks = [_chunk("d1", 0, 0.5), _chunk("d2", 1, 0.4)]
    out = apply_feedback_boosts_to_chunks(chunks, {"d2": 0.3})
    assert out[0].metadata.doc_id == "d2"  # 0.4 + 0.3 = 0.7 > 0.5
    assert abs(out[0].score - 0.7) < 1e-9


def test_apply_boosts_empty_is_noop():
    from src.retrieval.feedback import apply_feedback_boosts_to_chunks
    chunks = [_chunk("d1", 0, 0.5), _chunk("d2", 1, 0.9)]
    out = apply_feedback_boosts_to_chunks(chunks, {})
    assert [c.metadata.doc_id for c in out] == ["d1", "d2"]  # order untouched


def test_get_feedback_boosts_sync_failopen(monkeypatch):
    from src.retrieval import feedback
    async def boom(*a, **k):
        raise RuntimeError("db down")
    monkeypatch.setattr(feedback, "get_feedback_boosts", boom)
    # Sync wrapper must swallow the error and return {}
    assert feedback.get_feedback_boosts_sync([0.1, 0.2], ["ALL"]) == {}


def test_cosine_similarity_ignores_malformed_stored_vectors():
    import numpy as np
    from src.retrieval.feedback import _cosine_similarity, _deserialize_vector, _serialize_vector
    query = np.ones(768, dtype=np.float32)
    # Rows written from a missing (None) vector hold a single NaN (4 bytes).
    legacy = _deserialize_vector(np.array(None, dtype=np.float32).tobytes())
    assert _cosine_similarity(query, legacy) == 0.0
    assert _cosine_similarity(query, np.ones(384, dtype=np.float32)) == 0.0
    assert _cosine_similarity(query, _deserialize_vector(_serialize_vector([1.0] * 768))) > 0.99


def test_feedback_vector_embeds_question_when_missing(monkeypatch):
    import asyncio
    import numpy as np
    from src.ingestion import embedder
    from src.retrieval import feedback
    calls = []
    def fake_embed(text):
        calls.append(text)
        return [0.5] * 768
    monkeypatch.setattr(embedder, "embed_query", fake_embed)
    blob = asyncio.run(feedback._query_vector_blob("describe sdwan", None))
    assert calls == ["describe sdwan"]
    assert np.frombuffer(blob, dtype=np.float32).size == 768
    # A supplied vector is used as-is, without re-embedding.
    blob = asyncio.run(feedback._query_vector_blob("q", [0.1, 0.2]))
    assert calls == ["describe sdwan"] and len(blob) == 8
    # No usable vector at all: nothing is stored.
    monkeypatch.setattr(embedder, "embed_query", lambda text: [])
    assert asyncio.run(feedback._query_vector_blob("q", None)) is None
