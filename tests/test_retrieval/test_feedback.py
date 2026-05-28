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
