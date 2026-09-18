import pytest
from unittest.mock import MagicMock, patch

from src.retrieval.vector_store import VectorStore
from src.retrieval.models import ChunkMetadata


@pytest.fixture
def vector_store(tmp_path, monkeypatch):
    from src.config import settings
    monkeypatch.setattr(settings, "lancedb_path", str(tmp_path / "lance"))
    monkeypatch.setattr(settings, "embedding_dimension", 3)
    store = VectorStore()
    meta = ChunkMetadata(doc_id="doc-1", filename="test.pdf", doc_type="pdf", chunk_index=0,
                         start_char=0, acl_groups=["finance"])
    store.upsert(["hello world"], [[0.1, 0.2, 0.3]], [meta])
    return store


def test_upsert_chunks(vector_store):
    assert vector_store.table.count_rows() == 1


def test_search_enforces_empty_and_nonmatching_scope(vector_store):
    for groups, ids in [([], None), (["engineering"], None), (["ALL"], [])]:
        assert vector_store.search([0.1, 0.2, 0.3], groups, doc_ids=ids) == []
    assert len(vector_store.search([0.1, 0.2, 0.3], ["finance"])) == 1


def test_filter_quotes_cannot_expand_scope(vector_store):
    assert vector_store.search([0.1, 0.2, 0.3], ["finance' OR true --"]) == []
    assert vector_store.search([0.1, 0.2, 0.3], ["ALL"], doc_ids=["doc-1' OR true --"]) == []


def test_delete_by_doc_id(vector_store):
    vector_store.delete_by_doc_id("doc-1")
    assert vector_store.table.count_rows() == 0


# ---------------------------------------------------------------------------
# rerank_chunks tests
# ---------------------------------------------------------------------------
from src.retrieval.models import RetrievedChunk


def _chunk(doc_id, idx, score, text):
    return RetrievedChunk(
        text=text, score=score,
        metadata=ChunkMetadata(
            doc_id=doc_id, filename=f"{doc_id}.txt", doc_type="text",
            chunk_index=idx, start_char=0, acl_groups=["ALL"],
        ),
    )


class _FakeCE:
    """Fake CrossEncoder: score = 1.0 if 'match' in text else 0.0."""
    def predict(self, pairs):
        return [1.0 if "match" in text else 0.0 for _q, text in pairs]


def test_rerank_chunks_reorders_by_crossencoder(monkeypatch):
    from src.retrieval.vector_store import VectorStore
    vs = VectorStore.__new__(VectorStore)  # no DB init needed
    chunks = [
        _chunk("d1", 0, 0.9, "irrelevant text"),
        _chunk("d2", 1, 0.1, "this is a match"),
    ]
    with patch.object(VectorStore, "_get_cross_encoder_model", return_value=_FakeCE()):
        out = vs.rerank_chunks(chunks, "find the match", top_n=50, boosts=None)
    by_id = {c.metadata.doc_id: c.score for c in out}
    assert by_id["d2"] > by_id["d1"]


def test_rerank_chunks_applies_feedback_boost(monkeypatch):
    from src.retrieval.vector_store import VectorStore
    vs = VectorStore.__new__(VectorStore)
    chunks = [_chunk("d1", 0, 0.5, "a match"), _chunk("d2", 1, 0.5, "another match")]
    with patch.object(VectorStore, "_get_cross_encoder_model", return_value=_FakeCE()):
        out = vs.rerank_chunks(chunks, "match", top_n=50, boosts={"d2": 0.5})
    by_id = {c.metadata.doc_id: c.score for c in out}
    assert by_id["d2"] > by_id["d1"]  # equal CE score, d2 wins on boost


def test_rerank_chunks_skips_synthetic(monkeypatch):
    from src.retrieval.vector_store import VectorStore
    vs = VectorStore.__new__(VectorStore)
    synth = _chunk("map-reduce", 0, 0.42, "extracted data")
    reg1 = _chunk("d1", 0, 0.3, "a match")
    reg2 = _chunk("d2", 1, 0.2, "another match")
    original_reg_scores = (reg1.score, reg2.score)
    # 2 regular chunks → scoring loop runs; synthetic is present but excluded
    with patch.object(VectorStore, "_get_cross_encoder_model", return_value=_FakeCE()):
        vs.rerank_chunks([synth, reg1, reg2], "match", top_n=50, boosts=None)
    assert synth.score == 0.42  # synthetic chunk score untouched by reranking
    # Regular chunks must have been rescored (scores changed from originals)
    assert reg1.score != original_reg_scores[0] or reg2.score != original_reg_scores[1]


def test_rerank_chunks_failopen_on_predict_error(monkeypatch):
    from src.retrieval.vector_store import VectorStore
    vs = VectorStore.__new__(VectorStore)

    class _RaisingCE:
        """Fake CrossEncoder whose predict always raises."""
        def predict(self, pairs):
            raise RuntimeError("predict exploded")

    chunks = [_chunk("d1", 0, 0.9, "a match"), _chunk("d2", 1, 0.4, "another match")]
    original_scores = [c.score for c in chunks]
    with patch.object(VectorStore, "_get_cross_encoder_model", return_value=_RaisingCE()):
        out = vs.rerank_chunks(chunks, "match", top_n=50, boosts=None)
    assert [c.score for c in out] == original_scores  # scores unchanged on predict error


def test_rerank_chunks_failopen_on_model_error(monkeypatch):
    from src.retrieval.vector_store import VectorStore
    vs = VectorStore.__new__(VectorStore)
    chunks = [_chunk("d1", 0, 0.9, "x"), _chunk("d2", 1, 0.1, "y")]
    with patch.object(VectorStore, "_get_cross_encoder_model", side_effect=RuntimeError("boom")):
        out = vs.rerank_chunks(chunks, "q", top_n=50, boosts=None)
    assert [c.score for c in out] == [0.9, 0.1]  # unchanged
