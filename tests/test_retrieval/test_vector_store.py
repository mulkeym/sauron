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


def test_read_citation_candidates_exact_location_and_acl(vector_store):
    args = dict(chunk_index=0, chunk_size_tier='medium', start_char=0)
    assert [c.text for c in vector_store.read_citation_candidates('doc-1', ['finance'], **args)] == ['hello world']
    assert vector_store.read_citation_candidates('doc-1', ['other'], **args) == []
    assert vector_store.read_citation_candidates('doc-1', [], **args) == []
    assert vector_store.read_citation_candidates('doc-1', ['finance'], **{**args, 'chunk_index': 1}) == []
    assert vector_store.read_citation_candidates('doc-1', ['finance'], **{**args, 'start_char': 1}) == []
    assert vector_store.read_citation_candidates('doc-1', ['finance'], **{**args, 'chunk_size_tier': "medium' OR true --"}) == []


def test_document_acl_sync_updates_all_tiers_and_preserves_content(vector_store):
    for i, tier in enumerate(['small', 'large', 'table_row']):
        meta = ChunkMetadata(doc_id='doc-1', filename='test.pdf', doc_type='pdf',
                             chunk_index=i + 1, start_char=0, acl_groups=['finance'],
                             chunk_size_tier=tier)
        vector_store.upsert([f'passage {tier}'], [[.1, .2, .3]], [meta])
    other = ChunkMetadata(doc_id='other', filename='other.pdf', doc_type='pdf',
                         chunk_index=0, start_char=0, acl_groups=['finance'])
    vector_store.upsert(['other passage'], [[.1, .2, .3]], [other])
    before = vector_store.table.search().where("doc_id = 'doc-1'").limit(100).to_list()
    assert vector_store.synchronize_document_acl('doc-1', ['engineering'])
    after = vector_store.table.search().where("doc_id = 'doc-1'").limit(100).to_list()
    assert len(after) == 4 and all(r['acl_groups'] == ['engineering'] for r in after)
    assert {r['id']: (r['text'], r['vector']) for r in before} == {r['id']: (r['text'], r['vector']) for r in after}
    assert not vector_store.search([.1, .2, .3], ['finance'], doc_ids=['doc-1'])
    assert len(vector_store.search([.1, .2, .3], ['engineering'], doc_ids=['doc-1'])) == 4
    assert len(vector_store.search([.1, .2, .3], ['finance'], doc_ids=['other'])) == 1
    version = vector_store.table.version
    assert not vector_store.synchronize_document_acl('doc-1', ['engineering', 'engineering'])
    assert vector_store.table.version == version
    assert not vector_store.synchronize_document_acl("doc-1' OR true --", ['bad'])
    assert vector_store.synchronize_document_acl('doc-1', [])
    assert not vector_store.search([.1, .2, .3], ['engineering'], doc_ids=['doc-1'])


def test_document_acl_sync_detects_mismatch_past_first_batch(vector_store):
    metadata = [ChunkMetadata(doc_id='doc-1', filename='test.pdf', doc_type='pdf',
                             chunk_index=i + 1, start_char=0,
                             acl_groups=['engineering'] if i == 1000 else ['finance'])
                for i in range(1001)]
    vector_store.upsert(['passage'] * 1001, [[.1, .2, .3]] * 1001, metadata)
    assert vector_store.synchronize_document_acl('doc-1', ['finance'])
    assert not vector_store.synchronize_document_acl('doc-1', ['finance'])


def test_figure_source_expansion_reads_original_pages_with_acl_scope_and_budget(vector_store):
    from src.retrieval.models import RetrievedChunk
    def meta(doc='doc-1',page=39,kind='text',groups=None,tier='medium',index=1):
        return ChunkMetadata(doc_id=doc,filename='guide.pdf',doc_type='pdf',chunk_index=index,
            start_char=0,acl_groups=groups or ['finance'],page=page,content_type=kind,chunk_size_tier=tier,
            figure_id='f' if kind=='figure' else '')
    rows=[('applicability',meta()),('restrictions',meta(page=40,index=2)),
          ('commands',meta(page=41,index=3)),('far away',meta(page=47,index=4)),
          ('denied',meta(groups=['private'],index=5)),('other edition',meta(doc='other',index=6)),
          ('generated',meta(kind='figure',index=7)),('summary',meta(tier='summary',index=8)),
          ('prior',meta(page=38,index=9)),('Section: footer\n\n35',meta(index=10))]
    vector_store.upsert([t for t,m in rows],[[.1,.2,.3]]*len(rows),[m for t,m in rows])
    anchor=RetrievedChunk(text='generated MPLS overview',score=.8,metadata=meta(kind='figure',index=99))
    output=vector_store.expand_figure_source_pages([anchor],['finance'],['doc-1'])
    assert {c.text for c in output} == {'generated MPLS overview','applicability','restrictions','commands','prior'}
    assert len(vector_store.expand_figure_source_pages(output,['finance'],['doc-1'])) == len(output)
    assert vector_store.expand_figure_source_pages([anchor],['finance'],[]) == []
    assert len(vector_store.expand_figure_source_pages([anchor],['private'],['doc-1'])) == 2  # only that group's row
    limited=vector_store.expand_figure_source_pages([anchor],['finance'],['doc-1'],max_chars=len('applicability'))
    assert [c.text for c in limited] == ['generated MPLS overview','applicability']
    forward=vector_store.expand_figure_source_pages([anchor],['finance'],['doc-1'],max_chars=len('applicabilityrestrictions'))
    assert [c.text for c in forward] == ['generated MPLS overview','applicability','restrictions']
