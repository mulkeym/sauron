from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from src.agent.state import chunk_key
from src.agent.synthesizer import _evidence_id
from src.retrieval.models import ChunkMetadata, RetrievedChunk
from src.sources.passages import cited_passage


@pytest.fixture
def setup():
    chunk = RetrievedChunk(text='The exact cited passage.', score=1, metadata=ChunkMetadata(
        doc_id='doc-1', filename='guide.pdf', doc_type='pdf', chunk_index=126,
        start_char=180076, acl_groups=['team']))
    doc = SimpleNamespace(filename='guide.pdf', acl_groups=['team'], content_hash='a' * 64, dataset_id=None)
    store = SimpleNamespace(get_document=AsyncMock(return_value=doc), get_dataset=AsyncMock())
    vector = MagicMock()
    vector.read_citation_candidates.return_value = [chunk]
    return chunk, doc, store, vector


async def read(setup, **changes):
    chunk, doc, store, vector = setup
    args = dict(doc_id='doc-1', revision='a' * 64, evidence_id=_evidence_id(repr(chunk_key(chunk)), chunk.text),
                chunk_index=126, chunk_size_tier='medium', start_char=180076,
                groups=['team'], metadata_store=store, vector_store=vector)
    args.update(changes)
    return await cited_passage(**args)


@pytest.mark.asyncio
async def test_exact_passage(setup):
    result = await read(setup)
    assert result['available'] and result['snippet'] == setup[0].text
    setup[3].read_citation_candidates.assert_called_once_with('doc-1', ['team'], chunk_index=126, chunk_size_tier='medium', start_char=180076)


@pytest.mark.asyncio
@pytest.mark.parametrize('groups', [[], ['ALL'], ['other']])
async def test_denied_before_index_read(setup, groups):
    assert (await read(setup, groups=groups))['status'] == 404
    setup[3].read_citation_candidates.assert_not_called()


@pytest.mark.asyncio
async def test_revision_and_hash_cannot_substitute_passage(setup):
    assert (await read(setup, revision='b' * 64))['status'] == 409
    setup[3].read_citation_candidates.assert_not_called()
    assert (await read(setup, evidence_id='E' + '0' * 12))['status'] == 404


@pytest.mark.asyncio
async def test_disabled_dataset(setup):
    setup[1].dataset_id = 1
    setup[2].get_dataset.return_value = SimpleNamespace(active=False)
    assert (await read(setup))['status'] == 404
    setup[3].read_citation_candidates.assert_not_called()


@pytest.mark.asyncio
async def test_revoked_during_read(setup):
    def revoke(*args, **kwargs):
        setup[1].acl_groups = ['other']
        return [setup[0]]
    setup[3].read_citation_candidates.side_effect = revoke
    assert (await read(setup))['status'] == 404
