import pytest

from src.citations import citation_label, citation_markdown, citation_url, render_citations, CitationStream
from src.retrieval.models import Citation


def source(**changes):
    return Citation(**{
        'doc_id': 'd1', 'filename': 'sdwan-for-gov.pdf', 'doc_type': 'pdf',
        'chunk_index': 12, 'page': 7, 'section_title': 'Security and Compliance',
        'snippet': 'FIPS mode is enabled.', 'relevance': .9,
        'evidence_id': 'E6c6a058d49b3', **changes,
    })


def test_filename_first_repeated_refs_link_to_exact_local_passage_without_mutation():
    c = source()
    original = c.model_dump()
    answer = 'FIPS [E6c6a058d49b3]. Encryption [E6c6a058d49b3].'
    rendered = render_citations(answer, [c], local_links=True)
    assert rendered.count('[sdwan-for-gov.pdf — page 7, Security and Compliance](#citation-1)') == 2
    assert c.model_dump() == original
    assert '[E6c6a058d49b3]' in answer
    assert render_citations(rendered, [c], local_links=True) == rendered


def test_no_invented_urls_and_fallback_descriptors():
    assert citation_url(source()) == ''
    assert citation_label(source(page=None, section_title=None)) == 'sdwan-for-gov.pdf — passage 13'
    assert 'derived summary' in citation_label(source(source_kind='derived'))
    assert 'query result' in citation_label(source(source_kind='query_result', page=None, section_title=None))
    assert 'slide 3, diagram' in citation_label(source(page=None, slide=3, figure_id='hidden-hash', section_title=None))


def test_distinct_documents_and_passages_get_distinct_targets():
    a, b = source(), source(doc_id='d2', page=8, evidence_id='Eother')
    rendered = render_citations('First [E6c6a058d49b3], second [Eother].', [a,b], local_links=True)
    assert '#citation-1' in rendered and '#citation-2' in rendered
    assert 'page 7' in rendered and 'page 8' in rendered


@pytest.mark.parametrize('url', ['javascript:alert(1)', 'data:text/html,bad', 'file:///tmp/a.pdf', '//example.com', 'https://user:password@docs.test/a.pdf', 'https://docs.test/\nx', 'https://docs.test/\\evil'])
def test_unsafe_source_urls_never_become_links(url):
    c = source(source_url=url)
    assert citation_url(c) == ''
    assert '](' not in citation_markdown(c)


def test_source_url_pdf_page_and_markdown_escaping():
    c = source(filename='[bad](x)<img>.pdf', source_url='https://docs.test/a(b).pdf')
    assert citation_url(c) == 'https://docs.test/a%28b%29.pdf#page=7'
    assert '<img>' not in citation_markdown(c)
    assert r'\[bad\]' in citation_markdown(c)
    assert citation_url(source(source_url='https://docs.test/a.pdf#section')) == 'https://docs.test/a.pdf#section'


def test_literal_code_existing_links_unknown_ids_and_ordinals():
    c = source()
    literal = '`[E6c6a058d49b3]`\n```text\n[E6c6a058d49b3]\n```\n[Example](https://example.com) [Eunknown]'
    assert render_citations(literal, [c]) == literal
    assert render_citations('[1]', [source(evidence_id='')]).startswith('[sdwan-for-gov.pdf')
    assert render_citations('[1]', [c]) == '[1]'  # Never reinterpret arbitrary numbers as canonical IDs.


def test_stream_citation_split_at_every_character_and_literal_fences():
    c = source()
    answer = 'FIPS [E1].\n```text\n[E1]\n```\nMore [E1].'
    expected = render_citations(answer, [c], aliases={'E1': c.evidence_id})
    for split in range(len(answer)+1):
        stream = CitationStream([c], aliases={'E1':c.evidence_id})
        assert stream.feed(answer[:split]) + stream.feed(answer[split:]) + stream.feed('', final=True) == expected
