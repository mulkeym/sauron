from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from src.agent import synthesizer
from src.agent.profiles import AnswerProfile, snapshot
from src.agent.quote_support import unsupported_quotes, has_source_quote
from src.retrieval.models import Citation, ChunkMetadata, RetrievedChunk


def source():
    return Citation(doc_id='d', filename='topology.vsdx', doc_type='vsdx', chunk_index=0,
                    snippet='Shape 1: Azure storage', evidence_id='Ecanonical', relevance=1)


def test_mermaid_syntax_is_not_a_citation_or_verbatim_quotation():
    state = {'question': 'draw a diagram in mermaid', 'answer_profile': snapshot('test', 1, AnswerProfile(name='Test'))}
    pack = synthesizer.EvidencePack(citations=[source()], aliases={'E1': 'Ecanonical'})
    code = '```mermaid\nflowchart TD\n A[Endpoint]\n B["Storage components"]\n C[E1]\n```'
    result = synthesizer.finalize_answer(state, 'SAURON_STATUS: answer\nSupported components [E1].\n\n' + code,
                                         pack, aliases=pack.aliases)
    assert result['response_kind'] == 'answer'
    assert code in result['answer']
    assert '[Ecanonical]' in result['answer']
    assert not has_source_quote(code)
    assert unsupported_quotes('Run `erase all` [Ecanonical]', [source()])
    assert unsupported_quotes('"Invented label" [Ecanonical]', [source()])
    assert not synthesizer.finalize_answer(state, code, pack, aliases=pack.aliases)['citations']


@pytest.mark.parametrize('modern', [False, True])
def test_native_figures_match_saved_source_and_never_promote_ocr_or_generated_text(monkeypatch, modern):
    from src.api import routes_ingest
    native = 'Shape 1: Azure storage'
    record = dict(doc_id='d', figure_id='f', source='visio_page_render', analysis_status='source_extracted',
                  description=native, source_text=native if modern else '')
    store = SimpleNamespace(list_figures=AsyncMock(return_value=[record]))
    monkeypatch.setattr(routes_ingest, 'get_metadata_store', lambda: store)
    monkeypatch.setattr(synthesizer, '_source_urls', lambda ids: {})
    prefix = 'Figure: f\nCaption: page\nSource information:\n' if modern else 'Figure: f\nVisio page: page\nSource page ID: 1\n'
    def chunk(text, index=0, doc='d'):
        return RetrievedChunk(text='Document: topology.vsdx\nSummary: Invented overview\n\n' + text, score=1,
            metadata=ChunkMetadata(doc_id=doc, filename='topology.vsdx', doc_type='vsdx', chunk_index=index,
                start_char=0, acl_groups=['team'], content_type='figure', figure_id='f'))
    chunks = [chunk(prefix+native), chunk(prefix+'Unverified',1),
              chunk('Figure: f\nOCR text (recognition may contain errors):\n'+native,2),
              chunk(prefix+native,3,'denied')]
    pack = synthesizer.build_evidence_pack({'question':'summarize diagram', 'retrieved_chunks':chunks, 'allowed_doc_ids':['d']})
    assert [c.source_kind for c in pack.citations] == ['document','derived','derived']
    assert pack.citations[0].snippet == native
    assert not unsupported_quotes('"Azure storage" [' + pack.citations[0].evidence_id + ']', pack.citations)
    assert unsupported_quotes('"Invented overview" [' + pack.citations[0].evidence_id + ']', pack.citations)
    store.list_figures.assert_awaited_once_with(['d'])


def test_missing_native_records_fail_closed(monkeypatch):
    from src.api import routes_ingest
    store = SimpleNamespace(list_figures=AsyncMock(side_effect=RuntimeError('unavailable')))
    monkeypatch.setattr(routes_ingest, 'get_metadata_store', lambda: store)
    c=RetrievedChunk(text='Shape 1: label', score=1, metadata=ChunkMetadata(doc_id='d',filename='test.vsdx',
        doc_type='vsdx',chunk_index=0,start_char=0,acl_groups=['team'],content_type='figure',figure_id='f'))
    assert synthesizer.native_figure_passages([c]) == {}


def test_component_view_uses_native_labels_without_inventing_connections():
    from src.agent.diagram_answers import native_mermaid_answer
    state = {'question': 'draw this diagram in mermaid format',
             'answer_profile': snapshot('test', 1, AnswerProfile(name='Test'))}
    c = source().model_copy(update={'figure_id':'f', 'caption':'Logical view', 'page':1,
        'snippet':'Shape 1 () in group 9: Firewall\nShape 2 (): VPN GW\nShape 3 ()\nShape 4 (): "] --> BAD["'})
    pack = synthesizer.EvidencePack(citations=[c])
    r = native_mermaid_answer(state, pack)
    assert r['response_kind'] == 'answer'
    assert 'S1["Firewall"]' in r['answer']
    assert 'S2["VPN GW"]' in r['answer']
    assert 'S3[' not in r['answer']
    assert '-->' not in r['answer']
    assert 'partial view' in r['answer']
    assert native_mermaid_answer({**state,'question':'show exact connections in mermaid'}, pack) is None
    assert native_mermaid_answer(state, synthesizer.EvidencePack(citations=[c.model_copy(update={'source_kind':'derived'})])) is None
    assert native_mermaid_answer({**state,'answer_profile':snapshot('strict',1,AnswerProfile(name='Strict',insufficient_evidence='abstain'))}, pack) is None


def test_cited_filename_and_question_punctuation_are_not_failed_source_quotes():
    c=source()
    assert not unsupported_quotes('In `topology.vsdx` [Ecanonical]', [c])
    assert unsupported_quotes('In `different.vsdx` [Ecanonical]', [c])
    assert not unsupported_quotes('Regarding "summarize the diagram."', [c], 'summarize the diagram')
    assert unsupported_quotes('"Azure storage," [Ecanonical]', [c])


def test_quote_comma_repair_requires_exact_cited_words():
    from src.agent.quote_support import repair_quotation_punctuation
    c=source()
    assert repair_quotation_punctuation('"Azure storage," [Ecanonical]',[c]) == '"Azure storage", [Ecanonical]'
    for answer in ['"azure storage," [Ecanonical]', '"Azure storage," [Ewrong]',
                   '`Azure storage,` [Ecanonical]', '"Azure storage 9," [Ecanonical]']:
        assert repair_quotation_punctuation(answer,[c]) == answer
    code='```mermaid\nA["Azure storage,"]\n```\n[Ecanonical]'
    assert repair_quotation_punctuation(code,[c]) == code


def test_diagram_summary_excludes_unrelated_graph_context(monkeypatch):
    c=RetrievedChunk(text='native',score=1,metadata=ChunkMetadata(doc_id='d',filename='diagram.vsdx',
        doc_type='vsdx',chunk_index=0,start_char=0,acl_groups=['team'],content_type='figure',figure_id='f'))
    graph=c.model_copy(update={'text':'unrelated network assertions','metadata':c.metadata.model_copy(update={
        'doc_id':'knowledge-graph','figure_id':'','content_type':'text'})})
    monkeypatch.setattr(synthesizer,'native_figure_passages',lambda chunks, **kw:{('d','f','native'):'Shape 1: Firewall'})
    monkeypatch.setattr(synthesizer,'_source_urls',lambda ids:{})
    pack=synthesizer.build_evidence_pack({'question':'summarize the diagram','retrieved_chunks':[c,graph]})
    assert 'unrelated network assertions' not in pack.context
    assert [x.doc_id for x in pack.citations] == ['d']


def test_mermaid_output_format_is_not_a_retrieval_topic():
    from src.agent.strategies.technical import retrieval_subject, retrieval_question
    original='draw me the zero trust diagram in mermaid format'
    assert retrieval_subject(original) == 'the zero trust diagram'
    assert retrieval_question({'question':original}) == 'the zero trust diagram'
    assert retrieval_subject('summarize the zero trust diagram') == 'summarize the zero trust diagram'
    assert retrieval_subject('show Mermaid syntax examples') == 'Mermaid syntax examples'


def test_mermaid_does_not_choose_template_resource_page_over_architecture():
    from src.agent.diagram_answers import native_mermaid_answer
    state={'question':'draw zero trust in mermaid format','answer_profile':snapshot('test',1,AnswerProfile(name='Test'))}
    original=source().model_copy(update={'figure_id':'architecture','caption':'Spoke VNets','snippet':'Shape 1 (): VM','relevance':.8})
    font=original.model_copy(update={'figure_id':'font','caption':'Resources: Updating Font','relevance':1.,'snippet':'Shape 2 (): Font'})
    r=native_mermaid_answer(state,synthesizer.EvidencePack(citations=[font,original]))
    assert 'Spoke VNets' in r['answer']
    assert 'Font' not in r['answer']


def test_complete_page_expansion_requires_a_verified_native_chunk(monkeypatch):
    from src.api import routes_ingest
    text='Shape 1 (): Firewall\nShape 2 (): VPN GW'
    store=SimpleNamespace(list_figures=AsyncMock(return_value=[{'doc_id':'d','figure_id':'f',
        'source':'visio_page_render','analysis_status':'source_extracted','description':text}]))
    monkeypatch.setattr(routes_ingest,'get_metadata_store',lambda:store)
    c=RetrievedChunk(text='Figure: f\nSource page ID: 1\nShape 1 (): Firewall',score=1,
        metadata=ChunkMetadata(doc_id='d',filename='test.vsdx',doc_type='vsdx',chunk_index=0,
            start_char=0,acl_groups=['team'],content_type='figure',figure_id='f'))
    assert list(synthesizer.native_figure_passages([c],complete=True).values()) == [text]
    bad=c.model_copy(update={'text':c.text.replace('Firewall','Forged')})
    assert synthesizer.native_figure_passages([bad],complete=True) == {}


def test_component_view_prefers_saved_group_labels_over_page_titles():
    from src.agent.diagram_answers import native_mermaid_answer
    state={'question':'draw in mermaid','answer_profile':snapshot('test',1,AnswerProfile(name='Test'))}
    c=source().model_copy(update={'figure_id':'f','snippet':'Shape 1 (): Heading\n'+ '\n'.join(
        f'Shape {i+2} () in group 9: {label}' for i,label in enumerate(['Firewall','VPN GW','User','Admin','Firewall']))})
    r=native_mermaid_answer(state,synthesizer.EvidencePack(citations=[c]))
    assert 'Heading' not in r['answer']
    assert r['answer'].count('["Firewall"]') == 1
    assert '["VPN GW"]' in r['answer']
