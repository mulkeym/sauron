from copy import deepcopy
from unittest.mock import patch

import pytest
from lxml import etree as ET

from src.ingestion.visio_text import source_layout, repair_text, _wrap, V, S


def test_saved_box_styles_and_unsupported_transforms():
    shape = ET.fromstring(f'<Shape xmlns="{V}" Type="Shape" TextStyle="2"><Cell N="Width" V="2"/><Cell N="Height" V="1"/><Cell N="PinX" V="3"/><Cell N="PinY" V="4"/></Shape>')
    styles = {'2': ET.fromstring(f'<StyleSheet xmlns="{V}" TextStyle="0"><Cell N="LeftMargin" V="0.1"/><Cell N="VerticalAlign" V="0"/></StyleSheet>'),
              '0': ET.fromstring(f'<StyleSheet xmlns="{V}"><Section N="Paragraph"><Row IX="0"><Cell N="HorzAlign" V="2"/><Cell N="SpLine" V="-1.5"/></Row></Section></StyleSheet>')}
    result = source_layout(shape, None, styles, '0', '', 10)
    assert result['left'] == 144 and result['top'] == 396
    assert result['width'] == 144 and result['height'] == 72
    assert result['align'] == 2 and result['line_spacing'] == -1.5
    assert result['margins'][0] == pytest.approx(7.2)
    assert source_layout(shape, None, styles, '0', 'parent-group', 10) is None
    ET.SubElement(shape, f'{{{V}}}Cell', N='Angle', V='1.57')
    assert source_layout(shape, None, styles, '0', '', 10) is None


def fixture():
    svg = ET.fromstring(f'<svg xmlns="{S}"><image x="5" y="4" width="12" height="12" href="#icon"/><text x="10" y="20"><tspan font-size="10" fill="blue">BRANCH A</tspan><tspan font-size="10" fill="blue">Router</tspan></text><path d="M0 0 L100 0"/></svg>')
    shape = {'text': 'BRANCH A\nRouter', 'text_layout': {'left': 10, 'top': 20, 'width': 60, 'height': 40, 'margins': [0,0,0,0], 'align': 1, 'vertical': 0, 'line_spacing': -1.2}}
    return svg, [{'shapes': [shape]}]


class Font:
    def getlength(self, text): return len(text)*20
    def getmetrics(self): return (32, 8)
    def getname(self): return ('Synthetic Font', 'Regular')


def test_line_breaks_wrap_alignment_and_image_unchanged():
    svg, pages = fixture()
    image, path = ET.tostring(svg[0]), ET.tostring(svg[2])
    source = deepcopy(pages)
    with patch('src.ingestion.visio_text._font', return_value=Font()): warnings = repair_text(svg, pages)
    assert warnings and '1 label' in warnings[0]
    node = svg[1]
    assert node.get('text-anchor') == 'start'
    assert [s.text for s in node] == ['BRANCH A', 'Router']
    assert [float(s.get('y')) for s in node] == [28, 40]
    assert [float(s.get('x')) for s in node] == [20, 25]
    assert ET.tostring(svg[0]) == image and ET.tostring(svg[2]) == path and pages == source


def test_automatic_wrapping_does_not_lose_words():
    lines = _wrap('BRANCH A\nVirtual Edge Router\n\nidentifier12345', 40, lambda t: len(t)*5)
    assert lines[:2] == ['BRANCH A', 'Virtual']
    assert '' in lines
    assert ''.join(''.join(lines).split()) == ''.join('BRANCH A Virtual Edge Router identifier12345'.split())
    assert all(len(line)*5 <= 40 for line in lines)


@pytest.mark.parametrize('change', ['rotated', 'duplicate', 'scaled', 'positioned'])
def test_unsupported_layout_preserves_original(change):
    svg, pages = fixture()
    if change == 'rotated': svg[1].set('transform', 'rotate(45)')
    elif change == 'duplicate': pages[0]['shapes'].append(deepcopy(pages[0]['shapes'][0]))
    elif change == 'scaled': svg[1].set('x', '100')
    else: svg[1][1].set('y', '40')
    before = ET.tostring(svg)
    with patch('src.ingestion.visio_text._font', return_value=Font()): repair_text(svg, pages)
    assert ET.tostring(svg) == before


def test_repeated_labels_use_exact_saved_origin():
    svg, pages = fixture()
    second = deepcopy(pages[0]['shapes'][0]); second['text_layout']['left'] = 200
    pages[0]['shapes'].append(second)
    with patch('src.ingestion.visio_text._font', return_value=Font()): warnings = repair_text(svg, pages)
    assert '1 label' in warnings[0]
    assert svg[1][1].get('y') == '40.0000'


def test_height_overflow_preserves_text_with_warning():
    svg, pages = fixture(); pages[0]['shapes'][0]['text_layout']['height'] = 10
    with patch('src.ingestion.visio_text._font', return_value=Font()): warnings = repair_text(svg, pages)
    assert any('exceed' in w for w in warnings)
    assert [s.text for s in svg[1]] == ['BRANCH A', 'Router']


def test_rich_text_styles_are_preserved():
    svg, pages = fixture(); svg[1][1].set('fill', 'red')
    with patch('src.ingestion.visio_text._font', return_value=Font()): repair_text(svg, pages)
    assert [s.get('fill') for s in svg[1]] == ['blue', 'red']
    assert [s.text for s in svg[1]] == ['BRANCH A', 'Router']
    assert float(svg[1][1].get('y')) > float(svg[1][0].get('y'))


def test_scaled_drawing_uses_cached_inherited_coordinates():
    from src.ingestion.visio_text import group_origin
    shape = ET.fromstring(f'<Shape xmlns="{V}" Type="Shape"><Cell N="PinX" V="24" F="Inh"/><Cell N="PinY" V="48" F="Inh"/><Cell N="Width" V="12" F="Inh"/><Cell N="Height" V="12" F="Inh"/></Shape>')
    base = ET.fromstring(f'<Shape xmlns="{V}"><Cell N="PinX" V="2"/><Cell N="PinY" V="4"/><Cell N="Width" V="1"/><Cell N="Height" V="1"/></Shape>')
    result = source_layout(shape, base, {}, None, 'group', 120, (12, 12), 1/12)
    assert result['left'] == 180 and result['top'] == 324
    assert result['width'] == 72 and result['height'] == 72
    assert group_origin(shape, base, (12, 12)) == (30, 54)
    ET.SubElement(shape, f'{{{V}}}Cell', N='FlipX', V='1')
    assert group_origin(shape, base, (12, 12)) is None


def test_repeated_center_aligned_labels_match_containing_source_box():
    svg, pages = fixture()
    svg[1].set('y', '40') # converter puts a vertically centered label halfway down
    second = deepcopy(pages[0]['shapes'][0]); second['text_layout']['top'] = 100
    pages[0]['shapes'].append(second)
    with patch('src.ingestion.visio_text._font', return_value=Font()): warnings = repair_text(svg, pages)
    assert '1 label' in warnings[0]
    assert float(svg[1][0].get('y')) == 28


def test_adjacent_overlapping_boxes_narrow_without_moving_centers():
    from src.ingestion.visio_text import _separate_label_boxes
    _, pages = fixture()
    first = pages[0]['shapes'][0]; second = deepcopy(first)
    second['text_layout']['left'] = 50
    pages[0]['shapes'].append(second)
    widths = _separate_label_boxes(pages)
    assert widths[id(first)] == widths[id(second)] == 38
    # Original text-box geometry remains source evidence, not overwritten.
    assert first['text_layout']['width'] == second['text_layout']['width'] == 60


def test_rich_run_mapping_retains_character_content_and_breaks():
    from src.ingestion.visio_text import _source_runs
    svg, _ = fixture()
    source = 'BRANCH A\nRouter'
    mapped = _source_runs(source, list(svg[1]))
    assert ''.join(c for c, _ in mapped) == source
    assert mapped[0][1] == 0 and mapped[-1][1] == 1
    with pytest.raises(ValueError): _source_runs('A different label', list(svg[1]))


class ScaledFont(Font):
    def __init__(self, size): self.size = size
    def getlength(self, text): return len(text) * self.size * 2
    def getmetrics(self): return (self.size * 3.2, self.size * .8)


@pytest.mark.parametrize('height,minimum,expected_overflow', [(16, .5, False), (5, .25, True), (16, 1, True)])
def test_font_fitting_respects_bounds_and_preserves_source(height, minimum, expected_overflow):
    from src.config import settings
    svg, pages = fixture()
    pages[0]['shapes'][0]['text_layout']['height'] = height
    source = deepcopy(pages)
    with patch('src.ingestion.visio_text._font', side_effect=lambda family, bold, italic, size: ScaledFont(size)), patch.object(settings, 'visio_text_min_scale', minimum):
        warnings = repair_text(svg, pages)
    spans = list(svg[1])
    sizes = [float(s.get('font-size')) for s in spans]
    assert all(max(4, 10*minimum) <= size <= 10 for size in sizes)
    assert ''.join(''.join(s.text for s in spans).split()) == 'BRANCHARouter'
    assert pages == source
    assert any('still exceed' in warning for warning in warnings) == expected_overflow
    if not expected_overflow:
        assert max(float(s.get('y')) + size*.2 for s, size in zip(spans, sizes)) <= 20+height+.001


def test_missing_font_metrics_preserves_native_text():
    svg, pages = fixture(); before = ET.tostring(svg)
    with patch('src.ingestion.visio_text._font', side_effect=OSError('font unavailable')):
        warnings = repair_text(svg, pages)
    assert ET.tostring(svg) == before
    assert any('retain the native converter output' in w for w in warnings)


@pytest.mark.parametrize('label', ['SPINE', 'VTEP', 'gyp'])
def test_compact_centered_badges_use_visible_glyph_center(label):
    class BadgeFont(ScaledFont):
        def getbbox(self, text, anchor):
            assert anchor == 'ls'
            # Distinguish uppercase from labels with visible descenders.
            return (0, -self.size*2.8, self.getlength(text), self.size*.8 if text=='gyp' else 0)
    svg, pages = fixture()
    node = svg[1]
    for span in list(node): node.remove(span)
    ET.SubElement(node, f'{{{S}}}tspan', {'font-size':'7', 'fill':'green'}).text = label
    node.set('x', '14'); node.set('y','24')
    shape = pages[0]['shapes'][0]; shape['text'] = label
    shape['text_layout'].update(width=31, height=8, margins=[4,4,4,4], vertical=1)
    source = deepcopy(pages)
    with patch('src.ingestion.visio_text._font', side_effect=lambda family,bold,italic,size: BadgeFont(size)):
        warnings = repair_text(svg, pages)
    spans = list(svg[1]); assert len(spans)==1
    span = spans[0]; size = float(span.get('font-size')); baseline = float(span.get('y'))
    bounds = BadgeFont(size).getbbox(label, 'ls')
    assert baseline+(bounds[1]+bounds[3])/8 == pytest.approx(24, abs=.001)
    assert 20 <= baseline+bounds[1]/4 < baseline+bounds[3]/4 <= 28
    assert span.text == label and span.get('fill') == 'green'
    assert pages == source and any('Centered 1 compact' in w for w in warnings)


def test_compact_badges_do_not_override_top_alignment_or_unequal_padding():
    for vertical, margins in [(0,[4,4,4,4]), (1,[4,4,6,2])]:
        svg, pages = fixture()
        shape = pages[0]['shapes'][0]; shape['text'] = 'BRANCH A Router'
        shape['text_layout'].update(height=8, margins=margins, vertical=vertical)
        svg[1].set('x','14')
        before = ET.tostring(svg)
        with patch('src.ingestion.visio_text._font', return_value=Font()): repair_text(svg,pages)
        assert ET.tostring(svg) == before


def poster_shape(text_xml='<pp IX="0"/>First paragraph\n<pp IX="1"/>First item\nSecond item'):
    return ET.fromstring(f'''<Shape xmlns="{V}" Type="Shape" TextStyle="0">
        <Cell N="Width" V="3"/><Cell N="Height" V="2"/>
        <Cell N="PinX" V="2"/><Cell N="PinY" V="3"/>
        <Section N="Paragraph"><Row IX="1"><Cell N="Bullet" V="1"/>
        <Cell N="IndFirst" V="-0.25"/><Cell N="IndLeft" V="0.25"/></Row></Section>
        <Text>{text_xml}</Text></Shape>''')


def poster_styles():
    return {'0': ET.fromstring(f'''<StyleSheet xmlns="{V}"><Section N="Paragraph"><Row IX="0">
        <Cell N="SpAfter" V="0.08333333333333333"/>
        <Cell N="SpLine" V="-1.2"/><Cell N="HorzAlign" V="0"/>
        </Row></Section></StyleSheet>''')}


def test_saved_paragraph_spacing_and_active_hanging_bullets():
    shape=poster_shape()
    result=source_layout(shape,None,poster_styles(),'0','',10)
    assert result is not None
    assert [p['Bullet'] for p in result['paragraphs']] == [0,1,1]
    assert result['paragraphs'][0]['SpAfter'] == pytest.approx(1/12)
    assert result['paragraphs'][1]['IndFirst'] == -.25
    # Unused style rows (including unsupported ones) cannot disable active layout.
    shape.find('v:Section/v:Row/v:Cell[@N="Bullet"]',{'v':V}).set('V','9')
    assert source_layout(shape,None,poster_styles(),'0','',10) is None
    shape.find('v:Text',{'v':V}).clear()
    shape.find('v:Text',{'v':V}).text='Only the default paragraph'
    assert source_layout(shape,None,poster_styles(),'0','',10) is not None


def test_paragraph_styles_follow_run_markers_and_blank_lines():
    from src.ingestion.visio_text import _paragraph_rows
    text=ET.fromstring(f'<Text xmlns="{V}"><pp IX="0"/><cp IX="1"/> Intro\n\n<pp IX="1"/>Item A\nItem B\n</Text>')
    assert _paragraph_rows(text) == ['0','0','1','1']


def test_paragraph_layout_spacing_wrapping_and_indents():
    from src.ingestion.visio_text import _layout_paragraphs
    shape=poster_shape()
    layout=source_layout(shape,None,poster_styles(),'0','',10)
    chars=[(c,0) for c in 'First paragraph\nFirst item\nSecond item']
    styles=[{'font-size':'10'}]
    with patch('src.ingestion.visio_text._font',return_value=Font()):
        lines,height,_=_layout_paragraphs(chars,styles,80,layout,1)
        # 6pt paragraph spacing is applied between paragraphs, not after the last.
        assert lines[1]['baseline'] == 24
        assert lines[1]['offset'] == 18 and lines[1]['bullet_offset'] == 0
        assert height == 42
        single={**layout,'paragraphs':layout['paragraphs'][:1]}
        _,single_height,_=_layout_paragraphs([(c,0)for c in 'label'],styles,80,single,1)
        assert single_height == 10


def test_page_scale_applies_to_text_margins():
    shape=poster_shape('Label')
    ET.SubElement(shape,f'{{{V}}}Cell',N='LeftMargin',V='0.25')
    result=source_layout(shape,None,poster_styles(),'0','',10,page_scale=.5)
    assert result['margins'][0] == 9


def test_source_paragraphs_normalize_cr_and_crlf():
    from src.ingestion.visio_text import _paragraph_rows
    node=ET.Element(f'{{{V}}}Text');node.text='first\rsecond\r\nthird'
    assert _paragraph_rows(node) == ['0','0','0']


def test_paragraph_repair_keeps_images_and_source_text_and_adds_saved_bullets():
    svg,pages=fixture()
    shape=pages[0]['shapes'][0]
    shape['text_layout']['paragraphs']=[
        dict(IndLeft=0,IndRight=0,IndFirst=0,SpBefore=0,SpAfter=1/12,SpLine=-1.2,HorzAlign=0,Bullet=0),
        dict(IndLeft=.2,IndRight=0,IndFirst=-.2,SpBefore=0,SpAfter=0,SpLine=-1.2,HorzAlign=0,Bullet=1)]
    before=deepcopy(pages);image=ET.tostring(svg[0]);path=ET.tostring(svg[2])
    with patch('src.ingestion.visio_text._font',return_value=Font()):repair_text(svg,pages)
    assert [s.text for s in svg[1]] == ['BRANCH A','•','Router']
    assert ET.tostring(svg[0]) == image and ET.tostring(svg[2]) == path
    assert pages == before
