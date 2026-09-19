import io
import shutil
import sys
import zipfile
from pathlib import Path
from unittest.mock import patch

import pytest
from PIL import Image, ImageDraw

from src.config import settings
from src.figures import storage
from src.ingestion import visio
from src.ingestion.parser import parse_document
from src.ingestion.prepared import prepare_document, encode_prepared, decode_prepared


def fixture(path):
    """Synthetic source, no customer files or Microsoft Visio installation required."""
    v, r = visio.V, visio.R
    def cell(n, value):
        return f'<Cell N="{n}" V="{value}"/>'
    def box(sid, text, x, y, w, h, color):
        cells = ''.join(cell(n, val) for n, val in [('PinX', x), ('PinY', y), ('Width', w), ('Height', h),
            ('LocPinX', w/2), ('LocPinY', h/2), ('FillForegnd', color), ('FillPattern', 1), ('LineColor', '#000000'), ('LineWeight', .02)])
        return f'<Shape ID="{sid}" Type="Shape" NameU="Device">{cells}<Section N="Geometry"><Cell N="NoFill" V="0"/><Cell N="NoLine" V="0"/><Row T="MoveTo" IX="1"><Cell N="X" V="0"/><Cell N="Y" V="0"/></Row><Row T="LineTo" IX="2"><Cell N="X" V="{w}"/><Cell N="Y" V="0"/></Row><Row T="LineTo" IX="3"><Cell N="X" V="{w}"/><Cell N="Y" V="{h}"/></Row><Row T="LineTo" IX="4"><Cell N="X" V="0"/><Cell N="Y" V="{h}"/></Row><Row T="LineTo" IX="5"><Cell N="X" V="0"/><Cell N="Y" V="0"/></Row></Section><Text>{text}</Text></Shape>'
    dimensions = '<PageSheet><Cell N="PageWidth" V="10"/><Cell N="PageHeight" V="6"/><Cell N="PageScale" V="1"/><Cell N="DrawingScale" V="1"/></PageSheet>'
    line = '<Shape ID="3" Type="Shape" OneD="1" NameU="WAN link">'+''.join(cell(n,val) for n,val in [('PinX',5),('PinY',3),('Width',4),('Height',0),('LocPinX',2),('LocPinY',0),('BeginX',3),('BeginY',3),('EndX',7),('EndY',3),('LineColor','#00aa00'),('LineWeight',.06),('EndArrow',4)])+'<Section N="Geometry"><Cell N="NoFill" V="1"/><Row T="MoveTo" IX="1"><Cell N="X" V="0"/><Cell N="Y" V="0"/></Row><Row T="LineTo" IX="2"><Cell N="X" V="4"/><Cell N="Y" V="0"/></Row></Section><Text>WAN</Text></Shape>'
    photo = '<Shape ID="4" Type="Foreign" NameU="Embedded picture">'+''.join(cell(n,val) for n,val in [('PinX',8),('PinY',1),('Width',1),('Height',1),('LocPinX',.5),('LocPinY',.5),('ImgWidth',1),('ImgHeight',1),('ImgOffsetX',0),('ImgOffsetY',0)])+'<ForeignData ForeignType="Bitmap" CompressionType="PNG"><Rel r:id="image1"/></ForeignData></Shape>'
    group = '<Shape ID="10" Type="Group" NameU="Site group">'+''.join(cell(n,val) for n,val in [('PinX',5),('PinY',5),('Width',2),('Height',1),('LocPinX',1),('LocPinY',.5)])+'<Shapes>'+box(11,'SITE GROUP',1,.5,2,1,'#dddddd')+'</Shapes></Shape>'
    page = f'<PageContents xmlns="{v}" xmlns:r="{r}"><Shapes>'+box(1,'BRANCH A',2,3,2,1,'#5599ff')+box(2,'BRANCH B',8,3,2,1,'#ff7755')+line+photo+group+'</Shapes><Connects><Connect FromSheet="3" FromCell="BeginX" ToSheet="1" ToCell="PinX"/><Connect FromSheet="3" FromCell="EndX" ToSheet="2" ToCell="PinX"/></Connects></PageContents>'
    back = f'<PageContents xmlns="{v}"><Shapes>'+box(1,'BACKGROUND LEGEND',5,.25,9,.4,'#ffff99')+'</Shapes></PageContents>'
    image = Image.new('RGB',(80,80),'#ff00ff'); ImageDraw.Draw(image).rectangle((0,0,39,79),fill='#00ffff')
    raw=io.BytesIO();image.save(raw,'PNG')
    relns='http://schemas.openxmlformats.org/package/2006/relationships'
    def rels(items):
        result = []
        for rid, kind, target in items:
            namespace = r if kind == 'image' else 'http://schemas.microsoft.com/visio/2010/relationships'
            result.append(f'<Relationship Id="{rid}" Type="{namespace}/{kind}" Target="{target}"/>')
        return '<Relationships xmlns="'+relns+'">'+''.join(result)+'</Relationships>'
    files={
        '[Content_Types].xml':'<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types"><Default Extension="xml" ContentType="application/xml"/><Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/><Default Extension="png" ContentType="image/png"/><Override PartName="/visio/document.xml" ContentType="application/vnd.ms-visio.drawing.main+xml"/></Types>',
        '_rels/.rels':rels([('r1','document','visio/document.xml')]),
        'visio/document.xml':f'<VisioDocument xmlns="{v}" xmlns:r="{r}"><DocumentSettings DefaultTextStyle="0" DefaultLineStyle="0" DefaultFillStyle="0" DefaultGuideStyle="0"/><StyleSheets><StyleSheet ID="0" NameU="No Style"><Cell N="FillPattern" V="1"/><Cell N="LinePattern" V="1"/><Section N="Character"><Row IX="0"><Cell N="Size" V="0.14"/><Cell N="Color" V="#000000"/></Row></Section></StyleSheet></StyleSheets></VisioDocument>',
        'visio/_rels/document.xml.rels':rels([('pages','pages','pages/pages.xml')]),
        'visio/pages/pages.xml':f'<Pages xmlns="{v}" xmlns:r="{r}"><Page ID="5" Name="Shared background" Background="1">{dimensions}<Rel r:id="back"/></Page><Page ID="1" Name="Branch topology" BackPage="5">{dimensions}<Rel r:id="front"/></Page></Pages>',
        'visio/pages/_rels/pages.xml.rels':rels([('back','page','page2.xml'),('front','page','page1.xml')]),
        'visio/pages/page1.xml':page,'visio/pages/page2.xml':back,
        'visio/pages/_rels/page1.xml.rels':rels([('image1','image','../media/image1.png')]),
        'visio/media/image1.png':raw.getvalue(),
    }
    with zipfile.ZipFile(path,'w',zipfile.ZIP_DEFLATED) as archive:
        for name,value in files.items():archive.writestr(name,value)
    return path


def test_source_groups_connections_backgrounds(tmp_path):
    parsed = parse_document(fixture(tmp_path/'sample.vsdx'))
    assert parsed.doc_type == 'vsdx'
    assert 'BRANCH A' in parsed.text and 'SITE GROUP' in parsed.text
    front = parsed.metadata['visio_pages'][1]
    assert 'BACKGROUND LEGEND' in front['text']
    assert 'in group 10' in front['text']
    assert front['connections'][0]['ToSheet'] == '1'
    assert 'traffic direction' in front['text']


def rewrite(path, name, transform):
    with zipfile.ZipFile(path) as source: files={n:source.read(n) for n in source.namelist()}
    files[name]=transform(files[name])
    with zipfile.ZipFile(path,'w') as target:
        for key,value in files.items():target.writestr(key,value)


@pytest.mark.parametrize('damage', ['doctype','traversal','cycle','size','pages','shapes'])
def test_invalid_source_limits(tmp_path,monkeypatch,damage):
    source=fixture(tmp_path/'bad.vsdx')
    if damage=='doctype':
        rewrite(source,'visio/pages/page1.xml',lambda b:b'<!DOCTYPE PageContents [<!ENTITY x SYSTEM "file:///etc/passwd">]>'+b)
    elif damage=='traversal':
        rewrite(source,'visio/pages/_rels/pages.xml.rels',lambda b:b.replace(b'page1.xml',b'../../../escape.xml'))
    elif damage=='cycle':
        rewrite(source,'visio/pages/pages.xml',lambda b:b.replace(b'Background="1"',b'Background="1" BackPage="1"'))
    elif damage=='size':monkeypatch.setattr(settings,'visio_max_unpacked_mb',0)
    elif damage=='pages':monkeypatch.setattr(settings,'visio_max_pages',1)
    else:monkeypatch.setattr(settings,'visio_max_shapes',1)
    with pytest.raises(ValueError):parse_document(source)


@pytest.mark.asyncio
async def test_missing_renderer_keeps_searchable_text(tmp_path):
    with patch('src.ingestion.visio.shutil.which',return_value=None):
        result=await prepare_document(fixture(tmp_path/'sample.vsdx'),'topology.vsdx')
    assert 'BRANCH A' in result.parsed.text and result.warnings
    assert not result.office.figures
    assert decode_prepared(encode_prepared(result)).parsed.filename=='topology.vsdx'


def test_svg_limits_and_external_resources():
    svg=visio.ET.fromstring(f'<svg xmlns="{visio.S}" viewBox="0 0 100 50"/>')
    w,h=visio._dimensions(svg)
    assert w==2*h and w*h<=settings.figure_max_pixels
    for fragment in ('<image href="file:///etc/passwd"/>','<image href="https://example.com/image.png"/>','<style>@import "https://example.com/x.css";</style>','<script/>'):
        svg=visio.ET.fromstring(f'<svg xmlns="{visio.S}">{fragment}</svg>')
        with pytest.raises(ValueError):visio._safe_svg(svg)


def test_converter_timeout_and_failure(tmp_path):
    with pytest.raises(TimeoutError):visio._run([sys.executable,'-c','import time; time.sleep(10)'],tmp_path/'out',.05,1000)
    with pytest.raises(ValueError):visio._run([sys.executable,'-c','print("x"*10000)'],tmp_path/'out',3,100)
    with pytest.raises(ValueError,match='exit 2'):visio._run([sys.executable,'-c','raise SystemExit(2)'],tmp_path/'out',3,100)


@pytest.mark.skipif(not shutil.which('vsd2xhtml') or not shutil.which('rsvg-convert'),reason='native Visio tools not installed')
@pytest.mark.asyncio
async def test_native_full_pages_png_handoff(tmp_path,monkeypatch):
    monkeypatch.setattr(storage,'FIGURE_ROOT',tmp_path/'published')
    with storage.extraction_assets(tmp_path/'worker'):
        prepared=await prepare_document(fixture(tmp_path/'sample.vsdx'),'source.vsdx')
    assert not prepared.warnings,prepared.warnings
    assert [r.caption for r in prepared.office.figures]==['Branch topology','Shared background']
    assert [r.page for r in prepared.office.figures]==[1,0]
    staging=storage.FigureStore().handoff(tmp_path/'worker')
    records=storage.FigureStore().publish('doc-1',prepared.office.figures,staging)
    assert len(records)==2
    image=Image.open(storage.FigureStore().asset_path('doc-1',records[0]['assets']['full']['key'])).convert('RGB')
    colors=image.getcolors(image.width*image.height)
    for expected in [(85,153,255),(255,119,85),(255,255,153),(255,0,255),(0,255,255),(0,170,0),(221,221,221)]:
        assert sum(n for n,color in colors if color==expected)>100,expected
    assert image.width/image.height==pytest.approx(10/6,abs=.002)
    assert decode_prepared(encode_prepared(prepared)).office.figures[0].assets


def test_cached_inherited_fields_and_formatting():
    base = visio.ET.fromstring(f'''<Shape xmlns="{visio.V}" TextStyle="3"><Section N="Character"><Row IX="0"><Cell N="Size" V="0.111111"/></Row></Section><Text><cp IX="0"/><fld IX="0">Placeholder</fld></Text></Shape>''')
    shape = visio.ET.fromstring(f'''<Shape xmlns="{visio.V}"><Section N="Field"><Row IX="0"><Cell N="Value" V="10.0.1.5" U="STR" F="Inh"/></Row></Section></Shape>''')
    resolved=visio._resolved_text(shape,base)
    assert ''.join(resolved.itertext())=='10.0.1.5'
    visio._materialize_text_style(shape,base)
    assert shape.get('TextStyle')=='3'
    assert shape.find('v:Section[@N="Character"]/v:Row/v:Cell',visio.NS).get('V')=='0.111111'
    # Existing display text, including formatted dates/page numbers, wins over raw values.
    shape.append(visio.ET.fromstring(f'<Text xmlns="{visio.V}"><fld IX="0">Saved display</fld></Text>'))
    assert ''.join(visio._resolved_text(shape,base).itertext())=='Saved display'


def test_renderer_page_mismatch_and_partial_page(tmp_path,monkeypatch):
    parsed=parse_document(fixture(tmp_path/'source.vsdx'))
    monkeypatch.setattr(visio.shutil,'which',lambda name:name)
    def mismatched(argv,output,*args):output.write_text(f'<html><body><svg xmlns="{visio.S}" width="10" height="6"/></body></html>')
    monkeypatch.setattr(visio,'_run',mismatched)
    result,warnings=visio.render_visio(tmp_path/'source.vsdx',parsed)
    assert not result.figures and 'Page count mismatch' in warnings[0]
    calls=[]
    def partial(argv,output,*args):
        calls.append(argv[0])
        if argv[0]=='vsd2xhtml':
            output.write_text(f'<html><body><svg xmlns="{visio.S}" width="10" height="6"/><svg xmlns="{visio.S}" width="10" height="6"/></body></html>')
        elif calls.count('rsvg-convert')==1:raise ValueError('bad page')
        else:Image.new('RGB',(100,60),'white').save(output)
    monkeypatch.setattr(visio,'_run',partial)
    with storage.extraction_assets(tmp_path/'figures'):
        result,warnings=visio.render_visio(tmp_path/'source.vsdx',parsed)
    assert len(result.figures)==1 and result.figures[0].page==0
    assert any('bad page' in w for w in warnings)
    assert result.figures[0].render_warnings


@pytest.mark.skipif(not shutil.which('vsd2xhtml') or not shutil.which('rsvg-convert'),reason='native Visio tools not installed')
@pytest.mark.asyncio
async def test_native_worker_and_persistent_index(tmp_path,monkeypatch):
    from src.ingestion.isolation import extract_in_worker
    from src.ingestion.prepared_index import index_prepared
    from src.db.metadata import MetadataStore
    from src.figures.service import image_bytes, answer_images
    monkeypatch.setattr(settings,'extraction_work_dir',str(tmp_path/'jobs'))
    monkeypatch.setattr(storage,'FIGURE_ROOT',tmp_path/'assets')
    source=fixture(tmp_path/'source.vsdx')
    prepared=await extract_in_worker(source,'branch.vsdx')
    assert not prepared.warnings and prepared.figure_staging
    ms=MetadataStore('sqlite+aiosqlite:///'+str(tmp_path/'metadata.db'))
    await ms.init()
    try:
        _,_,figures=await index_prepared(prepared,'visio-doc',['network'],'',None,ms)
        await ms.add_document('visio-doc','branch.vsdx','vsdx',['network'],1,'test')
        figure=next(r for r in figures if r.caption=='Branch topology')
        assert 'BRANCH A' in figure.retrieval_text() and 'BeginX' in figure.retrieval_text()
        ref=await answer_images('show topology',[{'doc_id':'visio-doc','figure_id':figure.figure_id}],['network'],ms)
        assert ref[0]['filename']=='branch.vsdx' and ref[0]['page']==2
        raw,_=await image_bytes('visio-doc',figure.figure_id,['network'],ms)
        assert raw.startswith(storage.PNG)
        with pytest.raises(FileNotFoundError):await image_bytes('visio-doc',figure.figure_id,['other'],ms)
    finally:
        storage.FigureStore().discard(prepared.figure_staging)
        await ms.engine.dispose()


def test_large_page_labels_keep_figure_provenance():
    from src.ingestion.figure_extract import FigureRecord
    from src.ingestion.prepared_index import figure_index_entries
    figure=FigureRecord('visio-page-9','\n\n'.join(f'Router {i}: 10.1.2.{i}' for i in range(250)),
        'diagram',source='visio_page_render',page=2,caption='WAN',source_page_id='9')
    entries=list(figure_index_entries([figure]))
    assert len(entries)>2 and all(record is figure for record,_ in entries)
    assert any('Router 249:' in text for _,text in entries)
    assert all(len(text)<1800 and 'visio-page-9' in text for _,text in entries)


def arrow_fixture(path):
    """Known endpoint markers, separated from device graphics and text."""
    fixture(path)
    def make_page(raw):
        root=visio.ET.fromstring(raw)
        shapes=root.find('v:Shapes',visio.NS)
        template=shapes.find('v:Shape[@ID="3"]',visio.NS)
        shapes.clear()
        import copy
        for sid,y,color,begin,end,inherited in [
            (1,5,'#008800',0,4,False),
            (2,3.5,'#cc0000',4,4,False),
            (3,2,'#0000cc',4,4,True),
        ]:
            shape=copy.deepcopy(template);shape.set('ID',str(sid))
            shape.remove(shape.find('v:Text',visio.NS))
            cells={'PinX':5,'PinY':y,'Width':6,'LocPinX':3,'BeginX':2,'BeginY':y,
                   'EndX':8,'EndY':y,'LineColor':color,'LineWeight':.04,
                   'BeginArrow':begin,'EndArrow':end,'BeginArrowSize':3,'EndArrowSize':3}
            for name,value in cells.items():
                cell=shape.find(f'v:Cell[@N="{name}"]',visio.NS)
                if cell is None:cell=visio.ET.SubElement(shape,'{'+visio.V+'}Cell',N=name)
                cell.set('V',str(value))
            shape.find('v:Section[@N="Geometry"]/v:Row[@IX="2"]/v:Cell[@N="X"]',visio.NS).set('V','6')
            if inherited:
                shape.set('LineStyle','1')
                for name in ['BeginArrow','EndArrow']:
                    shape.remove(shape.find(f'v:Cell[@N="{name}"]',visio.NS))
            shapes.append(shape)
        root.remove(root.find('v:Connects',visio.NS))
        return visio.ET.tostring(root)
    rewrite(path,'visio/pages/page1.xml',make_page)
    def add_style(raw):
        root=visio.ET.fromstring(raw)
        parent=root.find('v:StyleSheets',visio.NS)
        style=visio.ET.SubElement(parent,'{'+visio.V+'}StyleSheet',ID='1',LineStyle='0')
        for name in ['BeginArrow','EndArrow']:visio.ET.SubElement(style,'{'+visio.V+'}Cell',N=name,V='4')
        return visio.ET.tostring(root)
    rewrite(path,'visio/document.xml',add_style)
    return path


@pytest.mark.skipif(not shutil.which('vsd2xhtml') or not shutil.which('rsvg-convert'),reason='native Visio tools not installed')
@pytest.mark.asyncio
async def test_native_directional_arrowheads(tmp_path):
    with storage.extraction_assets(tmp_path/'worker'):
        prepared=await prepare_document(arrow_fixture(tmp_path/'arrows.vsdx'),'arrows.vsdx')
    assert not prepared.warnings,prepared.warnings
    figure=next(r for r in prepared.office.figures if r.caption=='Branch topology')
    # Inspect the actual full PNG before publication; no generated substitute graphic.
    key=figure.assets['full']['key']
    image=Image.open(tmp_path/'worker'/key).convert('RGB')
    for y,color,left_arrow in [(5,(0,136,0),False),(3.5,(204,0,0),True),(2,(0,0,204),True)]:
        def thickness(x0,x1):
            crop=image.crop((int(x0/10*image.width),int((6-y-.4)/6*image.height),
                             int(x1/10*image.width),int((6-y+.4)/6*image.height)))
            return max((sum(crop.getpixel((x,j))!=(255,255,255) for j in range(crop.height)) for x in range(crop.width)),default=0)
        # libvisio emits black markers even on colored lines; test endpoint geometry.
        assert image.getpixel((image.width//2,round((6-y)/6*image.height)))==color
        shaft=thickness(4,4.5)
        left,right=thickness(1.6,2.5),thickness(7.5,8.4)
        assert shaft>0
        assert right>shaft*2  # Broad arrowhead, not merely a surviving line.
        assert (left>shaft*2) if left_arrow else (left<=shaft+2)
