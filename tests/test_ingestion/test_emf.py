import base64
import io
import shutil
import struct
import sys
from unittest.mock import patch

import pytest
from PIL import Image

from src.config import settings
from src.figures import storage
from src.ingestion import emf, visio
from src.ingestion.parser import parse_document
from src.ingestion.prepared import prepare_document, encode_prepared, decode_prepared


def sample_emf():
    """Synthetic GDI rectangle and right-pointing arrow, no licensed fixture."""
    def record(kind, body):
        return struct.pack('<II', kind, 8 + len(body)) + body
    records = [record(37, struct.pack('<I', 0x80000004)), # stock black brush
               record(43, struct.pack('<4i', 50, 150, 250, 450)),
               record(3, struct.pack('<4iI14i', 250, 200, 900, 400, 7,
                    250, 280, 700, 280, 700, 200, 900, 300, 700, 400, 700, 320, 250, 320)),
               record(14, struct.pack('<III', 0, 0, 20))]
    header = bytearray(88)
    struct.pack_into('<II4i4iIIIIHHIII4i', header, 0, 1, 88,
        0, 0, 1000, 600, 0, 0, 26458, 15875, 0x464d4520, 0x10000,
        88 + sum(map(len, records)), 1 + len(records), 1, 0, 0, 0, 0,
        1000, 600, 265, 159)
    return bytes(header) + b''.join(records)


def png_bytes(width=120, height=80):
    buf = io.BytesIO()
    Image.new('RGBA', (width, height), (0, 0, 0, 0)).save(buf, 'PNG')
    return buf.getvalue()


@pytest.mark.parametrize('damage', ['signature', 'truncated', 'size', 'record_length', 'count', 'frame', 'end'])
def test_rejects_malformed_before_native(damage, tmp_path):
    raw = bytearray(sample_emf())
    if damage == 'signature': raw[40:44] = b'evil'
    elif damage == 'truncated': raw = raw[:60]
    elif damage == 'size': struct.pack_into('<I', raw, 48, len(raw) + 4)
    elif damage == 'record_length': struct.pack_into('<I', raw, 92, 0)
    elif damage == 'count': struct.pack_into('<I', raw, 52, 999)
    elif damage == 'frame': struct.pack_into('<i', raw, 32, 0)
    elif damage == 'end': struct.pack_into('<I', raw, len(raw) - 20, 9)
    with patch('src.ingestion.visio._run') as run:
        with pytest.raises(ValueError): emf.EmfConverter(tmp_path).convert(raw)
        run.assert_not_called()


def test_parser_disabled_and_size(tmp_path, monkeypatch):
    path = tmp_path / 'topology.emf'; path.write_bytes(sample_emf())
    parsed = parse_document(path)
    assert parsed.doc_type == 'emf' and parsed.metadata['emf']['records'] == 5
    monkeypatch.setattr(settings, 'emf_enabled', False)
    with pytest.raises(ValueError, match='disabled'): parse_document(path)
    monkeypatch.setattr(settings, 'emf_max_input_mb', 0)
    with pytest.raises(ValueError, match='size limit'): emf.validate_emf(sample_emf())


def test_svg_replacement_preserves_geometry_and_guard(tmp_path):
    raw = sample_emf(); encoded = base64.b64encode(raw).decode()
    svg = visio.ET.fromstring(f'<svg xmlns="{visio.S}" xmlns:xlink="http://www.w3.org/1999/xlink"><g transform="rotate(30)"><image x="2" y="3" width="40" height="20" clip-path="url(#clip)" preserveAspectRatio="none" xlink:href="data:image/emf;base64,{encoded}"/></g></svg>')
    node = svg[0][0]; before = dict(node.attrib)
    with patch.object(emf.EmfConverter, 'convert', return_value=png_bytes()) as convert:
        assert emf.replace_embedded_emfs(svg, emf.EmfConverter(tmp_path)) == 1
        convert.assert_called_once_with(raw)
    key = '{http://www.w3.org/1999/xlink}href'
    assert {k: v for k, v in node.attrib.items() if k != key} == {k: v for k, v in before.items() if k != key}
    assert svg[0].get('transform') == 'rotate(30)'
    assert node.get(key).startswith('data:image/png;base64,')
    visio._safe_svg(svg)
    node.set(key, 'https://invalid.test/icon.emf')
    assert emf.replace_embedded_emfs(svg, emf.EmfConverter(tmp_path)) == 0
    with pytest.raises(ValueError, match='External'): visio._safe_svg(svg)


def test_disk_cache_limits_and_timeout(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, 'emf_embedded_max_edge', 500)
    raw = sample_emf()
    def rasterize(argv, output, timeout, max_bytes, **kwargs):
        w = int(next(a for a in argv if a.startswith('--export-width=')).split('=')[1])
        h = int(next(a for a in argv if a.startswith('--export-height=')).split('=')[1])
        output.write_bytes(png_bytes(w, h))
    converter = emf.EmfConverter(tmp_path)
    with patch('src.ingestion.emf.shutil.which', return_value='/bin/inkscape'), patch('src.ingestion.visio._run', side_effect=rasterize) as run:
        first = converter.convert(raw)
        assert converter.convert(raw) == first and run.call_count == 1
        monkeypatch.setattr(settings, 'emf_max_conversions_per_doc', 1)
        with pytest.raises(ValueError, match='count limit'): converter.convert(raw, embedded=False)
    converter = emf.EmfConverter(tmp_path / 'next'); converter.elapsed = settings.emf_timeout_seconds
    with pytest.raises(TimeoutError): converter.convert(raw)
    with pytest.raises(TimeoutError):
        visio._run([sys.executable, '-c', 'import time; time.sleep(5)'], tmp_path / 'timeout', .05, 1024)


@pytest.mark.asyncio
async def test_standalone_assets_search_and_failed_vision(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, 'emf_vision_enabled', True)
    path = tmp_path / 'branch.emf'; path.write_bytes(sample_emf())
    with storage.extraction_assets(tmp_path / 'assets'), patch.object(emf.EmfConverter, 'convert', return_value=png_bytes()), patch('pytesseract.image_to_string', return_value='BRANCH A WAN'), patch('src.generation.llm_client.generate_vision', side_effect=RuntimeError('model unavailable')):
        result = await prepare_document(path, 'branch.emf')
    decoded = decode_prepared(encode_prepared(result))
    figure = decoded.office.figures[0]
    assert figure.assets and figure.ocr_text == 'BRANCH A WAN' and not figure.vision_description
    assert figure.analysis_status == 'ocr_only'
    assert 'OCR text' in figure.retrieval_text() and 'BRANCH A WAN' in figure.retrieval_text()
    assert any('vision analysis unavailable' in w for w in result.warnings)
    with Image.open(tmp_path / 'assets' / figure.assets['full']['key']) as image:
        assert image.getpixel((0, 0)) == (255, 255, 255)
    from src.ingestion.prepared_index import figure_index_entries
    entries = list(figure_index_entries(decoded.office.figures))
    assert entries and any('BRANCH A WAN' in text for _, text in entries)


def test_ocr_and_vision_provenance_separate():
    from src.ingestion.figure_extract import FigureRecord
    record = FigureRecord('diagram', 'Source: explicit connector A to B.', 'diagram')
    with patch('pytesseract.image_to_string', return_value='BRANCH'), patch('src.generation.llm_client.generate_vision', return_value='Arrow appears to point right.'):
        emf.analyze_diagram(record, png_bytes())
    assert record.ocr_text == 'BRANCH'
    assert record.vision_description == 'Arrow appears to point right.'
    assert 'Source: explicit' in record.description and 'not source-verified' in record.description


@pytest.mark.skipif(not shutil.which('inkscape'), reason='native EMF renderer required')
def test_native_arrow_geometry(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, 'emf_render_max_edge', 1000)
    raw = emf.EmfConverter(tmp_path).convert(sample_emf(), embedded=False)
    with Image.open(io.BytesIO(raw)) as image:
        rgba = image.convert('RGBA')
        # The right tip is inked; the area past the tip remains transparent.
        assert rgba.width == 1000 and 590 <= rgba.height <= 610
        assert rgba.getpixel((850, 300))[3] > 200
        assert rgba.getpixel((950, 300))[3] == 0
        assert rgba.getpixel((850, 230))[3] == 0


def test_emf_settings_available_and_original_retention(tmp_path, monkeypatch):
    from src.admin.settings_catalog import settings_catalog
    from src.sources.storage import OriginalStore
    import hashlib
    names = {item['name'] for group in settings_catalog().values() for item in group}
    assert {name for name in type(settings).model_fields if name.startswith('emf_')} <= names
    monkeypatch.setattr(settings, 'source_originals_dir', str(tmp_path / 'originals'))
    path = tmp_path / 'diagram.emf'; raw = sample_emf(); path.write_bytes(raw)
    revision = hashlib.sha256(raw).hexdigest()
    store = OriginalStore()
    assert store.retain(path, 'doc-emf', revision, 'diagram.emf')
    stream, _ = store.open_verified('doc-emf', revision)
    with stream: assert stream.read() == raw


@pytest.mark.skipif(not all(shutil.which(t) for t in ('inkscape', 'vsd2xhtml', 'rsvg-convert')), reason='native diagram tools required')
def test_embedded_emf_topology_keeps_source_connectors_and_renders(tmp_path, monkeypatch):
    import zipfile
    from tests.test_ingestion.test_visio import fixture
    path = fixture(tmp_path / 'topology.vsdx')
    with zipfile.ZipFile(path) as archive:
        parts = {name: archive.read(name) for name in archive.namelist()}
    parts['visio/media/image1.emf'] = sample_emf()
    del parts['visio/media/image1.png']
    parts['visio/pages/page1.xml'] = parts['visio/pages/page1.xml'].replace(b'ForeignType="Bitmap" CompressionType="PNG"', b'ForeignType="EnhMetaFile"')
    parts['visio/pages/_rels/page1.xml.rels'] = parts['visio/pages/_rels/page1.xml.rels'].replace(b'image1.png', b'image1.emf')
    parts['[Content_Types].xml'] = parts['[Content_Types].xml'].replace(b'Extension="png" ContentType="image/png"', b'Extension="emf" ContentType="image/x-emf"')
    with zipfile.ZipFile(path, 'w') as archive:
        for name, raw in parts.items(): archive.writestr(name, raw)
    original = path.read_bytes()
    parsed = parse_document(path)
    connector = parsed.metadata['visio_pages'][1]['shapes'][2]['connector']
    monkeypatch.setattr(settings, 'emf_vision_enabled', False)
    with storage.extraction_assets(tmp_path / 'assets'), patch('pytesseract.image_to_string', return_value='BRANCH A'):
        office, warnings = visio.render_visio(path, parsed)
    assert len(office.figures) == 2
    assert any('1 embedded EMF' in warning for warning in warnings)
    assert path.read_bytes() == original
    assert parse_document(path).metadata['visio_pages'][1]['shapes'][2]['connector'] == connector
    front = next(f for f in office.figures if f.source_page_id == '1')
    assert front.assets and front.ocr_text == 'BRANCH A'
    with Image.open(tmp_path / 'assets' / front.assets['full']['key']) as image:
        # Embedded icon remains in the lower right, preserving placement.
        area = image.crop((int(image.width*.75), int(image.height*.75), int(image.width*.85), int(image.height*.92)))
        assert any(max(rgb[:3]) < 40 for rgb in area.getdata())


def test_conversion_failure_withholds_page_but_keeps_source(tmp_path):
    svg = visio.ET.fromstring(f'<svg xmlns="{visio.S}"><image href="data:image/emf;base64,{base64.b64encode(sample_emf()).decode()}"/></svg>')
    with patch.object(emf.EmfConverter, 'convert', side_effect=ValueError('unsupported')):
        with pytest.raises(ValueError, match='unsupported'):
            emf.replace_embedded_emfs(svg, emf.EmfConverter(tmp_path))
    assert svg[0].get('href').startswith('data:image/emf;')


@pytest.mark.skipif(not shutil.which('inkscape'), reason='native EMF renderer required')
@pytest.mark.asyncio
async def test_native_emf_worker_persistence_and_image_acl(tmp_path, monkeypatch):
    from src.ingestion.isolation import extract_in_worker
    from src.ingestion.prepared_index import index_prepared
    from src.db.metadata import MetadataStore
    from src.figures.service import image_bytes, answer_images
    monkeypatch.setattr(settings, 'emf_vision_enabled', False)
    monkeypatch.setattr(settings, 'extraction_work_dir', str(tmp_path / 'jobs'))
    monkeypatch.setattr(storage, 'FIGURE_ROOT', tmp_path / 'assets')
    path = tmp_path / 'diagram.emf'; path.write_bytes(sample_emf())
    prepared = await extract_in_worker(path, 'topology.emf')
    assert prepared.figure_staging and prepared.office.figures[0].assets
    ms = MetadataStore('sqlite+aiosqlite:///' + str(tmp_path / 'metadata.db'))
    await ms.init()
    try:
        _, _, figures = await index_prepared(prepared, 'emf-doc', ['network'], '', None, ms)
        await ms.add_document('emf-doc', 'topology.emf', 'emf', ['network'], 1, 'test')
        figure = figures[0]
        refs = await answer_images('show diagram', [{'doc_id': 'emf-doc', 'figure_id': figure.figure_id}], ['network'], ms)
        assert refs[0]['filename'] == 'topology.emf' and refs[0]['mime_type'] == 'image/png'
        raw, _ = await image_bytes('emf-doc', figure.figure_id, ['network'], ms)
        assert raw.startswith(storage.PNG)
        with pytest.raises(FileNotFoundError):
            await image_bytes('emf-doc', figure.figure_id, ['other'], ms)
    finally:
        storage.FigureStore().discard(prepared.figure_staging)
        await ms.engine.dispose()


def test_long_visual_descriptions_keep_provenance_on_every_chunk():
    from src.ingestion.figure_extract import FigureRecord
    from src.ingestion.prepared_index import figure_index_entries
    figure = FigureRecord('emf-one', '', 'diagram', source='emf_render',
        source_text='No native connector data.', ocr_text='BRANCH A',
        vision_description=' '.join(f'Visible router {i}.' for i in range(500)))
    entries = list(figure_index_entries([figure]))
    visual = [text for _, text in entries if 'Visible router' in text]
    assert len(visual) > 2 and all('not source-verified connector facts' in t for t in visual)
    assert all('Figure: emf-one' in t and len(t) < 1800 for t in visual)
