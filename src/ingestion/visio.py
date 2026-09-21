"""VSDX source evidence and bounded libvisio/librsvg page rendering.

All entry points that read/render documents run in the disposable extractor.
The XML reader extracts source facts; it does not evaluate ShapeSheet formulas
or replace libvisio's layout, master, group, and background rendering.
"""
from __future__ import annotations

import math
import posixpath
import re
import shutil
import subprocess
import tempfile
import time
import zipfile
from pathlib import Path
from urllib.parse import unquote

from lxml import etree as ET

from src.config import settings

V = "http://schemas.microsoft.com/office/visio/2012/main"
R = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
S = "http://www.w3.org/2000/svg"
NS = {"v": V}


def _xml(raw):
    root = ET.fromstring(raw, ET.XMLParser(resolve_entities=False, no_network=True, load_dtd=False))
    if root.getroottree().docinfo.internalDTD or any(isinstance(e, ET._Entity) for e in root.iter()):
        raise ValueError("Visio XML must not contain a DTD or entities")
    return root


class Package:
    def __init__(self, archive):
        self.archive = archive
        entries = archive.infolist()
        if len(entries) > 10000 or len({i.filename for i in entries}) != len(entries):
            raise ValueError("Visio package contains too many or duplicate entries")
        if sum(i.file_size for i in entries) > settings.visio_max_unpacked_mb * 1024**2:
            raise ValueError("Visio package exceeds the expanded size limit")
        for item in entries:
            name = item.filename
            if name.startswith('/') or '\\' in name or '..' in name.split('/') or item.flag_bits & 1:
                raise ValueError("Invalid or encrypted Visio package entry")
            if name.endswith(('.xml', '.rels')):
                if item.file_size > 16 * 1024**2:
                    raise ValueError("Visio XML part exceeds 16 MiB")
                _xml(archive.read(name))  # Validate before handing original bytes to native tools.
        self.names = {i.filename for i in entries}

    def xml(self, name):
        return _xml(self.archive.read(name))

    def rels(self, part):
        directory, name = posixpath.split(part)
        rel = posixpath.join(directory, '_rels', name + '.rels')
        result = {}
        if rel not in self.names:
            return result
        for item in self.xml(rel):
            if item.get('TargetMode') == 'External':
                continue  # Hyperlinks/external data are not fetched.
            target = unquote(item.get('Target', ''))
            if ':' in target or '\\' in target:
                raise ValueError("Invalid Visio relationship target")
            target = posixpath.normpath(posixpath.join(directory, target)) if not target.startswith('/') else target.lstrip('/')
            if target.startswith('../') or target not in self.names:
                raise ValueError("Visio relationship points outside or to a missing package part")
            result[item.get('Id')] = target
        return result


def _resolved_text(shape, base=None):
    """Use saved field values, never evaluate formulas or external data links."""
    import copy
    text = shape.find('v:Text', NS)
    inherited = text is None
    if text is None and base is not None:
        text = base.find('v:Text', NS)
    if text is None:
        return None
    text = copy.deepcopy(text)
    fields = {}
    for owner in (base, shape):
        if owner is not None:
            for row in owner.findall('v:Section[@N="Field"]/v:Row', NS):
                cell = row.find('v:Cell[@N="Value"]', NS)
                if cell is not None and cell.get('V') is not None:
                    fields[row.get('IX', '')] = cell
    for field in list(text.findall('.//v:fld', NS)):
        cell = fields.get(field.get('IX', ''))
        if not inherited and (field.text or '').strip():
            continue  # A saved display string takes precedence over unformatted numeric values.
        if cell is None or cell.get('U') != 'STR':
            continue  # Date/number formatting requires Visio; retain saved display text.
        value = cell.get('V', '') + (field.tail or '')
        parent, previous = field.getparent(), field.getprevious()
        if previous is None:
            parent.text = (parent.text or '') + value
        else:
            previous.tail = (previous.tail or '') + value
        parent.remove(field)
    return text


def _materialize_text_style(shape, base):
    """Copy inherited text formatting without replacing instance overrides."""
    import copy
    if base is None:
        return
    if shape.get('TextStyle') is None and base.get('TextStyle') is not None:
        shape.set('TextStyle', base.get('TextStyle'))
    text_cells = {'LeftMargin', 'RightMargin', 'TopMargin', 'BottomMargin', 'VerticalAlign',
                  'TextBkgnd', 'TextBkgndTrans', 'HideText', 'DefaultTabStop'}
    existing = {c.get('N') for c in shape.findall('v:Cell', NS)}
    for cell in base.findall('v:Cell', NS):
        if cell.get('N') in text_cells and cell.get('N') not in existing:
            shape.insert(0, copy.deepcopy(cell))
    for name in ('Character', 'Paragraph'):
        source = base.find(f'v:Section[@N="{name}"]', NS)
        if source is None:
            continue
        target = shape.find(f'v:Section[@N="{name}"]', NS)
        if target is None:
            shape.insert(0, copy.deepcopy(source))
            continue
        for row in source.findall('v:Row', NS):
            own = next((r for r in target.findall('v:Row', NS) if r.get('IX') == row.get('IX')), None)
            if own is None:
                target.append(copy.deepcopy(row))
            else:
                present = {c.get('N') for c in own.findall('v:Cell', NS)}
                for cell in row.findall('v:Cell', NS):
                    if cell.get('N') not in present:
                        own.append(copy.deepcopy(cell))


def _properties(shape):
    values = {}
    if shape is None:
        return values
    for row in shape.findall('v:Section[@N="Property"]/v:Row', NS):
        cells = {c.get('N'): c.get('V', '').strip('"') for c in row.findall('v:Cell', NS)}
        values[row.get('N', row.get('IX', ''))] = (cells.get('Label') or row.get('N', 'Property'), cells.get('Value', ''))
    return values


_LINE_END_CELLS = ('BeginArrow', 'EndArrow', 'BeginArrowSize', 'EndArrowSize')


def _line_end_cell(shape, base, styles, default_style, name):
    """Read saved values; resolve absent cells through masters/line styles.

    V is Visio's saved value, including for formula/inherited cells. No formulas
    are evaluated. Custom line ends remain unresolved for this evidence reader.
    """
    inherited_cache = []
    def read(owner, origin):
        if owner is None:
            return None
        cell = owner.find(f'v:Cell[@N="{name}"]', NS)
        if cell is None:
            return None
        raw, formula = cell.get('V', ''), cell.get('F', '')
        result = {'value': None, 'raw': raw, 'formula': formula, 'origin': origin}
        if re.search(r'\bUSE\s*\(', formula, re.I):
            result['status'] = 'custom_line_end'
            return result
        if re.fullmatch(r'\d+', raw):
            value = int(raw)
            maximum = 6 if name.endswith('Size') else 45
            if 0 <= value <= maximum:
                result.update(value=value, status='saved_value')
                if formula.lower() == 'inh':
                    inherited_cache.append(result)
                    return None
                return result
        if formula.lower() == 'inh' and not raw:
            return None
        result['status'] = 'unresolved'
        return result

    for owner, origin in ((shape, 'shape'), (base, 'master')):
        result = read(owner, origin)
        if result is not None:
            return result
    sid = shape.get('LineStyle')
    if sid is None and base is not None:
        sid = base.get('LineStyle')
    sid = sid if sid is not None else default_style
    seen = set()
    while sid is not None and sid not in seen:
        seen.add(sid)
        style = styles.get(sid)
        if style is None:
            break
        result = read(style, 'line_style:' + sid)
        if result is not None:
            return result
        sid = style.get('LineStyle')
    if inherited_cache:
        return inherited_cache[0]
    if sid is None and not name.endswith('Size'):
        return {'value': 0, 'raw': '', 'formula': '', 'origin': 'visio_default', 'status': 'default'}
    return {'value': None, 'raw': '', 'formula': '', 'origin': 'unresolved', 'status': 'unresolved'}


def _connector_evidence(shape, base, line_ends):
    coordinates = {}
    for name in ('BeginX', 'BeginY', 'EndX', 'EndY'):
        cell = shape.find(f'v:Cell[@N="{name}"]', NS)
        # Instance endpoint coordinates must not be replaced by master geometry.
        if cell is not None:
            try:
                value = float(cell.get('V', ''))
                if math.isfinite(value):
                    coordinates[name] = value
            except ValueError:
                pass
    one_d = shape.get('OneD')
    if one_d is None and base is not None:
        one_d = base.get('OneD')
    if one_d != '1' and not coordinates:
        return None
    return {'coordinates': coordinates, 'coordinate_space': 'containing_shape',
            'begin': {'attachments': [], 'arrow': line_ends['BeginArrow'], 'arrow_size': line_ends['BeginArrowSize']},
            'end': {'attachments': [], 'arrow': line_ends['EndArrow'], 'arrow_size': line_ends['EndArrowSize']}}


def _describe_connectors(shapes, connections):
    """Keep endpoint attachment separate from line-end decoration/meaning."""
    for connection in connections:
        shape = shapes.get(connection['FromSheet'])
        connector = shape.get('connector') if shape else None
        endpoint = {'BeginX': 'begin', 'BeginY': 'begin', 'EndX': 'end', 'EndY': 'end'}.get(connection['FromCell'])
        if connector is not None and endpoint:
            target = shapes.get(connection['ToSheet'])
            connector[endpoint]['attachments'].append({
                'shape_id': connection['ToSheet'], 'cell': connection['ToCell'],
                'shape_name': (target['text'] or target['name']) if target else '',
                'resolved': target is not None})
    lines = []
    for shape in shapes.values():
        connector = shape.get('connector')
        if connector is None:
            continue
        parts = []
        for endpoint in ('begin', 'end'):
            info = connector[endpoint]
            value = info['arrow']['value']
            marker = ('none' if value == 0 else f'Visio line-end style {value}') if value is not None else 'unresolved'
            attached = ', '.join('shape ' + a['shape_id'] + ' (' + a['shape_name'] + ')' for a in info['attachments']) or 'no explicit attachment recorded'
            parts.append(f"{endpoint}: {attached}; line end: {marker}; source: {info['arrow']['origin']}")
        lines.append(f"Connector {shape['id']}: " + '; '.join(parts) + '.')
    return lines


def parse_visio(path, render_copy=None):
    from src.ingestion.parser import ParsedDocument, DocumentBlock
    if not settings.visio_enabled:
        raise ValueError("Visio ingestion is disabled in admin settings")
    with zipfile.ZipFile(path) as archive:
        package = Package(archive)
        document = package.xml('visio/document.xml')
        styles = {s.get('ID'): s for s in document.findall('v:StyleSheets/v:StyleSheet', NS)}
        defaults = document.find('v:DocumentSettings', NS)
        default_line = defaults.get('DefaultLineStyle', '0') if defaults is not None else '0'
        default_text = defaults.get('DefaultTextStyle', '0') if defaults is not None else '0'
        masters = {}
        if 'visio/masters/masters.xml' in package.names:
            rels = package.rels('visio/masters/masters.xml')
            for master in package.xml('visio/masters/masters.xml').findall('v:Master', NS):
                rel = master.find('v:Rel', NS)
                target = rels.get(rel.get(f'{{{R}}}id')) if rel is not None else None
                if target:
                    masters[master.get('ID')] = package.xml(target)
        rels = package.rels('visio/pages/pages.xml')
        pages, materialized = [], {}
        declarations = package.xml('visio/pages/pages.xml').findall('v:Page', NS)
        if not declarations or len(declarations) > settings.visio_max_pages:
            raise ValueError("Visio document is empty or exceeds the page limit")
        for number, declaration in enumerate(declarations):
            page_id = declaration.get('ID', '')
            if not page_id.isdigit() or any(p['id'] == page_id for p in pages):
                raise ValueError("Visio page IDs must be unique integers")
            rel = declaration.find('v:Rel', NS)
            target = rels.get(rel.get(f'{{{R}}}id')) if rel is not None else None
            if not target:
                raise ValueError("Visio page relationship is missing")
            root = package.xml(target)
            shapes, count = {}, [0]
            from src.ingestion.visio_text import source_layout, group_origin
            page_height_cell = declaration.find('v:PageSheet/v:Cell[@N="PageHeight"]', NS)
            try:
                page_height = float(page_height_cell.get('V')) if page_height_cell is not None else None
                scale_cells = {c.get('N'): c.get('V') for c in declaration.findall('v:PageSheet/v:Cell', NS)}
                page_scale = float(scale_cells.get('PageScale', 1)) / float(scale_cells.get('DrawingScale', 1))
            except (ValueError, TypeError, ZeroDivisionError):
                page_height, page_scale = None, None
            def visit(element, parent='', inherited_master=None, parent_offset=(0, 0)):
                count[0] += 1
                if count[0] > settings.visio_max_shapes:
                    raise ValueError("Visio page exceeds the shape limit")
                shape_id = element.get('ID', '')
                if not shape_id or shape_id in shapes:
                    raise ValueError("Invalid or duplicate Visio shape ID")
                master = masters.get(element.get('Master'), inherited_master)
                base = None
                if master is not None:
                    master_id = element.get('MasterShape')
                    candidates = master.findall('.//v:Shape', NS)
                    base = next((s for s in candidates if s.get('ID') == master_id), None) if master_id else next(iter(candidates), None)
                line_ends = {name: _line_end_cell(element, base, styles, default_line, name) for name in _LINE_END_CELLS}
                connector = _connector_evidence(element, base, line_ends)
                if render_copy is not None:
                    # Materialize only known saved line-end values. Preserve native
                    # geometry and unknown/custom formulas for libvisio to interpret.
                    for name, evidence in line_ends.items():
                        if evidence['value'] is None or evidence['status'] == 'default':
                            continue
                        cell = element.find(f'v:Cell[@N="{name}"]', NS)
                        if cell is not None:
                            continue  # Native parser retains existing formulas and overrides.
                        cell = ET.Element('{' + V + '}Cell', N=name, V=str(evidence['value']))
                        element.insert(0, cell)
                resolved_text = _resolved_text(element, base)
                text = ''.join(resolved_text.itertext()).strip() if resolved_text is not None else ''
                if render_copy is not None and resolved_text is not None:
                    _materialize_text_style(element, base)
                    original = element.find('v:Text', NS)
                    if original is not None:
                        element.remove(original)
                    children = element.find('v:Shapes', NS)
                    element.insert(element.index(children) if children is not None else len(element), resolved_text)
                props = _properties(base)
                props.update(_properties(element))
                shapes[shape_id] = {'id': shape_id, 'name': element.get('NameU') or element.get('Name') or '',
                    'text': text or '', 'parent': parent, 'properties': list(props.values()),
                    'text_layout': source_layout(element, base, styles, default_text, parent, page_height, parent_offset, page_scale) if text else None,
                    'connector': connector, 'line_ends': line_ends,
                    'foreign_type': (element.find('v:ForeignData', NS).get('ForeignType', '') if element.find('v:ForeignData', NS) is not None else '')}
                for child in element.findall('v:Shapes/v:Shape', NS):
                    visit(child, shape_id, master, group_origin(element, base, parent_offset))
            for element in root.findall('v:Shapes/v:Shape', NS):
                visit(element)
            if render_copy is not None:
                materialized[target] = ET.tostring(root, encoding='UTF-8', xml_declaration=True)
            connections = [{k: c.get(k, '') for k in ('FromSheet', 'FromCell', 'ToSheet', 'ToCell')}
                           for c in root.findall('v:Connects/v:Connect', NS)]
            if len(connections) > settings.visio_max_shapes * 4:
                raise ValueError("Visio page exceeds the connection limit")
            name = declaration.get('Name') or declaration.get('NameU') or f'Page {number + 1}'
            lines = [f'Visio page {number + 1}: {name}']
            for shape in shapes.values():
                line = f"Shape {shape['id']} ({shape['name']})"
                if shape['parent']:
                    line += f" in group {shape['parent']}"
                if shape['text']:
                    line += ': ' + shape['text']
                if shape['properties']:
                    line += '; ' + '; '.join(f'{k}: {v}' for k, v in shape['properties'])
                lines.append(line)
            for connection in connections:
                def label(key):
                    sid = connection[key]
                    return f"shape {sid} ({shapes.get(sid, {}).get('text') or shapes.get(sid, {}).get('name') or 'unlabeled'})"
                lines.append(f"Documented attachment: {label('FromSheet')} {connection['FromCell']} connects to {label('ToSheet')} {connection['ToCell']}.")
            lines.extend(_describe_connectors(shapes, connections))
            lines.append('Connector attachments describe drawing structure; traffic direction, protocols and failover behavior are not inferred.')
            pages.append({'id': page_id, 'name': name, 'page': number,
                'background': declaration.get('Background', '0').lower() in ('1', 'true'),
                'back_page': declaration.get('BackPage', ''), 'text': '\n'.join(lines),
                'shapes': list(shapes.values()), 'connections': connections,
                'raster_images': sum(s['foreign_type'] == 'Bitmap' for s in shapes.values()),
                'complex_objects': sum(s['foreign_type'] in ('Object', 'MetaFile', 'EnhMetaFile') for s in shapes.values())})
        by_id = {p['id']: p for p in pages}
        for page in pages:
            background, seen = page['back_page'], {page['id']}
            page['background_ids'] = []
            while background and background != '4294967295':
                if background in seen or background not in by_id:
                    raise ValueError("Invalid or cyclic Visio background reference")
                seen.add(background)
                page['background_ids'].append(background)
                page['text'] += '\nBackground source: ' + by_id[background]['text'].split('\nBackground source:', 1)[0]
                background = by_id[background]['back_page']
        if render_copy is not None:
            with zipfile.ZipFile(render_copy, 'w', zipfile.ZIP_DEFLATED) as target_archive:
                for info in archive.infolist():
                    target_archive.writestr(info, materialized.get(info.filename, archive.read(info.filename)))
        blocks = [DocumentBlock('paragraph', p['page'], text=p['text'], page=p['page'], section_path=[p['name']]) for p in pages]
        return ParsedDocument(Path(path).name, 'vsdx', '\n\n'.join(p['text'] for p in pages),
                              metadata={'visio_pages': pages}, blocks=blocks)


def _run(argv, output, timeout, max_bytes, env=None):
    """Bound native output on disk as well as elapsed time, without PIPE RAM growth."""
    with output.open('wb') as stdout, tempfile.TemporaryFile() as stderr:
        process = subprocess.Popen(argv, stdout=stdout, stderr=stderr, cwd=output.parent, env=env)
        deadline = time.monotonic() + timeout
        try:
            while process.poll() is None:
                if time.monotonic() > deadline:
                    raise TimeoutError('Visio conversion timed out')
                if output.stat().st_size > max_bytes or stderr.seek(0, 2) > 1024**2:
                    raise ValueError('Visio converter output limit exceeded')
                time.sleep(.05)
            if output.stat().st_size > max_bytes:
                raise ValueError('Visio converter output limit exceeded')
            if process.returncode:
                raise ValueError(f'{Path(argv[0]).name} failed (exit {process.returncode})')
        finally:
            if process.poll() is None:
                process.kill()
            process.wait()


def _safe_svg(svg):
    # Never allow generated SVGs to fetch URLs or read files from outside a job.
    for element in svg.iter():
        if not isinstance(element.tag, str):
            continue
        if ET.QName(element).localname in ('script', 'foreignObject'):
            raise ValueError('Unsupported active content in converted SVG')
        for key, value in element.attrib.items():
            if ET.QName(key).localname in ('href', 'src') and not (value.startswith('#') or re.match(r'^data:image/(png|jpeg|gif|bmp);base64,', value)):
                if value.startswith('data:'):
                    raise ValueError('Unsupported embedded image format in converted SVG (for example EMF/WMF); a compatible image renderer is required')
                raise ValueError('External image reference in converted SVG')
        css = ' '.join(element.attrib.values()) + (element.text or '')
        if '@import' in css.lower() or any(not u.strip(' \t\"\'').startswith('#') for u in re.findall(r'url\((.*?)\)', css, re.I)):
            raise ValueError('External style reference in converted SVG')
    raw = ET.tostring(svg)
    return raw


def _dimensions(svg):
    box = svg.get('viewBox', '').replace(',', ' ').split()
    if len(box) == 4:
        width, height = float(box[2]), float(box[3])
    else:
        def length(value):
            match = re.fullmatch(r'([0-9.+eE-]+)(px|pt|in|cm|mm)?', value)
            if not match:
                raise ValueError('Missing or unsupported SVG page dimensions')
            return float(match[1]) * {'px': 1, 'pt': 96/72, 'in': 96, 'cm': 96/2.54, 'mm': 96/25.4, None: 1}[match[2]]
        width, height = length(svg.get('width', '')), length(svg.get('height', ''))
    if not all(math.isfinite(v) and v > 0 for v in (width, height)):
        raise ValueError('Invalid SVG page dimensions')
    scale = min(settings.visio_render_max_edge / max(width, height), math.sqrt(settings.figure_max_pixels / (width * height)))
    return max(1, int(width * scale)), max(1, int(height * scale))


def render_visio(path, parsed, progress=None):
    from PIL import Image
    from src.figures.storage import save_region
    from src.ingestion.figure_extract import FigureRecord, ImageRegion, OfficeFigureResult
    pages = parsed.metadata['visio_pages']
    warnings, records = [], []
    if not settings.figure_extraction_enabled or not settings.figure_store_enabled:
        return OfficeFigureResult(parsed.text), ['Visio PNG rendering is disabled by the figure extraction/storage settings.']
    converter, rasterizer = shutil.which('vsd2xhtml'), shutil.which('rsvg-convert')
    if not converter or not rasterizer:
        return OfficeFigureResult(parsed.text), ['Visio source text was extracted, but PNGs require libvisio-tools and librsvg2-bin.']
    # libvisio emits foreground pages in source order, then backgrounds sorted by ID.
    ordered = [p for p in pages if not p['background']] + sorted((p for p in pages if p['background']), key=lambda p: int(p['id']))
    by_id = {p['id']: p for p in pages}
    limit = settings.visio_converter_max_mb * 1024**2
    with tempfile.TemporaryDirectory(prefix='visio-', dir=path.parent) as temporary:
        work = Path(temporary)
        from src.ingestion.emf import EmfConverter, replace_embedded_emfs, analyze_diagram
        emf_converter = EmfConverter(work / 'emf')
        output = work / 'pages.xhtml'
        try:
            # libvisio can omit inherited data-graphic text. Materialize
            # saved text and line-end values in a temporary copy, keeping geometry,
            # pictures, connectors and the original source bytes unchanged.
            render_source = work / 'input.vsdx'
            parse_visio(path, render_copy=render_source)
            _run([converter, str(render_source.resolve())], output, settings.visio_timeout_seconds, limit)
            # libvisio writes an XHTML doctype. Do not load it; SVG subtrees alone are used.
            root = ET.parse(str(output), ET.XMLParser(resolve_entities=False, no_network=True, load_dtd=False)).getroot()
            svgs = root.xpath('//s:svg[not(ancestor::s:svg)]', namespaces={'s': S})
            if len(svgs) != len(ordered):
                raise ValueError(f'Page count mismatch: {len(ordered)} source pages, {len(svgs)} rendered pages; PNGs withheld to avoid incorrect citations')
        except (OSError, ValueError, ET.XMLSyntaxError, TimeoutError) as exc:
            return OfficeFigureResult(parsed.text), [f'Visio PNG conversion unavailable: {exc}']
        for index, (page, svg) in enumerate(zip(ordered, svgs)):
            if index >= settings.figure_store_max_per_doc:
                warnings.append('Visio PNG page limit reached; remaining pages retain source text only.')
                break
            if progress:
                progress(f"Rendering Visio page {page['page'] + 1}: {page['name']}")
            try:
                source_pages = [page] + [by_id[i] for i in page['background_ids']]
                page_warnings = []
                unresolved_ends = sum(1 for p in source_pages for shape in p['shapes']
                    if shape.get('connector') for name in ('BeginArrow', 'EndArrow')
                    if shape['line_ends'][name]['value'] is None)
                if unresolved_ends:
                    page_warnings.append(f'{unresolved_ends} connector line-end setting(s) could not be resolved from saved source values; arrowhead fidelity is unverified.')
                expected_images = sum(p['raster_images'] for p in source_pages)
                rendered_images = len(svg.findall(f'.//{{{S}}}image'))
                if expected_images > rendered_images:
                    page_warnings.append(f'Only {rendered_images} of at least {expected_images} embedded raster pictures appear in the converted page.')
                if any(p['complex_objects'] for p in source_pages):
                    page_warnings.append('Embedded OLE/metafile objects require visual review; converter support may be incomplete.')
                expected_text = [s['text'] for p in source_pages for s in p['shapes'] if s['text'].strip()]
                visible_text = ' '.join(' '.join(svg.itertext()).split())
                missing = [t for t in expected_text if ' '.join(t.split()) not in visible_text]
                if missing:
                    page_warnings.append(f'{len(missing)} source text label(s) were not found verbatim in the converted SVG; inspect the preview.')
                if settings.visio_repair_text_layout:
                    from src.ingestion.visio_text import repair_text
                    page_warnings.extend(repair_text(svg, source_pages))
                repairs_before = emf_converter.clip_repairs
                converted_emfs = replace_embedded_emfs(svg, emf_converter)
                if emf_converter.clip_repairs > repairs_before:
                    page_warnings.append(f"Corrected redundant full-frame clipping in {emf_converter.clip_repairs-repairs_before} embedded EMF+ image wrapper(s); source bytes unchanged.")
                source = work / 'page.svg'
                source.write_bytes(_safe_svg(svg))
                width, height = _dimensions(svg)
                png = work / 'page.png'
                _run([rasterizer, '--format=png', '--background-color=white', '--keep-aspect-ratio', '--width', str(width), '--height', str(height), str(source)], png, settings.visio_timeout_seconds, limit)
                with Image.open(png) as image:
                    if image.width * image.height > settings.figure_max_pixels:
                        raise ValueError('Rendered page exceeds the pixel limit')
                    width, height = image.size
                region = ImageRegion(page['page'], 0, png.read_bytes(), width, height,
                    source='visio_page_render', figure_id=f"visio-page-{page['id']}", caption=page['name'])
                assets = save_region(region)
                record = FigureRecord(region.figure_id, page['text'], 'diagram', page=page['page'],
                    caption=page['name'], section_path=[page['name']], source='visio_page_render', assets=assets,
                    alt_text='Converted Visio page; appearance may differ from Microsoft Visio.', analysis_status='source_extracted',
                    render_warnings=page_warnings, source_page_id=page['id'])
                if converted_emfs:
                    page_warnings.append(f'{converted_emfs} embedded EMF image(s) converted to PNG; visual fidelity requires review.')
                    if index < settings.figure_max_per_doc:
                        analyze_diagram(record, region.image_bytes, progress)
                    else:
                        page_warnings.append('EMF page OCR/vision analysis limit reached; source text and PNG retained.')
                records.append(record)
                warnings.extend(f"Visio page {page['page'] + 1} ({page['name']}): {w}" for w in page_warnings)
            except (OSError, ValueError, TimeoutError) as exc:
                warnings.append(f"Visio page {page['page'] + 1} ({page['name']}) PNG unavailable: {exc}")
    return OfficeFigureResult(parsed.text, figures=records), warnings
