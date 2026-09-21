"""Repair simple Visio text blocks that libvisio exports without SVG line layout.

Saved source coordinates, drawing scale, and styles drive layout. Translation-only
groups, styled runs, paragraph spacing and ordinary hanging bullets are supported; rotated/flipped groups and ambiguous source
matches remain with the converter. Preview font fitting never changes source text.
"""
from __future__ import annotations

import copy
import math
import re
import shutil
import subprocess
from functools import lru_cache

from src.config import settings

from lxml import etree as ET

V = 'http://schemas.microsoft.com/office/visio/2012/main'
S = 'http://www.w3.org/2000/svg'
NS = {'v': V}


def source_layout(shape, base, styles, default_style, parent, page_height, parent_offset=None, page_scale=1.0):
    if (parent and parent_offset is None) or shape.get('Type') not in ('Shape', 'Group') or shape.get('OneD') == '1':
        return None
    owners = [shape] + ([base] if base is not None else [])
    sid = shape.get('TextStyle', base.get('TextStyle', default_style) if base is not None else default_style)
    seen = set()
    while sid is not None:
        if sid in seen or sid not in styles:
            return None
        seen.add(sid)
        owners.append(styles[sid])
        sid = styles[sid].get('TextStyle')
    def value(name, default=None, paragraph=False, row="0"):
        query = f'v:Section[@N="Paragraph"]/v:Row[@IX="{row}"]/v:Cell[@N="{name}"]' if paragraph else f'v:Cell[@N="{name}"]'
        for owner in owners:
            cell = owner.find(query, NS)
            if cell is None and paragraph and row != '0':
                cell = owner.find(f'v:Section[@N="Paragraph"]/v:Row[@IX="0"]/v:Cell[@N="{name}"]', NS)
            if cell is not None and cell.get('V', ''):
                try:
                    result = float(cell.get('V', ''))
                    return result if math.isfinite(result) else None
                except ValueError:
                    return None
        return default
    try:
        if any(value(n, 0) != 0 for n in ('Angle', 'TxtAngle', 'FlipX', 'FlipY', 'TextDirection', 'HideText')):
            return None
        # Only paragraph rows actually used by the source text affect layout.
        text_node = next((o.find('v:Text', NS) for o in owners[:2] if o.find('v:Text', NS) is not None), None)
        paragraph_rows = _paragraph_rows(text_node)
        paragraphs = []
        for row in paragraph_rows:
            props = {name: value(name, default, True, row) for name, default in (
                ('IndFirst', 0), ('IndLeft', 0), ('IndRight', 0), ('SpBefore', 0), ('SpAfter', 0),
                ('Bullet', 0), ('HorzAlign', 1), ('SpLine', -1.2))}
            if (any(v is None or not math.isfinite(v) for v in props.values())
                    or props['Bullet'] not in (0, 1) or props['HorzAlign'] not in (0, 1, 2)
                    or any(props[n] < 0 for n in ('IndLeft','IndRight','SpBefore','SpAfter'))
                    or (props['IndFirst'] and not props['Bullet'])
                    or not -10 <= props['SpLine'] <= 10 or props['SpLine'] == 0
                    or (props['Bullet'] and (props['IndFirst'] >= 0 or props['IndLeft'] + props['IndFirst'] < 0))):
                return None
            paragraphs.append(props)
        width, height = value('Width'), value('Height')
        tw, th = value('TxtWidth', width), value('TxtHeight', height)
        x = value('PinX') - value('LocPinX', width/2) + value('TxtPinX', width/2) - value('TxtLocPinX', tw/2)
        y = value('PinY') - value('LocPinY', height/2) + value('TxtPinY', height/2) - value('TxtLocPinY', th/2)
        if parent_offset is not None:
            x += parent_offset[0]
            y += parent_offset[1]
        margins = [value(n, 0) * 72 * page_scale for n in ('LeftMargin', 'RightMargin', 'TopMargin', 'BottomMargin')]
        align, vertical, spacing = value('HorzAlign', 1, True), value('VerticalAlign', 1), value('SpLine', -1.2, True)
        if align not in (0, 1, 2) or vertical not in (0, 1, 2) or spacing is None or spacing == 0:
            return None
        result = {'left': x*72*page_scale, 'top': (page_height-y-th)*72*page_scale, 'width': tw*72*page_scale, 'height': th*72*page_scale,
                  'margins': margins, 'align': int(align), 'vertical': int(vertical), 'line_spacing': spacing, 'paragraphs': paragraphs, 'page_scale': page_scale}
        if not all(math.isfinite(v) for v in (x,y,tw,th,page_scale,*margins)) or page_scale <= 0 or min(tw, th) <= 0 or min(margins) < 0:
            return None
        return result
    except (TypeError, ValueError, ZeroDivisionError):
        return None


def _paragraph_rows(text_node):
    """Track pp markers through literal source newlines, without changing text."""
    if text_node is None:
        return ['0']
    stream, row = [], '0'
    stream.extend((ch, row) for ch in (text_node.text or ''))
    for child in text_node:
        if child.tag == f'{{{V}}}pp':
            row = child.get('IX', '0')
        stream.extend((ch, row) for ch in ''.join(child.itertext()))
        stream.extend((ch, row) for ch in (child.tail or ''))
    # Element.itertext does not include its own tail. Match parse_visio.strip().
    start, end = 0, len(stream)
    while start < end and stream[start][0].isspace(): start += 1
    while end > start and stream[end-1][0].isspace(): end -= 1
    stream = stream[start:end]
    rows, current = [], []
    for index, (char, style) in enumerate(stream):
        if char == '\r':
            if index+1 < len(stream) and stream[index+1][0] == '\n': continue
            char = '\n'
        if char == '\n':
            rows.append(next((r for c, r in current if not c.isspace()), style)); current = []
        else:
            current.append((char, style))
    rows.append(next((r for c, r in current if not c.isspace()), row))
    return rows


def group_origin(shape, base, parent_offset):
    """Translation-only group coordinates; unsupported rotations/flips fail closed."""
    if parent_offset is None:
        return None
    def value(name, default=None):
        for owner in (shape, base):
            if owner is None:
                continue
            cell = owner.find(f'v:Cell[@N="{name}"]', NS)
            if cell is not None:
                return float(cell.get('V', ''))
        return default
    try:
        if any(value(n, 0) != 0 for n in ('Angle', 'FlipX', 'FlipY')):
            return None
        width, height = value('Width'), value('Height')
        x = parent_offset[0] + value('PinX') - value('LocPinX', width / 2)
        y = parent_offset[1] + value('PinY') - value('LocPinY', height / 2)
        return (x, y) if all(math.isfinite(v) for v in (x, y)) else None
    except (TypeError, ValueError):
        return None


@lru_cache(maxsize=128)
def _font_path(family, bold, italic):
    matcher = shutil.which('fc-match')
    if not matcher:
        raise ValueError('fontconfig is unavailable for text layout')
    style = 'Bold Italic' if bold and italic else ('Bold' if bold else ('Italic' if italic else 'Regular'))
    found = subprocess.run([matcher, '-f', '%{file}', '--', family[:200] + ':style=' + style],
        capture_output=True, timeout=5, check=True).stdout.decode().strip()
    if not found or len(found) > 4096:
        raise ValueError('Invalid font match')
    return found


@lru_cache(maxsize=512)
def _font(family, bold, italic, size):
    from PIL import ImageFont
    return ImageFont.truetype(_font_path(family, bold, italic), max(1, round(size * 4)))


def _wrap(text, width, measure):
    lines = []
    for paragraph in text.replace('\r\n', '\n').replace('\r', '\n').split('\n'):
        if not paragraph.strip():
            lines.append('')
            continue
        line = ''
        for word in paragraph.split():
            if line and measure(line + ' ' + word) > width:
                lines.append(line); line = ''
            # Long identifiers must remain intact in text but can wrap by glyph.
            candidate = (line + ' ' + word).strip()
            if measure(candidate) <= width:
                line = candidate
                continue
            for char in word:
                if line and measure(line + char) > width:
                    lines.append(line); line = ''
                line += char
        lines.append(line)
    return lines


def _source_runs(text, spans):
    """Map source line breaks onto converter character styles without guessing text."""
    glyphs = [(char, index) for index, span in enumerate(spans)
              for char in (span.text or '') if not char.isspace()]
    if ''.join(c for c, _ in glyphs) != ''.join(text.split()):
        raise ValueError('Source/converter text mismatch')
    result, cursor = [], 0
    for char in text.replace('\r\n', '\n').replace('\r', '\n'):
        if not char.isspace():
            style = glyphs[cursor][1]; cursor += 1
        else:
            style = glyphs[min(cursor, len(glyphs)-1)][1] if glyphs else 0
        result.append((char, style))
    return result


def _layout_runs(chars, styles, width, spacing, scale, compact=False):
    fonts = [_font(s.get('font-family', 'sans-serif'), s.get('font-weight') == 'bold',
                   s.get('font-style') == 'italic', float(s['font-size']) * scale) for s in styles]
    def runs(line):
        result = []
        for char, style in line:
            if result and result[-1][1] == style:
                result[-1] = (result[-1][0] + char, style)
            else:
                result.append((char, style))
        return result
    def measure(line):
        return sum(fonts[i].getlength(text)/4 for text, i in runs(line))
    lines, line, word = [], [], []
    def add_word():
        nonlocal line, word
        if not word:
            return
        combined = line + ([(' ', word[0][1])] if line else []) + word
        if line and measure(combined) > width:
            lines.append(line); line = []
        if measure(word) <= width:
            if line: line.append((' ', word[0][1]))
            line += word
        else:
            for char in word:
                if line and measure(line + [char]) > width:
                    lines.append(line); line = []
                line.append(char)
        word = []
    for char in chars:
        if char[0].isspace():
            add_word()
            if char[0] == '\n':
                lines.append(line); line = []
        else:
            word.append(char)
    add_word(); lines.append(line)
    if len(lines) > 1000:
        raise ValueError('Text line limit exceeded')
    laid_out, baseline = [], 0.0
    for index, line in enumerate(lines):
        indices = {i for _, i in line} or {0}
        sizes = [float(styles[i]['font-size']) * scale for i in indices]
        asc = max(fonts[i].getmetrics()[0]/4 for i in indices)
        desc = max(fonts[i].getmetrics()[1]/4 for i in indices)
        if compact and line:
            # Center the visible glyphs, not unused font ascent/descent space.
            bounds = [fonts[i].getbbox(text, anchor='ls') for text, i in runs(line)]
            asc = max(-box[1]/4 for box in bounds)
            desc = max(box[3]/4 for box in bounds)
        advance = max(max(sizes) * -spacing if spacing < 0 else spacing*72, asc+desc)
        if index == 0:
            baseline = asc
        else:
            baseline += max(previous_advance, previous_desc + asc)
        laid_out.append({'runs': runs(line), 'width': measure(line), 'baseline': baseline})
        previous_advance, previous_desc = advance, desc
    return laid_out, baseline + desc, fonts


def _layout_paragraphs(chars, styles, width, layout, scale, compact=False):
    """Lay out saved paragraphs, including spacing and ordinary hanging bullets."""
    paragraphs = [[]]
    for char in chars:
        if char[0] == '\n': paragraphs.append([])
        else: paragraphs[-1].append(char)
    props = layout['paragraphs']
    if len(props) != len(paragraphs):
        raise ValueError('Source paragraph/style count mismatch')
    lines, y = [], 0.0
    units = 72 * layout.get('page_scale', 1)
    for paragraph_index, (chars, prop) in enumerate(zip(paragraphs, props)):
        left, right = prop['IndLeft'] * units, prop['IndRight'] * units
        available = width-left-right
        if available <= 0: raise ValueError('Paragraph indents exceed text box')
        spacing = prop['SpLine'] * layout.get('page_scale', 1) if prop['SpLine'] > 0 else prop['SpLine']
        laid, height, fonts = _layout_runs(chars, styles, available, spacing, scale, compact=compact)
        y += prop['SpBefore'] * units * scale
        for index, line in enumerate(laid):
            line.update(baseline=line['baseline']+y, offset=left, available=available, align=prop['HorzAlign'])
            if index == 0 and prop['Bullet'] == 1 and chars:
                line['bullet_offset'] = left + prop['IndFirst'] * units
            lines.append(line)
        y += height
        if paragraph_index + 1 < len(paragraphs):
            y += prop['SpAfter'] * units * scale
    return lines, y, fonts


def _separate_label_boxes(pages):
    """Narrow overlapping adjacent boxes on a shared row; never move their centers."""
    rows = {}
    for page_index, page in enumerate(pages):
        for shape in page['shapes']:
            box = shape.get('text_layout')
            if box and shape.get('text'):
                key = (page_index, shape.get('parent', ''), round(box['top']))
                rows.setdefault(key, []).append(shape)
    narrowed = {}
    for row in rows.values():
        row.sort(key=lambda s: s['text_layout']['left'] + s['text_layout']['width']/2)
        for a, b in zip(row, row[1:]):
            aa, bb = a['text_layout'], b['text_layout']
            ac, bc = aa['left'] + aa['width']/2, bb['left'] + bb['width']/2
            if bc-ac <= 2 or aa['left']+aa['width'] <= bb['left']:
                continue
            for shape, box in ((a, aa), (b, bb)):
                width = min(box['width'], bc-ac-2)
                if width <= box['margins'][0]+box['margins'][1]+4:
                    continue
                narrowed[id(shape)] = min(narrowed.get(id(shape), box['width']), width)
    return narrowed


def repair_text(svg, pages):
    """Wrap source-matched labels, preserve styled runs, and fit bounded overflow."""
    normalize = lambda text: re.sub(r'\s+', '', text)
    candidates = {}
    for page in pages:
        for shape in page['shapes']:
            if shape.get('text'):
                candidates.setdefault(normalize(shape['text']), []).append(shape)
    narrowed = _separate_label_boxes(pages)
    repaired = overflow = skipped = fitted = constrained = centered = 0
    for node in list(svg.iter(f'{{{S}}}text')):
        matches = candidates.get(normalize(''.join(node.itertext())), [])
        if len(matches) > 1:
            try:
                x, y = float(node.get('x', '')), float(node.get('y', ''))
                matches = [shape for shape in matches if shape.get('text_layout')
                    and abs(x-shape['text_layout']['left']-shape['text_layout']['margins'][0]) < 1
                    and shape['text_layout']['top']-1 <= y <= shape['text_layout']['top']+shape['text_layout']['height']+1]
            except ValueError:
                matches = []
        if len(matches) != 1 or not matches[0].get('text_layout'):
            continue
        shape = matches[0]; layout = shape['text_layout']; spans = list(node)
        if (not spans or (node.text or '').strip() or any(s.tag != f'{{{S}}}tspan' or len(s) for s in spans)
                or any(any(a in s.attrib for a in ('x', 'y', 'dx', 'dy', 'rotate')) for s in spans)
                or any(a in node.attrib for a in ('transform', 'rotate', 'dx', 'dy'))
                or any(p.get('transform') for p in node.iterancestors() if p is not svg)):
            skipped += 1
            continue
        try:
            styles = [dict(s.attrib) for s in spans]
            sizes = [float(s.get('font-size', '0')) for s in styles]
            if (not all(0 < size <= 1000 for size in sizes) or len(shape['text']) > 10000
                    or not all(math.isfinite(float(node.get(n, 'nan'))) for n in ('x', 'y'))
                    or abs(float(node.get('x'))-layout['left']-layout['margins'][0]) > 1
                    or not layout['top']-max(sizes) <= float(node.get('y')) <= layout['top']+layout['height']+max(sizes)):
                raise ValueError('Unsupported text geometry')
            if id(shape) in narrowed:
                new_width = narrowed[id(shape)]
                layout = dict(layout, left=layout['left']+(layout['width']-new_width)/2, width=new_width)
                constrained += 1
            chars = _source_runs(shape['text'], spans)
            left, right, top, bottom = layout['margins']
            compact = (layout['vertical'] == 1 and abs(top-bottom) < .001
                and layout['height'] <= max(sizes)*1.5
                and not any(c in shape['text'] for c in ('\n', '\r')))
            if compact:
                # Symmetric padding cancels in centered placement. In badges it
                # can consume the entire box: use the saved physical height and
                # actual glyph bounds instead of leaving the native baseline.
                top = bottom = 0
            width, height = layout['width']-left-right, layout['height']-top-bottom
            if width <= 0 or height <= 0:
                raise ValueError('No room in saved text block')
            spacing = layout['line_spacing']
            if not math.isfinite(spacing) or not -10 <= spacing <= 10 or spacing == 0:
                raise ValueError('Unsupported line spacing')
            def render(scale):
                if layout.get('paragraphs'):
                    return _layout_paragraphs(chars, styles, width, layout, scale, compact=compact)
                return _layout_runs(chars, styles, width, spacing, scale, compact=compact)
            lines, total, fonts = render(1)
            scale = 1.0
            if total > height or any(line['width'] > line.get('available', width) for line in lines):
                minimum = max(settings.visio_text_min_scale, min(1.0, 4 / min(sizes)))
                low, high = minimum, 1.0
                low_result = render(low)
                if low_result[1] <= height and all(line['width'] <= line.get('available', width) for line in low_result[0]):
                    for _ in range(7):
                        middle = (low+high)/2
                        candidate = render(middle)
                        if candidate[1] <= height and all(line['width'] <= line.get('available', width) for line in candidate[0]): low = middle
                        else: high = middle
                scale = low
                lines, total, fonts = render(scale)
                if scale < .999: fitted += 1
            if total > height or any(line['width'] > line.get('available', width) for line in lines): overflow += 1
            y = layout['top']+top + max(0, height-total)*(layout['vertical']/2)
            new = copy.deepcopy(node)
            for child in list(new): new.remove(child)
            new.text = None
            new.set('text-anchor', 'start')
            for line in lines:
                x = layout['left']+left + line.get('offset', 0) + max(0, line.get('available', width)-line['width'])*(line.get('align', layout['align'])/2)
                if 'bullet_offset' in line:
                    style = dict(styles[0]); style['font-size'] = f'{sizes[0]*scale:.4f}'
                    bullet = ET.SubElement(new, f'{{{S}}}tspan', style)
                    bullet.set('x', f"{layout['left']+left+line['bullet_offset']:.4f}")
                    bullet.set('y', f"{y+line['baseline']:.4f}")
                    bullet.text = '•'
                for text, index in line['runs']:
                    style = dict(styles[index]); style['font-size'] = f'{sizes[index]*scale:.4f}'
                    # Use the measured installed face, avoiding a second, different
                    # fallback choice in the SVG renderer.
                    style['font-family'] = fonts[index].getname()[0]
                    span = ET.SubElement(new, f'{{{S}}}tspan', style)
                    span.set('x', f'{x:.4f}'); span.set('y', f'{y+line["baseline"]:.4f}')
                    span.text = text
                    x += fonts[index].getlength(text)/4
            node.getparent().replace(node, new)
            repaired += 1
            centered += compact
        except (ValueError, OSError, subprocess.SubprocessError):
            skipped += 1
    warnings = []
    if repaired: warnings.append(f'Restored source text-box wrapping/alignment for {repaired} label(s); installed fonts may differ from the original.')
    if centered: warnings.append(f'Centered {centered} compact label(s) within saved text boxes using visible glyph bounds.')
    if constrained: warnings.append(f'{constrained} adjacent label box(es) narrowed around their original centers to avoid text overlap.')
    if fitted: warnings.append(f'{fitted} label(s) used a reduced preview font size to fit their saved text boxes; source text is unchanged.')
    if overflow: warnings.append(f'{overflow} label(s) still exceed their saved text boxes at the minimum preview scale; all text retained for review.')
    if skipped: warnings.append(f'{skipped} label(s) use unsupported text layout or font metrics and retain the native converter output.')
    return warnings
