"""Synthetic clipping fixtures; no third-party artwork in the test suite."""
import io
import shutil
import struct

import pytest
from PIL import Image, ImageChops

from src.ingestion.emf import EmfConverter, validate_emf
from src.ingestion.emf_clipping import normalize_frame_clips, _records
from tests.test_ingestion.test_emf import sample_emf


def record(kind, body=b''):
    return struct.pack('<II', kind, 8+len(body)) + body


def plus(kind, flags=0, data=b''):
    return struct.pack('<HHII', kind, flags, 12+len(data), len(data)) + data


def package(header, records):
    raw = bytearray(header + b''.join(records))
    struct.pack_into('<II', raw, 48, len(raw), len(records)+1)
    return bytes(raw)


def wrapper(*, redundant=True, partial=False, extra_draw=False, unknown=False):
    source = sample_emf()
    header = bytearray(source[:88])
    struct.pack_into('<4i4i', header, 8, 0,0,499,299,0,0,13229,7938)
    inner = struct.pack('<4I', 0xdbc01002, 2, 4, len(source)) + source
    identity = struct.pack('<6f', 1,0,0,1,0,0)
    transform = struct.pack('<6f', .5,0,0,.5,0,0)
    commands = [plus(0x4001, 1, struct.pack('<4I', 0xdbc01002, 1, 120,120)),
        plus(0x4030, 2, struct.pack('<f', 1)),
        plus(0x402a, 0, identity),
        plus(0x4032, 0x100, struct.pack('<4f',0,0,500,300)),
        plus(0x402a, 0, transform),
        plus(0x4008, 0x500, inner),
        plus(0x4008, 0x801, struct.pack('<6I',0xdbc01002,1,3,0,0,0))]
    drawing = plus(0x401b, 0, struct.pack('<II4fI6f',1,2,0,0,1000,600,3,0,0,1000 if not partial else 800,0,0,600))
    commands.append(drawing)
    if extra_draw: commands.append(drawing)
    if unknown: commands.append(plus(0x4999))
    commands.append(plus(0x4002))
    comment = b'EMF+' + b''.join(commands)
    records = [record(70, struct.pack('<I', len(comment))+comment),
        record(30, struct.pack('<4i',0,0,500,300)),
        record(35, transform)]
    if redundant: records.append(record(30, struct.pack('<4i',0,0,500,300)))
    records.extend(r for k,r in list(_records(source))[1:])
    return package(header, records)


def test_repair_only_redundant_wrapper_clips():
    raw = wrapper()
    repaired, count = normalize_frame_clips(raw)
    assert count == 1
    assert repaired == wrapper(redundant=False)
    assert normalize_frame_clips(repaired) == (repaired, 0)
    assert validate_emf(repaired)['records'] == validate_emf(raw)['records']-1
    # All drawing, frame, transform, EMF+ metadata and initial clipping retained.
    assert [(k,r) for k,r in _records(raw) if k not in (1,30)] == [(k,r) for k,r in _records(repaired) if k not in (1,30)]


@pytest.mark.parametrize('variant', ['partial', 'extra_draw', 'unknown', 'different_clip', 'clip_path', 'viewport', 'bad_plus_length', 'nan', 'compressed', 'continuation'])
def test_uncertain_or_intentional_clipping_is_not_rewritten(variant):
    raw = wrapper(**{variant: True}) if variant in ('partial','extra_draw','unknown') else wrapper()
    if variant in ('different_clip','clip_path','viewport'):
        records = [r for _,r in list(_records(raw))[1:]]
        extra = {'different_clip': record(30,struct.pack('<4i',0,0,200,100)),
                 'clip_path': record(67,struct.pack('<I',1)),
                 'viewport': record(12,struct.pack('<2i',2,3))}[variant]
        records.insert(-1, extra)
        raw = package(raw[:88],records)
    elif variant in ('bad_plus_length','nan','compressed','continuation'):
        data = bytearray(raw)
        if variant == 'bad_plus_length': struct.pack_into('<I',data,112,0xffffffff)
        else:
            marker = { 'nan':struct.pack('<HH',0x402a,0), 'compressed':struct.pack('<HH',0x401b,0), 'continuation':struct.pack('<HH',0x4008,0x500)}[variant]
            offset = data.index(marker)
            if variant=='nan': struct.pack_into('<f',data,offset+12,float('nan'))
            elif variant=='compressed':struct.pack_into('<H',data,offset+2,0x4000)
            else:struct.pack_into('<H',data,offset+2,0x8500)
        raw = bytes(data)
    assert normalize_frame_clips(raw) == (raw, 0)


def test_plain_emf_clipping_is_unchanged():
    source = sample_emf(); records = [r for _,r in list(_records(source))[1:]]
    records.insert(0, record(30,struct.pack('<4i',0,0,300,300)))
    raw = package(source[:88],records)
    assert normalize_frame_clips(raw) == (raw,0)


@pytest.mark.skipif(not shutil.which('inkscape'), reason='native Inkscape required')
def test_native_repair_matches_full_arrow_and_preserves_intentional_crop(tmp_path, monkeypatch):
    from src.config import settings
    monkeypatch.setattr(settings, 'emf_embedded_max_edge', 500)
    converter = EmfConverter(tmp_path/'repaired')
    png = converter.convert(wrapper())
    reference = EmfConverter(tmp_path/'reference').convert(wrapper(redundant=False))
    with Image.open(io.BytesIO(png)) as im, Image.open(io.BytesIO(reference)) as ref:
        assert ImageChops.difference(im.convert('RGBA'), ref.convert('RGBA')).getbbox() is None
        # Arrow tip at 90% width survives; background beyond it remains transparent.
        assert im.convert('RGBA').getpixel((440,150))[3] > 0
        assert im.convert('RGBA').getpixel((490,150))[3] == 0
    assert converter.clip_repairs == 1
    source = sample_emf(); records = [r for _,r in list(_records(source))[1:]]
    records.insert(0, record(30,struct.pack('<4i',0,0,300,600)))
    cropped = EmfConverter(tmp_path/'cropped').convert(package(source[:88],records))
    with Image.open(io.BytesIO(cropped)) as im:
        assert im.convert('RGBA').getpixel((440,150))[3] == 0
