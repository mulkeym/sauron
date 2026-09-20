"""Compatibility repair for full-frame EMF+ image wrappers.

Some Office dual EMFs repeat a device-sized frame rectangle in their GDI fallback
under a nonidentity world transform. Inkscape intersects that rectangle as logical
coordinates, cutting away valid artwork. Use the authoritative EMF+ image mapping
to recognize ONLY a single image filling the unchanged frame. Keep the initial
frame clip, every drawing record, and every transform; omit redundant later frame
intersections in the temporary renderer input. Original bytes are never changed.

This is deliberately not a general EMF+ interpreter. Unknown records, partial
image placement, additional drawings, or any other GDI clipping fail closed.
"""
from __future__ import annotations

import math
import struct

IDENTITY = (1., 0., 0., 1., 0., 0.)


def _records(raw):
    offset = 0
    while offset < len(raw):
        kind, size = struct.unpack_from('<II', raw, offset)
        if size < 8 or size % 4 or offset + size > len(raw):
            raise ValueError('Invalid EMF record')
        yield kind, raw[offset:offset+size]
        offset += size


def _plus_records(records):
    for kind, record in records:
        if kind != 70 or record[12:16] != b'EMF+':
            continue
        count, = struct.unpack_from('<I', record, 8)
        if count < 4 or 12+count > len(record):
            raise ValueError('Invalid EMF+ comment')
        offset, end = 16, 12+count
        while offset < end:
            kind, flags, size, length = struct.unpack_from('<HHII', record, offset)
            if size < 12 or size % 4 or size != length+12 or offset+size > end:
                raise ValueError('Invalid EMF+ record')
            yield kind, flags, record[offset+12:offset+size]
            offset += size


def _close(values, expected, tolerance=.01):
    return len(values) == len(expected) and all(math.isfinite(x) and abs(x-y) <= tolerance for x,y in zip(values, expected))


def _full_frame_image(records, width, height):
    world = IDENTITY
    clipped = False
    objects, saved = {}, {}
    draws = 0
    header = ended = False
    # These change raster quality, not coordinates or clipping.
    quality = {0x401e, 0x401f, 0x4021, 0x4022, 0x4023, 0x4024}
    for kind, flags, data in _plus_records(records):
        if ended:
            return False
        if kind == 0x4001:
            if header or len(data) != 16 or flags != 1:
                return False
            header = True
        elif not header:
            return False
        elif kind == 0x4002:
            ended = True
        elif kind == 0x4008:  # uncontinued object, only known wrapper object types
            if flags & 0x8000 or (flags & 255) > 63 or len(data) < 4:
                return False
            object_type = (flags >> 8) & 127
            if object_type not in (5, 8, 4):
                return False
            objects[flags & 255] = (object_type, data)
        elif kind == 0x4030:  # pixel page units and unit scale only
            if flags != 2 or len(data) != 4 or struct.unpack('<f', data)[0] != 1:
                return False
        elif kind == 0x402a:
            if flags or len(data) != 24:
                return False
            world = struct.unpack('<6f', data)
            if not all(math.isfinite(x) for x in world):
                return False
        elif kind == 0x402b:
            world = IDENTITY
        elif kind == 0x402c:  # only identity multiplication is needed by wrappers
            if flags not in (0, 0x2000) or len(data) != 24 or not _close(struct.unpack('<6f', data), IDENTITY, 0):
                return False
        elif kind == 0x4025:
            if len(data) != 4 or len(saved) >= 64:
                return False
            saved[struct.unpack('<I', data)[0]] = (world, clipped)
        elif kind == 0x4026:
            if len(data) != 4 or struct.unpack('<I', data)[0] not in saved:
                return False
            world, clipped = saved[struct.unpack('<I', data)[0]]
        elif kind == 0x4032:
            if draws:
                continue  # no subsequent drawing may pass the single-image check
            if flags not in (0, 0x100) or len(data) != 16 or world != IDENTITY:
                return False
            if not _close(struct.unpack('<4f', data), (0, 0, width, height)):
                return False
            clipped = True
        elif kind == 0x4034 and draws:
            continue
        elif kind in quality:
            continue
        elif kind == 0x401b:
            if draws or not clipped or flags > 63 or len(data) != 52:
                return False
            attributes, unit, sx, sy, sw, sh, count, x0, y0, x1, y1, x2, y2 = struct.unpack('<II4fI6f', data)
            image = objects.get(flags)
            attrs = objects.get(attributes)
            if not image or image[0] != 5 or not attrs or attrs[0] != 8 or unit != 2 or count != 3:
                return False
            # Accept the normal no-effect ImageAttributes object only.
            if len(attrs[1]) != 24 or struct.unpack_from('<5I', attrs[1], 4) != (1, 3, 0, 0, 0):
                return False
            metadata = image[1]
            if len(metadata) < 104:
                return False
            image_type, metafile_type, byte_count = struct.unpack_from('<III', metadata, 4)
            if image_type != 2 or metafile_type not in (3, 4, 5) or byte_count != len(metadata)-16 or metadata[56:60] != b' EMF':
                return False
            # No source-to-destination crop/warp. World transform must map this
            # image to precisely the outer frame, with no rotation/reflection.
            if sw <= 0 or sh <= 0 or not _close((x0,y0,x1,y1,x2,y2), (sx,sy,sx+sw,sy,sx,sy+sh)):
                return False
            a,b,c,d,e,f = world
            if a <= 0 or d <= 0 or b != 0 or c != 0:
                return False
            if not _close((a*x0+e,d*y0+f,a*x1+e,d*y1+f,a*x2+e,d*y2+f), (0,0,width,0,0,height)):
                return False
            draws += 1
        else:
            return False
    return header and ended and draws == 1


def normalize_frame_clips(raw):
    """Return temporary renderer bytes and repair count; unsupported files unchanged.

    Called only after validate_emf. No unchecked allocation from record sizes.
    """
    try:
        left, top, right, bottom = struct.unpack_from('<4i', raw, 8)
        if left != 0 or top != 0 or right <= 0 or bottom <= 0:
            return raw, 0
        frame = (0, 0, right+1, bottom+1)
        records = list(_records(raw))
        clips = []
        transformed = False
        for index, (kind, record) in enumerate(records):
            # Other clip and coordinate mapping operations need a full engine.
            if kind in (9, 10, 11, 12, 17, 26, 28, 29, 31, 32, 67, 75):
                # Office fixes the initial frame as the meta region before
                # installing any world transform; keep these records intact.
                if kind == 28 and not transformed:
                    continue
                return raw, 0
            if kind in (35, 36):
                transformed = True
            if kind == 30:
                if len(record) != 24 or struct.unpack_from('<4i', record, 8) != frame:
                    return raw, 0
                if not clips and transformed:
                    return raw, 0
                clips.append(index)
        if len(clips) < 2 or not _full_frame_image(records, frame[2], frame[3]):
            return raw, 0
        omit = set(clips[1:])
        result = bytearray(b''.join(record for index, (_, record) in enumerate(records) if index not in omit))
        struct.pack_into('<II', result, 48, len(result), len(records)-len(omit))
        return bytes(result), len(omit)
    except (ValueError, struct.error, OverflowError):
        return raw, 0
