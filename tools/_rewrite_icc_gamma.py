#!/usr/bin/env python3
"""
Replace rTRC/gTRC/bTRC in a JPEG's embedded ICC profile with a pure
power-law gamma curve (ICC v4 para type 0: Y = X^g).

Usage:
    uv run python tools/_rewrite_icc_gamma.py <input.jpg> [output.jpg]

If output is omitted, saves as <stem>_g22<ext> in the same folder.
"""
import os
import struct
import sys


# ---------------------------------------------------------------------------
# JPEG parser / serialiser
# ---------------------------------------------------------------------------

def parse_jpeg(data: bytes) -> tuple[list[dict], bytes]:
    """Split the *primary* JPEG into segments and return (segments, trailing).

    Each segment dict has 'marker' (int) and 'data' (bytes).
    Scan data after SOS is stored as marker=0x00.
    *trailing* is everything after the primary JPEG's EOI (e.g. the secondary
    JPEG in an Ultra HDR file); it is preserved verbatim."""
    if data[:2] != b'\xff\xd8':
        raise ValueError("Not a JPEG (missing FF D8 SOI)")
    result: list[dict] = [{'marker': 0xD8, 'data': b''}]
    i = 2
    while i < len(data):
        if data[i] != 0xFF:
            raise ValueError(f"Expected 0xFF at offset {i:#x}, got {data[i]:#04x}")
        while data[i] == 0xFF:
            i += 1
        marker = data[i]; i += 1
        if marker == 0xD9:
            result.append({'marker': 0xD9, 'data': b''})
            return result, data[i:]   # return trailing (secondary JPEG, etc.)
        if 0xD0 <= marker <= 0xD7:
            result.append({'marker': marker, 'data': b''})
            continue
        seg_len = struct.unpack('>H', data[i:i+2])[0]
        payload = data[i+2:i+seg_len]
        result.append({'marker': marker, 'data': payload})
        i += seg_len
        if marker == 0xDA:
            scan_start = i
            while i < len(data) - 1:
                if data[i] == 0xFF and data[i+1] != 0x00 and not (0xD0 <= data[i+1] <= 0xD7):
                    break
                i += 1
            result.append({'marker': 0x00, 'data': data[scan_start:i]})
    return result, b''


def serialise_jpeg(segments: list[dict]) -> bytes:
    out = bytearray()
    for seg in segments:
        m, d = seg['marker'], seg['data']
        if m == 0x00:
            out += d
        elif m in (0xD8, 0xD9) or 0xD0 <= m <= 0xD7:
            out += b'\xff' + bytes([m])
        else:
            out += b'\xff' + bytes([m]) + struct.pack('>H', len(d) + 2) + d
    return bytes(out)


# ---------------------------------------------------------------------------
# ICC helpers
# ---------------------------------------------------------------------------

_ICC_HDR = b'ICC_PROFILE\x00'
_MAX_ICC_BODY = 65519   # 65535 - 2 (length field) - 14 (ICC_PROFILE\x00 + chunk bytes)


def extract_icc(segments: list[dict]) -> bytes | None:
    chunks: dict[int, bytes] = {}
    total = None
    for seg in segments:
        if seg['marker'] != 0xE2:
            continue
        p = seg['data']
        if not p.startswith(_ICC_HDR):
            continue
        idx, cnt = p[12], p[13]
        chunks[idx] = p[14:]
        total = cnt
    if total is None:
        return None
    return b''.join(chunks[i] for i in range(1, total + 1))


def build_icc_segments(icc: bytes) -> list[dict]:
    """Wrap ICC bytes in one or more APP2 (0xE2) payloads."""
    out = []
    offset = 0
    n = -(-len(icc) // _MAX_ICC_BODY)   # ceiling division
    idx = 1
    while offset < len(icc):
        chunk = icc[offset:offset + _MAX_ICC_BODY]
        out.append({'marker': 0xE2,
                    'data': _ICC_HDR + bytes([idx, n]) + chunk})
        offset += _MAX_ICC_BODY
        idx += 1
    return out


# ---------------------------------------------------------------------------
# ICC profile TRC rewriter
# ---------------------------------------------------------------------------

def _para_gamma(g: float) -> bytes:
    """ICC v4 para type 0 tag: Y = X^g  (16 bytes)."""
    # s15Fixed16Number: value = (integer << 16) | frac  (unsigned 32-bit storage)
    g_fixed = round(g * 65536)
    return struct.pack('>4sI2H I', b'para', 0, 0, 0, g_fixed)


def modify_icc_trc(icc: bytes, gamma: float = 2.2) -> bytes:
    """Return a copy of *icc* with rTRC/gTRC/bTRC/kTRC replaced by para type 0."""
    if len(icc) < 132:
        raise ValueError("ICC profile too short")

    tag_count = struct.unpack('>I', icc[128:132])[0]
    tags: list[tuple[str, int, int]] = []
    for i in range(tag_count):
        base = 132 + i * 12
        sig = icc[base:base+4].decode('latin-1')
        off = struct.unpack('>I', icc[base+4:base+8])[0]
        sz  = struct.unpack('>I', icc[base+8:base+12])[0]
        tags.append((sig, off, sz))

    trc_sigs = {'rTRC', 'gTRC', 'bTRC', 'kTRC'}
    table_end = 132 + tag_count * 12

    new_curv = _para_gamma(gamma)

    # Rebuild the data section preserving non-TRC tag data (handles shared blocks).
    seen: dict[int, int] = {}     # old_offset -> new_offset  (for dedup)
    old_to_new: dict[tuple[int,int], int] = {}
    new_data = bytearray()

    for sig, off, sz in tags:
        if sig in trc_sigs or off < table_end:
            continue
        if off in seen:
            old_to_new[(off, sz)] = seen[off]
            continue
        pad = (-len(new_data)) % 4
        new_data += b'\x00' * pad
        new_off = table_end + len(new_data)
        old_to_new[(off, sz)] = new_off
        seen[off] = new_off
        new_data += icc[off:off+sz]

    # Append the one shared TRC block.
    pad = (-len(new_data)) % 4
    new_data += b'\x00' * pad
    new_trc_off = table_end + len(new_data)
    new_data += new_curv

    # New tag table.
    table = bytearray(struct.pack('>I', tag_count))
    for sig, off, sz in tags:
        if sig in trc_sigs:
            new_off, new_sz = new_trc_off, len(new_curv)
        else:
            new_off = old_to_new.get((off, sz), off)
            new_sz = sz
        table += sig.encode('latin-1') + struct.pack('>II', new_off, new_sz)

    total = table_end + len(new_data)
    header = bytearray(icc[:128])
    struct.pack_into('>I', header, 0, total)
    header[84:100] = b'\x00' * 16   # clear profile ID (MD5 no longer valid)
    return bytes(header) + bytes(table) + bytes(new_data)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    if len(sys.argv) < 2:
        print(__doc__, file=sys.stderr)
        sys.exit(1)

    in_path = sys.argv[1]
    if len(sys.argv) >= 3:
        out_path = sys.argv[2]
    else:
        stem, ext = os.path.splitext(in_path)
        out_path = stem + '_g22' + ext

    data = open(in_path, 'rb').read()
    segments, trailing = parse_jpeg(data)

    icc = extract_icc(segments)
    if icc is None:
        print("ERROR: no ICC_PROFILE found", file=sys.stderr)
        sys.exit(1)

    print(f"Original ICC: {len(icc)} bytes")
    new_icc = modify_icc_trc(icc)
    print(f"Modified ICC: {len(new_icc)} bytes  (gamma 2.2 para type 0)")

    # Replace ICC segments in-place (keep position, discard old chunks).
    new_segs: list[dict] = []
    injected = False
    for seg in segments:
        is_icc_app2 = (seg['marker'] == 0xE2 and seg['data'].startswith(_ICC_HDR))
        if is_icc_app2:
            if not injected:
                new_segs.extend(build_icc_segments(new_icc))
                injected = True
        else:
            new_segs.append(seg)

    if not injected:
        print("ERROR: could not find ICC_PROFILE APP2 to replace", file=sys.stderr)
        sys.exit(1)

    out_data = serialise_jpeg(new_segs) + trailing
    open(out_path, 'wb').write(out_data)
    print(f"Saved → {out_path}  ({len(out_data):,} bytes)")


if __name__ == '__main__':
    main()
