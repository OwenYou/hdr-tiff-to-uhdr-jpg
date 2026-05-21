"""Dump XMP and ISO 21496-1 gain map metadata from an Ultra HDR JPEG."""
import struct, sys

def parse_iso21496_payload(payload):
    p = payload
    if p[:32] != b'urn:iso:std:iso:ts:21496:-1' + b'\x00' * (32 - len(b'urn:iso:std:iso:ts:21496:-1')):
        nz = p.find(b'\x00')
        body = p[nz+1:] if nz >= 0 else p
    else:
        body = p[32:]
    print(f"  ISO 21496-1 body: {len(body)} bytes")
    print(f"  hex: {body.hex()}")
    if len(body) >= 2:
        ver = (body[0] << 8) | body[1]
        print(f"  minimum_version={ver >> 12}  writer_version={(ver >> 8) & 0xF}  ?")
        # Per ISO 21496-1: 2-byte minimum_version (BE), 1-byte flags ...
        # Then per channel: 4-byte gainMapMin/Max, 4-byte gamma, 4-byte baseOffset/altOffset, etc.
        # Layout is intricate; we just dump bytes for cross-comparison.

def walk(path):
    print(f"=== {path} ===")
    data = open(path, 'rb').read()
    i = 2
    while i < len(data):
        if data[i] != 0xFF: break
        while data[i] == 0xFF: i += 1
        marker = data[i]; i += 1
        if marker == 0xD9: break
        if marker == 0xDA:
            seglen = struct.unpack('>H', data[i:i+2])[0]
            i += seglen
            j = i
            while j < len(data) - 1:
                if data[j] == 0xFF and data[j+1] not in (0x00, 0xFF) and not (0xD0 <= data[j+1] <= 0xD7):
                    break
                j += 1
            i = j
            continue
        if marker in (0xD0, 0xD1, 0xD2, 0xD3, 0xD4, 0xD5, 0xD6, 0xD7):
            continue
        seglen = struct.unpack('>H', data[i:i+2])[0]
        payload = data[i+2:i+seglen]
        if marker == 0xE1 and payload.startswith(b'http://ns.adobe.com/xap/1.0/\x00'):
            xmp = payload[29:].decode('utf-8', 'replace')
            print(f"-- XMP packet ({len(xmp)} bytes) --")
            print(xmp)
        if marker == 0xE2 and b'urn:iso:std:iso:ts:21496:-1' in payload[:40]:
            print(f"-- ISO 21496-1 segment ({len(payload)} bytes) --")
            parse_iso21496_payload(payload)
        if marker == 0xE2 and payload[:4] == b'MPF\x00':
            print(f"-- MPF segment ({len(payload)} bytes) --")
            print(f"  hex (first 96): {payload[:96].hex()}")
        i += seglen
    print()

if __name__ == '__main__':
    for p in sys.argv[1:]:
        walk(p)
