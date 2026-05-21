"""Walk JPEG segments for an Ultra HDR JPEG. Print marker map + identifiers."""
import struct, sys, re

APP_NAMES = {0xE0:"APP0",0xE1:"APP1",0xE2:"APP2",0xE3:"APP3",0xE4:"APP4",0xE5:"APP5",
             0xE6:"APP6",0xE7:"APP7",0xE8:"APP8",0xE9:"APP9",0xEA:"APP10",0xEB:"APP11",
             0xEC:"APP12",0xED:"APP13",0xEE:"APP14",0xEF:"APP15"}

def walk(path):
    data = open(path, 'rb').read()
    print(f"=== {path}  ({len(data):,} bytes) ===")
    if data[:2] != b'\xff\xd8':
        print("not a JPEG"); return
    i = 2
    n = 0
    while i < len(data):
        if data[i] != 0xFF:
            print(f"  [{i}] not a marker (byte={data[i]:02x}) — stop"); break
        while i < len(data) and data[i] == 0xFF: i += 1
        if i >= len(data): break
        marker = data[i]; i += 1
        if marker == 0xD9:
            print(f"  [@{i-2}] EOI")
            if i < len(data):
                print(f"      trailing bytes after primary EOI: {len(data)-i:,} (MPF secondary image)")
            break
        if marker == 0xDA:
            seglen = struct.unpack('>H', data[i:i+2])[0]
            print(f"  [@{i-2}] SOS header_len={seglen}")
            i += seglen
            j = i
            while j < len(data)-1:
                if data[j] == 0xFF and data[j+1] not in (0x00, 0xFF) and not (0xD0 <= data[j+1] <= 0xD7):
                    break
                j += 1
            print(f"      scan data: {j-i:,} bytes")
            i = j
            continue
        if marker in (0xD0,0xD1,0xD2,0xD3,0xD4,0xD5,0xD6,0xD7):
            continue
        if i+2 > len(data):
            print(f"  [@{i-2}] truncated marker {marker:02x}"); break
        seglen = struct.unpack('>H', data[i:i+2])[0]
        payload = data[i+2:i+seglen]
        tag = APP_NAMES.get(marker, f"M{marker:02X}")
        prefix = ""
        if marker in APP_NAMES:
            null = payload.find(b'\x00')
            if 0 < null < 80:
                ident = payload[:null].decode('latin-1','replace')
            else:
                ident = repr(payload[:48])
            prefix = f"  id={ident!r}"
        print(f"  [@{i-2}] {tag} len={seglen}{prefix}")
        if marker == 0xE1 and payload.startswith(b'http://ns.adobe.com/xap/1.0/\x00'):
            xmp = payload[29:]
            print(f"      XMP packet: {len(xmp)} bytes")
            text = xmp.decode('utf-8','replace')
            for key in ('GainMapMin','GainMapMax','GainMapVersion','HDRCapacityMin','HDRCapacityMax',
                        'BaseRenditionIsHDR','OffsetSDR','OffsetHDR','Gamma','hdrgm:Version',
                        'Version'):
                for m in re.finditer(rf'{key}="?([^"<\s]+)"?', text):
                    print(f"        {key} = {m.group(1)}")
        if marker == 0xE1 and payload.startswith(b'http://ns.adobe.com/xmp/extension/'):
            print(f"      extended XMP: {len(payload)} bytes")
        if marker == 0xE2:
            for token in (b'urn:iso:std:iso:ts:21496:-1', b'MPF', b'ICC_PROFILE'):
                if token in payload[:64]:
                    print(f"      contains: {token!r}")
        i += seglen
        n += 1
        if n > 80:
            print("  ... (truncated)"); break

if __name__ == '__main__':
    for p in sys.argv[1:]:
        walk(p)
