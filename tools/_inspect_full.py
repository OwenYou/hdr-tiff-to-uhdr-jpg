"""Walk BOTH primary and secondary (gain map) JPEGs inside an Ultra HDR JPEG."""
import struct, sys, re

APP = {0xE0:"APP0",0xE1:"APP1",0xE2:"APP2",0xE3:"APP3",0xEB:"APP11",0xED:"APP13",0xEE:"APP14"}

def find_secondary_offset(data):
    """Return offset of secondary SOI after the primary EOI."""
    i = 2
    while i < len(data):
        if data[i] != 0xFF: return None
        while data[i] == 0xFF: i += 1
        marker = data[i]; i += 1
        if marker == 0xD9:
            # primary EOI - secondary starts at i (must begin with FFD8)
            if i+1 < len(data) and data[i] == 0xFF and data[i+1] == 0xD8:
                return i
            return None
        if marker == 0xDA:
            seglen = struct.unpack('>H', data[i:i+2])[0]
            i += seglen
            j = i
            while j < len(data)-1:
                if data[j] == 0xFF and data[j+1] not in (0x00,0xFF) and not (0xD0<=data[j+1]<=0xD7):
                    break
                j += 1
            i = j
            continue
        if marker in (0xD0,0xD1,0xD2,0xD3,0xD4,0xD5,0xD6,0xD7): continue
        seglen = struct.unpack('>H', data[i:i+2])[0]; i += seglen
    return None

def walk_segments(data, start, label):
    print(f"--- {label} (starts @ {start}) ---")
    i = start + 2  # skip SOI
    while i < len(data):
        if data[i] != 0xFF: return i
        while data[i] == 0xFF: i += 1
        marker = data[i]; i += 1
        if marker == 0xD9:
            print(f"  [@{i-2}] EOI"); return i
        if marker == 0xDA:
            seglen = struct.unpack('>H', data[i:i+2])[0]
            print(f"  [@{i-2}] SOS header_len={seglen}")
            i += seglen
            j = i
            while j < len(data)-1:
                if data[j] == 0xFF and data[j+1] not in (0x00,0xFF) and not (0xD0<=data[j+1]<=0xD7):
                    break
                j += 1
            print(f"      scan: {j-i:,} bytes")
            i = j; continue
        if marker in (0xD0,0xD1,0xD2,0xD3,0xD4,0xD5,0xD6,0xD7): continue
        seglen = struct.unpack('>H', data[i:i+2])[0]
        payload = data[i+2:i+seglen]
        tag = APP.get(marker, f"M{marker:02X}")
        ident = ""
        if marker >= 0xE0 and marker <= 0xEF:
            null = payload.find(b'\x00')
            if 0 < null < 60:
                ident = " id=" + repr(payload[:null].decode('latin-1','replace'))
        print(f"  [@{i-2}] {tag} len={seglen}{ident}")
        if marker == 0xE1 and payload.startswith(b'http://ns.adobe.com/xap/1.0/\x00'):
            xmp = payload[29:].decode('utf-8','replace')
            # Pull hdrgm fields (legacy single-channel format)
            attrs = {}
            for m in re.finditer(r'(\w+:[A-Za-z]+)\s*=\s*"([^"]*)"', xmp):
                k, v = m.group(1), m.group(2)
                if k.startswith('hdrgm:') or k.startswith('xmp:'):
                    attrs[k] = v
            for k, v in attrs.items():
                print(f"      {k} = {v}")
        if marker == 0xE2 and b'urn:iso:std:iso:ts:21496:-1' in payload[:40]:
            null = payload.find(b'\x00')
            body = payload[null+1:]
            print(f"      ISO 21496-1 body ({len(body)}B): {body.hex()}")
            parse_iso21496(body)
        i += seglen

def parse_iso21496(b):
    """Decode the ISO 21496-1 gain map metadata payload.

    Layout per libultrahdr/lib/src/gainmapmetadata.cpp `encodeGainmapMetadata`:
      u16 minimum_version (BE)
      u16 writer_version
      u8  flags  bit7 = kIsMultiChannelMask     (channelCount = 3 if set else 1)
                 bit6 = kUseBaseColorSpaceMask
                 bit2 = backwardDirection
                 bit3 = useCommonDenominator
      if useCommonDenominator:
        u32 common_denominator
        u32 baseHdrHeadroomN
        u32 alternateHdrHeadroomN
        per channel:
          s32 gainMapMinN, s32 gainMapMaxN, u32 gammaN, s32 baseOffsetN, s32 altOffsetN
      else:
        u32 baseHdrHeadroomN, u32 baseHdrHeadroomD
        u32 alternateHdrHeadroomN, u32 alternateHdrHeadroomD
        per channel (40 bytes each):
          s32 minN, u32 minD, s32 maxN, u32 maxD,
          u32 gammaN, u32 gammaD, s32 baseOffN, u32 baseOffD,
          s32 altOffN, u32 altOffD
    """
    if len(b) < 7:
        print("        (stub / marker only)"); return
    p = [0]
    def u16(): v = int.from_bytes(b[p[0]:p[0]+2],'big'); p[0]+=2; return v
    def u8():  v = b[p[0]]; p[0]+=1; return v
    def s32(): v = int.from_bytes(b[p[0]:p[0]+4],'big',signed=True); p[0]+=4; return v
    def u32(): v = int.from_bytes(b[p[0]:p[0]+4],'big',signed=False); p[0]+=4; return v

    minver = u16(); wrver = u16(); flags = u8()
    is_multi    = bool(flags & 0x80)
    use_base_cs = bool(flags & 0x40)
    backward    = bool(flags & 0x04)
    common_den  = bool(flags & 0x08)
    print(f"        min_version={minver}  writer_version={wrver}  flags=0x{flags:02X} (0b{flags:08b})")
    print(f"        is_multichannel={is_multi}  use_base_colour_space={use_base_cs}  "
          f"useCommonDenominator={common_den}  backwardDirection={backward}")

    if common_den:
        denom = u32()
        bhN, ahN = u32(), u32()
        bh = bhN/denom; ah = ahN/denom
        print(f"        base_hdr_headroom      = {bhN}/{denom} = {bh:+.6f} (log2)  -> linear {2**bh:.3f}x")
        print(f"        alternate_hdr_headroom = {ahN}/{denom} = {ah:+.6f} (log2)  -> linear {2**ah:.3f}x")
    else:
        bhN, bhD = u32(), u32()
        ahN, ahD = u32(), u32()
        bh = bhN/bhD if bhD else float('nan')
        ah = ahN/ahD if ahD else float('nan')
        print(f"        base_hdr_headroom      = {bhN}/{bhD} = {bh:+.6f} (log2)  -> linear {2**bh:.3f}x")
        print(f"        alternate_hdr_headroom = {ahN}/{ahD} = {ah:+.6f} (log2)  -> linear {2**ah:.3f}x")

    n_channels = 3 if is_multi else 1
    for ch in range(n_channels):
        if common_den:
            mnN = s32(); mxN = s32(); gN = u32(); boN = s32(); aoN = s32()
            denom_local = denom
            mn, mx, g, bo, ao = mnN/denom, mxN/denom, gN/denom, boN/denom, aoN/denom
            print(f"        ch{ch}  min={mnN}/{denom}={mn:+.4f} (log2)  "
                  f"max={mxN}/{denom}={mx:+.4f} (log2,  linear {2**mx:.3f}x)")
            print(f"             gamma={g:.4f}  baseOff={bo:.6g}  altOff={ao:.6g}")
        else:
            mnN, mnD = s32(), u32()
            mxN, mxD = s32(), u32()
            gN,  gD  = u32(), u32()
            boN, boD = s32(), u32()
            aoN, aoD = s32(), u32()
            mn = mnN/mnD if mnD else float('nan')
            mx = mxN/mxD if mxD else float('nan')
            g  = gN/gD   if gD  else float('nan')
            bo = boN/boD if boD else float('nan')
            ao = aoN/aoD if aoD else float('nan')
            print(f"        ch{ch}  min={mn:+.4f} (log2)  max={mx:+.4f} (log2,  linear {2**mx:.3f}x)")
            print(f"             gamma={g:.4f}  baseOff={bo:.4g}  altOff={ao:.4g}")

def main(path):
    print(f"\n========================= {path} =========================")
    data = open(path,'rb').read()
    print(f"file size: {len(data):,}")
    sec = find_secondary_offset(data)
    print(f"secondary SOI offset: {sec}")
    walk_segments(data, 0, "PRIMARY")
    if sec is not None:
        walk_segments(data, sec, "SECONDARY (gain map)")

if __name__ == '__main__':
    for p in sys.argv[1:]:
        main(p)
