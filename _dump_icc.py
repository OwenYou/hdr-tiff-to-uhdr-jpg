"""Extract every ICC profile from an UltraHDR JPEG (primary + secondary) and
decode key tags. ICC profiles in JPEG can be split across multiple APP2 segments
under the ICC_PROFILE identifier; we reassemble them."""
import struct, sys, re

def find_segments(data, start):
    i = start + 2  # skip SOI
    out = []
    while i < len(data):
        if data[i] != 0xFF: return out
        while data[i] == 0xFF: i += 1
        m = data[i]; i += 1
        if m == 0xD9: return out
        if m == 0xDA:
            seglen = struct.unpack('>H', data[i:i+2])[0]
            i += seglen
            j = i
            while j < len(data)-1:
                if data[j] == 0xFF and data[j+1] not in (0x00,0xFF) and not (0xD0<=data[j+1]<=0xD7):
                    break
                j += 1
            i = j
            continue
        if m in (0xD0,0xD1,0xD2,0xD3,0xD4,0xD5,0xD6,0xD7): continue
        seglen = struct.unpack('>H', data[i:i+2])[0]
        out.append((i-2, m, data[i+2:i+seglen]))
        i += seglen
    return out

def find_secondary_offset(data):
    i = 2
    while i < len(data):
        if data[i] != 0xFF: return None
        while data[i] == 0xFF: i += 1
        m = data[i]; i += 1
        if m == 0xD9:
            if i+1 < len(data) and data[i] == 0xFF and data[i+1] == 0xD8:
                return i
            return None
        if m == 0xDA:
            seglen = struct.unpack('>H', data[i:i+2])[0]
            i += seglen
            j = i
            while j < len(data)-1:
                if data[j] == 0xFF and data[j+1] not in (0x00,0xFF) and not (0xD0<=data[j+1]<=0xD7):
                    break
                j += 1
            i = j; continue
        if m in (0xD0,0xD1,0xD2,0xD3,0xD4,0xD5,0xD6,0xD7): continue
        seglen = struct.unpack('>H', data[i:i+2])[0]; i += seglen
    return None

def reassemble_icc(segs):
    chunks = {}
    total = None
    for off, m, payload in segs:
        if m != 0xE2: continue
        if not payload.startswith(b'ICC_PROFILE\x00'): continue
        chunk_idx = payload[12]
        chunk_cnt = payload[13]
        body = payload[14:]
        chunks[chunk_idx] = body
        total = chunk_cnt
    if total is None: return None
    return b''.join(chunks[i] for i in range(1, total+1))

def decode_icc_text_descriptions(icc):
    """ICC v2 has 'desc' tag with text; ICC v4 has 'desc' with mluc. We just
    extract any printable strings near recognized tag signatures."""
    out = {}
    if len(icc) < 128: return out
    # Header parsing
    size = struct.unpack('>I', icc[0:4])[0]
    cmm  = icc[4:8].decode('latin-1','replace')
    prof_ver_raw = icc[8:12]
    profile_ver = f"{prof_ver_raw[0]}.{prof_ver_raw[1]>>4}.{prof_ver_raw[1]&0xF}"
    dev_class = icc[12:16].decode('latin-1','replace')
    color_space = icc[16:20].decode('latin-1','replace')
    pcs        = icc[20:24].decode('latin-1','replace')
    out['_header'] = dict(size=size, cmm=cmm, profile_version=profile_ver,
                          device_class=dev_class, color_space=color_space, pcs=pcs)
    # Tag table at offset 128: u32 tag_count then (sig, off, size) entries
    tag_cnt = struct.unpack('>I', icc[128:132])[0]
    tags = []
    for i in range(tag_cnt):
        base = 132 + i*12
        sig = icc[base:base+4].decode('latin-1','replace')
        off = struct.unpack('>I', icc[base+4:base+8])[0]
        sz  = struct.unpack('>I', icc[base+8:base+12])[0]
        tags.append((sig, off, sz))
    out['_tags'] = tags
    # Pull descriptions from interesting tags
    for sig, off, sz in tags:
        if sig in ('desc', 'cprt', 'dmnd', 'dmdd'):
            blob = icc[off:off+sz]
            text = ''
            # ICC v2 descType is 'desc' record with ASCII length+text after sig+pad
            type_sig = blob[0:4].decode('latin-1','replace')
            if type_sig == 'desc':
                ascii_len = struct.unpack('>I', blob[8:12])[0]
                text = blob[12:12+ascii_len].rstrip(b'\x00').decode('latin-1','replace')
            elif type_sig == 'mluc':
                rec_cnt = struct.unpack('>I', blob[8:12])[0]
                rec_sz  = struct.unpack('>I', blob[12:16])[0]
                if rec_cnt:
                    lang = blob[16:20].decode('latin-1','replace')
                    str_len = struct.unpack('>I', blob[20:24])[0]
                    str_off = struct.unpack('>I', blob[24:28])[0]
                    text = blob[str_off:str_off+str_len].decode('utf-16-be','replace')
                    text = f"[{lang}] {text}"
            elif type_sig == 'text':
                text = blob[8:].rstrip(b'\x00').decode('latin-1','replace')
            out[sig] = text
    return out

def main(path):
    print(f"\n========================= {path} =========================")
    data = open(path,'rb').read()
    sec_off = find_secondary_offset(data)

    print(f"--- PRIMARY image (offset 0) ---")
    segs = find_segments(data, 0)
    icc = reassemble_icc(segs)
    if icc:
        print(f"  ICC profile: {len(icc)} bytes")
        info = decode_icc_text_descriptions(icc)
        h = info.pop('_header'); tags = info.pop('_tags', [])
        for k,v in h.items(): print(f"    {k}: {v}")
        print(f"    tag count: {len(tags)}  tags: {[t[0] for t in tags]}")
        for k,v in info.items(): print(f"    {k}: {v}")
    else:
        print("  (no ICC profile)")

    if sec_off:
        print(f"\n--- SECONDARY image / gain map (offset {sec_off}) ---")
        segs = find_segments(data, sec_off)
        icc = reassemble_icc(segs)
        if icc:
            print(f"  ICC profile: {len(icc)} bytes")
            info = decode_icc_text_descriptions(icc)
            h = info.pop('_header'); tags = info.pop('_tags', [])
            for k,v in h.items(): print(f"    {k}: {v}")
            print(f"    tag count: {len(tags)}  tags: {[t[0] for t in tags]}")
            for k,v in info.items(): print(f"    {k}: {v}")
        else:
            print("  (no ICC profile)")

if __name__ == '__main__':
    for p in sys.argv[1:]:
        main(p)
