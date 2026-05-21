"""Side-by-side compare ICC profiles from two JPEGs: primaries (rXYZ/gXYZ/bXYZ),
white point, chromatic adaptation, and TRC curves (parametric or table-LUT).

Goal: figure out whether 'Display P3' and 'Display P3 Gamut with sRGB Transfer'
are mathematically identical or numerically different, and what shows up in the
tag table that downstream renderers might key off."""
import struct, sys
from _dump_icc import find_secondary_offset, find_segments, reassemble_icc

def s15f16(x):
    return x / 65536.0

def parse_header(icc):
    out = {}
    out['size'] = struct.unpack('>I', icc[0:4])[0]
    out['cmm']  = icc[4:8].decode('latin-1','replace')
    out['ver']  = f"{icc[8]}.{icc[9]>>4}.{icc[9]&0xF}"
    out['devclass'] = icc[12:16].decode('latin-1','replace')
    out['cs']  = icc[16:20].decode('latin-1','replace')
    out['pcs'] = icc[20:24].decode('latin-1','replace')
    # Illuminant @ offset 68 (3 x s15Fixed16)
    iX = struct.unpack('>i', icc[68:72])[0]
    iY = struct.unpack('>i', icc[72:76])[0]
    iZ = struct.unpack('>i', icc[76:80])[0]
    out['illuminant'] = (s15f16(iX), s15f16(iY), s15f16(iZ))
    return out

def parse_tag_table(icc):
    n = struct.unpack('>I', icc[128:132])[0]
    out = []
    for i in range(n):
        b = 132 + i*12
        sig = icc[b:b+4].decode('latin-1','replace')
        off = struct.unpack('>I', icc[b+4:b+8])[0]
        sz  = struct.unpack('>I', icc[b+8:b+12])[0]
        out.append((sig, off, sz))
    return out

def parse_xyz(blob):
    sig = blob[0:4]
    if sig != b'XYZ ':
        return None
    x = struct.unpack('>i', blob[8:12])[0]
    y = struct.unpack('>i', blob[12:16])[0]
    z = struct.unpack('>i', blob[16:20])[0]
    return (s15f16(x), s15f16(y), s15f16(z))

def parse_trc(blob):
    sig = blob[0:4]
    if sig == b'para':
        # ICC parametric curve
        func = struct.unpack('>H', blob[8:10])[0]
        # Reserved >H @ 10
        params = []
        # number of parameters depends on func
        nparam = {0:1, 1:3, 2:4, 3:5, 4:7}.get(func, 0)
        for i in range(nparam):
            p = struct.unpack('>i', blob[12+i*4:16+i*4])[0]
            params.append(s15f16(p))
        return ('para', func, params)
    elif sig == b'curv':
        n = struct.unpack('>I', blob[8:12])[0]
        if n == 0:
            return ('curv', 0, [])  # identity
        if n == 1:
            # single u8.8 gamma
            g = struct.unpack('>H', blob[12:14])[0] / 256.0
            return ('curv', 1, [g])
        vals = struct.unpack('>'+'H'*n, blob[12:12+2*n])
        return ('curv', n, list(vals))
    return ('?', sig.decode('latin-1','replace'), [])

def srgb_paracurve_value(t, params):
    """Evaluate ICC parametric curve type 3 (sRGB-style):
       y = ((a*x + b)^g)  if x >= d
       y = c*x            if x <  d
    """
    g, a, b, c, d = params[:5]
    if t < d:
        return c*t
    v = a*t + b
    if v < 0:
        return 0.0
    return v**g

def sample_curve(trc, samples=11):
    kind, n, params = trc
    pts = []
    for i in range(samples):
        t = i / (samples - 1.0)
        if kind == 'para' and n == 3 and len(params) >= 5:
            y = srgb_paracurve_value(t, params)
        elif kind == 'curv' and n == 0:
            y = t
        elif kind == 'curv' and n == 1:
            y = t ** params[0]
        elif kind == 'curv' and n > 1:
            # LUT lookup
            idx = t * (n - 1)
            lo = int(idx); hi = min(lo+1, n-1)
            frac = idx - lo
            y = (params[lo]*(1-frac) + params[hi]*frac) / 65535.0
        else:
            y = float('nan')
        pts.append((t, y))
    return pts

def dump(path):
    print(f"\n========== {path} ==========")
    data = open(path,'rb').read()
    segs = find_segments(data, 0)
    icc = reassemble_icc(segs)
    if not icc:
        print("  no ICC"); return
    h = parse_header(icc)
    print(f"  {h}")
    tags = parse_tag_table(icc)
    print(f"  tags: {[(t[0], t[2]) for t in tags]}")
    by_sig = {}
    for sig, off, sz in tags:
        blob = icc[off:off+sz]
        by_sig[sig] = blob
    for sig in ('rXYZ','gXYZ','bXYZ','wtpt'):
        if sig in by_sig:
            xyz = parse_xyz(by_sig[sig])
            print(f"  {sig}: {xyz}")
    if 'chad' in by_sig:
        b = by_sig['chad']
        type_sig = b[0:4]
        # sf32Type: header (sig+reserved=8 bytes) then array of s15Fixed16
        n = (len(b) - 8) // 4
        nums = [s15f16(struct.unpack('>i', b[8+i*4:12+i*4])[0])
                for i in range(n)]
        print(f"  chad ({type_sig.decode()}, n={n}): {nums}")
    for sig in ('rTRC','gTRC','bTRC'):
        if sig in by_sig:
            trc = parse_trc(by_sig[sig])
            print(f"  {sig}: kind={trc[0]} n={trc[1]} params={trc[2][:8]}{'...' if len(trc[2])>8 else ''}")
            print(f"      curve samples: {[f'{t:.2f}->{y:.5f}' for (t,y) in sample_curve(trc, 11)]}")

if __name__ == '__main__':
    for p in sys.argv[1:]:
        dump(p)
