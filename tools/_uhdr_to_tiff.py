"""Decode an Ultra HDR JPEG to a BT.2020 PQ uint16 TIFF.

Usage:
    uv run python tools/_uhdr_to_tiff.py <input.uhdr.jpg> [output.tif]

Reconstructs the HDR image at full hdr_capacity_max headroom (gainmap_weight=1),
converts Display P3 linear -> BT.2020 linear, applies PQ OETF, and saves as a
16-bit BT.2020 PQ TIFF matching the format expected by convert.py.
"""

import sys
import struct
import io
import warnings
warnings.filterwarnings("ignore", module=r"colour")

import numpy as np
import tifffile
import colour
from PIL import Image

# ---------------------------------------------------------------------------
# PQ OETF (inverse EOTF): linear [0,1] (1.0=10000 nits) -> signal [0,1]
# ---------------------------------------------------------------------------
_PQ_M1 = np.float64(2610.0 / 16384.0)
_PQ_M2 = np.float64(2523.0 / 4096.0 * 128.0)
_PQ_C1 = np.float64(3424.0 / 4096.0)
_PQ_C2 = np.float64(2413.0 / 4096.0 * 32.0)
_PQ_C3 = np.float64(2392.0 / 4096.0 * 32.0)
_PQ_PEAK   = 10000.0
_SDR_WHITE = 203.0


def pq_oetf(linear: np.ndarray) -> np.ndarray:
    """PQ OETF: linear [0,1] (1.0 = 10 000 nits) -> PQ signal [0,1]."""
    L = np.clip(linear.astype(np.float64), 0.0, 1.0)
    Lm1 = np.power(L, _PQ_M1)
    num = _PQ_C1 + _PQ_C2 * Lm1
    den = 1.0   + _PQ_C3 * Lm1
    return np.power(num / den, _PQ_M2)


def gamma22_eotf(v: np.ndarray) -> np.ndarray:
    return np.power(np.clip(v.astype(np.float64), 0.0, 1.0), 2.2)


# ---------------------------------------------------------------------------
# UHDR helpers (same as _compare_hdr_linear.py)
# ---------------------------------------------------------------------------
def extract_primary_jpeg(uhdr_bytes: bytes) -> bytes:
    boundary = uhdr_bytes.find(b'\xff\xd9\xff\xd8')
    if boundary >= 0:
        return uhdr_bytes[:boundary + 2]
    eoi = uhdr_bytes.find(b'\xff\xd9')
    if eoi >= 0:
        return uhdr_bytes[:eoi + 2]
    raise ValueError("no JPEG EOI found")


def extract_gainmap_jpeg(uhdr_bytes: bytes) -> bytes:
    boundary = uhdr_bytes.find(b'\xff\xd9\xff\xd8')
    if boundary >= 0:
        return uhdr_bytes[boundary + 2:]
    raise ValueError("no gain-map JPEG found")


def parse_iso21496(data: bytes) -> dict:
    p = [0]

    def u8():
        v = data[p[0]]; p[0] += 1; return v

    def u16():
        v = int.from_bytes(data[p[0]:p[0]+2], 'big'); p[0] += 2; return v

    def s32():
        v = int.from_bytes(data[p[0]:p[0]+4], 'big', signed=True); p[0] += 4; return v

    def u32():
        v = int.from_bytes(data[p[0]:p[0]+4], 'big', signed=False); p[0] += 4; return v

    _minver = u16(); _wrver = u16(); flags = u8()
    multi_channel = bool(flags & 0x80)
    use_base_cg   = bool(flags & 0x40)
    common_den    = bool(flags & 0x08)
    n_ch = 3 if multi_channel else 1

    if common_den:
        denom = u32()
        bh = u32() / denom
        ah = u32() / denom
        channels = []
        for _ in range(n_ch):
            channels.append({
                "gainmap_min": s32() / denom,
                "gainmap_max": s32() / denom,
                "gamma":       u32() / denom,
                "base_offset": s32() / denom,
                "alt_offset":  s32() / denom,
            })
    else:
        bhN, bhD = u32(), u32()
        ahN, ahD = u32(), u32()
        bh = bhN / bhD if bhD else 0.0
        ah = ahN / ahD if ahD else 0.0
        channels = []
        for _ in range(n_ch):
            mnN, mnD = s32(), u32()
            mxN, mxD = s32(), u32()
            gN,  gD  = u32(), u32()
            boN, boD = s32(), u32()
            aoN, aoD = s32(), u32()
            channels.append({
                "gainmap_min": mnN / mnD if mnD else 0.0,
                "gainmap_max": mxN / mxD if mxD else 0.0,
                "gamma":       gN  / gD  if gD  else 1.0,
                "base_offset": boN / boD if boD else 0.0,
                "alt_offset":  aoN / aoD if aoD else 0.0,
            })

    return {
        "multi_channel": multi_channel,
        "channels": channels,
        "base_hdr_headroom": bh,
        "alt_hdr_headroom":  ah,
    }


def read_gainmap_metadata(gainmap_jpeg: bytes) -> dict:
    URN = b"urn:iso:std:iso:ts:21496:-1\x00"
    pos = 0
    while pos < len(gainmap_jpeg) - 4:
        if gainmap_jpeg[pos] != 0xFF:
            pos += 1; continue
        marker = gainmap_jpeg[pos:pos+2]
        if marker == b'\xFF\xD8':
            pos += 2; continue
        if marker == b'\xFF\xD9':
            break
        if len(gainmap_jpeg) < pos + 4:
            break
        seg_len = struct.unpack(">H", gainmap_jpeg[pos+2:pos+4])[0]
        payload = gainmap_jpeg[pos+4:pos+2+seg_len]
        if marker == b'\xFF\xE2' and payload.startswith(URN):
            return parse_iso21496(payload[len(URN):])
        pos += 2 + seg_len
    raise ValueError("ISO 21496-1 APP2 not found")


def reconstruct_hdr_linear_p3(sdr_img, gm_img, meta, display_boost_log2):
    """Reconstruct linear P3, SDR white = 1.0. Matches libultrahdr applyGain."""
    channels = meta["channels"]
    n_ch = len(channels)
    bh = meta["base_hdr_headroom"]
    ah = meta["alt_hdr_headroom"]

    gainmap_weight = (1.0 if ah == bh else
                      np.clip((display_boost_log2 - bh) / (ah - bh), 0.0, 1.0))

    sdr_linear = gamma22_eotf(sdr_img)
    hdr = np.empty_like(sdr_linear)

    for c in range(3):
        ch = channels[c if n_ch == 3 else 0]
        gamma    = ch["gamma"]
        gm_c     = gm_img[..., c].astype(np.float64)
        gain_norm = np.power(np.clip(gm_c, 0.0, 1.0), 1.0/gamma) if gamma != 1.0 else np.clip(gm_c, 0.0, 1.0)
        log_boost = ch["gainmap_min"] * (1.0 - gain_norm) + ch["gainmap_max"] * gain_norm
        gain_factor = np.power(2.0, log_boost * gainmap_weight)
        hdr[..., c] = (sdr_linear[..., c] + ch["base_offset"]) * gain_factor - ch["alt_offset"]

    return hdr


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    uhdr_path = sys.argv[1]
    out_path  = sys.argv[2] if len(sys.argv) > 2 else uhdr_path.replace(".jpg", ".decoded.tif")

    # Colour matrices
    bt2020 = colour.RGB_COLOURSPACES["ITU-R BT.2020"]
    p3d65  = colour.RGB_COLOURSPACES["Display P3"]
    M_p3_bt2020 = colour.matrix_RGB_to_RGB(p3d65, bt2020, "Bradford")

    print(f"Loading {uhdr_path} …")
    with open(uhdr_path, "rb") as f:
        uhdr_bytes = f.read()

    primary_jpeg = extract_primary_jpeg(uhdr_bytes)
    gainmap_jpeg = extract_gainmap_jpeg(uhdr_bytes)
    meta         = read_gainmap_metadata(gainmap_jpeg)

    ah = meta["alt_hdr_headroom"]
    bh = meta["base_hdr_headroom"]
    print(f"  hdr_capacity_min: {2**bh * _SDR_WHITE:.0f} nits  "
          f"hdr_capacity_max: {2**ah * _SDR_WHITE:.0f} nits")
    print(f"  multi_channel: {meta['multi_channel']}")
    for i, ch in enumerate(meta["channels"]):
        label = ["R","G","B"][i] if meta["multi_channel"] else "luma"
        print(f"  ch[{label}]: min={ch['gainmap_min']:.4f} log2  "
              f"max={ch['gainmap_max']:.4f} log2  gamma={ch['gamma']:.3f}")

    # Decode SDR base and gain map
    sdr_arr = np.array(Image.open(io.BytesIO(primary_jpeg)).convert("RGB"),
                       dtype=np.float64) / 255.0
    H, W = sdr_arr.shape[:2]
    print(f"  SDR base: {W}x{H}")

    gm_pil = Image.open(io.BytesIO(gainmap_jpeg)).convert("RGB")
    gm_arr = np.array(gm_pil, dtype=np.float64) / 255.0
    if gm_arr.shape[:2] != (H, W):
        gm_arr = np.array(gm_pil.resize((W, H), Image.BILINEAR),
                          dtype=np.float64) / 255.0
        print(f"  Gain map upscaled to {W}x{H}")

    # Reconstruct linear P3 at full headroom (gainmap_weight = 1.0)
    print(f"  Reconstructing HDR at full headroom ({2**ah * _SDR_WHITE:.0f} nits) …")
    hdr_p3 = reconstruct_hdr_linear_p3(sdr_arr, gm_arr, meta, display_boost_log2=ah)
    hdr_p3_nits = hdr_p3 * _SDR_WHITE   # -> nits, linear P3

    # Convert P3 linear -> BT.2020 linear
    flat_p3 = hdr_p3_nits.reshape(-1, 3)
    flat_bt2020 = (M_p3_bt2020 @ flat_p3.T).T
    hdr_bt2020_nits = flat_bt2020.reshape(H, W, 3)

    # Clip negative values (P3->BT.2020 can produce tiny negatives at gamut boundary)
    hdr_bt2020_nits = np.clip(hdr_bt2020_nits, 0.0, _PQ_PEAK)

    # Apply PQ OETF: nits -> [0,1] linear -> PQ signal [0,1]
    hdr_bt2020_linear_norm = hdr_bt2020_nits / _PQ_PEAK   # [0,1], 1.0=10000 nits
    pq_signal = pq_oetf(hdr_bt2020_linear_norm)            # [0,1]

    # Quantise to uint16
    u16 = np.clip(pq_signal * 65535.0 + 0.5, 0, 65535).astype(np.uint16)

    print(f"  PQ signal range: {pq_signal.min():.4f} – {pq_signal.max():.4f}")
    print(f"  Peak nits (BT.2020 linear): {hdr_bt2020_nits.max():.1f}")

    print(f"  Saving -> {out_path}")
    tifffile.imwrite(out_path, u16, photometric="rgb",
                     description="BT.2020 PQ HDR, decoded from Ultra HDR JPEG")
    print("  Done.")


if __name__ == "__main__":
    main()
