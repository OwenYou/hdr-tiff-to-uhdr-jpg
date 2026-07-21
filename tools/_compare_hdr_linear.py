"""Compare linear BT.2020 values between a PQ TIFF and an Ultra HDR JPEG.

Usage:
    uv run python tools/_compare_hdr_linear.py <input.tif> <input.uhdr.jpg>

The UHDR JPEG is decoded by reading the gain map metadata from the secondary JPEG,
then reconstructing the HDR linear signal from the primary SDR JPEG + gain map.
The TIF is decoded directly via PQ EOTF.

Both are converted to linear BT.2020 nits for a fair comparison.
"""

import sys
import struct
import warnings
warnings.filterwarnings("ignore", module=r"colour")

import numpy as np
import tifffile
import colour
from PIL import Image
import io

# ---------------------------------------------------------------------------
# PQ EOTF (matches gainmapmath.cpp pqInvOetf)
# ---------------------------------------------------------------------------
_PQ_M1 = np.float32(2610.0 / 16384.0)
_PQ_M2 = np.float32(2523.0 / 4096.0 * 128.0)
_PQ_C1 = np.float32(3424.0 / 4096.0)
_PQ_C2 = np.float32(2413.0 / 4096.0 * 32.0)
_PQ_C3 = np.float32(2392.0 / 4096.0 * 32.0)
_PQ_PEAK = 10000.0  # nits at signal=1.0


def pq_eotf(signal: np.ndarray) -> np.ndarray:
    """PQ EOTF: signal [0,1] -> linear [0,1], 1.0 = 10 000 nits."""
    s = np.clip(signal.astype(np.float64), 0.0, 1.0)
    val = np.power(s, 1.0 / _PQ_M2)
    num = np.maximum(val - _PQ_C1, 0.0)
    den = _PQ_C2 - _PQ_C3 * val
    return np.power(np.maximum(num / den, 0.0), 1.0 / _PQ_M1)


def gamma22_eotf(v: np.ndarray) -> np.ndarray:
    """γ 2.2 EOTF: encoded [0,1] -> linear [0,1]."""
    return np.power(np.clip(v.astype(np.float64), 0.0, 1.0), 2.2)


# ---------------------------------------------------------------------------
# Colour-space matrices (via colour-science, Bradford CAT)
# ---------------------------------------------------------------------------
def _get_matrices():
    bt2020 = colour.RGB_COLOURSPACES["ITU-R BT.2020"]
    p3d65  = colour.RGB_COLOURSPACES["Display P3"]
    M_p3_bt2020 = colour.matrix_RGB_to_RGB(p3d65, bt2020, "Bradford")
    return M_p3_bt2020


# ---------------------------------------------------------------------------
# JPEG / UHDR helpers
# ---------------------------------------------------------------------------
def _find_marker(data: bytes, marker: bytes, start: int = 0):
    while True:
        pos = data.find(marker, start)
        if pos == -1:
            return -1, 0
        # skip length field (2 bytes big-endian including the length field itself)
        length = struct.unpack(">H", data[pos+2:pos+4])[0]
        return pos, length


def extract_primary_jpeg(uhdr_bytes: bytes) -> bytes:
    boundary = uhdr_bytes.find(b'\xff\xd9\xff\xd8')
    if boundary >= 0:
        return uhdr_bytes[:boundary + 2]
    eoi = uhdr_bytes.find(b'\xff\xd9')
    if eoi >= 0:
        return uhdr_bytes[:eoi + 2]
    raise ValueError("no JPEG EOI found in UHDR output")


def extract_gainmap_jpeg(uhdr_bytes: bytes) -> bytes:
    boundary = uhdr_bytes.find(b'\xff\xd9\xff\xd8')
    if boundary >= 0:
        return uhdr_bytes[boundary + 2:]
    raise ValueError("no gain-map JPEG found in UHDR output")


def parse_iso21496(data: bytes) -> dict:
    """Parse ISO 21496-1 binary gain map metadata from APP2 payload (after 'urn:...\0').

    Layout mirrors libultrahdr gainmapmetadata.cpp encodeGainmapMetadata and
    _inspect_full.py parse_iso21496:
      u16 min_version, u16 writer_version, u8 flags
      if useCommonDenominator (flags & 0x08):
        u32 denom, u32 baseHdrHeadroomN, u32 altHdrHeadroomN
        per channel: s32 minN, s32 maxN, u32 gammaN, s32 baseOffN, s32 altOffN
      else:
        u32 baseHdrHeadroomN, u32 baseHdrHeadroomD
        u32 altHdrHeadroomN,  u32 altHdrHeadroomD
        per channel: s32 minN, u32 minD, s32 maxN, u32 maxD,
                     u32 gammaN, u32 gammaD, s32 baseOffN, u32 baseOffD,
                     s32 altOffN, u32 altOffD
    """
    p = [0]

    def u8():
        v = data[p[0]]; p[0] += 1; return v

    def u16():
        v = int.from_bytes(data[p[0]:p[0]+2], 'big'); p[0] += 2; return v

    def s32():
        v = int.from_bytes(data[p[0]:p[0]+4], 'big', signed=True); p[0] += 4; return v

    def u32():
        v = int.from_bytes(data[p[0]:p[0]+4], 'big', signed=False); p[0] += 4; return v

    _min_ver = u16(); _wri_ver = u16(); flags = u8()
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
        "use_base_cg": use_base_cg,
        "channels": channels,
        "base_hdr_headroom": bh,   # log2
        "alt_hdr_headroom":  ah,   # log2
    }


def read_gainmap_metadata(gainmap_jpeg: bytes) -> dict:
    """Extract ISO 21496-1 metadata from the gain-map JPEG APP2 segment."""
    URN = b"urn:iso:std:iso:ts:21496:-1\x00"
    pos = 0
    while pos < len(gainmap_jpeg) - 4:
        if gainmap_jpeg[pos] != 0xFF:
            pos += 1
            continue
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
    raise ValueError("ISO 21496-1 APP2 segment not found in gain-map JPEG")


# ---------------------------------------------------------------------------
# Reconstruct HDR from SDR + gain map (ISO 21496-1 §7 apply algorithm)
# ---------------------------------------------------------------------------
def reconstruct_hdr_linear_p3(
    sdr_img: np.ndarray,   # (H,W,3) float64 [0,1], gamma-2.2 encoded, Display P3
    gm_img:  np.ndarray,   # (H,W,3) float64 [0,1], gain map pixels
    meta:    dict,
    weight:  float,        # display weight in [0,1]: 0=SDR, 1=full HDR boost
) -> np.ndarray:
    """Reconstruct linear Display P3 per ISO 21496-1 §7 / libultrahdr applyGainMap.

    weight = (log2(display_peak/sdr_white) - base_hdr_headroom)
             / (alt_hdr_headroom - base_hdr_headroom), clamped to [0,1].
    At weight=1 the full alt_hdr_headroom boost is applied.
    """
    channels = meta["channels"]
    n_ch = len(channels)
    bh = meta["base_hdr_headroom"]   # log2
    ah = meta["alt_hdr_headroom"]    # log2

    # γ 2.2 decode SDR base → linear [0,1]
    sdr_linear = gamma22_eotf(sdr_img)

    hdr_linear = np.empty_like(sdr_linear)
    for c in range(3):
        ch_idx = c if n_ch == 3 else 0
        ch = channels[ch_idx]
        gm_min   = ch["gainmap_min"]    # log2
        gm_max   = ch["gainmap_max"]    # log2
        gamma    = ch["gamma"]
        base_off = ch["base_offset"]
        alt_off  = ch["alt_offset"]

        gm_c = gm_img[..., c].astype(np.float64)

        # Decode gain map pixel → log2 gain (libultrahdr RecoveryMap::applyGainMap)
        # log_gain = gainMapMin + pixel^(1/gamma) * (gainMapMax - gainMapMin)
        log_gain = gm_min + np.power(np.clip(gm_c, 0.0, 1.0),
                                     1.0 / gamma) * (gm_max - gm_min)

        # Interpolate between base and alt headroom by display weight
        # gainLog2 = log_gain * weight + gainMapMin * (1 - weight)  [simplified]
        # Full formula: gain = 2^( log_gain * weight * (ah - bh) + bh_contribution )
        # libultrahdr: gain = 2^(log_gain * w) where w scales to display headroom
        gain_log2 = log_gain * weight * (ah - bh) + bh
        # base_hdr_headroom contribution cancels in ratio — apply as in applyGainMap:
        # hdr = (sdr + baseOffset) * 2^(log_gain*weight*(ah-bh)) - altOffset
        gain = np.power(2.0, log_gain * weight * (ah - bh))

        hdr_linear[..., c] = (sdr_linear[..., c] + base_off) * gain - alt_off

    return hdr_linear  # linear Display P3, units: SDR white = 1.0


# ---------------------------------------------------------------------------
# Main comparison
# ---------------------------------------------------------------------------
def main():
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(1)

    tif_path  = sys.argv[1]
    uhdr_path = sys.argv[2]

    M_p3_bt2020 = _get_matrices()

    # ------------------------------------------------------------------
    # 1. Decode TIF -> linear BT.2020 nits
    # ------------------------------------------------------------------
    print("Loading TIF …")
    with tifffile.TiffFile(tif_path) as tif:
        tif_u16 = tif.asarray()  # (H,W,3) uint16

    tif_pq  = tif_u16.astype(np.float64) / 65535.0          # PQ signal [0,1]
    tif_lin = pq_eotf(tif_pq) * _PQ_PEAK                    # linear BT.2020 nits
    H_tif, W_tif = tif_u16.shape[:2]
    print(f"  TIF size: {W_tif}x{H_tif}")

    # ------------------------------------------------------------------
    # 2. Decode UHDR JPEG
    # ------------------------------------------------------------------
    print("Loading UHDR JPEG …")
    with open(uhdr_path, "rb") as f:
        uhdr_bytes = f.read()

    primary_jpeg = extract_primary_jpeg(uhdr_bytes)
    gainmap_jpeg = extract_gainmap_jpeg(uhdr_bytes)
    meta         = read_gainmap_metadata(gainmap_jpeg)

    base_hdr_headroom = meta["base_hdr_headroom"]   # log2 of base display max / SDR white
    alt_hdr_headroom  = meta["alt_hdr_headroom"]    # log2 of alt (HDR display) max / SDR white
    display_boost = 2.0 ** (alt_hdr_headroom - base_hdr_headroom)  # gain map full-scale boost

    print(f"  base_hdr_headroom (log2): {base_hdr_headroom:.4f}  "
          f"({2**base_hdr_headroom * 203:.0f} nits)")
    print(f"  alt_hdr_headroom  (log2): {alt_hdr_headroom:.4f}  "
          f"({2**alt_hdr_headroom * 203:.0f} nits)")
    print(f"  display_boost: {display_boost:.4f}x  ({display_boost * 203:.0f} nits peak)")
    print(f"  multi_channel: {meta['multi_channel']}  use_base_cg: {meta['use_base_cg']}")

    for i, ch in enumerate(meta["channels"]):
        label = ["R","G","B"][i] if meta["multi_channel"] else "luma"
        print(f"  ch[{label}]: gainmap_min={ch['gainmap_min']:.4f}  "
              f"gainmap_max={ch['gainmap_max']:.4f}  gamma={ch['gamma']:.4f}  "
              f"base_offset={ch['base_offset']:.6f}  alt_offset={ch['alt_offset']:.6f}")

    # Decode primary JPEG -> SDR pixels [0,255] uint8
    sdr_pil  = Image.open(io.BytesIO(primary_jpeg)).convert("RGB")
    sdr_arr  = np.array(sdr_pil, dtype=np.float64) / 255.0  # (H,W,3) [0,1]
    H_jpg, W_jpg = sdr_arr.shape[:2]
    print(f"  Primary JPEG size: {W_jpg}x{H_jpg}")

    # Decode gain map JPEG -> float [0,1]
    gm_pil  = Image.open(io.BytesIO(gainmap_jpeg)).convert("RGB")
    gm_arr  = np.array(gm_pil, dtype=np.float64) / 255.0   # (H,W,3) [0,1]
    H_gm, W_gm = gm_arr.shape[:2]

    # Resize gain map to match primary if scaled
    if (H_gm, W_gm) != (H_jpg, W_jpg):
        from PIL import Image as _PIL
        gm_pil_rs = gm_pil.resize((W_jpg, H_jpg), _PIL.BILINEAR)
        gm_arr = np.array(gm_pil_rs, dtype=np.float64) / 255.0
        print(f"  Gain map resized from {W_gm}x{H_gm} to {W_jpg}x{H_jpg}")

    # Reconstruct HDR linear Display P3 [0, display_boost]
    hdr_p3_linear = reconstruct_hdr_linear_p3(sdr_arr, gm_arr, meta, display_boost)

    # Convert Display P3 linear -> BT.2020 linear
    # hdr_p3_linear is in units of [0, display_boost] (SDR white = 1.0)
    sdr_white_nits = 203.0
    hdr_p3_nits = hdr_p3_linear * sdr_white_nits  # nits

    # Reshape for matrix multiply: (H,W,3) -> (N,3)
    flat_p3 = hdr_p3_nits.reshape(-1, 3)
    flat_bt2020 = (M_p3_bt2020 @ flat_p3.T).T
    uhdr_bt2020_nits = flat_bt2020.reshape(H_jpg, W_jpg, 3)

    # ------------------------------------------------------------------
    # 3. Resize TIF to match UHDR if needed (sample at same pixels)
    # ------------------------------------------------------------------
    if (H_tif, W_tif) != (H_jpg, W_jpg):
        print(f"\nWARNING: TIF ({W_tif}x{H_tif}) != UHDR primary ({W_jpg}x{H_jpg}). "
              "Down-sampling TIF for pixel-accurate comparison.")
        # Simple block-average by reshaping (assume integer scale factor)
        sf_y = H_tif // H_jpg
        sf_x = W_tif // W_jpg
        if sf_y > 1 and sf_x > 1:
            tif_lin_ds = tif_lin[:sf_y*H_jpg, :sf_x*W_jpg, :]\
                .reshape(H_jpg, sf_y, W_jpg, sf_x, 3).mean(axis=(1,3))
        else:
            from PIL import Image as _PIL
            tif_lin_pil = _PIL.fromarray(
                np.clip(tif_lin / _PQ_PEAK * 65535, 0, 65535).astype(np.uint16), "I;16")
            tif_lin_ds = np.array(
                tif_lin_pil.resize((W_jpg, H_jpg), _PIL.BILINEAR),
                dtype=np.float64
            ) / 65535.0 * _PQ_PEAK
        tif_lin_cmp = tif_lin_ds
    else:
        tif_lin_cmp = tif_lin

    # ------------------------------------------------------------------
    # 4. Statistics: compare luminance (max channel) across the image
    # ------------------------------------------------------------------
    print("\n--- Overall image luminance (max-channel nits, BT.2020 linear) ---")

    tif_lum  = tif_lin_cmp.max(axis=-1)    # (H,W) peak nits per pixel
    uhdr_lum = uhdr_bt2020_nits.max(axis=-1)

    def stats(name, arr):
        print(f"  {name:20s}  "
              f"p50={np.percentile(arr,50):8.2f}  "
              f"p95={np.percentile(arr,95):8.2f}  "
              f"p99={np.percentile(arr,99):8.2f}  "
              f"p99.9={np.percentile(arr,99.9):8.2f}  "
              f"max={arr.max():8.2f}  nits")

    stats("TIF (source)", tif_lum)
    stats("UHDR reconstr.", uhdr_lum)

    ratio = np.where(uhdr_lum > 0, tif_lum / np.maximum(uhdr_lum, 1e-6), 1.0)
    print(f"\n  TIF/UHDR ratio    "
          f"p50={np.percentile(ratio,50):.4f}  "
          f"p95={np.percentile(ratio,95):.4f}  "
          f"p99={np.percentile(ratio,99):.4f}  "
          f"p99.9={np.percentile(ratio,99.9):.4f}  "
          f"max={ratio.max():.4f}")

    # ------------------------------------------------------------------
    # 5. Focus on the brightest 1% of TIF pixels
    # ------------------------------------------------------------------
    threshold = np.percentile(tif_lum, 99.0)
    mask = tif_lum >= threshold
    n_bright = mask.sum()
    print(f"\n--- Bright-highlight analysis (TIF lum >= p99 = {threshold:.1f} nits, {n_bright} pixels) ---")

    tif_bright_r  = tif_lin_cmp[mask]           # (N,3) BT.2020 linear nits
    uhdr_bright_r = uhdr_bt2020_nits[mask]

    print("  Channel stats (nits) in those pixels:")
    for c, ch_name in enumerate(["R","G","B"]):
        t_med = np.median(tif_bright_r[:,c])
        u_med = np.median(uhdr_bright_r[:,c])
        t_max = tif_bright_r[:,c].max()
        u_max = uhdr_bright_r[:,c].max()
        print(f"    {ch_name}: TIF median={t_med:8.2f}  UHDR median={u_med:8.2f}  "
              f"ratio={t_med/(u_med+1e-9):.4f} | "
              f"TIF max={t_max:8.2f}  UHDR max={u_max:8.2f}  ratio={t_max/(u_max+1e-9):.4f}")

    # ------------------------------------------------------------------
    # 6. Diagnose gain map headroom
    # ------------------------------------------------------------------
    print("\n--- Gain-map headroom analysis ---")
    gm_max_encoded = meta["channels"][0]["gainmap_max"]  # R channel (or luma)
    max_gain_log2 = gm_max_encoded  # gainMapMax is log2 gain at full white
    max_recoverable_nits = (2.0 ** max_gain_log2) * display_boost * sdr_white_nits
    print(f"  gainMapMax (R, log2 gain): {gm_max_encoded:.4f}")
    print(f"  Theoretical max recoverable nits from gain map: {max_recoverable_nits:.1f} nits")
    print(f"  TIF actual max luminance: {tif_lum.max():.1f} nits")
    if tif_lum.max() > max_recoverable_nits * 1.01:
        excess = tif_lum.max() - max_recoverable_nits
        print(f"  *** CLIPPED: TIF has {excess:.1f} nits MORE than gain map can represent! ***")
        print(f"  This is the root cause of the highlight brightness loss.")
    else:
        print(f"  Gain map headroom is sufficient (no clipping).")

    # ------------------------------------------------------------------
    # 7. Check the PQ signal ceiling at 1010 code (uint10 max usable)
    # ------------------------------------------------------------------
    print("\n--- RGBA1010102 packing ceiling ---")
    # packed_p3_hdr stores P3 PQ in [0,1023], so max signal = 1023/1023 = 1.0
    # PQ 1.0 = 10000 nits. But convert.py clips P3 PQ to [0,1] already.
    # The clip pipeline clips BT.2020 -> P3 then re-encodes to PQ. Check if
    # any BT.2020 primaries survive above P3 gamut boundary at full scale.
    bt2020_peak_pq = tif_pq.max()
    bt2020_peak_nits = tif_lin.max()
    print(f"  TIF PQ signal max: {bt2020_peak_pq:.6f}  -> {bt2020_peak_nits:.1f} nits")
    pq_1023 = pq_eotf(np.array([1023.0/1023.0])) * _PQ_PEAK
    print(f"  RGBA1010102 max code (1023) represents: {float(pq_1023[0]):.1f} nits in P3 PQ")
    print(f"  (No 10-bit quantisation clip in the HDR path — max code covers full PQ range)")

    # ------------------------------------------------------------------
    # 8. Diagnose the API-1 SDR tone map ceiling
    # ------------------------------------------------------------------
    print("\n--- API-1 SDR base ceiling (Reinhard tone map) ---")
    # API-1 supplies raw SDR RGBA8888 → max = 255 per channel.
    # The UHDR SDR base is at most 203-nits-normalised. The gain map must
    # bridge from SDR (≤1.0 in linear) up to peak_nits (set at encode time).
    # Check what peak_nits was. We infer it from alt_hdr_headroom.
    inferred_peak = 2.0 ** alt_hdr_headroom * sdr_white_nits
    print(f"  Inferred --peak-nits at encode time: {inferred_peak:.0f} nits "
          f"(from alt_hdr_headroom={alt_hdr_headroom:.4f})")
    print(f"  TIF max = {tif_lum.max():.1f} nits, "
          f"capped at {inferred_peak:.0f} nits in gain map")
    if tif_lum.max() > inferred_peak * 1.01:
        print(f"  *** GAIN MAP HEADROOM CEILING: source has {tif_lum.max():.0f} nits "
              f"but --peak-nits was {inferred_peak:.0f} nits. ***")
        print(f"  All pixels above {inferred_peak:.0f} nits are clipped to that value in the UHDR output.")
        print(f"  Fix: re-encode with --peak-nits {int(np.ceil(tif_lum.max()))}")
    else:
        print(f"  peak-nits ceiling is adequate for this image.")


if __name__ == "__main__":
    main()
