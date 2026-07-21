"""Compare linear BT.2020 values between a PQ TIFF and an Ultra HDR JPEG.

Usage:
    uv run python tools/_compare_hdr_linear.py <input.tif> <input.uhdr.jpg>

Decodes the TIFF via PQ EOTF and the UHDR JPEG via SDR base + gain map apply
(matching libultrahdr gainmapmath.cpp applyGain exactly), then compares both
in linear BT.2020 nits. Also compares in P3 linear to isolate gamut clip loss.
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
# PQ EOTF (matches gainmapmath.cpp pqInvOetf)
# ---------------------------------------------------------------------------
_PQ_M1 = np.float64(2610.0 / 16384.0)
_PQ_M2 = np.float64(2523.0 / 4096.0 * 128.0)
_PQ_C1 = np.float64(3424.0 / 4096.0)
_PQ_C2 = np.float64(2413.0 / 4096.0 * 32.0)
_PQ_C3 = np.float64(2392.0 / 4096.0 * 32.0)
_PQ_PEAK      = 10000.0   # nits at PQ signal 1.0
_SDR_WHITE    = 203.0     # nits, libultrahdr kSdrWhiteNits


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
# Colour-space matrices (Bradford CAT via colour-science)
# ---------------------------------------------------------------------------
def _get_matrices():
    bt2020 = colour.RGB_COLOURSPACES["ITU-R BT.2020"]
    p3d65  = colour.RGB_COLOURSPACES["Display P3"]
    M_bt2020_p3  = colour.matrix_RGB_to_RGB(bt2020, p3d65,  "Bradford")
    M_p3_bt2020  = colour.matrix_RGB_to_RGB(p3d65,  bt2020, "Bradford")
    return M_bt2020_p3, M_p3_bt2020


# ---------------------------------------------------------------------------
# UHDR helpers
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
    """Parse ISO 21496-1 metadata. Mirrors _inspect_full.py parse_iso21496."""
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
        "use_base_cg": use_base_cg,
        "channels": channels,
        "base_hdr_headroom": bh,   # log2, = log2(hdr_capacity_min)
        "alt_hdr_headroom":  ah,   # log2, = log2(hdr_capacity_max)
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
    raise ValueError("ISO 21496-1 APP2 segment not found")


# ---------------------------------------------------------------------------
# HDR reconstruction — exact match of libultrahdr gainmapmath.cpp applyGain
#
# applyGain (multi-channel, with gainmapWeight):
#   gain_norm[c] = pixel[c] ^ (1/gamma[c])              # gamma=1 here
#   logBoost[c]  = log2(min_boost[c]) * (1-gain_norm[c])
#                + log2(max_boost[c]) *    gain_norm[c]
#   gainFactor[c] = exp2(logBoost[c] * gainmapWeight)
#   hdr[c] = (sdr[c] + offset_sdr[c]) * gainFactor[c] - offset_hdr[c]
#
# gainmapWeight = (log2(display_boost) - log2(hdr_cap_min))
#               / (log2(hdr_cap_max)  - log2(hdr_cap_min))
#   clamped to [0,1].  At full display (display_boost = hdr_cap_max), weight=1.
#
# min_boost / max_boost are the XMP hdrgm: values, but for ISO 21496-1 they
# are stored as gainmap_min / gainmap_max in log2 units. Convert:
#   min_content_boost = exp2(gainmap_min)
#   max_content_boost = exp2(gainmap_max)
# ---------------------------------------------------------------------------
def reconstruct_hdr_linear_p3(
    sdr_img: np.ndarray,   # (H,W,3) float64 [0,1], γ2.2-encoded, Display P3
    gm_img:  np.ndarray,   # (H,W,3) float64 [0,1], gain map pixels [0,255]/255
    meta:    dict,
    display_boost_log2: float,  # log2 of target display peak / SDR white
) -> np.ndarray:
    """Return linear Display P3, SDR white = 1.0."""
    channels = meta["channels"]
    n_ch = len(channels)
    hdr_cap_min_log2 = meta["base_hdr_headroom"]   # log2(hdr_capacity_min)
    hdr_cap_max_log2 = meta["alt_hdr_headroom"]    # log2(hdr_capacity_max)

    # gainmapWeight: how far toward full HDR this display can show
    if hdr_cap_max_log2 != hdr_cap_min_log2:
        gainmap_weight = np.clip(
            (display_boost_log2 - hdr_cap_min_log2) /
            (hdr_cap_max_log2  - hdr_cap_min_log2),
            0.0, 1.0
        )
    else:
        gainmap_weight = 1.0

    # γ 2.2 decode SDR base → linear [0,1], SDR white = 1.0
    sdr_linear = gamma22_eotf(sdr_img)

    hdr_linear = np.empty_like(sdr_linear)
    for c in range(3):
        ch_idx = c if n_ch == 3 else 0
        ch = channels[ch_idx]
        log2_min_boost = ch["gainmap_min"]   # already log2
        log2_max_boost = ch["gainmap_max"]   # already log2
        gamma    = ch["gamma"]
        base_off = ch["base_offset"]
        alt_off  = ch["alt_offset"]

        gm_c = gm_img[..., c].astype(np.float64)   # [0,1]

        # decode gain map pixel to normalised [0,1] gain value
        if gamma != 1.0:
            gain_norm = np.power(np.clip(gm_c, 0.0, 1.0), 1.0 / gamma)
        else:
            gain_norm = np.clip(gm_c, 0.0, 1.0)

        # lerp in log2 space between min and max boost
        log_boost = log2_min_boost * (1.0 - gain_norm) + log2_max_boost * gain_norm

        # apply gainmap_weight (scale toward display headroom)
        gain_factor = np.power(2.0, log_boost * gainmap_weight)

        hdr_linear[..., c] = (sdr_linear[..., c] + base_off) * gain_factor - alt_off

    return hdr_linear  # linear P3, SDR white = 1.0


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(1)

    tif_path  = sys.argv[1]
    uhdr_path = sys.argv[2]

    M_bt2020_p3, M_p3_bt2020 = _get_matrices()

    # ------------------------------------------------------------------
    # 1. Decode TIF -> linear BT.2020 nits and linear P3 nits
    # ------------------------------------------------------------------
    print("Loading TIF …")
    with tifffile.TiffFile(tif_path) as tif:
        tif_u16 = tif.asarray()   # (H,W,3) uint16, BT.2020 PQ
    H_tif, W_tif = tif_u16.shape[:2]
    print(f"  TIF size: {W_tif}x{H_tif}")

    tif_pq      = tif_u16.astype(np.float64) / 65535.0
    tif_lin_bt  = pq_eotf(tif_pq) * _PQ_PEAK                 # linear BT.2020 nits

    # Convert BT.2020 linear -> P3 linear (clip gamut matches --gamut clip pipeline)
    flat_bt = tif_lin_bt.reshape(-1, 3)
    flat_p3 = (M_bt2020_p3 @ flat_bt.T).T
    tif_lin_p3 = np.clip(flat_p3, 0.0, None).reshape(H_tif, W_tif, 3)  # nits, P3

    # ------------------------------------------------------------------
    # 2. Decode UHDR JPEG -> reconstructed linear P3 nits
    # ------------------------------------------------------------------
    print("Loading UHDR JPEG …")
    with open(uhdr_path, "rb") as f:
        uhdr_bytes = f.read()

    primary_jpeg = extract_primary_jpeg(uhdr_bytes)
    gainmap_jpeg = extract_gainmap_jpeg(uhdr_bytes)
    meta         = read_gainmap_metadata(gainmap_jpeg)

    bh = meta["base_hdr_headroom"]   # log2
    ah = meta["alt_hdr_headroom"]    # log2
    print(f"  hdr_capacity_min (log2): {bh:.4f}  -> {2**bh * _SDR_WHITE:.0f} nits")
    print(f"  hdr_capacity_max (log2): {ah:.4f}  -> {2**ah * _SDR_WHITE:.0f} nits")
    print(f"  multi_channel: {meta['multi_channel']}  use_base_cg: {meta['use_base_cg']}")
    for i, ch in enumerate(meta["channels"]):
        label = ["R","G","B"][i] if meta["multi_channel"] else "luma"
        min_nits = 2**ch['gainmap_min'] * _SDR_WHITE
        max_nits = 2**ch['gainmap_max'] * _SDR_WHITE
        print(f"  ch[{label}]: min={ch['gainmap_min']:.4f} log2 ({min_nits:.1f} nits)  "
              f"max={ch['gainmap_max']:.4f} log2 ({max_nits:.1f} nits)  "
              f"gamma={ch['gamma']:.3f}")

    # Decode primary JPEG
    sdr_arr = np.array(Image.open(io.BytesIO(primary_jpeg)).convert("RGB"),
                       dtype=np.float64) / 255.0   # (H,W,3) γ2.2 [0,1]
    H_jpg, W_jpg = sdr_arr.shape[:2]
    print(f"  Primary JPEG size: {W_jpg}x{H_jpg}")

    # Decode gain map JPEG
    gm_pil = Image.open(io.BytesIO(gainmap_jpeg)).convert("RGB")
    gm_arr = np.array(gm_pil, dtype=np.float64) / 255.0
    H_gm, W_gm = gm_arr.shape[:2]
    if (H_gm, W_gm) != (H_jpg, W_jpg):
        gm_arr = np.array(gm_pil.resize((W_jpg, H_jpg), Image.BILINEAR),
                          dtype=np.float64) / 255.0
        print(f"  Gain map resized {W_gm}x{H_gm} -> {W_jpg}x{H_jpg}")

    # Reconstruct at full display headroom (gainmap_weight = 1.0)
    uhdr_p3_linear = reconstruct_hdr_linear_p3(
        sdr_arr, gm_arr, meta, display_boost_log2=ah
    ) * _SDR_WHITE   # -> nits, linear P3

    # Also convert UHDR P3 -> BT.2020 for cross-gamut comparison
    flat_uhdr_p3 = uhdr_p3_linear.reshape(-1, 3)
    uhdr_bt2020_nits = (M_p3_bt2020 @ flat_uhdr_p3.T).T.reshape(H_jpg, W_jpg, 3)

    # ------------------------------------------------------------------
    # 3. Align TIF to UHDR resolution
    # ------------------------------------------------------------------
    if (H_tif, W_tif) != (H_jpg, W_jpg):
        print(f"\nNote: TIF {W_tif}x{H_tif} != UHDR {W_jpg}x{H_jpg} — block-averaging TIF")
        sf_y, sf_x = H_tif // H_jpg, W_tif // W_jpg
        tif_lin_p3_cmp = (tif_lin_p3[:sf_y*H_jpg, :sf_x*W_jpg]
                          .reshape(H_jpg, sf_y, W_jpg, sf_x, 3).mean(axis=(1, 3)))
        tif_lin_bt_cmp = (tif_lin_bt[:sf_y*H_jpg, :sf_x*W_jpg]
                          .reshape(H_jpg, sf_y, W_jpg, sf_x, 3).mean(axis=(1, 3)))
    else:
        tif_lin_p3_cmp = tif_lin_p3
        tif_lin_bt_cmp = tif_lin_bt

    # ------------------------------------------------------------------
    # 4. Overall luminance statistics (max-channel nits)
    # ------------------------------------------------------------------
    def pstats(name, arr):
        lum = arr.max(axis=-1)
        print(f"  {name:28s}  "
              f"p50={np.percentile(lum,50):7.1f}  "
              f"p90={np.percentile(lum,90):7.1f}  "
              f"p95={np.percentile(lum,95):7.1f}  "
              f"p99={np.percentile(lum,99):7.1f}  "
              f"p99.9={np.percentile(lum,99.9):7.1f}  "
              f"max={lum.max():7.1f}  nits")

    print("\n--- Luminance statistics (max-channel nits) ---")
    print("  [P3 linear space — direct comparison, no gamut conversion loss]")
    pstats("TIF → P3 linear (clipped)", tif_lin_p3_cmp)
    pstats("UHDR reconstructed P3",     uhdr_p3_linear)

    print()
    print("  [BT.2020 linear space — TIF native, UHDR back-converted]")
    pstats("TIF BT.2020 linear",        tif_lin_bt_cmp)
    pstats("UHDR → BT.2020 linear",     uhdr_bt2020_nits)

    # ------------------------------------------------------------------
    # 5. Mid-range sky analysis: pixels 100–800 nits in TIF P3 space
    # ------------------------------------------------------------------
    tif_lum_p3 = tif_lin_p3_cmp.max(axis=-1)
    sky_mask = (tif_lum_p3 >= 100.0) & (tif_lum_p3 <= 800.0)
    n_sky = sky_mask.sum()
    print(f"\n--- Mid-range sky pixels (TIF P3 lum 100–800 nits, {n_sky:,} pixels) ---")

    tif_sky  = tif_lin_p3_cmp[sky_mask]
    uhdr_sky = uhdr_p3_linear[sky_mask]
    ratio_sky = tif_sky / np.maximum(uhdr_sky, 1e-6)

    print("  Per-channel median (nits in P3 linear):")
    for c, name in enumerate(["R", "G", "B"]):
        tm = np.median(tif_sky[:, c])
        um = np.median(uhdr_sky[:, c])
        print(f"    {name}: TIF={tm:7.2f}  UHDR={um:7.2f}  ratio={tm/(um+1e-9):.4f}")

    lum_ratio = tif_lum_p3[sky_mask] / np.maximum(uhdr_p3_linear.max(axis=-1)[sky_mask], 1e-6)
    print(f"  Lum ratio (TIF/UHDR): "
          f"p10={np.percentile(lum_ratio,10):.4f}  "
          f"p50={np.percentile(lum_ratio,50):.4f}  "
          f"p90={np.percentile(lum_ratio,90):.4f}  "
          f"mean={lum_ratio.mean():.4f}")

    # ------------------------------------------------------------------
    # 6. Gamut clip loss: how much BT.2020->P3 clipping costs
    # ------------------------------------------------------------------
    print("\n--- BT.2020→P3 gamut clip loss in TIF ---")
    tif_lum_bt = tif_lin_bt_cmp.max(axis=-1)
    flat_bt_all = tif_lin_bt_cmp.reshape(-1, 3)
    flat_p3_unclipped = (M_bt2020_p3 @ flat_bt_all.T).T.reshape(H_jpg, W_jpg, 3)
    out_of_gamut = (flat_p3_unclipped < 0).any(axis=-1).reshape(H_jpg, W_jpg)
    n_oog = out_of_gamut.sum()
    print(f"  Pixels with any negative P3 component (out-of-gamut): "
          f"{n_oog:,} / {H_jpg*W_jpg:,} = {100*n_oog/(H_jpg*W_jpg):.2f}%")

    # Clip loss: how much nits are lost by clipping negative channels
    clipped_vals = np.minimum(flat_p3_unclipped.reshape(H_jpg, W_jpg, 3), 0.0)
    clip_loss_nits = (-clipped_vals).max(axis=-1)   # nits lost per pixel
    print(f"  Clip loss (max negative channel, nits):  "
          f"p50={np.percentile(clip_loss_nits[out_of_gamut],50) if n_oog else 0:.1f}  "
          f"p95={np.percentile(clip_loss_nits[out_of_gamut],95) if n_oog else 0:.1f}  "
          f"max={clip_loss_nits.max():.1f}")

    # ------------------------------------------------------------------
    # 7. Gain map encode/decode round-trip fidelity in P3 space
    # ------------------------------------------------------------------
    print("\n--- Gain map round-trip fidelity (TIF P3 vs UHDR P3, ignoring gamut clip) ---")
    # Focus on pixels where TIF P3 is fully in-gamut (no clip loss)
    in_gamut = ~out_of_gamut
    n_ig = in_gamut.sum()
    print(f"  In-gamut pixels: {n_ig:,} ({100*n_ig/(H_jpg*W_jpg):.1f}%)")

    tif_ig   = tif_lin_p3_cmp[in_gamut]
    uhdr_ig  = uhdr_p3_linear[in_gamut]
    rt_ratio = tif_ig / np.maximum(uhdr_ig, 1e-6)

    # Exclude near-black pixels (< 1 nit) to avoid divide-by-zero noise
    bright_mask = tif_ig.max(axis=-1) > 1.0
    if bright_mask.sum() > 0:
        rt_r = rt_ratio[bright_mask]
        print(f"  Ratio TIF/UHDR (bright in-gamut pixels, lum > 1 nit):")
        for c, name in enumerate(["R", "G", "B"]):
            print(f"    {name}: p10={np.percentile(rt_r[:,c],10):.4f}  "
                  f"p50={np.percentile(rt_r[:,c],50):.4f}  "
                  f"p90={np.percentile(rt_r[:,c],90):.4f}  "
                  f"mean={rt_r[:,c].mean():.4f}")

    # ------------------------------------------------------------------
    # 8. Diagnose JPEG quantisation loss in SDR base
    # ------------------------------------------------------------------
    print("\n--- JPEG SDR base quantisation check ---")
    # Reconstruct what the SDR should be: Reinhard tone map of P3 PQ
    # Instead, compare the reconstructed HDR at gainmap_weight=0 (pure SDR decode)
    # against the γ2.2 SDR base.
    uhdr_sdr_only = reconstruct_hdr_linear_p3(
        sdr_arr, gm_arr, meta, display_boost_log2=bh  # weight=0
    ) * _SDR_WHITE
    print(f"  SDR base (gainmap_weight=0) peak nits: {uhdr_sdr_only.max():.1f}")
    print(f"  SDR base p99 nits: {np.percentile(uhdr_sdr_only.max(axis=-1), 99):.1f}")
    print(f"  (SDR white = {_SDR_WHITE:.0f} nits; values > {_SDR_WHITE:.0f} indicate"
          f" Reinhard headroom bleed-through)")

    # ------------------------------------------------------------------
    # 9. Summary
    # ------------------------------------------------------------------
    print("\n=== SUMMARY ===")
    tif_p3_max   = tif_lum_p3.max()
    uhdr_p3_max  = uhdr_p3_linear.max(axis=-1).max()
    encoded_peak = 2**ah * _SDR_WHITE
    print(f"  TIF P3 peak:          {tif_p3_max:.1f} nits")
    print(f"  UHDR reconstructed P3 peak: {uhdr_p3_max:.1f} nits")
    print(f"  Gain map hdr_capacity_max:  {encoded_peak:.0f} nits  (--peak-nits at encode time)")
    if tif_p3_max > encoded_peak * 1.01:
        print(f"  *** --peak-nits too low: source has {tif_p3_max:.0f} nits but "
              f"gain map ceiling is {encoded_peak:.0f} nits ***")
        print(f"  Re-encode with --peak-nits {int(np.ceil(tif_p3_max))}")
    else:
        print(f"  peak-nits ceiling is sufficient.")

    sky_ratio_med = np.median(lum_ratio)
    if sky_ratio_med > 1.05:
        print(f"\n  Sky pixels (100-800 nits) are {sky_ratio_med:.3f}x brighter in TIF than UHDR.")
        print(f"  Likely causes (in order of impact):")
        print(f"    1. JPEG quantisation loss in the SDR base (8-bit, γ2.2).")
        print(f"    2. Gain map 8-bit quantisation (255 levels across {2**ah - 2**bh:.1f} log2 stops).")
        print(f"    3. BT.2020→P3 gamut clip removing out-of-gamut energy.")
    else:
        print(f"\n  Sky luminance matches well (TIF/UHDR median = {sky_ratio_med:.3f}).")


if __name__ == "__main__":
    main()
