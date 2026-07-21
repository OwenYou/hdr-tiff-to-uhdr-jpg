"""Convert a BT.2020 PQ HDR TIFF to a UHDR JPEG with B&W SDR base and colour HDR alternate.

Pipeline
--------
1. Load uint16 BT.2020 PQ TIFF.
2. Convert BT.2020 PQ -> Display P3 PQ (ACES 1.3 gamut compression, LUT mode).
3. Tone-map P3 PQ -> P3 linear SDR (Reinhard), then desaturate to luma-only
   greyscale (BT.709 luma coefficients applied in P3 linear space).
   Result: RGBA8888 greyscale image as the SDR base.
4. Pack the colour P3 PQ as RGBA1010102 for the HDR intent.
5. Encode via libultrahdr API-1 with --force-rgb-gainmap so the 3-channel RGB
   gain map metadata is always written, even though the per-channel statistics
   will be identical (greyscale SDR with colour HDR produces symmetric channel
   gain ranges by construction).

Output: <output>.uhdr.jpg
  SDR rendition (base JPEG): black-and-white
  HDR rendition (gain map applied): full colour

Usage:
    uv run python tools/_bw_sdr_colour_hdr.py <input.tif> <output.uhdr.jpg> [options]

Options:
    --quality N          JPEG quality 0-100 (default: 95)
    --gainmap-scale N    gain map downscale factor 1-128 (default: 1)
    --gainmap-gamma G    gain map encoding gamma (default: 1.0)
    --peak-nits N        target HDR display peak in nits (default: 1000)
    --pipeline lut|parametric  colour pipeline mode (default: lut)
    --gamut compress|clip      gamut handling (default: compress)
    --force              overwrite output if it exists
    --verbose
"""

import argparse
import ctypes
import os
import sys
import time
from pathlib import Path

import numpy as np
import tifffile

sys.path.insert(0, str(Path(__file__).parent.parent))

import color
from uhdr_ctypes import (
    uhdr, check, byref,
    UhdrRawImage,
    UHDR_IMG_FMT_32bppRGBA1010102, UHDR_IMG_FMT_32bppRGBA8888,
    UHDR_CG_DISPLAY_P3,
    UHDR_CT_PQ, UHDR_CT_SRGB,
    UHDR_CR_FULL_RANGE,
    UHDR_HDR_IMG, UHDR_SDR_IMG, UHDR_BASE_IMG, UHDR_GAIN_MAP_IMG,
)


# BT.709 luma coefficients — standard for perceptual greyscale in linear light.
# P3 primaries differ from BT.709, but the difference is small and these
# coefficients match what most tone-map pipelines use for perceived brightness.
_LUMA_R = np.float32(0.2126)
_LUMA_G = np.float32(0.7152)
_LUMA_B = np.float32(0.0722)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("input",  help="Input 16-bit BT.2020 PQ TIFF")
    p.add_argument("output", help="Output Ultra HDR JPEG path")
    p.add_argument("--quality",       type=int,   default=95)
    p.add_argument("--gainmap-scale", type=int,   default=1)
    p.add_argument("--gainmap-gamma", type=float, default=1.0)
    p.add_argument("--peak-nits",     type=float, default=1000.0)
    p.add_argument("--pipeline", choices=["lut", "parametric"], default="lut")
    p.add_argument("--gamut",    choices=["compress", "clip"],  default="compress")
    p.add_argument("--force",  "-f", action="store_true")
    p.add_argument("--verbose", "-v", action="store_true")
    return p.parse_args()


def load_pq_tiff(path: str) -> np.ndarray:
    with tifffile.TiffFile(path) as tif:
        page = tif.pages[0]
        if int(page.photometric) != 2:
            raise ValueError(f"expected RGB TIFF (photometric=2), got {int(page.photometric)}")
        arr = tif.asarray()
    if arr.dtype != np.uint16:
        raise ValueError(f"expected uint16 TIFF, got {arr.dtype}")
    if arr.ndim != 3 or arr.shape[2] != 3:
        raise ValueError(f"expected (H, W, 3), got {arr.shape}")
    return arr


def pack_rgba1010102(rgb_f32: np.ndarray) -> np.ndarray:
    H, W = rgb_f32.shape[:2]
    u32 = (rgb_f32 * np.float32(1023.0) + np.float32(0.5)).astype(np.uint32)
    u32[..., 1] <<= np.uint32(10)
    u32[..., 2] <<= np.uint32(20)
    packed = np.empty((H, W), dtype=np.uint32)
    np.bitwise_or(u32[..., 0], u32[..., 1], out=packed)
    np.bitwise_or(packed, u32[..., 2], out=packed)
    packed |= np.uint32(0xC0000000)
    return packed


def make_greyscale_sdr(p3_pq_f32: np.ndarray, use_lut: bool) -> tuple[np.ndarray, dict]:
    """Tone-map P3 PQ -> SDR RGBA8888, then desaturate to luma-only greyscale.

    Returns (rgba_u32, timings) — same dtype/shape as p3_pq_to_sdr_rgba8888.
    """
    rgba_u32, timings = color.p3_pq_to_sdr_rgba8888(p3_pq_f32, use_lut=use_lut)

    # Unpack (H, W) uint32 RGBA8888 -> per-channel uint8
    r_u8 = ( rgba_u32        & 0xFF).astype(np.float32)
    g_u8 = ((rgba_u32 >>  8) & 0xFF).astype(np.float32)
    b_u8 = ((rgba_u32 >> 16) & 0xFF).astype(np.float32)
    a_u8 = ((rgba_u32 >> 24) & 0xFF)

    # sRGB gamma decode -> linear, compute luma, re-encode -> sRGB gamma
    def srgb_eotf(u: np.ndarray) -> np.ndarray:
        v = u / 255.0
        return np.where(v <= 0.04045, v / 12.92, ((v + 0.055) / 1.055) ** 2.4)

    def srgb_oetf(lin: np.ndarray) -> np.ndarray:
        return np.where(lin <= 0.0031308, lin * 12.92, 1.055 * lin ** (1.0/2.4) - 0.055)

    luma_lin = (srgb_eotf(r_u8) * _LUMA_R +
                srgb_eotf(g_u8) * _LUMA_G +
                srgb_eotf(b_u8) * _LUMA_B)
    luma_u8 = np.clip(srgb_oetf(luma_lin) * 255.0 + 0.5, 0, 255).astype(np.uint32)

    # Repack as RGBA8888 uint32 with R=G=B=luma, original A
    grey_u32 = (luma_u8
                | (luma_u8 << np.uint32(8))
                | (luma_u8 << np.uint32(16))
                | (a_u8.astype(np.uint32) << np.uint32(24)))

    return grey_u32, timings


def _raw_image(fmt, cg, ct, arr):
    h, w = arr.shape[:2]
    img = UhdrRawImage()
    img.fmt = fmt; img.cg = cg; img.ct = ct; img.range = UHDR_CR_FULL_RANGE
    img.w = w; img.h = h
    img.planes[0] = arr.ctypes.data_as(ctypes.c_void_p).value
    img.planes[1] = img.planes[2] = 0
    img.stride[0] = w; img.stride[1] = img.stride[2] = 0
    return img


def encode(packed_hdr: np.ndarray, grey_sdr: np.ndarray,
           args: argparse.Namespace) -> bytes:
    hdr_img = _raw_image(UHDR_IMG_FMT_32bppRGBA1010102,
                         UHDR_CG_DISPLAY_P3, UHDR_CT_PQ, packed_hdr)
    sdr_img = _raw_image(UHDR_IMG_FMT_32bppRGBA8888,
                         UHDR_CG_DISPLAY_P3, UHDR_CT_SRGB, grey_sdr)

    enc = uhdr.uhdr_create_encoder()
    if not enc:
        raise RuntimeError("uhdr_create_encoder returned NULL")
    try:
        check(uhdr.uhdr_enc_set_raw_image(enc, byref(hdr_img), UHDR_HDR_IMG),
              "set_raw_image(HDR)")
        check(uhdr.uhdr_enc_set_raw_image(enc, byref(sdr_img), UHDR_SDR_IMG),
              "set_raw_image(SDR)")
        # Always force 3-ch metadata: greyscale SDR + colour HDR produces
        # symmetric per-channel gain statistics, so without this flag the
        # library would collapse to 1-channel metadata despite RGB888 pixel data.
        check(uhdr.uhdr_enc_set_using_multi_channel_gainmap(enc, 1),
              "set_multi_channel_gainmap")
        check(uhdr.uhdr_enc_set_force_rgb_gainmap_metadata(enc, 1),
              "set_force_rgb_gainmap_metadata")
        check(uhdr.uhdr_enc_set_gainmap_scale_factor(enc, args.gainmap_scale),
              "set_gainmap_scale_factor")
        check(uhdr.uhdr_enc_set_gainmap_gamma(enc, args.gainmap_gamma),
              "set_gainmap_gamma")
        check(uhdr.uhdr_enc_set_quality(enc, args.quality, UHDR_BASE_IMG),
              "set_quality(base)")
        check(uhdr.uhdr_enc_set_quality(enc, args.quality, UHDR_GAIN_MAP_IMG),
              "set_quality(gainmap)")
        check(uhdr.uhdr_enc_set_target_display_peak_brightness(enc, args.peak_nits),
              "set_peak_brightness")
        check(uhdr.uhdr_encode(enc), "encode")
        out = uhdr.uhdr_get_encoded_stream(enc)
        if not out:
            raise RuntimeError("uhdr_get_encoded_stream returned NULL")
        result = ctypes.string_at(out.contents.data, out.contents.data_sz)
        _ = packed_hdr, grey_sdr
        return result
    finally:
        uhdr.uhdr_release_encoder(enc)


def main() -> int:
    args = parse_args()
    if os.path.exists(args.output) and not args.force:
        print(f"error: {args.output} exists (use --force to overwrite)", file=sys.stderr)
        return 2

    steps: list[tuple[str, float]] = []

    def lap(label, t0):
        elapsed = time.perf_counter() - t0
        steps.append((label, elapsed))
        return time.perf_counter()

    t = time.perf_counter()
    rgb16 = load_pq_tiff(args.input)
    H, W = rgb16.shape[:2]
    t = lap(f"load TIFF ({W}x{H})", t)

    use_lut = (args.pipeline == "lut")
    gamut_compress = (args.gamut == "compress")

    p3_pq_f32, color_timings = color.bt2020_pq_to_p3_pq(
        rgb16, use_lut=use_lut, gamut_compress=gamut_compress)
    for name, secs in color_timings.items():
        steps.append((f"  color/{name}", secs))
    t = time.perf_counter()

    packed_hdr = pack_rgba1010102(p3_pq_f32)
    t = lap("pack RGBA1010102 (HDR)", t)

    grey_sdr, tm_timings = make_greyscale_sdr(p3_pq_f32, use_lut=use_lut)
    for name, secs in tm_timings.items():
        steps.append((f"  tonemap/{name}", secs))
    t = time.perf_counter()
    t = lap("greyscale SDR", t)

    jpg = encode(packed_hdr, grey_sdr, args)
    t = lap(f"encode ({len(jpg)/1e3:.0f} KB)", t)

    with open(args.output, "wb") as f:
        f.write(jpg)
    lap("write output", t)

    total = sum(s for _, s in steps)
    col_w = max(len(n) for n, _ in steps)
    print(f"\n{'Step':<{col_w}}   Time (s)   %", file=sys.stderr)
    print("-" * (col_w + 18), file=sys.stderr)
    for name, secs in steps:
        print(f"{name:<{col_w}}   {secs:7.3f}s  {100*secs/total:5.1f}%",
              file=sys.stderr)
    print("-" * (col_w + 18), file=sys.stderr)
    print(f"{'TOTAL':<{col_w}}   {total:7.3f}s  100.0%", file=sys.stderr)
    print(args.output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
