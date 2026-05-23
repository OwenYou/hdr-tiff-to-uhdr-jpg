"""Convert a BT.2020 PQ HDR TIFF to a Google Ultra HDR JPEG.

Default pipeline (API-1):
  1. Convert BT.2020 PQ -> Display P3 PQ (ACES 1.3 gamut compression).
  2. Python Reinhard tone map (replicating libultrahdr's internal ReinhardMap)
     converts the P3 PQ image to raw RGBA8888 SDR pixels — no JPEG encode/decode.
  3. API-1: feed raw P3 PQ HDR + raw P3 sRGB SDR; libultrahdr computes the
     gain map from unquantised pixels, then encodes both layers.
  The gain map benefits from a cleaner SDR input (no DCT block artefacts).
  Both renditions are Display P3 -> use_base_cg=true.

With --use-api3 (API-3 path):
  Steps 2-3 are replaced by a two-pass pipeline:
  2. App-0: feed P3 PQ to libultrahdr HDR-only encoder; extract the
     tone-mapped primary JPEG (Display P3 SDR base).
  3. API-3: feed P3 PQ raw HDR + the App-0 primary JPEG as compressed SDR;
     libultrahdr computes a multi-channel (RGB) gain map and assembles the
     final UHDR JPEG.
"""

import argparse
import ctypes
import os
import struct
import sys
import time

import numpy as np
import tifffile

import color
from uhdr_ctypes import (
    uhdr, check, byref,
    UhdrRawImage, UhdrCompressedImage,
    UHDR_IMG_FMT_32bppRGBA1010102, UHDR_IMG_FMT_32bppRGBA8888,
    UHDR_CG_DISPLAY_P3,
    UHDR_CT_PQ, UHDR_CT_SRGB,
    UHDR_CR_FULL_RANGE,
    UHDR_HDR_IMG, UHDR_SDR_IMG, UHDR_BASE_IMG, UHDR_GAIN_MAP_IMG,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("input",  help="Input 16-bit BT.2020 PQ TIFF")
    p.add_argument("output", help="Output Ultra HDR JPEG path")
    p.add_argument("--quality", type=int, default=95,
                   help="JPEG quality 0-100 for base and gain map (default: 95)")
    p.add_argument("--gainmap-scale", type=int, default=1,
                   help="Gain map downscale factor 1-128 (default: 1)")
    p.add_argument("--gainmap-gamma", type=float, default=1.0,
                   help="Gain map encoding gamma (default: 1.0)")
    p.add_argument("--peak-nits", type=float, default=1000.0,
                   help="Target HDR display peak in nits 203-10000 (default: 1000)")
    p.add_argument("--sdr-tonemap", choices=["lut", "parametric"], default="lut",
                   help="SDR tone-map mode: 'lut' (default) uses a fused OCIO 3D LUT "
                        "(fast, tetrahedral interpolation); 'parametric' uses explicit "
                        "NumPy steps that exactly reproduce libultrahdr's formulae")
    p.add_argument("--use-api3", action="store_true",
                   help="API-3 mode: App-0 JPEG tone map -> compressed SDR + raw HDR encode "
                        "(two-pass; gain map computed from JPEG-quantised SDR pixels)")
    p.add_argument("--force", "-f", action="store_true",
                   help="Overwrite output if it exists")
    p.add_argument("--verbose", "-v", action="store_true")
    return p.parse_args()


def load_pq_tiff(path: str) -> np.ndarray:
    with tifffile.TiffFile(path) as tif:
        page = tif.pages[0]
        if int(page.photometric) != 2:
            raise ValueError(
                f"expected RGB TIFF (photometric=2), got {int(page.photometric)}"
            )
        arr = tif.asarray()
    if arr.dtype != np.uint16:
        raise ValueError(f"expected uint16 TIFF, got {arr.dtype}")
    if arr.ndim != 3 or arr.shape[2] != 3:
        raise ValueError(f"expected (H, W, 3), got {arr.shape}")
    return arr


def pack_rgba1010102(rgb16: np.ndarray) -> np.ndarray:
    """uint16 (H,W,3) -> RGBA1010102 uint32 (H,W).

    Bit layout per ultrahdr_api.h: R[9:0] G[19:10] B[29:20] A[31:30].
    """
    u32 = rgb16.astype(np.uint32)
    r10 = (u32[..., 0] * np.uint32(1023) + np.uint32(32767)) // np.uint32(65535)
    g10 = (u32[..., 1] * np.uint32(1023) + np.uint32(32767)) // np.uint32(65535)
    b10 = (u32[..., 2] * np.uint32(1023) + np.uint32(32767)) // np.uint32(65535)
    packed = r10 | (g10 << np.uint32(10)) | (b10 << np.uint32(20)) | (np.uint32(3) << np.uint32(30))
    return np.ascontiguousarray(packed)


def _raw_image(fmt: int, cg: int, ct: int, packed: np.ndarray) -> UhdrRawImage:
    h, w = packed.shape
    img = UhdrRawImage()
    img.fmt    = fmt
    img.cg     = cg
    img.ct     = ct
    img.range  = UHDR_CR_FULL_RANGE
    img.w      = w
    img.h      = h
    img.planes[0] = packed.ctypes.data_as(ctypes.c_void_p).value
    img.planes[1] = 0
    img.planes[2] = 0
    img.stride[0] = w
    img.stride[1] = 0
    img.stride[2] = 0
    return img


def _compressed_image(data: bytes, cg: int, ct: int) -> tuple:
    """Return (UhdrCompressedImage, backing_buffer) — caller must keep buffer alive."""
    buf = (ctypes.c_ubyte * len(data)).from_buffer_copy(data)
    img = UhdrCompressedImage()
    img.data     = ctypes.cast(buf, ctypes.c_void_p)
    img.data_sz  = len(data)
    img.capacity = len(data)
    img.cg       = cg
    img.ct       = ct
    img.range    = UHDR_CR_FULL_RANGE
    return img, buf


def extract_primary_jpeg(uhdr_bytes: bytes) -> bytes:
    """Return just the primary JPEG from a UHDR multi-picture JPEG.

    A UHDR JPEG is two back-to-back JPEGs.  The primary ends with FF D9 (EOI)
    immediately followed by FF D8 (SOI) of the gain-map JPEG.  FF D9 cannot
    appear inside JPEG scan data (FF bytes are byte-stuffed as FF 00 there),
    so FF D9 FF D8 is an unambiguous boundary.
    """
    boundary = uhdr_bytes.find(b'\xff\xd9\xff\xd8')
    if boundary >= 0:
        return uhdr_bytes[:boundary + 2]   # include FF D9
    # Fallback: single JPEG or no immediate succession — return up to first EOI.
    eoi = uhdr_bytes.find(b'\xff\xd9')
    if eoi >= 0:
        return uhdr_bytes[:eoi + 2]
    raise ValueError("no JPEG EOI found in App-0 UHDR output")


def _encode_app0(packed_p3_hdr: np.ndarray, quality: int) -> bytes:
    """API-0: HDR-only encode with Display P3 PQ input.

    Returns the raw UHDR JPEG bytes.  The primary JPEG embedded inside is the
    libultrahdr-internal tone-mapped Display P3 SDR base.
    """
    hdr_img = _raw_image(UHDR_IMG_FMT_32bppRGBA1010102,
                         UHDR_CG_DISPLAY_P3, UHDR_CT_PQ, packed_p3_hdr)
    enc = uhdr.uhdr_create_encoder()
    if not enc:
        raise RuntimeError("uhdr_create_encoder returned NULL (App-0)")
    try:
        check(uhdr.uhdr_enc_set_raw_image(enc, byref(hdr_img), UHDR_HDR_IMG),
              "app0 set_raw_image")
        check(uhdr.uhdr_enc_set_quality(enc, quality, UHDR_BASE_IMG),
              "app0 set_quality(base)")
        check(uhdr.uhdr_enc_set_quality(enc, quality, UHDR_GAIN_MAP_IMG),
              "app0 set_quality(gainmap)")
        check(uhdr.uhdr_encode(enc), "app0 encode")
        out = uhdr.uhdr_get_encoded_stream(enc)
        if not out:
            raise RuntimeError("app0: uhdr_get_encoded_stream returned NULL")
        result = ctypes.string_at(out.contents.data, out.contents.data_sz)
        _ = packed_p3_hdr  # keep NumPy buffer alive until after string_at
        return result
    finally:
        uhdr.uhdr_release_encoder(enc)


def _encode_api3(packed_p3_hdr: np.ndarray,
                 sdr_jpeg: bytes,
                 args: argparse.Namespace) -> bytes:
    """API-3: raw P3 PQ HDR + compressed Display P3 SDR -> UHDR with RGB gain map.

    Both renditions are Display P3, so use_base_cg=true in the output.
    The SDR JPEG from App-0 is used directly as the primary image (no re-encode).
    """
    hdr_img = _raw_image(UHDR_IMG_FMT_32bppRGBA1010102,
                         UHDR_CG_DISPLAY_P3, UHDR_CT_PQ, packed_p3_hdr)
    sdr_cimg, sdr_buf = _compressed_image(sdr_jpeg, UHDR_CG_DISPLAY_P3, UHDR_CT_SRGB)

    enc = uhdr.uhdr_create_encoder()
    if not enc:
        raise RuntimeError("uhdr_create_encoder returned NULL (API-3)")
    try:
        check(uhdr.uhdr_enc_set_raw_image(enc, byref(hdr_img), UHDR_HDR_IMG),
              "api3 set_raw_image(HDR)")
        check(uhdr.uhdr_enc_set_compressed_image(enc, byref(sdr_cimg), UHDR_SDR_IMG),
              "api3 set_compressed_image(SDR)")
        check(uhdr.uhdr_enc_set_using_multi_channel_gainmap(enc, 1),
              "api3 set_multi_channel_gainmap")
        check(uhdr.uhdr_enc_set_gainmap_scale_factor(enc, args.gainmap_scale),
              "api3 set_gainmap_scale_factor")
        check(uhdr.uhdr_enc_set_gainmap_gamma(enc, args.gainmap_gamma),
              "api3 set_gainmap_gamma")
        check(uhdr.uhdr_enc_set_quality(enc, args.quality, UHDR_GAIN_MAP_IMG),
              "api3 set_quality(gainmap)")
        check(uhdr.uhdr_enc_set_target_display_peak_brightness(enc, args.peak_nits),
              "api3 set_target_display_peak_brightness")
        check(uhdr.uhdr_encode(enc), "api3 encode")
        out = uhdr.uhdr_get_encoded_stream(enc)
        if not out:
            raise RuntimeError("api3: uhdr_get_encoded_stream returned NULL")
        result = ctypes.string_at(out.contents.data, out.contents.data_sz)
        _ = packed_p3_hdr, sdr_buf  # keep buffers alive until after string_at
        return result
    finally:
        uhdr.uhdr_release_encoder(enc)


def _encode_api1(packed_p3_hdr: np.ndarray,
                 sdr_rgba8888: np.ndarray,
                 args: argparse.Namespace) -> bytes:
    """API-1: raw P3 PQ HDR + raw P3 sRGB SDR -> UHDR JPEG with RGB gain map.

    Both inputs are raw (uncompressed).  libultrahdr computes the gain map from
    clean pixels and re-encodes the SDR base JPEG at args.quality.
    Both renditions are Display P3 -> use_base_cg=true.
    """
    hdr_img = _raw_image(UHDR_IMG_FMT_32bppRGBA1010102,
                         UHDR_CG_DISPLAY_P3, UHDR_CT_PQ, packed_p3_hdr)
    sdr_img = _raw_image(UHDR_IMG_FMT_32bppRGBA8888,
                         UHDR_CG_DISPLAY_P3, UHDR_CT_SRGB, sdr_rgba8888)

    enc = uhdr.uhdr_create_encoder()
    if not enc:
        raise RuntimeError("uhdr_create_encoder returned NULL (API-1)")
    try:
        check(uhdr.uhdr_enc_set_raw_image(enc, byref(hdr_img), UHDR_HDR_IMG),
              "api1 set_raw_image(HDR)")
        check(uhdr.uhdr_enc_set_raw_image(enc, byref(sdr_img), UHDR_SDR_IMG),
              "api1 set_raw_image(SDR)")
        check(uhdr.uhdr_enc_set_using_multi_channel_gainmap(enc, 1),
              "api1 set_multi_channel_gainmap")
        check(uhdr.uhdr_enc_set_gainmap_scale_factor(enc, args.gainmap_scale),
              "api1 set_gainmap_scale_factor")
        check(uhdr.uhdr_enc_set_gainmap_gamma(enc, args.gainmap_gamma),
              "api1 set_gainmap_gamma")
        check(uhdr.uhdr_enc_set_quality(enc, args.quality, UHDR_BASE_IMG),
              "api1 set_quality(base)")
        check(uhdr.uhdr_enc_set_quality(enc, args.quality, UHDR_GAIN_MAP_IMG),
              "api1 set_quality(gainmap)")
        check(uhdr.uhdr_enc_set_target_display_peak_brightness(enc, args.peak_nits),
              "api1 set_target_display_peak_brightness")
        check(uhdr.uhdr_encode(enc), "api1 encode")
        out = uhdr.uhdr_get_encoded_stream(enc)
        if not out:
            raise RuntimeError("api1: uhdr_get_encoded_stream returned NULL")
        result = ctypes.string_at(out.contents.data, out.contents.data_sz)
        _ = packed_p3_hdr, sdr_rgba8888  # keep NumPy buffers alive until after string_at
        return result
    finally:
        uhdr.uhdr_release_encoder(enc)


def main() -> int:
    args = parse_args()
    if os.path.exists(args.output) and not args.force:
        print(f"error: {args.output} exists (use --force to overwrite)", file=sys.stderr)
        return 2

    steps: list[tuple[str, float]] = []

    def lap(label: str, t_start: float) -> float:
        elapsed = time.perf_counter() - t_start
        steps.append((label, elapsed))
        return time.perf_counter()

    t = time.perf_counter()
    rgb16 = load_pq_tiff(args.input)
    H, W = rgb16.shape[:2]
    t = lap(f"load TIFF ({W}x{H})", t)

    # Step 1: BT.2020 PQ -> Display P3 PQ (ACES 1.3 gamut compression)
    p3_pq_16, color_timings = color.bt2020_pq_to_p3_pq_uint16(rgb16)
    for name, secs in color_timings.items():
        steps.append((f"  color/{name}", secs))
    t = time.perf_counter()

    # Pack HDR (needed by both paths)
    packed_p3_hdr = pack_rgba1010102(p3_pq_16)
    t = lap("pack RGBA1010102", t)

    if args.use_api3:
        # API-3 path: App-0 tone map -> extract primary JPEG -> API-3
        app0_bytes = _encode_app0(packed_p3_hdr, args.quality)
        t = lap(f"App-0 encode ({len(app0_bytes)/1e3:.0f} KB)", t)
        sdr_jpeg = extract_primary_jpeg(app0_bytes)
        t = lap(f"extract primary JPEG ({len(sdr_jpeg)/1e3:.0f} KB)", t)
        jpg = _encode_api3(packed_p3_hdr, sdr_jpeg, args)
        t = lap(f"API-3 encode ({len(jpg)/1e3:.0f} KB)", t)
    else:
        # Default API-1 path: Python Reinhard tone map -> raw HDR + raw SDR -> libultrahdr
        sdr_rgba8888, tm_timings = color.p3_pq_to_sdr_rgba8888(
            packed_p3_hdr, use_lut=(args.sdr_tonemap == "lut")
        )
        for name, secs in tm_timings.items():
            steps.append((f"  tonemap/{name}", secs))
        t = time.perf_counter()
        jpg = _encode_api1(packed_p3_hdr, sdr_rgba8888, args)
        t = lap(f"API-1 encode ({len(jpg)/1e3:.0f} KB)", t)

    with open(args.output, "wb") as f:
        f.write(jpg)
    t = lap("write output", t)

    total = sum(s for _, s in steps)
    col_w = max(len(n) for n, _ in steps)
    print(f"\n{'Step':<{col_w}}   Time (s)   %", file=sys.stderr)
    print("-" * (col_w + 18), file=sys.stderr)
    for name, secs in steps:
        print(f"{name:<{col_w}}   {secs:7.3f}s  {100*secs/total:5.1f}%", file=sys.stderr)
    print("-" * (col_w + 18), file=sys.stderr)
    print(f"{'TOTAL':<{col_w}}   {total:7.3f}s  100.0%", file=sys.stderr)

    print(args.output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
