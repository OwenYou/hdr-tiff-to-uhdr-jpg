"""Fabricate a UHDR test image exposing the 1-ch vs 3-ch metadata mismatch.

Image design
------------
SDR base:    uniform neutral grey  (Display P3, sRGB transfer, RGBA8888)
HDR intent:  red | green | blue squares  (Display P3, PQ, RGBA1010102)

Why the per-channel gain statistics are identical
-------------------------------------------------
The gain map encodes log2(HDR / SDR) per channel. With uniform grey SDR
and symmetric pure-primary HDR squares:

  Red square   HDR (R=H, G=0, B=0):
      gain_R large, gain_G tiny, gain_B tiny
  Green square HDR (R=0, G=H, B=0):
      gain_R tiny,  gain_G large, gain_B tiny
  Blue square  HDR (R=0, G=0, B=H):
      gain_R tiny,  gain_G tiny,  gain_B large

By symmetry:
  max(gain_R) == max(gain_G) == max(gain_B)   (each has one bright square)
  min(gain_R) == min(gain_G) == min(gain_B)   (each has two dark squares)
  -> allChannelsIdentical() == true -> ISO metadata collapses to 1-channel

The gain map JPEG is still RGB888. The spatial distributions differ per
channel: R is hot on the left square only, G on the middle, B on the right.

Effect on decoding
------------------
Conformant 1-ch decode (uses only first JPEG component for all channels):
  All three squares get the same luma boost -> uniform grey HDR (wrong)

Correct 3-ch decode (uses per-channel JPEG data):
  Each channel boosted by its own spatial gain -> red|green|blue HDR (correct)

The _3ch file carries is_multichannel=True so a conformant decoder uses
per-channel gain and reproduces the correct coloured HDR image.

Usage:
    uv run python tools/_make_rgb_squares_uhdr.py [out_stem] [--quality N]
"""

import argparse
import ctypes
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from uhdr_ctypes import (
    uhdr, check, byref,
    UhdrRawImage,
    UHDR_IMG_FMT_32bppRGBA1010102, UHDR_IMG_FMT_32bppRGBA8888,
    UHDR_CG_DISPLAY_P3,
    UHDR_CT_PQ, UHDR_CT_SRGB,
    UHDR_CR_FULL_RANGE,
    UHDR_HDR_IMG, UHDR_SDR_IMG, UHDR_BASE_IMG, UHDR_GAIN_MAP_IMG,
)

# ---------------------------------------------------------------------------
SQ = 256
W, H = SQ * 3, SQ

SDR_GREY_LINEAR = 0.18      # neutral grey for SDR base
HDR_PRIMARY_NITS = 406.0    # ~2x SDR white (203 nits)
PEAK_NITS = 1000.0

HDR_PRIMARIES = [
    (1.0, 0.0, 0.0),   # red   (left)
    (0.0, 1.0, 0.0),   # green (middle)
    (0.0, 0.0, 1.0),   # blue  (right)
]


# ---------------------------------------------------------------------------
def srgb_oetf(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, 0.0, 1.0)
    return np.where(x <= 0.0031308, 12.92 * x, 1.055 * x ** (1.0 / 2.4) - 0.055)


def pq_oetf(x: np.ndarray) -> np.ndarray:
    m1, m2 = 0.1593017578125, 78.84375
    c1, c2, c3 = 0.8359375, 18.8515625, 18.6875
    xp = np.maximum(x, 0.0) ** m1
    return ((c1 + c2 * xp) / (1.0 + c3 * xp)) ** m2


def make_sdr() -> np.ndarray:
    """(H, W, 4) uint8 RGBA8888 -- uniform neutral grey, sRGB gamma, A=255."""
    grey_enc = int(np.clip(srgb_oetf(np.array([SDR_GREY_LINEAR]))[0] * 255 + 0.5, 0, 255))
    img = np.full((H, W, 4), grey_enc, dtype=np.uint8)
    img[..., 3] = 255
    return img


def make_hdr() -> np.ndarray:
    """(H, W) uint32 RGBA1010102 -- three primary-colour squares, PQ."""
    norm = HDR_PRIMARY_NITS / 10000.0
    f32 = np.zeros((H, W, 3), dtype=np.float32)
    for i, (r, g, b) in enumerate(HDR_PRIMARIES):
        for ch_idx, val in enumerate((r, g, b)):
            pq_val = float(pq_oetf(np.array([val * norm]))[0]) if val > 0 else 0.0
            f32[:, i * SQ:(i + 1) * SQ, ch_idx] = pq_val
    u32 = (f32 * np.float32(1023.0) + np.float32(0.5)).astype(np.uint32)
    u32[..., 1] <<= np.uint32(10)
    u32[..., 2] <<= np.uint32(20)
    packed = np.empty((H, W), dtype=np.uint32)
    np.bitwise_or(u32[..., 0], u32[..., 1], out=packed)
    np.bitwise_or(packed, u32[..., 2], out=packed)
    packed |= np.uint32(0xC0000000)
    return packed


def _raw_image(fmt, cg, ct, arr):
    h, w = arr.shape[:2]
    img = UhdrRawImage()
    img.fmt = fmt; img.cg = cg; img.ct = ct; img.range = UHDR_CR_FULL_RANGE
    img.w = w; img.h = h
    img.planes[0] = arr.ctypes.data_as(ctypes.c_void_p).value
    img.planes[1] = img.planes[2] = 0
    img.stride[0] = w; img.stride[1] = img.stride[2] = 0
    return img


def encode(sdr, hdr, quality, force_rgb_metadata):
    sdr_img = _raw_image(UHDR_IMG_FMT_32bppRGBA8888, UHDR_CG_DISPLAY_P3, UHDR_CT_SRGB, sdr)
    hdr_img = _raw_image(UHDR_IMG_FMT_32bppRGBA1010102, UHDR_CG_DISPLAY_P3, UHDR_CT_PQ, hdr)
    enc = uhdr.uhdr_create_encoder()
    if not enc:
        raise RuntimeError("uhdr_create_encoder returned NULL")
    try:
        check(uhdr.uhdr_enc_set_raw_image(enc, byref(hdr_img), UHDR_HDR_IMG), "HDR")
        check(uhdr.uhdr_enc_set_raw_image(enc, byref(sdr_img), UHDR_SDR_IMG), "SDR")
        check(uhdr.uhdr_enc_set_using_multi_channel_gainmap(enc, 1), "multichannel")
        check(uhdr.uhdr_enc_set_force_rgb_gainmap_metadata(enc, int(force_rgb_metadata)), "force_rgb")
        check(uhdr.uhdr_enc_set_quality(enc, quality, UHDR_BASE_IMG), "quality base")
        check(uhdr.uhdr_enc_set_quality(enc, quality, UHDR_GAIN_MAP_IMG), "quality gm")
        check(uhdr.uhdr_enc_set_target_display_peak_brightness(enc, PEAK_NITS), "peak")
        check(uhdr.uhdr_encode(enc), "encode")
        out = uhdr.uhdr_get_encoded_stream(enc)
        if not out:
            raise RuntimeError("uhdr_get_encoded_stream returned NULL")
        result = ctypes.string_at(out.contents.data, out.contents.data_sz)
        _ = sdr, hdr
        return result
    finally:
        uhdr.uhdr_release_encoder(enc)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("output", nargs="?", default="rgb_squares")
    ap.add_argument("--quality", type=int, default=95)
    args = ap.parse_args()

    out_dir = Path(args.output).parent
    stem = Path(args.output).stem
    sdr = make_sdr()
    hdr = make_hdr()

    grey_u8 = int(np.clip(srgb_oetf(np.array([SDR_GREY_LINEAR]))[0] * 255 + 0.5, 0, 255))
    print(f"Image:    {W}x{H}  ({SQ}px squares)")
    print(f"SDR base: uniform grey  (linear {SDR_GREY_LINEAR}, sRGB u8={grey_u8})")
    print(f"HDR alt:  red | green | blue  ({HDR_PRIMARY_NITS:.0f} nits, PQ)")
    print()

    for force, suffix in [(False, "1ch"), (True, "3ch")]:
        path = out_dir / f"{stem}_{suffix}.uhdr.jpg"
        data = encode(sdr, hdr, args.quality, force)
        path.write_bytes(data)
        label = ("1-ch metadata (collapse) -- conformant decoder: grey HDR (wrong)"
                 if not force else
                 "3-ch metadata (forced)   -- conformant decoder: colour HDR (correct)")
        print(f"  {label}")
        print(f"    {path}  ({len(data)/1024:.1f} KB)")

    print()
    print("Inspect:")
    print(f"  uv run python tools/_inspect_full.py {stem}_1ch.uhdr.jpg")
    print(f"  uv run python tools/_inspect_full.py {stem}_3ch.uhdr.jpg")


if __name__ == "__main__":
    main()
