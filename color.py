"""BT.2020 PQ color conversion utilities used by the Ultra HDR pipeline."""

import warnings
warnings.filterwarnings("ignore", module=r"colour")

import time
import numpy as np
import colour
import PyOpenColorIO as OCIO

_HDR_PEAK_NITS  = np.float32(10000.0)
_SDR_WHITE_NITS = np.float32(203.0)

# ACES 1.3 Reference Gamut Compression parameters (AMPAS spec defaults).
_RGC_PARAMS = [1.147, 1.264, 1.312, 0.815, 0.803, 0.880, 1.2]

_CPU_PROC: OCIO.CPUProcessor | None = None
_TONEMAP_PROC: OCIO.CPUProcessor | None = None

# PQ EOTF constants matching gainmapmath.cpp (kPqM1, kPqM2, kPqC1-C3).
# pqInvOetf returns [0,1] where 1.0 = 10 000 nits — same scale libultrahdr uses.
_PQ_M1 = np.float32(2610.0 / 16384.0)
_PQ_M2 = np.float32(2523.0 / 4096.0 * 128.0)
_PQ_C1 = np.float32(3424.0 / 4096.0)
_PQ_C2 = np.float32(2413.0 / 4096.0 * 32.0)
_PQ_C3 = np.float32(2392.0 / 4096.0 * 32.0)


def _pq_inv_oetf(signal: np.ndarray) -> np.ndarray:
    """PQ EOTF (pqInvOetf from gainmapmath.cpp): signal [0,1] → linear [0,1], 1.0 = 10 000 nits."""
    val = np.power(np.clip(signal, np.float32(0.0), np.float32(1.0)), np.float32(1.0) / _PQ_M2)
    num = np.maximum(val - _PQ_C1, np.float32(0.0))
    den = _PQ_C2 - _PQ_C3 * val
    return np.power(np.maximum(num / den, np.float32(0.0)), np.float32(1.0) / _PQ_M1)


def _mat3_to_ocio44(m33: np.ndarray) -> list:
    """3×3 ndarray → OCIO MatrixTransform flat 16-element row-major list."""
    m = np.eye(4, dtype=np.float64)
    m[:3, :3] = m33
    return list(m.flatten())


def _build_processor() -> OCIO.CPUProcessor:
    """Build (once) a fused OCIO CPU processor for the full BT.2020 PQ -> P3 PQ pipeline.

    Transform chain (all in one AVX-vectorised pass):
      ST.2084 EOTF  ->  BT.2020->ACEScg matrix  ->  ACES 1.3 RGC
      ->  ACEScg->P3 matrix  ->  clip [0,inf)  ->  ST.2084 inverse EOTF  ->  clip [0,1]

    OCIO normalises linear light so that 1.0 == 10 000 nits, which is scale-invariant
    through the linear matrix and RGC operations.
    """
    global _CPU_PROC
    if _CPU_PROC is not None:
        return _CPU_PROC

    bt2020 = colour.RGB_COLOURSPACES["ITU-R BT.2020"]
    acescg = colour.RGB_COLOURSPACES["ACEScg"]
    p3d65  = colour.RGB_COLOURSPACES["Display P3"]

    M_bt2020_acescg = colour.matrix_RGB_to_RGB(bt2020, acescg, "Bradford")
    M_acescg_p3     = colour.matrix_RGB_to_RGB(acescg, p3d65, "Bradford")

    cfg   = OCIO.Config.CreateRaw()
    group = OCIO.GroupTransform()

    # 1. ST.2084 EOTF: PQ signal [0,1] -> linear [0,1]  (1.0 = 10 000 nits)
    group.appendTransform(OCIO.BuiltinTransform(style="CURVE - ST-2084_to_LINEAR"))

    # 2. BT.2020 -> ACEScg  (Bradford chromatic adaptation)
    group.appendTransform(OCIO.MatrixTransform(matrix=_mat3_to_ocio44(M_bt2020_acescg)))

    # 3. ACES 1.3 Reference Gamut Compression
    group.appendTransform(
        OCIO.FixedFunctionTransform(
            OCIO.FIXED_FUNCTION_ACES_GAMUT_COMP_13, _RGC_PARAMS
        )
    )

    # 4. ACEScg -> Display P3  (Bradford chromatic adaptation)
    group.appendTransform(OCIO.MatrixTransform(matrix=_mat3_to_ocio44(M_acescg_p3)))

    # 5. Clip residual out-of-gamut hues to [0, inf)
    clip_low = OCIO.RangeTransform()
    clip_low.setMinInValue(0.0)
    clip_low.setMinOutValue(0.0)
    group.appendTransform(clip_low)

    # 6. ST.2084 inverse EOTF: linear [0,1] -> PQ signal [0,1]
    group.appendTransform(OCIO.BuiltinTransform(style="CURVE - LINEAR_to_ST-2084"))

    # 7. Clamp final PQ to [0, 1]  (guards against float rounding at peak)
    clip_full = OCIO.RangeTransform()
    clip_full.setMinInValue(0.0)
    clip_full.setMinOutValue(0.0)
    clip_full.setMaxInValue(1.0)
    clip_full.setMaxOutValue(1.0)
    group.appendTransform(clip_full)

    _CPU_PROC = cfg.getProcessor(group).getDefaultCPUProcessor()
    return _CPU_PROC


def _build_tonemap_processor() -> OCIO.CPUProcessor:
    """Build (once) a fused OCIO CPU processor for the full P3 PQ → SDR tone mapping.

    Encodes the complete pipeline into a 65³ 3D LUT in PQ signal space so that
    all five tone-map stages run in a single AVX-vectorised OCIO traversal:
      PQ EOTF (10-bit signal → linear × headroom)
      → per-pixel max-channel Reinhard: y*(1+y/h²)/(1+y)
      → γ 2.2 OETF (linear [0,1] → gamma-encoded [0,1])

    65 grid points per channel (spacing ≈ 0.016 in PQ signal) is sufficient for
    sub-LSB accuracy at 8-bit output after OCIO trilinear interpolation.
    """
    global _TONEMAP_PROC
    if _TONEMAP_PROC is not None:
        return _TONEMAP_PROC

    N = 65  # 65³ ≈ 274 K LUT samples; b varies fastest to match OCIO's layout
    headroom    = float(_HDR_PEAK_NITS / _SDR_WHITE_NITS)   # ≈ 49.26
    headroom_sq = np.float32(headroom * headroom)

    coords = np.linspace(0.0, 1.0, N, dtype=np.float32)
    r_g, g_g, b_g = np.meshgrid(coords, coords, coords, indexing='ij')
    flat = np.stack([r_g.ravel(), g_g.ravel(), b_g.ravel()], axis=-1)  # (N³, 3)

    # PQ EOTF → linear [0,1] (1.0 = 10 000 nits), then scale to Reinhard headroom
    linear = _pq_inv_oetf(flat) * np.float32(headroom)  # (N³, 3), range [0, headroom]

    # Reinhard tone mapping (ReinhardMap in jpegr.cpp)
    y_max = np.max(linear, axis=-1, keepdims=True)
    y_max_sdr = (y_max * (np.float32(1.0) + y_max / headroom_sq)
                 / (np.float32(1.0) + y_max))
    with np.errstate(invalid="ignore", divide="ignore"):
        scale = np.where(y_max > np.float32(0.0), y_max_sdr / y_max, np.float32(0.0))
    sdr = np.clip(linear * scale, np.float32(0.0), np.float32(1.0))

    # γ 2.2 OETF: linear [0,1] → gamma-encoded [0,1]
    sdr_gamma = np.power(
        np.clip(sdr, np.float32(0.0), np.float32(1.0)),
        np.float32(1.0 / 2.2),
    )  # (N³, 3)

    lut3d = OCIO.Lut3DTransform()
    lut3d.setGridSize(N)
    lut3d.setInterpolation(OCIO.INTERP_TETRAHEDRAL)
    lut3d.setData(sdr_gamma.ravel())

    cfg = OCIO.Config.CreateRaw()
    _TONEMAP_PROC = cfg.getProcessor(lut3d).getDefaultCPUProcessor()
    return _TONEMAP_PROC


def bt2020_pq_to_p3_pq_uint16(
    arr_u16: np.ndarray,
) -> tuple[np.ndarray, dict[str, float]]:
    """BT.2020 PQ uint16 (H,W,3) -> (Display P3 PQ uint16 (H,W,3), timings).

    Single-pass OCIO pipeline — all ops fused into one AVX-vectorised traversal:
      ST.2084 EOTF -> BT.2020->ACEScg -> ACES 1.3 RGC -> ACEScg->P3 -> ST.2084 OEOTF

    Returns (result, timings) where timings is a dict of per-substep seconds.
    """
    if arr_u16.dtype != np.uint16:
        raise TypeError(f"expected uint16, got {arr_u16.dtype}")
    if arr_u16.ndim != 3 or arr_u16.shape[2] != 3:
        raise ValueError(f"expected (H, W, 3), got {arr_u16.shape}")

    H, W = arr_u16.shape[:2]

    t = time.perf_counter()
    f32 = np.ascontiguousarray(arr_u16.astype(np.float32) * np.float32(1.0 / 65535.0))
    t_to_f32 = time.perf_counter() - t

    t = time.perf_counter()
    img = OCIO.PackedImageDesc(f32, W, H, OCIO.CHANNEL_ORDERING_RGB)
    _build_processor().apply(img)
    t_ocio = time.perf_counter() - t

    t = time.perf_counter()
    result = (f32 * np.float32(65535.0) + np.float32(0.5)).astype(np.uint16)
    t_to_u16 = time.perf_counter() - t

    timings = {
        "u16->f32":   t_to_f32,
        "ocio_apply": t_ocio,
        "f32->u16":   t_to_u16,
    }
    return result, timings


def p3_pq_to_sdr_rgba8888(
    packed_p3_hdr: np.ndarray,
    use_lut: bool = True,
) -> tuple[np.ndarray, dict[str, float]]:
    """RGBA1010102 uint32 (H,W) → RGBA8888 uint32 (H,W) using libultrahdr's tone map.

    Replicates the Reinhard tone mapper that libultrahdr applies internally during App-0
    (ReinhardMap in jpegr.cpp), operating on exactly the same 10-bit signal values that
    libultrahdr sees (getRgba1010102Pixel divides by 1023.0, not 65535.0).

    use_lut=True  (default): fused OCIO 3D LUT with tetrahedral interpolation —
      unpack + PQ EOTF × headroom + Reinhard + γ 2.2 OETF in one AVX pass.
    use_lut=False (parametric): explicit NumPy steps that exactly reproduce the
      libultrahdr formulae; slower but no LUT approximation.

    Returns (rgba_u32, timings) where rgba_u32 is (H,W) uint32 C-contiguous.
    """
    H, W = packed_p3_hdr.shape
    timings: dict[str, float] = {}

    if use_lut:
        proc = _build_tonemap_processor()  # pre-warm cache before timing starts

    # Unpack 10-bit channels and normalise to [0, 1] PQ signal space.
    # Matches getRgba1010102Pixel in gainmapmath.cpp: float(packed & 0x3ff) / 1023.0f
    t = time.perf_counter()
    r10 = (packed_p3_hdr & np.uint32(0x3FF)).astype(np.float32)
    g10 = ((packed_p3_hdr >> np.uint32(10)) & np.uint32(0x3FF)).astype(np.float32)
    b10 = ((packed_p3_hdr >> np.uint32(20)) & np.uint32(0x3FF)).astype(np.float32)
    f32 = np.ascontiguousarray(
        np.stack([r10, g10, b10], axis=-1) * np.float32(1.0 / 1023.0)
    )
    timings["unpack_10bit"] = time.perf_counter() - t

    if use_lut:
        # Fused OCIO 3D LUT: PQ EOTF + Reinhard + γ 2.2 OETF in one AVX pass.
        # f32 is modified in-place; output is SDR gamma-encoded values in [0, 1].
        t = time.perf_counter()
        img = OCIO.PackedImageDesc(f32, W, H, OCIO.CHANNEL_ORDERING_RGB)
        proc.apply(img)
        timings["ocio_lut3d"] = time.perf_counter() - t
    else:
        # Parametric path: explicit NumPy steps matching libultrahdr's formulae exactly.
        # (OCIO's CURVE - ST-2084_to_LINEAR returns 1.0 = 100 nits, which is 100× off,
        #  so we use _pq_inv_oetf directly to match gainmapmath.cpp pqInvOetf.)
        t = time.perf_counter()
        linear = _pq_inv_oetf(f32)
        timings["pq_decode"] = time.perf_counter() - t

        t = time.perf_counter()
        headroom = _HDR_PEAK_NITS / _SDR_WHITE_NITS
        y = linear * headroom
        y_max = np.max(y, axis=-1, keepdims=True)
        y_max_sdr = (y_max * (np.float32(1.0) + y_max / (headroom * headroom))
                     / (np.float32(1.0) + y_max))
        with np.errstate(invalid="ignore", divide="ignore"):
            scale = np.where(y_max > np.float32(0.0), y_max_sdr / y_max, np.float32(0.0))
        f32 = np.clip(y * scale, np.float32(0.0), np.float32(1.0))
        timings["reinhard"] = time.perf_counter() - t

        t = time.perf_counter()
        f32 = np.power(f32, np.float32(1.0 / 2.2))
        timings["srgb_oetf"] = time.perf_counter() - t

    # Quantise to uint8 and pack as RGBA8888 (R@byte0, G@byte1, B@byte2, A=255@byte3).
    t = time.perf_counter()
    u8 = np.clip(f32 * np.float32(255.0) + np.float32(0.5), 0, 255).astype(np.uint8)
    rgba = np.empty((H, W, 4), dtype=np.uint8)
    rgba[..., :3] = u8
    rgba[..., 3] = 255
    rgba_u32 = np.ascontiguousarray(rgba).view(np.uint32).reshape(H, W)
    timings["pack_rgba8888"] = time.perf_counter() - t

    return rgba_u32, timings
