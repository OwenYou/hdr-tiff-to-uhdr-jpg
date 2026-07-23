"""BT.2020 PQ color conversion utilities used by the Ultra HDR pipeline."""

import warnings
warnings.filterwarnings("ignore", module=r"colour")

import time
import numpy as np
import colour
import PyOpenColorIO as OCIO

_HDR_PEAK_NITS  = np.float32(10000.0)
# _SDR_WHITE_NITS is not used by any Python code — the gain map computation is
# internal to libultrahdr (C library) and assumes 203 nit SDR diffuse white per
# ISO 21496-1.  This constant is kept as a reference so the 203/100 split is
# visible in one place alongside _SDR_TM_WHITE_NITS.
_SDR_WHITE_NITS = np.float32(203.0)
# Target white level for the SDR base tone map only.  Set to 100 nit (typical
# consumer display) rather than 203 nit so that diffuse white in the SDR image
# maps to peak code value, giving a correct-looking result on SDR-only displays.
# This does NOT affect the gain map: libultrahdr always uses 203 nit internally.
_SDR_TM_WHITE_NITS = np.float32(120.0)

# ACES 1.3 Reference Gamut Compression parameters (AMPAS spec defaults).
_RGC_PARAMS = [1.147, 1.264, 1.312, 0.815, 0.803, 0.880, 1.2]

_CPU_PROC: OCIO.CPUProcessor | None = None                   # LUT-based gamut-compress processor
_CPU_PROC_ANALYTICAL: OCIO.CPUProcessor | None = None         # analytical gamut-compress processor
_CPU_PROC_CLIP: OCIO.CPUProcessor | None = None               # LUT-based gamut-clip processor
_CPU_PROC_ANALYTICAL_CLIP: OCIO.CPUProcessor | None = None    # analytical gamut-clip processor
_TONEMAP_PROC: OCIO.CPUProcessor | None = None
_TONEMAP_GREY_PROC: OCIO.CPUProcessor | None = None


def configure_sdr_tm_white(nits: float) -> None:
    """Override the SDR base tone-map white point and invalidate cached LUTs.

    Call once before the first encode.  Has no effect on gain map computation —
    libultrahdr's internal gain map math always uses 203 nit SDR diffuse white.
    """
    global _SDR_TM_WHITE_NITS, _TONEMAP_PROC, _TONEMAP_GREY_PROC
    _SDR_TM_WHITE_NITS = np.float32(nits)
    _TONEMAP_PROC = None
    _TONEMAP_GREY_PROC = None

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


def _build_analytical_processor() -> OCIO.CPUProcessor:
    """Build (once) the analytical BT.2020 PQ -> P3 PQ OCIO pipeline.

    Used directly when use_lut=False, and to bake the 3D LUT in _build_processor().
    Transform chain: ST.2084 EOTF -> BT.2020->ACEScg -> ACES 1.3 RGC
      -> ACEScg->P3 -> clip [0,∞) -> ST.2084 OETF -> clip [0,1]
    """
    global _CPU_PROC_ANALYTICAL
    if _CPU_PROC_ANALYTICAL is not None:
        return _CPU_PROC_ANALYTICAL

    bt2020 = colour.RGB_COLOURSPACES["ITU-R BT.2020"]
    acescg = colour.RGB_COLOURSPACES["ACEScg"]
    p3d65  = colour.RGB_COLOURSPACES["Display P3"]

    M_bt2020_acescg = colour.matrix_RGB_to_RGB(bt2020, acescg, "Bradford")
    M_acescg_p3     = colour.matrix_RGB_to_RGB(acescg, p3d65, "Bradford")

    cfg   = OCIO.Config.CreateRaw()
    group = OCIO.GroupTransform()

    group.appendTransform(OCIO.BuiltinTransform(style="CURVE - ST-2084_to_LINEAR"))
    group.appendTransform(OCIO.MatrixTransform(matrix=_mat3_to_ocio44(M_bt2020_acescg)))
    group.appendTransform(
        OCIO.FixedFunctionTransform(
            OCIO.FIXED_FUNCTION_ACES_GAMUT_COMP_13, _RGC_PARAMS
        )
    )
    group.appendTransform(OCIO.MatrixTransform(matrix=_mat3_to_ocio44(M_acescg_p3)))

    clip_low = OCIO.RangeTransform()
    clip_low.setMinInValue(0.0)
    clip_low.setMinOutValue(0.0)
    group.appendTransform(clip_low)

    group.appendTransform(OCIO.BuiltinTransform(style="CURVE - LINEAR_to_ST-2084"))

    clip_full = OCIO.RangeTransform()
    clip_full.setMinInValue(0.0)
    clip_full.setMinOutValue(0.0)
    clip_full.setMaxInValue(1.0)
    clip_full.setMaxOutValue(1.0)
    group.appendTransform(clip_full)

    _CPU_PROC_ANALYTICAL = cfg.getProcessor(group).getDefaultCPUProcessor()
    return _CPU_PROC_ANALYTICAL


def _build_processor() -> OCIO.CPUProcessor:
    """Build (once) a 3D LUT CPU processor for BT.2020 PQ -> P3 PQ.

    Bakes the full analytical pipeline (EOTF + RGC + matrix + OETF) into a 97³
    3D LUT with tetrahedral interpolation.  Per-pixel cost drops from per-pixel
    ACES RGC arithmetic to a single table lookup; 97 grid points give sub-LSB
    accuracy at 10-bit output.
    """
    global _CPU_PROC
    if _CPU_PROC is not None:
        return _CPU_PROC

    analytical = _build_analytical_processor()

    N = 97  # 97³ ≈ 912 K samples; b varies fastest to match OCIO's LUT layout
    coords = np.linspace(0.0, 1.0, N, dtype=np.float32)
    r_g, g_g, b_g = np.meshgrid(coords, coords, coords, indexing='ij')
    flat = np.ascontiguousarray(
        np.stack([r_g.ravel(), g_g.ravel(), b_g.ravel()], axis=-1)
    )  # (N³, 3) — b varies fastest

    img = OCIO.PackedImageDesc(flat, N * N * N, 1, OCIO.CHANNEL_ORDERING_RGB)
    analytical.apply(img)  # bake: flat is overwritten with P3 PQ output values

    lut3d = OCIO.Lut3DTransform()
    lut3d.setGridSize(N)
    lut3d.setInterpolation(OCIO.INTERP_TETRAHEDRAL)
    lut3d.setData(flat.ravel())

    cfg = OCIO.Config.CreateRaw()
    _CPU_PROC = cfg.getProcessor(lut3d).getDefaultCPUProcessor()
    return _CPU_PROC


def _build_analytical_processor_clip() -> OCIO.CPUProcessor:
    """Build (once) the analytical BT.2020 PQ -> P3 PQ pipeline with direct clip.

    Same chain as _build_analytical_processor() but skips ACES 1.3 RGC: out-of-gamut
    colors are hard-clipped to [0, ∞) in linear light after the BT.2020→P3 matrix.
    Transform chain: ST.2084 EOTF -> BT.2020->P3 (Bradford CAT)
      -> clip [0,∞) -> ST.2084 OETF -> clip [0,1]
    """
    global _CPU_PROC_ANALYTICAL_CLIP
    if _CPU_PROC_ANALYTICAL_CLIP is not None:
        return _CPU_PROC_ANALYTICAL_CLIP

    bt2020 = colour.RGB_COLOURSPACES["ITU-R BT.2020"]
    p3d65  = colour.RGB_COLOURSPACES["Display P3"]

    M_bt2020_p3 = colour.matrix_RGB_to_RGB(bt2020, p3d65, "Bradford")

    cfg   = OCIO.Config.CreateRaw()
    group = OCIO.GroupTransform()

    group.appendTransform(OCIO.BuiltinTransform(style="CURVE - ST-2084_to_LINEAR"))
    group.appendTransform(OCIO.MatrixTransform(matrix=_mat3_to_ocio44(M_bt2020_p3)))

    clip_low = OCIO.RangeTransform()
    clip_low.setMinInValue(0.0)
    clip_low.setMinOutValue(0.0)
    group.appendTransform(clip_low)

    group.appendTransform(OCIO.BuiltinTransform(style="CURVE - LINEAR_to_ST-2084"))

    clip_full = OCIO.RangeTransform()
    clip_full.setMinInValue(0.0)
    clip_full.setMinOutValue(0.0)
    clip_full.setMaxInValue(1.0)
    clip_full.setMaxOutValue(1.0)
    group.appendTransform(clip_full)

    _CPU_PROC_ANALYTICAL_CLIP = cfg.getProcessor(group).getDefaultCPUProcessor()
    return _CPU_PROC_ANALYTICAL_CLIP


def _build_processor_clip() -> OCIO.CPUProcessor:
    """Build (once) a 3D LUT CPU processor for BT.2020 PQ -> P3 PQ with direct clip.

    Bakes _build_analytical_processor_clip() into a 97³ 3D LUT with tetrahedral
    interpolation.  Same grid size as the gamut-compress LUT.
    """
    global _CPU_PROC_CLIP
    if _CPU_PROC_CLIP is not None:
        return _CPU_PROC_CLIP

    analytical = _build_analytical_processor_clip()

    N = 97
    coords = np.linspace(0.0, 1.0, N, dtype=np.float32)
    r_g, g_g, b_g = np.meshgrid(coords, coords, coords, indexing='ij')
    flat = np.ascontiguousarray(
        np.stack([r_g.ravel(), g_g.ravel(), b_g.ravel()], axis=-1)
    )

    img = OCIO.PackedImageDesc(flat, N * N * N, 1, OCIO.CHANNEL_ORDERING_RGB)
    analytical.apply(img)

    lut3d = OCIO.Lut3DTransform()
    lut3d.setGridSize(N)
    lut3d.setInterpolation(OCIO.INTERP_TETRAHEDRAL)
    lut3d.setData(flat.ravel())

    cfg = OCIO.Config.CreateRaw()
    _CPU_PROC_CLIP = cfg.getProcessor(lut3d).getDefaultCPUProcessor()
    return _CPU_PROC_CLIP


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
    headroom    = float(_HDR_PEAK_NITS / _SDR_TM_WHITE_NITS)   # ≈ 100.0 (10000/100)
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


# BT.709 luma coefficients for greyscale in linear light (matches _sdr_u32_to_greyscale)
_LUMA_R = np.float32(0.2126)
_LUMA_G = np.float32(0.7152)
_LUMA_B = np.float32(0.0722)


def _build_tonemap_grey_processor() -> OCIO.CPUProcessor:
    """Build (once) a fused 65³ LUT for P3 PQ -> greyscale SDR tone mapping.

    Identical to _build_tonemap_processor but applies BT.709 luma weighting
    in linear light (after Reinhard, before γ2.2) so all three output channels
    carry R=G=B=luma.  The RGBA8888 pack step then produces a true greyscale
    image with no additional post-processing required.
    """
    global _TONEMAP_GREY_PROC
    if _TONEMAP_GREY_PROC is not None:
        return _TONEMAP_GREY_PROC

    N = 65
    headroom    = float(_HDR_PEAK_NITS / _SDR_TM_WHITE_NITS)
    headroom_sq = np.float32(headroom * headroom)

    coords = np.linspace(0.0, 1.0, N, dtype=np.float32)
    r_g, g_g, b_g = np.meshgrid(coords, coords, coords, indexing='ij')
    flat = np.stack([r_g.ravel(), g_g.ravel(), b_g.ravel()], axis=-1)  # (N³, 3)

    linear = _pq_inv_oetf(flat) * np.float32(headroom)

    y_max = np.max(linear, axis=-1, keepdims=True)
    y_max_sdr = (y_max * (np.float32(1.0) + y_max / headroom_sq)
                 / (np.float32(1.0) + y_max))
    with np.errstate(invalid="ignore", divide="ignore"):
        scale = np.where(y_max > np.float32(0.0), y_max_sdr / y_max, np.float32(0.0))
    sdr = np.clip(linear * scale, np.float32(0.0), np.float32(1.0))

    # BT.709 luma weighting in linear light -> broadcast to all 3 channels
    luma = (sdr[..., 0:1] * _LUMA_R +
            sdr[..., 1:2] * _LUMA_G +
            sdr[..., 2:3] * _LUMA_B)
    sdr_grey = np.repeat(luma, 3, axis=-1)

    sdr_gamma = np.power(
        np.clip(sdr_grey, np.float32(0.0), np.float32(1.0)),
        np.float32(1.0 / 2.2),
    )

    lut3d = OCIO.Lut3DTransform()
    lut3d.setGridSize(N)
    lut3d.setInterpolation(OCIO.INTERP_TETRAHEDRAL)
    lut3d.setData(sdr_gamma.ravel())

    cfg = OCIO.Config.CreateRaw()
    _TONEMAP_GREY_PROC = cfg.getProcessor(lut3d).getDefaultCPUProcessor()
    return _TONEMAP_GREY_PROC


def bt2020_pq_to_p3_pq(
    arr_u16: np.ndarray,
    use_lut: bool = True,
    gamut_compress: bool = True,
) -> tuple[np.ndarray, dict[str, float]]:
    """BT.2020 PQ uint16 (H,W,3) -> Display P3 PQ float32 (H,W,3) in [0,1].

    use_lut=True  (default): 97³ 3D LUT with tetrahedral interpolation.
    use_lut=False (parametric): per-pixel analytical OCIO pipeline, no LUT approximation.
    gamut_compress=True  (default): ACES 1.3 Reference Gamut Compression via ACEScg.
    gamut_compress=False: direct hard clip of out-of-gamut values after BT.2020→P3 matrix.

    Returns (result_f32, timings).  result_f32 is a C-contiguous float32 array with PQ
    signal values in [0, 1].  Pass it directly to pack_rgba1010102() and
    p3_pq_to_sdr_rgba8888(); no uint16 intermediate is required.
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
    if gamut_compress:
        proc = _build_processor() if use_lut else _build_analytical_processor()
    else:
        proc = _build_processor_clip() if use_lut else _build_analytical_processor_clip()
    img = OCIO.PackedImageDesc(f32, W, H, OCIO.CHANNEL_ORDERING_RGB)
    proc.apply(img)
    t_ocio = time.perf_counter() - t

    return f32, {"u16->f32": t_to_f32, "ocio_apply": t_ocio}


def p3_pq_to_sdr_rgba8888(
    p3_pq_f32: np.ndarray,
    use_lut: bool = True,
    bw: bool = False,
) -> tuple[np.ndarray, dict[str, float]]:
    """Display P3 PQ float32 (H,W,3) in [0,1] → RGBA8888 uint32 (H,W).

    Replicates the Reinhard tone mapper that libultrahdr applies internally during App-0
    (ReinhardMap in jpegr.cpp), operating on the same PQ signal values.

    use_lut=True  (default): fused OCIO 3D LUT with tetrahedral interpolation —
      PQ EOTF × headroom + Reinhard + γ 2.2 OETF in one AVX pass.
    use_lut=False (parametric): explicit NumPy steps that exactly reproduce the
      libultrahdr formulae; slower but no LUT approximation.
    bw=True: desaturate to BT.709 luma greyscale in linear light before γ 2.2.
      In lut mode a separate pre-baked 65³ LUT is used — no post-processing needed.

    NOTE: when use_lut=True, OCIO modifies p3_pq_f32 in-place.  The caller must
    not use p3_pq_f32 after this call (pack_rgba1010102 should be called first).

    Returns (rgba_u32, timings) where rgba_u32 is (H,W) uint32 C-contiguous.
    """
    if p3_pq_f32.ndim != 3 or p3_pq_f32.shape[2] != 3:
        raise ValueError(f"expected (H, W, 3), got {p3_pq_f32.shape}")

    H, W = p3_pq_f32.shape[:2]
    timings: dict[str, float] = {}

    if use_lut:
        proc = _build_tonemap_grey_processor() if bw else _build_tonemap_processor()

    # Ensure C-contiguous layout required by OCIO PackedImageDesc.
    # np.ascontiguousarray returns the array itself when already C-contiguous,
    # so OCIO's in-place apply will overwrite p3_pq_f32 (caller is notified above).
    f32 = np.ascontiguousarray(p3_pq_f32)

    if use_lut:
        # Fused OCIO 3D LUT: PQ EOTF + Reinhard [+ luma] + γ 2.2 OETF in one AVX pass.
        # f32 is modified in-place; output is SDR gamma-encoded values in [0, 1].
        t = time.perf_counter()
        img = OCIO.PackedImageDesc(f32, W, H, OCIO.CHANNEL_ORDERING_RGB)
        proc.apply(img)
        timings["ocio_lut3d"] = time.perf_counter() - t
    else:
        # Parametric path: explicit NumPy steps matching libultrahdr's formulae exactly.
        t = time.perf_counter()
        linear = _pq_inv_oetf(f32)
        timings["pq_decode"] = time.perf_counter() - t

        t = time.perf_counter()
        headroom = _HDR_PEAK_NITS / _SDR_TM_WHITE_NITS
        y = linear * headroom
        y_max = np.max(y, axis=-1, keepdims=True)
        y_max_sdr = (y_max * (np.float32(1.0) + y_max / (headroom * headroom))
                     / (np.float32(1.0) + y_max))
        with np.errstate(invalid="ignore", divide="ignore"):
            scale = np.where(y_max > np.float32(0.0), y_max_sdr / y_max, np.float32(0.0))
        f32 = np.clip(y * scale, np.float32(0.0), np.float32(1.0))
        timings["reinhard"] = time.perf_counter() - t

        if bw:
            t = time.perf_counter()
            luma = (f32[..., 0:1] * _LUMA_R +
                    f32[..., 1:2] * _LUMA_G +
                    f32[..., 2:3] * _LUMA_B)
            f32 = np.repeat(luma, 3, axis=-1)
            timings["luma"] = time.perf_counter() - t

        t = time.perf_counter()
        f32 = np.power(f32, np.float32(1.0 / 2.2))
        timings["srgb_oetf"] = time.perf_counter() - t

    # Quantise to uint8 and pack as RGBA8888 (R@byte0, G@byte1, B@byte2, A=255@byte3).
    # Scale f32 in-place (out=) to avoid 3 × 414 MB float temporaries from the
    # expression chain f32*255 + 0.5 + clip.
    t = time.perf_counter()
    np.multiply(f32, np.float32(255.0), out=f32)
    np.add(f32, np.float32(0.5), out=f32)
    np.clip(f32, np.float32(0.0), np.float32(255.0), out=f32)
    u8 = f32.astype(np.uint8)
    rgba = np.empty((H, W, 4), dtype=np.uint8)
    rgba[..., :3] = u8
    rgba[..., 3] = 255
    rgba_u32 = rgba.view(np.uint32).reshape(H, W)
    timings["pack_rgba8888"] = time.perf_counter() - t

    return rgba_u32, timings
