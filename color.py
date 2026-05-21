"""BT.2020 PQ color conversion utilities used by the Ultra HDR pipeline."""

import warnings
warnings.filterwarnings("ignore", module=r"colour")

import time
import numpy as np
import colour
import PyOpenColorIO as OCIO

# ACES 1.3 Reference Gamut Compression parameters (AMPAS spec defaults).
_RGC_PARAMS = [1.147, 1.264, 1.312, 0.815, 0.803, 0.880, 1.2]

_CPU_PROC: OCIO.CPUProcessor | None = None


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
