"""BT.2020 PQ color conversion utilities used by the Ultra HDR pipeline."""

import numpy as np
import colour
import PyOpenColorIO as OCIO

# ACES 1.3 Reference Gamut Compression parameters (AMPAS spec defaults).
# Applied in ACEScg (AP1) space to soft-compress out-of-gamut hues.
_RGC_PARAMS = [1.147, 1.264, 1.312, 0.815, 0.803, 0.880, 1.2]
_RGC_CPU = None


def _rgc_cpu():
    global _RGC_CPU
    if _RGC_CPU is None:
        cfg = OCIO.Config.CreateRaw()
        ff = OCIO.FixedFunctionTransform(OCIO.FIXED_FUNCTION_ACES_GAMUT_COMP_13, _RGC_PARAMS)
        _RGC_CPU = cfg.getProcessor(ff).getDefaultCPUProcessor()
    return _RGC_CPU


def bt2020_pq_to_p3_pq_uint16(arr_u16: np.ndarray) -> np.ndarray:
    """BT.2020 PQ uint16 (H,W,3) -> Display P3 PQ uint16 (H,W,3).

    Pipeline:
      ST.2084 EOTF -> BT.2020 linear -> ACEScg (Bradford CAT)
      -> ACES 1.3 RGC (soft gamut compression toward AP1)
      -> ACEScg -> Display P3 (Bradford CAT)
      -> clip residuals -> ST.2084 inverse EOTF -> uint16
    """
    if arr_u16.dtype != np.uint16:
        raise TypeError(f"expected uint16, got {arr_u16.dtype}")
    if arr_u16.ndim != 3 or arr_u16.shape[2] != 3:
        raise ValueError(f"expected (H, W, 3), got {arr_u16.shape}")

    pq = arr_u16.astype(np.float32) * np.float32(1.0 / 65535.0)
    nits = np.asarray(colour.eotf(pq, function="ST 2084"), dtype=np.float32)

    bt2020 = colour.RGB_COLOURSPACES["ITU-R BT.2020"]
    acescg = colour.RGB_COLOURSPACES["ACEScg"]
    p3d65 = colour.RGB_COLOURSPACES["Display P3"]

    acescg_linear = colour.RGB_to_RGB(
        nits, bt2020, acescg, chromatic_adaptation_transform="Bradford"
    ).astype(np.float32, copy=False)
    acescg_linear = np.ascontiguousarray(acescg_linear)

    H, W = acescg_linear.shape[:2]
    ocio_img = OCIO.PackedImageDesc(acescg_linear, W, H, OCIO.CHANNEL_ORDERING_RGB)
    _rgc_cpu().apply(ocio_img)

    p3_linear = colour.RGB_to_RGB(
        acescg_linear, acescg, p3d65, chromatic_adaptation_transform="Bradford"
    ).astype(np.float32, copy=False)
    np.clip(p3_linear, 0.0, None, out=p3_linear)

    pq_out = np.asarray(colour.eotf_inverse(p3_linear, function="ST 2084"), dtype=np.float32)
    np.clip(pq_out, 0.0, 1.0, out=pq_out)
    return (pq_out * np.float32(65535.0) + np.float32(0.5)).astype(np.uint16)
