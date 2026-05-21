"""ctypes bindings for libultrahdr (uhdr.dll).

Loads uhdr.dll from the same directory as this module and exposes the minimum
encoder surface needed for the API-0 (HDR-only) path.
"""

import os
import ctypes
from ctypes import (
    Structure, POINTER, byref,
    c_int, c_uint, c_size_t, c_float, c_char, c_void_p,
)

# --- Enum values from ultrahdr_api.h --------------------------------------
UHDR_CODEC_OK = 0

UHDR_IMG_FMT_32bppRGBA1010102 = 5
UHDR_IMG_FMT_64bppRGBAHalfFloat = 4
UHDR_IMG_FMT_32bppRGBA8888 = 3

UHDR_CG_UNSPECIFIED = -1
UHDR_CG_BT_709 = 0
UHDR_CG_DISPLAY_P3 = 1
UHDR_CG_BT_2100 = 2

UHDR_CT_UNSPECIFIED = -1
UHDR_CT_LINEAR = 0
UHDR_CT_HLG = 1
UHDR_CT_PQ = 2
UHDR_CT_SRGB = 3

UHDR_CR_FULL_RANGE = 1

UHDR_HDR_IMG = 0
UHDR_SDR_IMG = 1
UHDR_BASE_IMG = 2
UHDR_GAIN_MAP_IMG = 3


# --- Structures -----------------------------------------------------------
class UhdrErrorInfo(Structure):
    _fields_ = [
        ("error_code", c_int),
        ("has_detail", c_int),
        ("detail",     c_char * 256),
    ]


class UhdrRawImage(Structure):
    _fields_ = [
        ("fmt",    c_int),
        ("cg",     c_int),
        ("ct",     c_int),
        ("range",  c_int),
        ("w",      c_uint),
        ("h",      c_uint),
        ("planes", c_void_p * 3),
        ("stride", c_uint   * 3),
    ]


class UhdrCompressedImage(Structure):
    _fields_ = [
        ("data",     c_void_p),
        ("data_sz",  c_size_t),
        ("capacity", c_size_t),
        ("cg",       c_int),
        ("ct",       c_int),
        ("range",    c_int),
    ]


class UhdrMemBlock(Structure):
    _fields_ = [
        ("data",     c_void_p),
        ("data_sz",  c_size_t),
        ("capacity", c_size_t),
    ]


assert ctypes.sizeof(UhdrRawImage) == 64, (
    f"UhdrRawImage size mismatch: {ctypes.sizeof(UhdrRawImage)} != 64 "
    "(check ABI / compiler padding)"
)


# --- DLL load -------------------------------------------------------------
_here = os.path.dirname(os.path.abspath(__file__))
_dll_dir_handle = os.add_dll_directory(_here)  # keep reference alive
_dll_path = os.path.join(_here, "uhdr.dll")
if not os.path.exists(_dll_path):
    raise FileNotFoundError(
        f"uhdr.dll not found at {_dll_path}. Build libultrahdr and copy "
        "uhdr.dll + jpeg62.dll next to this module."
    )
uhdr = ctypes.CDLL(_dll_path)


# --- Function prototypes --------------------------------------------------
_codec_p = c_void_p

uhdr.uhdr_create_encoder.argtypes = []
uhdr.uhdr_create_encoder.restype = _codec_p

uhdr.uhdr_release_encoder.argtypes = [_codec_p]
uhdr.uhdr_release_encoder.restype = None

uhdr.uhdr_enc_set_raw_image.argtypes = [_codec_p, POINTER(UhdrRawImage), c_int]
uhdr.uhdr_enc_set_raw_image.restype = UhdrErrorInfo

uhdr.uhdr_enc_set_quality.argtypes = [_codec_p, c_int, c_int]
uhdr.uhdr_enc_set_quality.restype = UhdrErrorInfo

uhdr.uhdr_enc_set_using_multi_channel_gainmap.argtypes = [_codec_p, c_int]
uhdr.uhdr_enc_set_using_multi_channel_gainmap.restype = UhdrErrorInfo

uhdr.uhdr_enc_set_gainmap_scale_factor.argtypes = [_codec_p, c_int]
uhdr.uhdr_enc_set_gainmap_scale_factor.restype = UhdrErrorInfo

uhdr.uhdr_enc_set_gainmap_gamma.argtypes = [_codec_p, c_float]
uhdr.uhdr_enc_set_gainmap_gamma.restype = UhdrErrorInfo

uhdr.uhdr_enc_set_exif_data.argtypes = [_codec_p, POINTER(UhdrMemBlock)]
uhdr.uhdr_enc_set_exif_data.restype = UhdrErrorInfo

uhdr.uhdr_enc_set_target_display_peak_brightness.argtypes = [_codec_p, c_float]
uhdr.uhdr_enc_set_target_display_peak_brightness.restype = UhdrErrorInfo

uhdr.uhdr_enc_set_compressed_image.argtypes = [_codec_p, POINTER(UhdrCompressedImage), c_int]
uhdr.uhdr_enc_set_compressed_image.restype = UhdrErrorInfo

uhdr.uhdr_encode.argtypes = [_codec_p]
uhdr.uhdr_encode.restype = UhdrErrorInfo

uhdr.uhdr_get_encoded_stream.argtypes = [_codec_p]
uhdr.uhdr_get_encoded_stream.restype = POINTER(UhdrCompressedImage)

uhdr.is_uhdr_image.argtypes = [c_void_p, c_int]
uhdr.is_uhdr_image.restype = c_int


def check(err: UhdrErrorInfo, where: str) -> None:
    """Raise RuntimeError if the libultrahdr call did not return OK."""
    if err.error_code != UHDR_CODEC_OK:
        detail = ""
        if err.has_detail:
            detail = err.detail.decode("utf-8", "replace").rstrip("\x00")
        raise RuntimeError(
            f"libultrahdr {where} failed (code={err.error_code}): {detail}"
        )


__all__ = [
    "uhdr", "check", "byref",
    "UhdrErrorInfo", "UhdrRawImage", "UhdrCompressedImage", "UhdrMemBlock",
    "UHDR_IMG_FMT_32bppRGBA1010102", "UHDR_IMG_FMT_64bppRGBAHalfFloat",
    "UHDR_IMG_FMT_32bppRGBA8888",
    "UHDR_CG_UNSPECIFIED", "UHDR_CG_BT_709", "UHDR_CG_DISPLAY_P3", "UHDR_CG_BT_2100",
    "UHDR_CT_UNSPECIFIED", "UHDR_CT_LINEAR", "UHDR_CT_HLG", "UHDR_CT_PQ", "UHDR_CT_SRGB",
    "UHDR_CR_FULL_RANGE",
    "UHDR_HDR_IMG", "UHDR_SDR_IMG", "UHDR_BASE_IMG", "UHDR_GAIN_MAP_IMG",
]
