# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this project is

A Python pipeline that converts a single-layer **PQ HDR TIFF** (16-bit RGB, BT.2020) into a **Google Ultra HDR JPEG** using stock `libultrahdr` (google/libultrahdr, no local patches). The pipeline runs two libultrahdr encode passes:

1. **App-0 pass** — the BT.2020→Display P3 re-gamuted image is fed to the HDR-only encoder (`uhdr_enc_set_raw_image(HDR_IMG)` only); libultrahdr internally tone-maps it and returns a UHDR JPEG whose embedded primary JPEG is the Display P3 SDR base.
2. **API-3 pass** — the same P3 PQ image is supplied as the raw HDR intent and the primary JPEG extracted from step 1 is supplied as the compressed SDR intent (`uhdr_enc_set_compressed_image(SDR_IMG)`); libultrahdr computes a multi-channel (RGB) gain map and assembles the final UHDR JPEG. Both renditions are Display P3 → `use_base_cg=true`.

The BT.2020→P3 re-gamut applies ACES 1.3 Reference Gamut Compression (OCIO `FIXED_FUNCTION_ACES_GAMUT_COMP_13`) so far-out-of-gamut hues are soft-compressed rather than hard-clipped.

## Commands

This is a `uv`-managed project. Always invoke Python through `uv run`:

```bash
# Run the main pipeline (BT.2020 PQ TIFF -> Display P3 Ultra HDR JPEG)
uv run python convert.py <input.tif> <output.jpg> [--force] [--verbose]

# Flags:
#   --quality 95         base + gainmap JPEG quality 0..100 (default: 95)
#   --gainmap-scale 1    gain map downscale factor 1..128 (default: 1)
#   --gainmap-gamma 1.0  encoding gamma applied to the gain map (default: 1.0)
#   --peak-nits 1000     target HDR display peak in nits 203..10000 (default: 1000)
```

### Building the native dependencies

The two DLLs (`uhdr.dll`, `jpeg62.dll`) sit next to `uhdr_ctypes.py` and are loaded at import time. They are built from the `libultrahdr/` and `libjpeg-turbo/` source trees (cloned separately; not tracked in this repo) using the helper batch files. **Both batch files call `vcvars64.bat` from `Visual Studio\18\Community` and use NASM at `C:\Program Files\NASM`; install paths are hard-coded to `C:\msvcinstalls`.**

```cmd
_build_jpegturbo.bat        :: builds libjpeg-turbo, installs to C:\msvcinstalls
_build_libultrahdr.bat      :: builds libultrahdr against C:\msvcinstalls
```

`libultrahdr` is used from **upstream google/libultrahdr with no local patches**. It must be configured with `-DUHDR_WRITE_XMP=ON` so the output carries the legacy `hdrgm:` XMP for older viewer compatibility. The batch file already sets this. After rebuilding, copy `libultrahdr/build/uhdr.dll` and `C:\msvcinstalls\bin\jpeg62.dll` next to `uhdr_ctypes.py` (or run `_decode_check.bat` / `_smoke.bat` which stage them automatically).

### Diagnostics / inspection helpers

Each underscore-prefixed `.py` file is a one-off diagnostic script, not part of the production pipeline:

```bash
uv run python _inspect_jpg.py  <file.uhdr.jpg>   # JPEG marker map (one image)
uv run python _inspect_full.py <file.uhdr.jpg>   # walks BOTH primary + secondary JPEGs;
                                                 #   decodes ISO 21496-1 gain map metadata
uv run python _dump_meta.py    <file.uhdr.jpg>   # XMP packets + raw ISO 21496-1 hex
uv run python _dump_icc.py     <file.uhdr.jpg>   # extracts ICC profiles from primary + gain map
_smoke.bat                                       # dumpbin /dependents + ctypes load test
_decode_check.bat                                # runs ultrahdr_app.exe -m 1 on a known file
```

There is no test suite.

## Architecture

### Pipeline overview (`convert.py`)

```
input.tif  (uint16 BT.2020 PQ, full range)
  |
  v
color.bt2020_pq_to_p3_pq_uint16()
  (ST.2084 EOTF -> BT.2020 -> ACEScg (Bradford CAT)
   -> ACES 1.3 RGC -> ACEScg -> Display P3 (Bradford CAT)
   -> clip residuals -> ST.2084 inverse EOTF)
  |
  v  p3_pq_16  (uint16 Display P3 PQ)
  |
  v
pack_rgba1010102()  ->  packed_p3_hdr  (uint32 H x W)
  |
  +---> App-0 encode  (HDR-only, UHDR_CG_DISPLAY_P3 / UHDR_CT_PQ):
  |       uhdr_enc_set_raw_image(packed_p3_hdr, HDR_IMG)
  |       uhdr_encode()
  |       -> UHDR JPEG bytes
  |       -> extract_primary_jpeg()  ->  sdr_jpeg  (Display P3 SDR base, no re-encode)
  |
  +---> API-3 encode  (raw HDR + compressed SDR):
          uhdr_enc_set_raw_image(packed_p3_hdr, DISPLAY_P3, PQ, HDR_IMG)
          uhdr_enc_set_compressed_image(sdr_jpeg, DISPLAY_P3, SRGB, SDR_IMG)
          uhdr_enc_set_using_multi_channel_gainmap(1)
          uhdr_encode()
          ->  output.uhdr.jpg
```

`extract_primary_jpeg` searches for `FF D9 FF D8` — the unambiguous boundary between the
primary and gain-map JPEGs in the UHDR container. JPEG scan data byte-stuffs `0xFF` as
`FF 00`, so `FF D9` only appears as the EOI marker and never inside scan data.

### Module responsibilities

- **`convert.py`** — argparse + I/O, `pack_rgba1010102` / `_compressed_image` helpers, App-0 and API-3 encoder calls, `extract_primary_jpeg`.
- **`color.py`** — `bt2020_pq_to_p3_pq_uint16`: BT.2020 PQ → Display P3 PQ with ACES 1.3 RGC gamut compression. Uses `colour-science` for ST.2084 EOTF / gamut-matrix conversions and `PyOpenColorIO` for the RGC fixed-function transform. OCIO CPU processor is lazily built and cached in a module-level global.
- **`uhdr_ctypes.py`** — `ctypes` binding for `uhdr.dll`. Loads the DLL via `os.add_dll_directory`, defines `UhdrRawImage` / `UhdrCompressedImage` / `UhdrErrorInfo` matching `ultrahdr_api.h`, asserts `sizeof(UhdrRawImage) == 64` to catch ABI drift, and exposes both the raw-image (`uhdr_enc_set_raw_image`) and compressed-image (`uhdr_enc_set_compressed_image`) encoder paths.

### Color-pipeline invariants worth knowing

- **Input must be `uint16` RGB with `PhotometricInterpretation == 2`** — `load_pq_tiff` enforces this; YCbCr or floating-point TIFFs are rejected.
- **The RGBA1010102 pack order is R(0..9), G(10..19), B(20..29), A(30..31)** — per `ultrahdr_api.h`. Don't reorder; libultrahdr will silently produce wrong colors.
- **Both HDR and SDR renditions are Display P3** — the re-gamut step maps BT.2020 into P3 before either encoder pass, so the output has `use_base_cg=true` (iOS Photos / Lightroom-friendly).
- **The App-0 primary JPEG is used as-is in API-3** — when the compressed SDR input codec and output codec are both JPEG, libultrahdr uses the JPEG directly as the primary image without re-encoding, avoiding double-encode quality loss.
- **The output buffer from `uhdr_get_encoded_stream` is owned by the encoder** — we `string_at` it to Python `bytes` *before* calling `uhdr_release_encoder`. The `_ = packed_p3_hdr, sdr_buf` line inside the `try` block in `_encode_api3` keeps both backing buffers alive until after the copy.

### Ultra HDR JPEG file structure (what the encoder produces)

An Ultra HDR JPEG is two concatenated JPEGs: primary (base SDR image) then secondary (gain map), tied together by MPF (APP2 `MPF\0`). Metadata that downstream readers care about:

- **APP1 XMP** on the primary with `hdrgm:` namespace (legacy Adobe gain map format — written because `UHDR_WRITE_XMP=ON`).
- **APP2 `urn:iso:std:iso:ts:21496:-1`** on the secondary carrying the ISO 21496-1 gain map metadata (per-channel `gainMapMin/Max`, `gamma`, `baseOffset/altOffset`, `baseHdrHeadroom`, `alternateHdrHeadroom`, multichannel flag, `use_base_colour_space` flag).
- **APP2 `ICC_PROFILE`** segments (often split across multiple APP2s; `_dump_icc.py` reassembles them by chunk index/count).

`_inspect_full.py` is the authoritative reference for the ISO 21496-1 binary layout — its `parse_iso21496` mirrors `libultrahdr/lib/src/gainmapmetadata.cpp::encodeGainmapMetadata`. Consult it before changing anything that touches gain map metadata.
