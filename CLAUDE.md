# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this project is

A Python pipeline that converts a single-layer **PQ HDR TIFF** (16-bit RGB, BT.2020) into a **Google Ultra HDR JPEG** using a locally patched `libultrahdr` (vendored in `libultrahdr/`, see `NOTICE.md`). The default pipeline is **API-1**:

1. **Tone-map pass** — Python Reinhard tone map (`color.p3_pq_to_sdr_rgba8888`) converts the P3 PQ image to raw RGBA8888 SDR pixels.
2. **API-1 pass** — raw P3 PQ HDR + raw P3 sRGB SDR are both supplied to libultrahdr (`uhdr_enc_set_raw_image` for both); libultrahdr computes a multi-channel (RGB) gain map from unquantised pixels and encodes both layers. Both renditions are Display P3 → `use_base_cg=true`.

With `--use-api3` the pipeline instead runs two passes:

1. **App-0 pass** — the BT.2020→Display P3 re-gamuted image is fed to the HDR-only encoder; libultrahdr internally tone-maps it and returns a UHDR JPEG whose embedded primary JPEG is the Display P3 SDR base.
2. **API-3 pass** — the same P3 PQ image is supplied as the raw HDR intent and the primary JPEG extracted from step 1 is supplied as the compressed SDR intent (`uhdr_enc_set_compressed_image(SDR_IMG)`); libultrahdr computes a multi-channel (RGB) gain map and assembles the final UHDR JPEG.

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
#   --pipeline lut       color pipeline mode: lut (default, baked OCIO 3D LUT, fast)
#                        or parametric (per-pixel analytical OCIO + NumPy Reinhard, slow)
#   --use-api3           use App-0 + API-3 two-pass pipeline instead of API-1 (default)

# Launch the batch GUI (tkinter; wraps convert.py via subprocess)
uv run python gui.py
```

### Building the native dependencies

Pre-built runtime libraries for both platforms are committed to the repo and sit next to `uhdr_ctypes.py`: `uhdr.dll` + `jpeg62.dll` on Windows, `libuhdr.dylib` + `libjpeg.62.dylib` on macOS. No build step is required to run the pipeline on either platform.

`libultrahdr/` is vendored in this repo with a local patch (see `NOTICE.md`). It must be configured with `-DUHDR_WRITE_XMP=ON` so the output carries the legacy `hdrgm:` XMP for older viewer compatibility. All build scripts already set this flag.

Rebuild only when updating or patching the native source. `libultrahdr/` is tracked in this repo. `libjpeg-turbo/` is cloned separately and not tracked.

#### Windows (rebuild)

**Both batch files call `vcvars64.bat` from `Visual Studio\18\Community` and use NASM at `C:\Program Files\NASM`; install paths are hard-coded to `C:\msvcinstalls`. CMake ships with Visual Studio at `C:\Program Files\Microsoft Visual Studio\18\Community\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin`.**

```cmd
scripts\_build_jpegturbo.bat        :: builds libjpeg-turbo, installs to C:\msvcinstalls
scripts\_build_libultrahdr.bat      :: builds libultrahdr against C:\msvcinstalls
```

After rebuilding, copy `libultrahdr/build/uhdr.dll` and `C:\msvcinstalls\bin\jpeg62.dll` next to `uhdr_ctypes.py` (or run `scripts\_decode_check.bat` / `scripts\_smoke.bat` which stage them automatically).

#### macOS (rebuild)

Prerequisites: Xcode Command Line Tools (`xcode-select --install`), `cmake`, `git`. Ninja is used automatically when available (`brew install ninja`).

```bash
bash scripts/_build_jpegturbo_macos.sh     # builds libjpeg-turbo, installs to ~/uhdr-deps
bash scripts/_build_libultrahdr_macos.sh   # builds libultrahdr, fixes rpath, copies to project dir
```

The libultrahdr script runs `install_name_tool` to rewrite the embedded JPEG dependency to `@loader_path/libjpeg.62.dylib`, so both dylibs resolve each other by relative path without `DYLD_LIBRARY_PATH`. `libultrahdr/` is already in the project directory; the macOS script builds from it directly. `libjpeg-turbo/` must be cloned separately (the jpegturbo build script handles this).

### Diagnostics / inspection helpers

Diagnostic Python scripts live in `tools/`; batch helpers live in `scripts/`:

```bash
uv run python tools/_inspect_jpg.py  <file.uhdr.jpg>   # JPEG marker map (one image)
uv run python tools/_inspect_full.py <file.uhdr.jpg>   # walks BOTH primary + secondary JPEGs;
                                                       #   decodes ISO 21496-1 gain map metadata
uv run python tools/_dump_meta.py    <file.uhdr.jpg>   # XMP packets + raw ISO 21496-1 hex
uv run python tools/_dump_icc.py     <file.uhdr.jpg>   # extracts ICC profiles from primary + gain map
uv run python tools/_compare_icc.py  <a.jpg> <b.jpg>  # side-by-side ICC tag comparison
uv run python tools/_rewrite_icc_gamma.py <file.jpg> [out.jpg]  # rewrite ICC rTRC/gTRC/bTRC to pure γ 2.2 (para type 0)
uv run python tools/_downscale_tiff.py <in.tif> <out.tif> <WxH>  # linear-light PQ TIFF downscale
scripts\_smoke.bat                                     # dumpbin /dependents + ctypes load test
scripts\_decode_check.bat                              # runs ultrahdr_app.exe -m 1 on a known file
```

There is no test suite.

## Architecture

### Pipeline overview (`convert.py`)

```
input.tif  (uint16 BT.2020 PQ, full range)
  |
  v
color.bt2020_pq_to_p3_pq()
  (ST.2084 EOTF -> BT.2020 -> ACEScg (Bradford CAT)
   -> ACES 1.3 RGC -> ACEScg -> Display P3 (Bradford CAT)
   -> clip residuals -> ST.2084 inverse EOTF)
  baked into a 97³ OCIO 3D LUT (lut mode) or run analytically (parametric mode)
  |
  v  p3_pq_f32  (float32 Display P3 PQ, no uint16 intermediate)
  |
  v
pack_rgba1010102(p3_pq_f32)  ->  packed_p3_hdr  (uint32 H x W)
  |
  +---> [DEFAULT] API-1 encode  (raw HDR + raw SDR):
  |       color.p3_pq_to_sdr_rgba8888(p3_pq_f32)  ->  sdr_rgba8888 (RGBA8888, sRGB)
  |         (baked 65³ LUT in lut mode; explicit NumPy Reinhard in parametric mode)
  |       uhdr_enc_set_raw_image(packed_p3_hdr, DISPLAY_P3, PQ, HDR_IMG)
  |       uhdr_enc_set_raw_image(sdr_rgba8888,  DISPLAY_P3, SRGB, SDR_IMG)
  |       uhdr_enc_set_using_multi_channel_gainmap(1)
  |       uhdr_encode()
  |       ->  output.uhdr.jpg
  |
  +---> [--use-api3] App-0 encode  (HDR-only, UHDR_CG_DISPLAY_P3 / UHDR_CT_PQ):
  |       uhdr_enc_set_raw_image(packed_p3_hdr, HDR_IMG)
  |       uhdr_encode()
  |       -> UHDR JPEG bytes
  |       -> extract_primary_jpeg()  ->  sdr_jpeg  (Display P3 SDR base, no re-encode)
  |
  +---> [--use-api3] API-3 encode  (raw HDR + compressed SDR):
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
- **`gui.py`** — tkinter batch GUI. Spawns `convert.py` as a subprocess per file, runs up to 8 jobs in parallel via `ThreadPoolExecutor`, and streams per-file log output back to the main thread via a `queue.Queue`. Screenshot at `docs/GUI.png`.
- **`color.py`** — `bt2020_pq_to_p3_pq`: BT.2020 PQ → Display P3 PQ with ACES 1.3 RGC gamut compression; returns float32 directly (no uint16 roundtrip). In `lut` mode the full analytical pipeline is baked into a 97³ OCIO 3D LUT at startup (tetrahedral interpolation). `p3_pq_to_sdr_rgba8888`: Reinhard tone map from float32 P3 PQ to RGBA8888; in `lut` mode uses a baked 65³ LUT. Both paths controlled by `--pipeline {lut,parametric}`; LUT processors are lazily built and cached in module-level globals.
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

`tools/_inspect_full.py` is the authoritative reference for the ISO 21496-1 binary layout — its `parse_iso21496` mirrors `libultrahdr/lib/src/gainmapmetadata.cpp::encodeGainmapMetadata`. Consult it before changing anything that touches gain map metadata.
