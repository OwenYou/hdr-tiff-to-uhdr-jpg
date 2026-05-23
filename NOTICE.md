# Third-Party Notices

## libultrahdr

Source: https://github.com/google/libultrahdr  
License: Apache License 2.0 (see `libultrahdr/LICENSE-APACHE`)  
Copyright: The Android Open Source Project

`libultrahdr/` is vendored in this repository with the following local modification.

### Modifications

**File:** `libultrahdr/lib/src/gainmapmath.cpp`  
**Functions changed:** `srgbInvOetf`, `srgbOetf`  
**Change:** Replaced the IEC 61966-2-1 piecewise sRGB transfer curve (γ 2.4 power +
linear segment for very dark values) with a pure power-2.2 function in both the
forward and inverse directions.

**Reason:** The Python pipeline encodes the raw RGBA8888 SDR input with a pure γ 2.2
OETF.  libultrahdr decodes the SDR image with the function registered under
`UHDR_CT_SRGB`, so that function must match the encoder's curve.  The IEC 61966-2-1
curve (γ 2.4 + linear segment) diverges from γ 2.2 in the shadow region — it decodes
the same encoded signal to a higher linear value — causing the computed gain map to
go below 1.0 in shadows and making HDR shadow reconstructions darker than the SDR
base image.  Aligning both sides to pure γ 2.2 eliminates this sign error.
