#!/usr/bin/env bash
# Build libultrahdr (Release, shared) against libjpeg-turbo in $HOME/uhdr-deps,
# fix the JPEG dylib rpath with install_name_tool, and copy both dylibs next to
# uhdr_ctypes.py.  Mirrors _build_libultrahdr.bat for macOS.
#
# Run _build_jpegturbo_macos.sh first.
#
# Prerequisites: cmake, git, install_name_tool (ships with Xcode Command Line Tools).
#
# Usage: bash scripts/_build_libultrahdr_macos.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
ROOT="$PROJECT_DIR"   # libultrahdr is vendored inside the project directory
INSTALL="$HOME/uhdr-deps"

echo "=== ROOT=$ROOT  INSTALL=$INSTALL ==="

cd "$ROOT/libultrahdr"
rm -rf build

CMAKE_GENERATOR_FLAG=""
command -v ninja &>/dev/null && CMAKE_GENERATOR_FLAG="-G Ninja"

echo "=== configuring libultrahdr (Release, against $INSTALL) ==="
cmake $CMAKE_GENERATOR_FLAG \
      -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_PREFIX_PATH="$INSTALL" \
      -DBUILD_SHARED_LIBS=ON \
      -DUHDR_BUILD_EXAMPLES=ON \
      -DUHDR_BUILD_TESTS=OFF \
      -DUHDR_WRITE_XMP=ON \
      -S . -B build

echo "=== building libultrahdr ==="
cmake --build build

echo "=== build outputs ==="
ls build/*.dylib build/*.a 2>/dev/null || true

# ── Rpath fix ──────────────────────────────────────────────────────────────
# Change the embedded JPEG install name from the build-time absolute path to
# @loader_path/libjpeg.62.dylib so that the two dylibs can live side-by-side
# next to uhdr_ctypes.py without any DYLD_LIBRARY_PATH.
JPEG_LINKED=$(otool -L build/libuhdr.dylib | awk '/libjpeg/{print $1}' | head -1)
if [ -n "$JPEG_LINKED" ]; then
  echo "=== fixing JPEG dependency: $JPEG_LINKED -> @loader_path/libjpeg.62.dylib ==="
  install_name_tool -change "$JPEG_LINKED" "@loader_path/libjpeg.62.dylib" build/libuhdr.dylib
else
  echo "WARNING: no libjpeg dependency found in build/libuhdr.dylib — skipping rpath fix"
fi

# ── Copy artifacts ──────────────────────────────────────────────────────────
echo "=== copying artifacts to $PROJECT_DIR ==="
cp build/libuhdr.dylib "$PROJECT_DIR/"
cp "$INSTALL/lib/libjpeg.62.dylib" "$PROJECT_DIR/"

echo "=== verifying ==="
otool -L "$PROJECT_DIR/libuhdr.dylib"
echo "=== DONE ==="
echo "Copied libuhdr.dylib + libjpeg.62.dylib to $PROJECT_DIR"
