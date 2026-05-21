#!/usr/bin/env bash
# Build libjpeg-turbo (Release) and install to $HOME/uhdr-deps.
# Mirrors _build_jpegturbo.bat for macOS.
#
# Prerequisites: cmake, git, and a C compiler (Xcode Command Line Tools).
# Ninja is used when available; falls back to Unix Makefiles.
#
# Usage: bash scripts/_build_jpegturbo_macos.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
ROOT="$(cd "$PROJECT_DIR/.." && pwd)"
INSTALL="$HOME/uhdr-deps"

echo "=== ROOT=$ROOT  INSTALL=$INSTALL ==="

if [ ! -d "$ROOT/libjpeg-turbo" ]; then
  echo "=== cloning libjpeg-turbo ==="
  git -C "$ROOT" clone --depth 1 https://github.com/libjpeg-turbo/libjpeg-turbo.git
fi

cd "$ROOT/libjpeg-turbo"
rm -rf build

CMAKE_GENERATOR_FLAG=""
command -v ninja &>/dev/null && CMAKE_GENERATOR_FLAG="-G Ninja"

echo "=== configuring libjpeg-turbo (Release) ==="
cmake $CMAKE_GENERATOR_FLAG \
      -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_INSTALL_PREFIX="$INSTALL" \
      -DENABLE_SHARED=ON \
      -DENABLE_STATIC=OFF \
      -S . -B build

echo "=== building libjpeg-turbo ==="
cmake --build build --target install

echo "=== verifying install ==="
if [ -f "$INSTALL/lib/libjpeg.62.dylib" ]; then
  echo "OK: libjpeg.62.dylib present"
else
  echo "MISSING: $INSTALL/lib/libjpeg.62.dylib"
  ls "$INSTALL/lib/" || true
  exit 1
fi
ls "$INSTALL/lib/"
echo "=== DONE ==="
