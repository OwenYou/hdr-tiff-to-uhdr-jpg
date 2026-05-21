"""Linear-light downscale a uint16 BT.2020 PQ TIFF for resolution-limit
testing against iOS Photos. Decodes PQ -> linear nits, box/bilinear
resamples, re-encodes PQ.

Usage:
  uv run python _downscale_tiff.py <in.tif> <out.tif> <new_W>x<new_H>
"""
import sys
import numpy as np
import tifffile
import colour


def _bilinear_resize_linear(linear: np.ndarray, new_w: int, new_h: int) -> np.ndarray:
    """Vectorized bilinear downsample of a (H, W, 3) float32 linear array."""
    H, W, _ = linear.shape
    # Target sample positions in source coordinate space (centered).
    sx = (np.arange(new_w, dtype=np.float32) + 0.5) * (W / new_w) - 0.5
    sy = (np.arange(new_h, dtype=np.float32) + 0.5) * (H / new_h) - 0.5
    x0 = np.clip(np.floor(sx).astype(np.int32), 0, W - 1)
    x1 = np.clip(x0 + 1, 0, W - 1)
    y0 = np.clip(np.floor(sy).astype(np.int32), 0, H - 1)
    y1 = np.clip(y0 + 1, 0, H - 1)
    fx = (sx - x0).astype(np.float32)
    fy = (sy - y0).astype(np.float32)

    # Row-by-row to keep peak memory reasonable on 42MP -> 24MP downsamples.
    out = np.empty((new_h, new_w, 3), dtype=np.float32)
    for j in range(new_h):
        yy0, yy1, ff = y0[j], y1[j], fy[j]
        row_top = linear[yy0]      # (W, 3)
        row_bot = linear[yy1]
        a = row_top[x0] * (1.0 - fx[:, None]) + row_top[x1] * fx[:, None]
        b = row_bot[x0] * (1.0 - fx[:, None]) + row_bot[x1] * fx[:, None]
        out[j] = a * (1.0 - ff) + b * ff
    return out


def main(in_path: str, out_path: str, size: str) -> int:
    new_w, new_h = (int(s) for s in size.lower().split('x'))

    with tifffile.TiffFile(in_path) as tif:
        page = tif.pages[0]
        if int(page.photometric) != 2:
            raise ValueError(f"expected RGB photometric, got {int(page.photometric)}")
        arr = tif.asarray()
    if arr.dtype != np.uint16 or arr.ndim != 3 or arr.shape[2] != 3:
        raise ValueError(f"expected uint16 (H,W,3) TIFF, got {arr.dtype} {arr.shape}")

    H, W = arr.shape[:2]
    print(f"{in_path}: {W}x{H} uint16 PQ ({W*H/1e6:.2f} MP) -> "
          f"{new_w}x{new_h} ({new_w*new_h/1e6:.2f} MP)")

    pq = arr.astype(np.float32) * np.float32(1.0 / 65535.0)
    nits = np.asarray(colour.eotf(pq, function="ST 2084"), dtype=np.float32)

    resized = _bilinear_resize_linear(nits, new_w, new_h)

    pq_out = np.asarray(
        colour.eotf_inverse(np.clip(resized, 0.0, 10000.0), function="ST 2084"),
        dtype=np.float32,
    )
    pq_out = np.clip(pq_out, 0.0, 1.0)
    out_u16 = (pq_out * np.float32(65535.0) + np.float32(0.5)).astype(np.uint16)

    tifffile.imwrite(out_path, out_u16, photometric='rgb')
    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print(__doc__)
        sys.exit(2)
    sys.exit(main(sys.argv[1], sys.argv[2], sys.argv[3]))
