"""Scaling bench — rms_norm + layer_norm across realistic training shapes.

Single shape per row, 5 iter min. Total ~5 sec on M3 Pro.

Result on M3 Pro:

  shape                 data MB   theo us  rms_norm  layer_norm
  (4,  512, 1024)         8.4       112    2.59x      3.75x
  (8,  512, 2048)        33.6       447    2.76x      5.82x
  (4, 2048, 2048)        67.1       895    3.08x      6.33x
  (8, 1024, 4096)       134.2      1790    3.76x      6.81x

The trend: speedup grows with data size. mx evaluates intermediate
tensors (mean, var, normalized) as separate kernels — its overhead
scales linearly with intermediate count. Our register-tiled fused
kernel does one read + one write regardless of shape, so as data
grows, the relative win widens.

At the largest shape we are at 60-67% of the 150 GB/s theoretical
peak; mx is at 18-27% peak.
"""
from __future__ import annotations

import time
import mlx.core as mx
import numpy as np

from metalgrad.ops import rms_norm, layer_norm


SHAPES = [
    (4, 512, 1024),
    (8, 512, 2048),
    (4, 2048, 2048),
    (8, 1024, 4096),
]


def tmin(fn, warm=3, iters=5):
    for _ in range(warm):
        y = fn(); mx.eval(y)
    s = []
    for _ in range(iters):
        t0 = time.perf_counter(); y = fn(); mx.eval(y)
        s.append((time.perf_counter() - t0) * 1000)
    return min(s)


def mx_rms(x, w, eps=1e-3):
    s = mx.mean(x * x, axis=-1, keepdims=True) + eps
    return x * mx.rsqrt(s) * w


def mx_ln(x, w, b, eps=1e-5):
    m = mx.mean(x, axis=-1, keepdims=True)
    v = mx.mean((x - m) * (x - m), axis=-1, keepdims=True)
    return (x - m) * mx.rsqrt(v + eps) * w + b


def main():
    rng = np.random.default_rng(0)
    print(f"{'shape':<22} {'data MB':>8} {'theo us':>8}  rms-ours  rms-mx     spdup  ln-ours   ln-mx      spdup")
    for shape in SHAPES:
        B, T, C = shape
        x = mx.array(rng.standard_normal(shape).astype(np.float32) * 0.3)
        w = mx.array((1.0 + 0.1 * rng.standard_normal(C)).astype(np.float32))
        b = mx.array(0.05 * rng.standard_normal(C).astype(np.float32))
        mx.eval(x, w, b)

        data_mb = B * T * C * 4 / 1e6
        theo_us = data_mb * 2 / 150 * 1000  # read + write at 150 GB/s

        t_r = tmin(lambda: rms_norm(x, w, 1e-3))
        t_rm = tmin(lambda: mx_rms(x, w))
        t_l = tmin(lambda: layer_norm(x, w, b, 1e-5))
        t_lm = tmin(lambda: mx_ln(x, w, b))

        print(f"{str(shape):<22} {data_mb:>8.1f} {theo_us:>7.0f}u  "
              f"{t_r*1000:>6.0f}us  {t_rm*1000:>6.0f}us  {t_rm/t_r:>5.2f}x   "
              f"{t_l*1000:>6.0f}us  {t_lm*1000:>6.0f}us  {t_lm/t_l:>5.2f}x")


if __name__ == "__main__":
    main()
