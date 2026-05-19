"""Per-op forward + backward time vs `mx.{op}` baseline.

Single representative shape per op (chosen to match common training
workloads). 5 measurement iters min, ~1 second total per op.

This is the v0.1 acceptance bench: every shipped op should be at least
≥1.5× over the mx baseline forward, and not slower on backward.
"""
from __future__ import annotations

import time
import numpy as np
import mlx.core as mx

from metalgrad.ops import (
    matmul, rms_norm, conv1d, conv2d, depthwise_conv2d, layer_norm, attention,
)

ITERS = 5
WARMUP = 5


def _arr(*shape, seed=0, scale=0.3):
    rng = np.random.default_rng(seed)
    return mx.array((rng.standard_normal(shape) * scale).astype(np.float32))


def tmin(fn) -> float:
    for _ in range(WARMUP):
        y = fn()
        mx.eval(y)
    samples = []
    for _ in range(ITERS):
        t0 = time.perf_counter()
        y = fn()
        mx.eval(y)
        samples.append((time.perf_counter() - t0) * 1000)
    return min(samples)


def tmin_grad(fn) -> float:
    """fn() returns a scalar loss. We time mx.grad(...) + eval."""
    for _ in range(WARMUP):
        gs = mx.grad(fn, argnums=tuple(range(fn.__code__.co_argcount)))(*fn.__defaults__) \
            if False else None
    # The above is awkward; we just inline-time below via the caller.
    raise NotImplementedError("use case_grad instead")


def case_fwd(name, ours_fn, mx_fn, *args):
    t_ours = tmin(lambda: ours_fn(*args))
    t_mx   = tmin(lambda: mx_fn(*args))
    speedup = t_mx / t_ours if t_ours > 0 else 0.0
    flag = "✓" if speedup >= 1.5 else ("≈" if speedup >= 0.95 else "✗")
    print(f"  {flag} {name:<22} ours {t_ours:>7.2f} ms  mx {t_mx:>7.2f} ms  speedup {speedup:.2f}×")
    return t_ours, t_mx


def case_grad(name, ours_fn, mx_fn, *args, argnums):
    """fn(*args) -> scalar. Time mx.grad with argnums."""
    grad_ours = mx.grad(ours_fn, argnums=argnums)
    grad_mx   = mx.grad(mx_fn,   argnums=argnums)
    t_ours = tmin(lambda: grad_ours(*args))
    t_mx   = tmin(lambda: grad_mx(*args))
    speedup = t_mx / t_ours if t_ours > 0 else 0.0
    flag = "✓" if speedup >= 0.95 else "✗"
    print(f"  {flag} {name:<22} ours {t_ours:>7.2f} ms  mx {t_mx:>7.2f} ms  speedup {speedup:.2f}×")
    return t_ours, t_mx


def main():
    print(f"Bench: metalgrad ops vs mx baseline (M3 Pro, {ITERS} iter min)")
    print()

    # ─── matmul ─────────────────────────────────────────────────────────────
    print("=== matmul (forward) ===  x: (256, 512), w: (512, 512)")
    x = _arr(256, 512, seed=1)
    w = _arr(512, 512, seed=2)
    case_fwd("matmul", matmul, mx.matmul, x, w)

    print("=== matmul (backward, sum of squares loss) ===")
    case_grad("matmul",
              lambda x, w: mx.sum(matmul(x, w) ** 2),
              lambda x, w: mx.sum(mx.matmul(x, w) ** 2),
              x, w, argnums=(0, 1))

    # ─── rms_norm ───────────────────────────────────────────────────────────
    print()
    print("=== rms_norm (forward) ===  x: (4, 512, 1024)")
    x = _arr(4, 512, 1024, seed=3)
    w = mx.array(np.ones(1024, dtype=np.float32))
    def mx_rms(x, w):
        s = mx.mean(x*x, axis=-1, keepdims=True) + 1e-6
        return x * mx.rsqrt(s) * w
    case_fwd("rms_norm", lambda x, w: rms_norm(x, w, 1e-6), mx_rms, x, w)

    print("=== rms_norm (backward) ===")
    case_grad("rms_norm",
              lambda x, w: mx.sum(rms_norm(x, w, 1e-6) ** 2),
              lambda x, w: mx.sum(mx_rms(x, w) ** 2),
              x, w, argnums=(0, 1))

    # ─── layer_norm ─────────────────────────────────────────────────────────
    print()
    print("=== layer_norm (forward) ===  x: (4, 512, 1024)")
    x = _arr(4, 512, 1024, seed=4)
    g = mx.array(np.ones(1024, dtype=np.float32))
    b = mx.zeros((1024,))
    def mx_ln(x, g, b):
        m = mx.mean(x, axis=-1, keepdims=True)
        v = mx.mean((x-m)**2, axis=-1, keepdims=True)
        return (x-m) * mx.rsqrt(v + 1e-5) * g + b
    case_fwd("layer_norm", lambda x, g, b: layer_norm(x, g, b, 1e-5), mx_ln, x, g, b)

    print("=== layer_norm (backward) ===")
    case_grad("layer_norm",
              lambda x, g, b: mx.sum(layer_norm(x, g, b, 1e-5) ** 2),
              lambda x, g, b: mx.sum(mx_ln(x, g, b) ** 2),
              x, g, b, argnums=(0, 1, 2))

    # ─── conv1d (LLVC-style depthwise) ──────────────────────────────────────
    print()
    print("=== conv1d (forward) ===  x: (1, 4094, 512), w: (512, 3, 1)  groups=512")
    x = _arr(1, 4094, 512, seed=5)
    w = _arr(512, 3, 1, seed=6, scale=0.1)
    b = mx.zeros((512,))
    def mx_conv1d_dw(x, w, b):
        return mx.conv1d(x, w, padding=1, groups=512) + b
    case_fwd("conv1d (depthwise)",
             lambda x, w, b: conv1d(x, w, b, padding=1, groups=512),
             mx_conv1d_dw, x, w, b)

    # ─── conv2d ─────────────────────────────────────────────────────────────
    print()
    print("=== conv2d (forward) ===  x: (1, 64, 64, 64), w: (128, 3, 3, 64)")
    x = _arr(1, 64, 64, 64, seed=7)
    w = _arr(128, 3, 3, 64, seed=8, scale=0.05)
    b = mx.zeros((128,))
    def mx_conv2d(x, w, b):
        return mx.conv2d(x, w, padding=1) + b
    case_fwd("conv2d",
             lambda x, w, b: conv2d(x, w, b, padding=1),
             mx_conv2d, x, w, b)

    # ─── depthwise_conv2d (FCDM/ConvNeXt K=7) ───────────────────────────────
    print()
    print("=== depthwise_conv2d K=7 (forward) ===  x: (1, 32, 32, 384)")
    C = 384
    x = _arr(1, 32, 32, C, seed=9)
    w = _arr(C, 7, 7, 1, seed=10, scale=0.05)
    b = mx.zeros((C,))
    def mx_dwconv2d(x, w, b):
        return mx.conv2d(x, w, padding=3, groups=C) + b
    case_fwd("depthwise_conv2d K=7",
             lambda x, w, b: depthwise_conv2d(x, w, b, padding=3),
             mx_dwconv2d, x, w, b)

    print("=== depthwise_conv2d K=7 (backward) ===")
    case_grad("depthwise_conv2d K=7",
              lambda x, w, b: mx.sum(depthwise_conv2d(x, w, b, padding=3) ** 2),
              lambda x, w, b: mx.sum(mx_dwconv2d(x, w, b) ** 2),
              x, w, b, argnums=(0, 1, 2))

    # ─── attention ──────────────────────────────────────────────────────────
    print()
    print("=== attention (forward) ===  B=2 H=8 T=256 D=64")
    B, H, T, D = 2, 8, 256, 64
    q = _arr(B, H, T, D, seed=11)
    k = _arr(B, H, T, D, seed=12)
    v = _arr(B, H, T, D, seed=13)
    scale = 1.0 / (D ** 0.5)
    def mx_attn(q, k, v):
        return mx.fast.scaled_dot_product_attention(q, k, v, scale=scale)
    case_fwd("attention",
             lambda q, k, v: attention(q, k, v, scale=scale),
             mx_attn, q, k, v)


if __name__ == "__main__":
    main()
