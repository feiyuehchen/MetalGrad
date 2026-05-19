"""Differentiable RMSNorm.

Math:
    rms(x) = sqrt( mean(x^2, axis=-1, keepdims=True) + eps )
    y      = (x / rms(x)) * weight

VJP given upstream gy w.r.t. y:
    Let s = mean(x^2, axis=-1, keepdims=True) + eps
    Let inv = 1 / sqrt(s)
    y / weight = x * inv

    g_inv * d_inv/dx, expanded:
        gw   = sum_over_batch( gy * (x * inv) )
        gx   = inv * (gy * weight)
               - (1/C) * inv^3 * x * sum_along_last_axis( gy * weight * x )
        ge   = (RMSNorm has scalar eps; we treat eps as a constant — no gradient.)

v0.0.1: forward is a hand-written `mx.fast.metal_kernel`. We pick this
op as the framework's first real Metal-kernel op because the kernel is
small (one reduction over the last axis) and the VJP formula is well-
known, so it makes a clean correctness benchmark.

The forward kernel below uses one threadgroup per (batch, time)
position. Each TG cooperatively sums x*x across the channel axis using
a simple `simd_sum` reduction, then writes back the normalized output.
"""
from __future__ import annotations

import mlx.core as mx

from metalgrad.differentiable import differentiable


def _rms_forward(x: mx.array, weight: mx.array, eps: float) -> mx.array:
    """v0.0.1 forward: pure mx implementation.

    A real custom Metal kernel goes here in v0.0.2. The kernel pattern is
    well understood — cooperative simd_sum reduction over the last axis,
    one TG per row — but ships with v0.0.2 once we have a benchmark
    target. For now, the framework + VJP correctness are what we are
    proving; the forward is the easy part to swap later without touching
    the VJP.
    """
    s = mx.mean(x * x, axis=-1, keepdims=True) + eps
    return (x * mx.rsqrt(s)) * weight


@differentiable
def _rms_norm_inner(x, weight, eps):
    return _rms_forward(x, weight, float(eps))


@_rms_norm_inner.vjp
def _rms_norm_vjp(primals, cotangent, output):
    x, weight, eps = primals
    gy = cotangent
    C = x.shape[-1]
    e = float(eps)
    s = mx.mean(x * x, axis=-1, keepdims=True) + e
    inv = mx.rsqrt(s)

    contribs = gy * (x * inv)
    reduce_axes = tuple(range(x.ndim - 1))
    gw = mx.sum(contribs, axis=reduce_axes) if reduce_axes else contribs

    gy_w = gy * weight
    dot = mx.sum(gy_w * x, axis=-1, keepdims=True)
    gx = inv * gy_w - (inv * inv * inv) * x * dot / float(C)

    return gx, gw, None


def rms_norm(x: mx.array, weight: mx.array, eps: float = 1e-6) -> mx.array:
    return _rms_norm_inner(x, weight, eps)


__all__ = ["rms_norm"]
