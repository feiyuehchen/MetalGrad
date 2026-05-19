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


_RMS_FWD_SRC = """
    // Each TG = one SIMD (32 threads), one row (B*T position).
    // Channels split across the 32 threads: each thread does C/32 channels.
    // Pass 1: sum x*x across the row (each thread sums its C/32 elements,
    //         then simd_sum reduces across the SIMD).
    // Pass 2: each thread writes y = x * inv * weight for its C/32 channels.
    //
    // `eps_arr` is a 1-element mx.array; mx.fast.metal_kernel only allows
    // int / bool / Dtype as template args so we pass float eps via a tiny
    // input buffer instead of templating it. That preserves bit-exact
    // numerics when eps is small (e.g. 1e-5).
    uint row = thread_position_in_grid.y;
    uint lane = thread_position_in_threadgroup.x;
    if (row >= N_ROWS) return;

    int C_PER_LANE = (int)C / 32;
    uint row_off = row * (uint)C + lane * (uint)C_PER_LANE;
    float eps = eps_arr[0];

    float sq = 0.0f;
    #pragma clang loop unroll(full)
    for (int i = 0; i < C_PER_LANE; ++i) {
        float v = x[row_off + (uint)i];
        sq = fma(v, v, sq);
    }
    float row_sq = simd_sum(sq);
    float inv = rsqrt(row_sq / float(C) + eps);

    #pragma clang loop unroll(full)
    for (int i = 0; i < C_PER_LANE; ++i) {
        uint cidx = lane * (uint)C_PER_LANE + (uint)i;
        float v = x[row_off + (uint)i];
        y[row_off + (uint)i] = v * inv * weight[cidx];
    }
"""

_rms_kernels: dict = {}


def _get_rms_kernel(C: int):
    k = _rms_kernels.get(C)
    if k is None:
        k = mx.fast.metal_kernel(
            name=f"metalgrad_rms_norm_fwd_C{C}",
            input_names=["x", "weight", "eps_arr"],
            output_names=["y"],
            source=_RMS_FWD_SRC,
        )
        _rms_kernels[C] = k
    return k


def _rms_forward(x: mx.array, weight: mx.array, eps: float) -> mx.array:
    """Fused Metal kernel forward: one TG per row, simd_sum reduction.

    Requires C divisible by 32 (one SIMD group per row, 32 lanes split
    the channel axis). For other C we fall back to the mx implementation.
    """
    C = x.shape[-1]
    if C % 32 != 0 or x.ndim < 2:
        return _rms_forward_mx(x, weight, eps)

    orig_shape = x.shape
    n_rows = 1
    for d in orig_shape[:-1]:
        n_rows *= d
    x_flat = x.reshape(n_rows, C)
    eps_arr = mx.array([float(eps)], dtype=x.dtype)

    kernel = _get_rms_kernel(C)
    (y_flat,) = kernel(
        inputs=[x_flat, weight, eps_arr],
        template=[("C", C), ("N_ROWS", n_rows)],
        grid=(32, n_rows, 1),
        threadgroup=(32, 1, 1),
        output_shapes=[(n_rows, C)],
        output_dtypes=[x.dtype],
    )
    return y_flat.reshape(orig_shape)


def _rms_forward_mx(x: mx.array, weight: mx.array, eps: float) -> mx.array:
    """Pure mx fallback for shapes the kernel does not support."""
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
