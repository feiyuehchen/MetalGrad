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
    // One SIMD per row. Striped float4 layout: thread `lane` holds the
    // float4 group at channels (i*32 + lane) * 4 + (0..3) for each iter
    // i in [0, N_ITERS_F4). For C=1024, N_ITERS_F4 = 1024/128 = 8.
    //
    // Float4 cuts the per-thread load/store instruction count by 4× —
    // each instruction moves 16 bytes instead of 4 — which frees up
    // SIMD issue slots and reduces compiler scheduling pressure. The
    // total bytes transferred stays the same.
    //
    // Requires C % 128 == 0. Host falls back to mx for other C.
    uint row = thread_position_in_grid.y;
    uint lane = thread_position_in_threadgroup.x;
    if (row >= N_ROWS) return;

    constexpr int N_ITERS_F4 = (int)C / 128;
    uint row_base_f4 = row * (uint)(C / 4);
    float eps = eps_arr[0];

    const device float4* x_v4 = (const device float4*)x;
    const device float4* w_v4 = (const device float4*)weight;
    device float4* y_v4 = (device float4*)y;

    float4 xs[N_ITERS_F4];
    float sq = 0.0f;
    #pragma clang loop unroll(full)
    for (int i = 0; i < N_ITERS_F4; ++i) {
        uint idx = (uint)(i * 32) + lane;
        float4 v = x_v4[row_base_f4 + idx];
        xs[i] = v;
        sq += dot(v, v);   // x²+y²+z²+w² → one SIMD-level op
    }
    float inv = rsqrt(simd_sum(sq) / float(C) + eps);

    #pragma clang loop unroll(full)
    for (int i = 0; i < N_ITERS_F4; ++i) {
        uint idx = (uint)(i * 32) + lane;
        float4 wv = w_v4[idx];
        y_v4[row_base_f4 + idx] = xs[i] * inv * wv;
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
    if C % 128 != 0 or x.ndim < 2:
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


@mx.compile
def _rms_norm_bwd_fused(x, weight, gy, eps_arr):
    """Closed-form backward, mx.compile-fused so the elementwise chain
    runs as one kernel sequence. Returns (gx, gw_contributions_per_elem)
    — the caller does the final batch-axis reduction for gw."""
    eps = eps_arr[0]
    C_inv = 1.0 / float(x.shape[-1])
    s = mx.mean(x * x, axis=-1, keepdims=True) + eps
    inv = mx.rsqrt(s)
    gy_w = gy * weight
    dot = mx.sum(gy_w * x, axis=-1, keepdims=True)
    gx = inv * gy_w - (inv * inv * inv) * x * dot * C_inv
    gw_contrib = gy * x * inv
    return gx, gw_contrib


@_rms_norm_inner.vjp
def _rms_norm_vjp(primals, cotangent, output):
    x, weight, eps = primals
    gy = cotangent
    eps_arr = mx.array([float(eps)], dtype=x.dtype)
    gx, gw_contrib = _rms_norm_bwd_fused(x, weight, gy, eps_arr)
    reduce_axes = tuple(range(x.ndim - 1))
    gw = mx.sum(gw_contrib, axis=reduce_axes) if reduce_axes else gw_contrib
    return gx, gw, None


def rms_norm(x: mx.array, weight: mx.array, eps: float = 1e-6) -> mx.array:
    return _rms_norm_inner(x, weight, eps)


__all__ = ["rms_norm"]
