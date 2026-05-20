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
    // One SIMD per row, striped scalar layout. C % 32 == 0 required.
    //
    // Accumulation is always in float internally, regardless of the
    // input dtype: reads from x are auto-promoted (half → float,
    // bfloat → float on M3+); writes to y are explicitly cast to T
    // (= x.dtype) so we work correctly on FP16 / BF16 / FP32.
    uint row = thread_position_in_grid.y;
    uint lane = thread_position_in_threadgroup.x;
    if (row >= N_ROWS) return;

    constexpr int N_ITERS = (int)C / 32;
    uint row_base = row * (uint)C;
    float eps = float(eps_arr[0]);

    float xs[N_ITERS];
    float sq = 0.0f;
    #pragma clang loop unroll(full)
    for (int i = 0; i < N_ITERS; ++i) {
        uint cidx = (uint)(i * 32) + lane;
        float v = float(x[row_base + cidx]);
        xs[i] = v;
        sq = fma(v, v, sq);
    }
    float inv = rsqrt(simd_sum(sq) / float(C) + eps);

    #pragma clang loop unroll(full)
    for (int i = 0; i < N_ITERS; ++i) {
        uint cidx = (uint)(i * 32) + lane;
        float result = xs[i] * inv * float(weight[cidx]);
        y[row_base + cidx] = T(result);
    }
"""

_rms_kernels: dict = {}


def _get_rms_kernel(C: int, dtype):
    key = (C, dtype)
    k = _rms_kernels.get(key)
    if k is None:
        k = mx.fast.metal_kernel(
            name=f"metalgrad_rms_norm_fwd_C{C}_{str(dtype).split('.')[-1]}",
            input_names=["x", "weight", "eps_arr"],
            output_names=["y"],
            source=_RMS_FWD_SRC,
        )
        _rms_kernels[key] = k
    return k


def _rms_forward(x: mx.array, weight: mx.array, eps: float) -> mx.array:
    """Fused Metal kernel forward: one TG per row, simd_sum reduction.

    Requires C % 32 == 0. The kernel works in float internally and
    casts to / from x.dtype at the boundaries (`T` template), so FP32 /
    FP16 / BF16 all produce correct results.
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

    kernel = _get_rms_kernel(C, x.dtype)
    (y_flat,) = kernel(
        inputs=[x_flat, weight, eps_arr],
        template=[("C", C), ("N_ROWS", n_rows), ("T", x.dtype)],
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
