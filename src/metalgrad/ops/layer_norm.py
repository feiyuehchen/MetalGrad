"""Differentiable LayerNorm (over the last axis).

Math:
    mean = mean(x, axis=-1, keepdims=True)
    var  = mean((x - mean)^2, axis=-1, keepdims=True)
    y    = (x - mean) / sqrt(var + eps) * weight + bias

VJP is the standard LayerNorm gradient. We delegate to `mx.vjp` on a
reference implementation written in mx ops, mirroring the conv1d/2d
pattern. v0.0.2 can swap the forward for a fused Metal kernel.
"""
from __future__ import annotations

import mlx.core as mx

from metalgrad.differentiable import differentiable


_LN_FWD_SRC = """
    // Each TG = one SIMD (32 threads), one row.
    // C must be divisible by 32. Each lane handles C/32 channels.
    // Two simd_sum reductions: one for mean, one for variance.
    uint row = thread_position_in_grid.y;
    uint lane = thread_position_in_threadgroup.x;
    if (row >= N_ROWS) return;

    int C_PER_LANE = (int)C / 32;
    uint row_off = row * (uint)C + lane * (uint)C_PER_LANE;
    float eps = eps_arr[0];

    // Pass 1: row sum -> mean.
    float lane_sum = 0.0f;
    #pragma clang loop unroll(full)
    for (int i = 0; i < C_PER_LANE; ++i) {
        lane_sum += x[row_off + (uint)i];
    }
    float mean = simd_sum(lane_sum) / float(C);

    // Pass 2: row sum of (x - mean)^2 -> variance.
    float lane_sq = 0.0f;
    #pragma clang loop unroll(full)
    for (int i = 0; i < C_PER_LANE; ++i) {
        float d = x[row_off + (uint)i] - mean;
        lane_sq = fma(d, d, lane_sq);
    }
    float inv = rsqrt(simd_sum(lane_sq) / float(C) + eps);

    // Pass 3: write y.
    #pragma clang loop unroll(full)
    for (int i = 0; i < C_PER_LANE; ++i) {
        uint cidx = lane * (uint)C_PER_LANE + (uint)i;
        float v = x[row_off + (uint)i];
        y[row_off + (uint)i] = (v - mean) * inv * weight[cidx] + bias[cidx];
    }
"""

_ln_kernels: dict = {}


def _get_ln_kernel(C: int):
    k = _ln_kernels.get(C)
    if k is None:
        k = mx.fast.metal_kernel(
            name=f"metalgrad_layer_norm_fwd_C{C}",
            input_names=["x", "weight", "bias", "eps_arr"],
            output_names=["y"],
            source=_LN_FWD_SRC,
        )
        _ln_kernels[C] = k
    return k


def _layer_norm_fast(x: mx.array, weight: mx.array, bias: mx.array,
                     eps: float) -> mx.array:
    """Fused fast forward — requires C % 32 == 0."""
    C = x.shape[-1]
    orig_shape = x.shape
    n_rows = 1
    for d in orig_shape[:-1]:
        n_rows *= d
    x_flat = x.reshape(n_rows, C)
    eps_arr = mx.array([float(eps)], dtype=x.dtype)
    kernel = _get_ln_kernel(C)
    (y_flat,) = kernel(
        inputs=[x_flat, weight, bias, eps_arr],
        template=[("C", C), ("N_ROWS", n_rows)],
        grid=(32, n_rows, 1),
        threadgroup=(32, 1, 1),
        output_shapes=[(n_rows, C)],
        output_dtypes=[x.dtype],
    )
    return y_flat.reshape(orig_shape)


def _layer_norm_mx(x: mx.array, weight: mx.array, bias: mx.array,
                   eps: float) -> mx.array:
    """Pure mx ops. Used as the autograd-traceable reference inside the VJP
    — mx.vjp through this gives correct (mx-equivalent) gradients."""
    mean = mx.mean(x, axis=-1, keepdims=True)
    var = mx.mean((x - mean) * (x - mean), axis=-1, keepdims=True)
    norm = (x - mean) * mx.rsqrt(var + eps)
    return norm * weight + bias


def _layer_norm_ref(x: mx.array, weight: mx.array, bias: mx.array,
                    eps: float) -> mx.array:
    """Forward dispatch: fast Metal kernel where supported, else mx."""
    C = x.shape[-1]
    if C % 32 == 0 and x.ndim >= 2:
        return _layer_norm_fast(x, weight, bias, float(eps))
    return _layer_norm_mx(x, weight, bias, eps)


@differentiable
def _layer_norm_inner(x, weight, bias, eps):
    return _layer_norm_ref(x, weight, bias, float(eps))


@_layer_norm_inner.vjp
def _layer_norm_vjp(primals, cotangent, output):
    x, weight, bias, eps = primals
    gy = cotangent
    e = float(eps)

    # Use the pure-mx reference inside the VJP. mx.vjp cannot backprop
    # through our custom Metal forward (CustomKernel has no built-in
    # vjp); the math is identical so this is correct.
    def _ref(xx, ww, bb):
        return _layer_norm_mx(xx, ww, bb, e)

    _, (gx, gw, gb) = mx.vjp(_ref, [x, weight, bias], [gy])
    return gx, gw, gb, None


def layer_norm(x: mx.array, weight: mx.array, bias: mx.array,
               eps: float = 1e-5) -> mx.array:
    return _layer_norm_inner(x, weight, bias, eps)


__all__ = ["layer_norm"]
