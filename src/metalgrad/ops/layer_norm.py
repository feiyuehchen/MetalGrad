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
    // One SIMD per row, striped channel layout (see rms_norm.py for the
    // coalescing rationale).
    //
    // Bandwidth: read x (8 MB) + write y (8 MB) = 16 MB.
    // ~110 μs theoretical at (4, 512, 1024) on 150 GB/s.
    uint row = thread_position_in_grid.y;
    uint lane = thread_position_in_threadgroup.x;
    if (row >= N_ROWS) return;

    constexpr int N_ITERS = (int)C / 32;
    uint row_base = row * (uint)C;
    float eps = eps_arr[0];

    // Load x once + accumulate lane sum.
    float xs[N_ITERS];
    float lane_sum = 0.0f;
    #pragma clang loop unroll(full)
    for (int i = 0; i < N_ITERS; ++i) {
        uint cidx = (uint)(i * 32) + lane;
        float v = x[row_base + cidx];
        xs[i] = v;
        lane_sum += v;
    }
    float mean = simd_sum(lane_sum) / float(C);

    // Sum of (x - mean)^2 from register.
    float lane_sq = 0.0f;
    #pragma clang loop unroll(full)
    for (int i = 0; i < N_ITERS; ++i) {
        float d = xs[i] - mean;
        lane_sq = fma(d, d, lane_sq);
    }
    float inv = rsqrt(simd_sum(lane_sq) / float(C) + eps);

    // Write y.
    #pragma clang loop unroll(full)
    for (int i = 0; i < N_ITERS; ++i) {
        uint cidx = (uint)(i * 32) + lane;
        y[row_base + cidx] = (xs[i] - mean) * inv * weight[cidx] + bias[cidx];
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


@mx.compile
def _layer_norm_bwd_fused(x, weight, gy, eps_arr):
    """Canonical LayerNorm backward, mx.compile-fused.

    Forward y = n_hat * w + b   where   n_hat = (x - mean) * inv
    Backward (Hinton 2016 / PyTorch reference):
        dL/dx = (1/C) * inv * (C * gn - sum(gn) - n_hat * sum(gn * n_hat))
        dL/dw = sum_batch( gy * n_hat )
        dL/db = sum_batch( gy )
    where  gn = gy * w.

    All ops are elementwise reductions over the last axis — well-suited
    to mx.compile fusion. Returns (gx, gw_contrib_per_elem,
    gb_contrib_per_elem); caller sums the contribs over batch axes."""
    eps = eps_arr[0]
    C = x.shape[-1]
    C_inv = 1.0 / float(C)
    mean = mx.mean(x, axis=-1, keepdims=True)
    var = mx.mean((x - mean) * (x - mean), axis=-1, keepdims=True)
    inv = mx.rsqrt(var + eps)
    n_hat = (x - mean) * inv

    gn = gy * weight
    sum_gn = mx.sum(gn, axis=-1, keepdims=True)
    sum_gn_nhat = mx.sum(gn * n_hat, axis=-1, keepdims=True)
    gx = C_inv * inv * (float(C) * gn - sum_gn - n_hat * sum_gn_nhat)

    gw_contrib = gy * n_hat
    gb_contrib = gy
    return gx, gw_contrib, gb_contrib


@_layer_norm_inner.vjp
def _layer_norm_vjp(primals, cotangent, output):
    x, weight, bias, eps = primals
    gy = cotangent
    eps_arr = mx.array([float(eps)], dtype=x.dtype)
    gx, gw_contrib, gb_contrib = _layer_norm_bwd_fused(x, weight, gy, eps_arr)
    reduce_axes = tuple(range(x.ndim - 1))
    if reduce_axes:
        gw = mx.sum(gw_contrib, axis=reduce_axes)
        gb = mx.sum(gb_contrib, axis=reduce_axes)
    else:
        gw = gw_contrib
        gb = gb_contrib
    return gx, gw, gb, None


def layer_norm(x: mx.array, weight: mx.array, bias: mx.array,
               eps: float = 1e-5) -> mx.array:
    return _layer_norm_inner(x, weight, bias, eps)


__all__ = ["layer_norm"]
