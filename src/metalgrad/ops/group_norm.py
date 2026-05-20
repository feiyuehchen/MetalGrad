"""GroupNorm — used in HuBERT / data2vec / WavLM feature extractors.

For x of shape (B, T, C), GroupNorm splits C into `num_groups` groups
and normalises each group's mean / var independently per (B, T)
position:

    reshape x to (B, T, G, C/G)
    mean_g, var_g = mean/var over the last axis
    y = (x - mean_g) / sqrt(var_g + eps) * weight + bias

`weight` and `bias` are per-channel — (C,) — so the affine step is
applied across all groups but with a single (C,) parameter pair.

mx does not ship a fused group_norm. Naive composition costs:
   read x      → mean intermediate (B, T, G, 1) materialised
   read x      → var intermediate
   read x      → normalised intermediate
   read+write  → final
Total: ≥3 reads + 1 write of x.

This fused kernel does one read of x into registers, two SIMD-level
reductions (mean + var), and one write of y. Same recipe as our
layer_norm but the reduction axis is C/G floats per group instead of
the full last axis. T_groups (= T * G total rows) replaces T as the
row dimension.

Requires C/G divisible by 32 so one SIMD covers one group's channels.
For other shapes we fall back to mx ops.
"""
from __future__ import annotations

import mlx.core as mx

from metalgrad.differentiable import differentiable


_GN_FWD_SRC = """
    // Each TG = one SIMD (32 threads).
    // One row of work = one (batch, time, group) triple.
    // Row index: row = thread_position_in_grid.y in [0, B*T*G).
    // Lane k handles channels {k, k+32, k+64, ...} within the group.
    //
    // Striped: lane k owns positions i*32 + k for i in [0, CPG/32).
    uint row = thread_position_in_grid.y;
    uint lane = thread_position_in_threadgroup.x;
    if (row >= N_ROWS) return;

    constexpr int CPG = (int)C_PER_GROUP;
    constexpr int N_ITERS = CPG / 32;
    uint group = row % (uint)G;
    uint base_x = row * (uint)CPG;
    uint base_param = group * (uint)CPG;        // (weight, bias) per channel
    float eps = float(eps_arr[0]);

    float xs[N_ITERS];
    float lane_sum = 0.0f;
    #pragma clang loop unroll(full)
    for (int i = 0; i < N_ITERS; ++i) {
        uint cidx = (uint)(i * 32) + lane;
        float v = float(x[base_x + cidx]);
        xs[i] = v;
        lane_sum += v;
    }
    float mean = simd_sum(lane_sum) / float(CPG);

    float lane_sq = 0.0f;
    #pragma clang loop unroll(full)
    for (int i = 0; i < N_ITERS; ++i) {
        float d = xs[i] - mean;
        lane_sq = fma(d, d, lane_sq);
    }
    float inv = rsqrt(simd_sum(lane_sq) / float(CPG) + eps);

    #pragma clang loop unroll(full)
    for (int i = 0; i < N_ITERS; ++i) {
        uint cidx = (uint)(i * 32) + lane;
        float w_ = float(weight[base_param + cidx]);
        float b_ = float(bias[base_param + cidx]);
        float result = (xs[i] - mean) * inv * w_ + b_;
        y[base_x + cidx] = T((result));
    }
"""

_gn_kernels: dict = {}


def _get_gn_kernel(C_per_group: int, G: int, dtype):
    key = (C_per_group, G, dtype)
    k = _gn_kernels.get(key)
    if k is None:
        k = mx.fast.metal_kernel(
            name=f"metalgrad_group_norm_fwd_CPG{C_per_group}_G{G}_{str(dtype).split('.')[-1]}",
            input_names=["x", "weight", "bias", "eps_arr"],
            output_names=["y"],
            source=_GN_FWD_SRC,
        )
        _gn_kernels[key] = k
    return k


def _group_norm_fast(x: mx.array, weight: mx.array, bias: mx.array,
                     num_groups: int, eps: float) -> mx.array:
    """Fused fast path. Requires (C/num_groups) divisible by 32."""
    orig_shape = x.shape
    C = orig_shape[-1]
    CPG = C // num_groups
    # Flatten (B, ..., C) into (B*spatial*G, CPG) where spatial is product
    # of all axes between the batch axis and the channel axis.
    n_rows = 1
    for d in orig_shape[:-1]:
        n_rows *= d
    # Each row covers CPG channels; total rows = n_rows * num_groups since
    # each (batch, spatial) row has G groups.
    n_groups_rows = n_rows * num_groups
    x_flat = x.reshape(n_groups_rows, CPG)
    eps_arr = mx.array([float(eps)], dtype=x.dtype)
    kernel = _get_gn_kernel(CPG, num_groups, x.dtype)
    (y_flat,) = kernel(
        inputs=[x_flat, weight, bias, eps_arr],
        template=[("C_PER_GROUP", CPG), ("G", num_groups),
                  ("N_ROWS", n_groups_rows), ("T", x.dtype)],
        grid=(32, n_groups_rows, 1),
        threadgroup=(32, 1, 1),
        output_shapes=[(n_groups_rows, CPG)],
        output_dtypes=[x.dtype],
    )
    return y_flat.reshape(orig_shape)


def _group_norm_mx(x: mx.array, weight: mx.array, bias: mx.array,
                   num_groups: int, eps: float) -> mx.array:
    """Pure mx fallback / reference for the VJP."""
    orig_shape = x.shape
    C = orig_shape[-1]
    new_shape = orig_shape[:-1] + (num_groups, C // num_groups)
    x_r = x.reshape(new_shape)
    mean = mx.mean(x_r, axis=-1, keepdims=True)
    var = mx.mean((x_r - mean) * (x_r - mean), axis=-1, keepdims=True)
    norm = (x_r - mean) * mx.rsqrt(var + eps)
    norm = norm.reshape(orig_shape)
    return norm * weight + bias


@differentiable
def _group_norm_inner(x, weight, bias, num_groups, eps):
    C = x.shape[-1]
    G = int(num_groups)
    CPG = C // G
    if CPG % 32 == 0 and x.ndim >= 2:
        return _group_norm_fast(x, weight, bias, G, float(eps))
    return _group_norm_mx(x, weight, bias, G, float(eps))


@_group_norm_inner.vjp
def _group_norm_vjp(primals, cotangent, output):
    x, weight, bias, num_groups, eps = primals
    gy = cotangent
    G = int(num_groups)
    e = float(eps)

    def _ref(xx, ww, bb):
        return _group_norm_mx(xx, ww, bb, G, e)

    _, (gx, gw, gb) = mx.vjp(_ref, [x, weight, bias], [gy])
    return gx, gw, gb, None, None


def group_norm(x: mx.array, weight: mx.array, bias: mx.array,
               num_groups: int, eps: float = 1e-5) -> mx.array:
    """GroupNorm. x is (..., C); weight and bias are (C,).
    `num_groups` must divide C.

    For the special case num_groups = C (per-channel normalisation =
    InstanceNorm), this reduces to LayerNorm over a single channel and
    the kernel still works.
    """
    if x.shape[-1] % num_groups != 0:
        raise ValueError(
            f"C={x.shape[-1]} not divisible by num_groups={num_groups}"
        )
    return _group_norm_inner(x, weight, bias, num_groups, eps)


__all__ = ["group_norm"]
