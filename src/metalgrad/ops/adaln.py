"""Adaptive LayerNorm — the DiT / FCDM modulation primitive.

For each token (DiT) or pixel (FCDM):

    y = norm_no_affine(x) * (1 + scale[batch, channel])
                          + shift[batch, channel]

`norm_no_affine` is LayerNorm without learnable γ/β — we get those
from `(scale, shift)` instead, conditioned on the diffusion timestep
or class label. `scale` and `shift` are produced once per batch by a
small MLP and broadcast across the spatial (T or H*W) axis.

mx composition has the chain materialise intermediate `(x - mean) /
sigma` and `(1 + scale[:, None, ...] * norm)` tensors — both full-
sized. The fused kernel below extends our layer_norm pattern with the
modulation, reading x once and writing y once.

Two layout conventions:
  - Sequence:  x (B, T, C), scale/shift (B, C). Modulation broadcasts over T.
  - 2-D image: x (B, H, W, C), scale/shift (B, C). Modulation broadcasts over (H, W).

The kernel handles both by treating x as (B, R, C) where R = product of
all spatial axes.

Backward: same canonical LayerNorm-with-affine formula as our
`layer_norm` op, except γ is the per-batch `(1 + scale)`. We re-use
mx.compile to fuse the elementwise chain.

API:
    y = adaln(x, scale, shift, eps=1e-5)
"""
from __future__ import annotations

import mlx.core as mx

from metalgrad.differentiable import differentiable


_ADALN_FWD_SRC = """
    // Each TG = one SIMD (32 threads), one row of x (length C).
    // x is logically (B, R, C); row index r in [0, B*R). batch = r / R.
    // scale, shift: (B, C). C must be divisible by 32.
    //
    // Striped float-load layout — see rms_norm/layer_norm for rationale.
    uint row = thread_position_in_grid.y;
    uint lane = thread_position_in_threadgroup.x;
    if (row >= N_ROWS) return;

    constexpr int N_ITERS = (int)C / 32;
    uint row_base = row * (uint)C;
    uint batch_idx = row / (uint)R;
    uint scale_base = batch_idx * (uint)C;
    float eps = eps_arr[0];

    // Load x into registers, sum.
    float xs[N_ITERS];
    float lane_sum = 0.0f;
    #pragma clang loop unroll(full)
    for (int i = 0; i < N_ITERS; ++i) {
        uint cidx = (uint)(i * 32) + lane;
        float v = float(x[row_base + cidx]);
        xs[i] = v;
        lane_sum += v;
    }
    float mean = simd_sum(lane_sum) / float(C);

    // Variance.
    float lane_sq = 0.0f;
    #pragma clang loop unroll(full)
    for (int i = 0; i < N_ITERS; ++i) {
        float d = xs[i] - mean;
        lane_sq = fma(d, d, lane_sq);
    }
    float inv = rsqrt(simd_sum(lane_sq) / float(C) + eps);

    // Write y = norm * (1 + scale[batch, c]) + shift[batch, c]. Cast to T
    // explicitly so BF16/FP16 outputs work (Metal won't implicitly
    // convert float -> bfloat16_t on store).
    #pragma clang loop unroll(full)
    for (int i = 0; i < N_ITERS; ++i) {
        uint cidx = (uint)(i * 32) + lane;
        float sc = float(scale[scale_base + cidx]);
        float sh = float(shift[scale_base + cidx]);
        float result = (xs[i] - mean) * inv * (1.0f + sc) + sh;
        y[row_base + cidx] = T(result);
    }
"""

_adaln_kernels: dict = {}


def _get_adaln_kernel(C: int, dtype):
    key = (C, dtype)
    k = _adaln_kernels.get(key)
    if k is None:
        k = mx.fast.metal_kernel(
            name=f"metalgrad_adaln_fwd_C{C}_{str(dtype).split('.')[-1]}",
            input_names=["x", "scale", "shift", "eps_arr"],
            output_names=["y"],
            source=_ADALN_FWD_SRC,
        )
        _adaln_kernels[key] = k
    return k


def _adaln_fast(x: mx.array, scale: mx.array, shift: mx.array,
                eps: float) -> mx.array:
    """Fused forward — requires C % 32 == 0 and x.shape[-1] == C."""
    orig_shape = x.shape
    B = scale.shape[0]
    C = orig_shape[-1]
    # All non-batch, non-channel axes are spatial; flatten them into R.
    R_total = 1
    for d in orig_shape[1:-1]:
        R_total *= d
    n_rows = B * R_total
    x_flat = x.reshape(n_rows, C)
    eps_arr = mx.array([float(eps)], dtype=x.dtype)
    kernel = _get_adaln_kernel(C, x.dtype)
    (y_flat,) = kernel(
        inputs=[x_flat, scale, shift, eps_arr],
        template=[("C", C), ("R", R_total), ("N_ROWS", n_rows), ("T", x.dtype)],
        grid=(32, n_rows, 1),
        threadgroup=(32, 1, 1),
        output_shapes=[(n_rows, C)],
        output_dtypes=[x.dtype],
    )
    return y_flat.reshape(orig_shape)


def _adaln_mx(x: mx.array, scale: mx.array, shift: mx.array,
              eps: float) -> mx.array:
    """Pure mx reference, also used in the VJP (mx.vjp on this chain)."""
    # Broadcast scale/shift from (B, C) to x's shape with mid axes unsqueezed.
    extra = x.ndim - 2          # number of spatial axes (T or H,W,...)
    target_shape = (scale.shape[0],) + (1,) * extra + (scale.shape[-1],)
    s = scale.reshape(target_shape)
    sh = shift.reshape(target_shape)
    mean = mx.mean(x, axis=-1, keepdims=True)
    var = mx.mean((x - mean) * (x - mean), axis=-1, keepdims=True)
    norm = (x - mean) * mx.rsqrt(var + eps)
    return norm * (1.0 + s) + sh


@differentiable
def _adaln_inner(x, scale, shift, eps):
    C = x.shape[-1]
    if C % 32 == 0 and x.ndim >= 2:
        return _adaln_fast(x, scale, shift, float(eps))
    return _adaln_mx(x, scale, shift, float(eps))


@_adaln_inner.vjp
def _adaln_vjp(primals, cotangent, output):
    x, scale, shift, eps = primals
    gy = cotangent
    e = float(eps)

    def _ref(xx, ss, sh):
        return _adaln_mx(xx, ss, sh, e)

    _, (gx, gscale, gshift) = mx.vjp(_ref, [x, scale, shift], [gy])
    return gx, gscale, gshift, None


def adaln(x: mx.array, scale: mx.array, shift: mx.array,
          eps: float = 1e-5) -> mx.array:
    """Adaptive LayerNorm: affineless normalise x, then scale + shift.

    Args:
      x:     (B, ..., C). Last axis is normalised.
      scale: (B, C). Broadcast across non-batch non-channel axes.
      shift: (B, C). Same broadcast.
      eps:   numerical safety on the variance.

    Returns: same shape as x.
    """
    return _adaln_inner(x, scale, shift, eps)


__all__ = ["adaln"]
