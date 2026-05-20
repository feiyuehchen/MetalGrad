"""Fused cross-entropy loss.

mx has no built-in fused cross_entropy. Naive composition through mx ops
materialises huge intermediates (a (N, V) one-hot or log-probs tensor),
costing 4-5× the bandwidth we actually need. For Llama-scale training
(N = batch*seqlen, V = 32K-128K vocab) this is the single most wasteful
op in the loss-side critical path.

This module ships a fused Metal kernel that computes per-row loss in
two passes over `logits` (max, then sum-exp) with no materialised
intermediates. Backward is also fused: a single pass writes the
gradient = (softmax - one_hot) / N directly into the gradient buffer.

API:
    loss = cross_entropy(logits, labels)           # scalar
    where logits: (N, V) float, labels: (N,) int32.

Bench at (N=1024, V=32000) on M3 Pro:
    mx ops chain (one_hot trick):  44 600 µs   (4% bw peak — abysmal)
    metalgrad fused forward:        ~2-3 ms    (target: 50% bw peak)
"""
from __future__ import annotations

import mlx.core as mx

from metalgrad.differentiable import differentiable


# ─── Forward kernel ──────────────────────────────────────────────────────────

_CE_FWD_SRC = """
    // One SIMD per row of logits, online softmax: each lane streams its
    // striped slice, maintaining (running_max, running_sum_exp) in a
    // numerically stable way:
    //
    //   m_new = max(m_old, x)
    //   s_new = s_old * exp(m_old - m_new) + exp(x - m_new)
    //
    // After the local stream, combine the 32 lanes' (m, s) pairs via a
    // tree reduction using simd_shuffle_xor. Each combine step:
    //
    //   m12 = max(m1, m2)
    //   s12 = s1 * exp(m1 - m12) + s2 * exp(m2 - m12)
    //
    // One pass over logits — half the bandwidth of the naive max-then-
    // sum-exp two-pass version.
    uint row = thread_position_in_grid.y;
    uint lane = thread_position_in_threadgroup.x;
    if (row >= N_ROWS) return;

    uint row_off = row * (uint)V;

    // Stream with online softmax stats. Internal computation in float
    // regardless of logits' dtype (FP16/BF16 inputs auto-promote).
    float m = -INFINITY;
    float s = 0.0f;
    for (uint i = lane; i < (uint)V; i += 32u) {
        float x = float(logits[row_off + i]);
        float m_new = max(m, x);
        s = s * exp(m - m_new) + exp(x - m_new);
        m = m_new;
    }

    // Cross-SIMD reduction via tree-of-shuffles.
    #pragma clang loop unroll(full)
    for (uint off = 16u; off > 0u; off >>= 1u) {
        float m2 = simd_shuffle_xor(m, off);
        float s2 = simd_shuffle_xor(s, off);
        float m12 = max(m, m2);
        s = s * exp(m - m12) + s2 * exp(m2 - m12);
        m = m12;
    }

    if (lane == 0) {
        float lse = log(s) + m;
        int label_idx = labels[row];
        float label_logit = float(logits[row_off + (uint)label_idx]);
        loss_per_row[row] = T(lse - label_logit);
    }
"""

_ce_kernels: dict = {}


def _get_ce_kernel(dtype):
    k = _ce_kernels.get(dtype)
    if k is None:
        k = mx.fast.metal_kernel(
            name=f"metalgrad_cross_entropy_fwd_{str(dtype).split('.')[-1]}",
            input_names=["logits", "labels"],
            output_names=["loss_per_row"],
            source=_CE_FWD_SRC,
        )
        _ce_kernels[dtype] = k
    return k


def _cross_entropy_per_row_fast(logits: mx.array, labels: mx.array) -> mx.array:
    """Returns (N,) per-row losses. Mean reduction happens at the caller."""
    N, V = logits.shape
    kernel = _get_ce_kernel(logits.dtype)
    (loss_per_row,) = kernel(
        inputs=[logits, labels],
        template=[("V", V), ("N_ROWS", N), ("T", logits.dtype)],
        grid=(32, N, 1),
        threadgroup=(32, 1, 1),
        output_shapes=[(N,)],
        output_dtypes=[logits.dtype],
    )
    return loss_per_row


def _cross_entropy_mx(logits: mx.array, labels: mx.array) -> mx.array:
    """Pure mx fallback / reference. Uses mx.take_along_axis to avoid the
    one-hot trick that mx code in the wild typically uses."""
    lse = mx.logsumexp(logits, axis=-1)                            # (N,)
    label_logits = mx.take_along_axis(logits, labels[:, None], axis=-1).squeeze(-1)
    return lse - label_logits                                      # per-row loss


# ─── Backward kernel ─────────────────────────────────────────────────────────
#
# d_loss/d_logits[i, j] = (softmax(logits[i])[j] - 1{j == label[i]}) / N
#
# Streaming kernel: per row, recompute (max, sum_exp), then write
# softmax - one_hot in a single pass. No materialised softmax tensor.

_CE_BWD_SRC = """
    uint row = thread_position_in_grid.y;
    uint lane = thread_position_in_threadgroup.x;
    if (row >= N_ROWS) return;

    uint row_off = row * (uint)V;
    int label_idx = labels[row];
    float inv_N = 1.0f / float(N_ROWS);
    float upstream = gy_arr[0];   // upstream gradient (mean loss is scalar; gy is scalar too)

    // Pass 1: max.
    float lane_max = -INFINITY;
    for (uint i = lane; i < (uint)V; i += 32u) {
        float v = float(logits[row_off + i]);
        lane_max = max(lane_max, v);
    }
    float row_max = simd_max(lane_max);

    // Pass 2: sum_exp.
    float lane_sum = 0.0f;
    for (uint i = lane; i < (uint)V; i += 32u) {
        float v = float(logits[row_off + i]);
        lane_sum += exp(v - row_max);
    }
    float row_sum_exp = simd_sum(lane_sum);
    float inv_sum = 1.0f / row_sum_exp;

    // Pass 3: write grad.
    for (uint i = lane; i < (uint)V; i += 32u) {
        float v = float(logits[row_off + i]);
        float sm = exp(v - row_max) * inv_sum;
        float indicator = ((int)i == label_idx) ? 1.0f : 0.0f;
        grad[row_off + i] = T((sm - indicator) * inv_N * upstream);
    }
"""

_ce_bwd_kernels: dict = {}


def _get_ce_bwd_kernel(dtype):
    k = _ce_bwd_kernels.get(dtype)
    if k is None:
        k = mx.fast.metal_kernel(
            name=f"metalgrad_cross_entropy_bwd_{str(dtype).split('.')[-1]}",
            input_names=["logits", "labels", "gy_arr"],
            output_names=["grad"],
            source=_CE_BWD_SRC,
        )
        _ce_bwd_kernels[dtype] = k
    return k


def _cross_entropy_grad_fast(logits: mx.array, labels: mx.array,
                              gy: mx.array) -> mx.array:
    N, V = logits.shape
    kernel = _get_ce_bwd_kernel(logits.dtype)
    gy_arr = gy.reshape((1,)) if gy.ndim == 0 else gy
    (grad,) = kernel(
        inputs=[logits, labels, gy_arr],
        template=[("V", V), ("N_ROWS", N), ("T", logits.dtype)],
        grid=(32, N, 1),
        threadgroup=(32, 1, 1),
        output_shapes=[(N, V)],
        output_dtypes=[logits.dtype],
    )
    return grad


# ─── Differentiable public op ────────────────────────────────────────────────

@differentiable
def _cross_entropy_inner(logits, labels):
    per_row = _cross_entropy_per_row_fast(logits, labels)
    return mx.mean(per_row)


@_cross_entropy_inner.vjp
def _cross_entropy_vjp(primals, cotangent, output):
    logits, labels = primals
    gy = cotangent
    gx = _cross_entropy_grad_fast(logits, labels, gy)
    return gx, None      # labels are int — no gradient


def cross_entropy(logits: mx.array, labels: mx.array) -> mx.array:
    """Fused softmax cross-entropy.

    Args:
      logits: (N, V) float32. Class scores; need not be normalised.
      labels: (N,) int32 — class index per row.

    Returns: scalar mean loss.
    """
    if logits.ndim != 2 or labels.ndim != 1:
        raise ValueError(f"expected logits (N,V) and labels (N,); got "
                         f"logits.shape={logits.shape}, labels.shape={labels.shape}")
    return _cross_entropy_inner(logits, labels)


__all__ = ["cross_entropy"]
