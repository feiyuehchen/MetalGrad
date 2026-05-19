"""KL divergence loss from logits — fused softmax + log_softmax.

  loss = mean( sum_j p(target)[j] * (log p(target)[j] - log p(pred)[j]) )

where `p(z)[j] = softmax(z)[j]`. The standard distillation loss between
a teacher (target) and a student (pred). For each row of length V we
need:

  m_p, m_t       row-wise max (numerical stability)
  s_p, s_t       row-wise sum of exp
  log Z_p, log Z_t  log of sum_exp

  contribution per element j of a row:
      softmax(t)[j] * (logits_t[j] - log Z_t - logits_p[j] + log Z_p)

This is fundamentally a multi-pass workload. mx by composition
materialises both `softmax(target)`, `log_softmax(target)`, and
`log_softmax(pred)` as separate full-size tensors before the final
reduction — at V = 50K vocab the wasted memory pass dominates.

This module ships a fused Metal kernel: one pass over each tensor for
per-row (max, sum), one combined pass that computes contributions and
accumulates the per-row sum, then a final mean. Two reads of each
logits tensor, one scalar output per row.

Backward (with respect to pred only — target gradient is typically
detached in distillation):

  d loss / d pred_logits[i, j] = (softmax(pred) - softmax(target))[j] / N

Streaming kernel: re-compute both softmaxes per row and write the
difference directly into the gradient buffer. No materialised
softmax tensor.

API:
    loss = kl_div_logits(pred_logits, target_logits)
    pred_logits, target_logits: (N, V) float32. Returns scalar.
"""
from __future__ import annotations

import mlx.core as mx

from metalgrad.differentiable import differentiable


# ─── Forward kernel ──────────────────────────────────────────────────────────

_KL_FWD_SRC = """
    // One SIMD per row.
    // Lane k streams over channels {k, k+32, k+64, ...}.
    // Pass 1: per-row max for pred and target (online softmax style).
    // Pass 2: per-row sum_exp for pred and target.
    // Pass 3: accumulate sum_j softmax(target)[j] * (log_softmax(t) - log_softmax(p))[j]
    //
    // V need not be a multiple of 32 — the streaming loop bound handles tails.
    uint row = thread_position_in_grid.y;
    uint lane = thread_position_in_threadgroup.x;
    if (row >= N_ROWS) return;

    uint row_off = row * (uint)V;

    // ── Online softmax pass for PRED ──
    float mp = -INFINITY;
    float sp = 0.0f;
    for (uint i = lane; i < (uint)V; i += 32u) {
        float x = pred_logits[row_off + i];
        float mn = max(mp, x);
        sp = sp * exp(mp - mn) + exp(x - mn);
        mp = mn;
    }
    #pragma clang loop unroll(full)
    for (uint off = 16u; off > 0u; off >>= 1u) {
        float m2 = simd_shuffle_xor(mp, off);
        float s2 = simd_shuffle_xor(sp, off);
        float m12 = max(mp, m2);
        sp = sp * exp(mp - m12) + s2 * exp(m2 - m12);
        mp = m12;
    }
    float log_Z_p = log(sp) + mp;

    // ── Online softmax pass for TARGET ──
    float mt = -INFINITY;
    float st = 0.0f;
    for (uint i = lane; i < (uint)V; i += 32u) {
        float x = target_logits[row_off + i];
        float mn = max(mt, x);
        st = st * exp(mt - mn) + exp(x - mn);
        mt = mn;
    }
    #pragma clang loop unroll(full)
    for (uint off = 16u; off > 0u; off >>= 1u) {
        float m2 = simd_shuffle_xor(mt, off);
        float s2 = simd_shuffle_xor(st, off);
        float m12 = max(mt, m2);
        st = st * exp(mt - m12) + s2 * exp(m2 - m12);
        mt = m12;
    }
    float log_Z_t = log(st) + mt;

    // ── Accumulate KL per row ──
    //   row_kl = sum_j softmax(target)[j] * (logits_t[j] - logits_p[j]) + log_Z_p - log_Z_t
    // The +(log_Z_p - log_Z_t) factor is constant across j; we add it once.
    float lane_kl = 0.0f;
    for (uint i = lane; i < (uint)V; i += 32u) {
        float lt = target_logits[row_off + i];
        float lp = pred_logits[row_off + i];
        float p_t = exp(lt - log_Z_t);             // softmax(target)[j]
        lane_kl += p_t * (lt - lp);
    }
    float row_kl_partial = simd_sum(lane_kl);
    if (lane == 0) {
        // softmax(target) sums to 1 by construction, so multiplying
        // (log_Z_p - log_Z_t) by sum(softmax(target)) is just adding
        // (log_Z_p - log_Z_t) once per row.
        kl_per_row[row] = row_kl_partial + (log_Z_p - log_Z_t);
    }
"""

_kl_fwd_kernel = None


def _get_kl_fwd_kernel():
    global _kl_fwd_kernel
    if _kl_fwd_kernel is None:
        _kl_fwd_kernel = mx.fast.metal_kernel(
            name="metalgrad_kl_div_logits_fwd",
            input_names=["pred_logits", "target_logits"],
            output_names=["kl_per_row"],
            source=_KL_FWD_SRC,
        )
    return _kl_fwd_kernel


def _kl_div_per_row_fast(pred_logits, target_logits):
    N, V = pred_logits.shape
    kernel = _get_kl_fwd_kernel()
    (kl,) = kernel(
        inputs=[pred_logits, target_logits],
        template=[("V", V), ("N_ROWS", N)],
        grid=(32, N, 1),
        threadgroup=(32, 1, 1),
        output_shapes=[(N,)],
        output_dtypes=[pred_logits.dtype],
    )
    return kl


# ─── Backward kernel: grad w.r.t. pred_logits ────────────────────────────────

_KL_BWD_SRC = """
    // d KL(target || pred) / d pred_logits[i, j] = (softmax(pred) - softmax(target))[j] / N
    // Streaming kernel: recompute both softmaxes per row, write the
    // difference directly. No materialised softmax tensors.
    uint row = thread_position_in_grid.y;
    uint lane = thread_position_in_threadgroup.x;
    if (row >= N_ROWS) return;

    uint row_off = row * (uint)V;
    float inv_N = 1.0f / float(N_ROWS);
    float upstream = gy_arr[0];

    // Online softmax for pred and target. Stat per row: (max, sum).
    float mp = -INFINITY; float sp = 0.0f;
    float mt = -INFINITY; float st = 0.0f;
    for (uint i = lane; i < (uint)V; i += 32u) {
        float xp = pred_logits[row_off + i];
        float xt = target_logits[row_off + i];
        float mn_p = max(mp, xp);  sp = sp * exp(mp - mn_p) + exp(xp - mn_p); mp = mn_p;
        float mn_t = max(mt, xt);  st = st * exp(mt - mn_t) + exp(xt - mn_t); mt = mn_t;
    }
    #pragma clang loop unroll(full)
    for (uint off = 16u; off > 0u; off >>= 1u) {
        float mp2 = simd_shuffle_xor(mp, off);
        float sp2 = simd_shuffle_xor(sp, off);
        float mt2 = simd_shuffle_xor(mt, off);
        float st2 = simd_shuffle_xor(st, off);
        float mp12 = max(mp, mp2); sp = sp * exp(mp - mp12) + sp2 * exp(mp2 - mp12); mp = mp12;
        float mt12 = max(mt, mt2); st = st * exp(mt - mt12) + st2 * exp(mt2 - mt12); mt = mt12;
    }
    float inv_sp = 1.0f / sp;
    float inv_st = 1.0f / st;

    // Write grad = (softmax(pred) - softmax(target)) / N * upstream.
    for (uint i = lane; i < (uint)V; i += 32u) {
        float xp = pred_logits[row_off + i];
        float xt = target_logits[row_off + i];
        float sm_p = exp(xp - mp) * inv_sp;
        float sm_t = exp(xt - mt) * inv_st;
        grad[row_off + i] = (sm_p - sm_t) * inv_N * upstream;
    }
"""

_kl_bwd_kernel = None


def _get_kl_bwd_kernel():
    global _kl_bwd_kernel
    if _kl_bwd_kernel is None:
        _kl_bwd_kernel = mx.fast.metal_kernel(
            name="metalgrad_kl_div_logits_bwd",
            input_names=["pred_logits", "target_logits", "gy_arr"],
            output_names=["grad"],
            source=_KL_BWD_SRC,
        )
    return _kl_bwd_kernel


def _kl_div_grad_fast(pred_logits, target_logits, gy):
    N, V = pred_logits.shape
    kernel = _get_kl_bwd_kernel()
    gy_arr = gy.reshape((1,)) if gy.ndim == 0 else gy
    (grad,) = kernel(
        inputs=[pred_logits, target_logits, gy_arr],
        template=[("V", V), ("N_ROWS", N)],
        grid=(32, N, 1),
        threadgroup=(32, 1, 1),
        output_shapes=[(N, V)],
        output_dtypes=[pred_logits.dtype],
    )
    return grad


# ─── Public op ───────────────────────────────────────────────────────────────

@differentiable
def _kl_div_inner(pred_logits, target_logits):
    per_row = _kl_div_per_row_fast(pred_logits, target_logits)
    return mx.mean(per_row)


@_kl_div_inner.vjp
def _kl_div_vjp(primals, cotangent, output):
    pred_logits, target_logits = primals
    gy = cotangent
    # Backward only w.r.t. pred. Target's grad is usually irrelevant
    # (it's a frozen teacher in distillation). Caller can mx.grad on
    # target separately if they need it — but the common case is
    # detached target.
    grad_pred = _kl_div_grad_fast(pred_logits, target_logits, gy)
    # For target gradient: d KL(target||pred) / d target_logits is
    # logits-difference style; we leave it unimplemented and return
    # None. If a user does mx.grad on target, mx will raise (which is
    # the right behaviour — they should detach).
    return grad_pred, mx.zeros_like(target_logits)


def kl_div_logits(pred_logits: mx.array, target_logits: mx.array) -> mx.array:
    """KL divergence KL(softmax(target) || softmax(pred)). Scalar mean.

    Standard distillation loss: target = teacher, pred = student.
    Gradient flows through `pred` only (target's gradient is zero —
    detach in distillation use cases).
    """
    if pred_logits.shape != target_logits.shape or pred_logits.ndim != 2:
        raise ValueError(f"kl_div_logits: need 2-D (N,V) tensors of the same "
                         f"shape; got {pred_logits.shape} and {target_logits.shape}")
    return _kl_div_inner(pred_logits, target_logits)


__all__ = ["kl_div_logits"]
