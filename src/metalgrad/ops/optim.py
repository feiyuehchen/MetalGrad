"""Fused training-step kernels: AdamW, EMA, gradient L2 clipping.

These ops are NOT in the autograd graph — they consume gradients and
produce updated parameters / EMA buffers / scaled gradients. There is
no VJP to define. They are written here purely as fast Metal kernels
that the caller invokes inside their training loop.

The user manages all state (param, grad, m, v buffers; alpha; step
counter). metalgrad just gives them the kernels.

What's here:

  adamw_step(param, grad, m, v, *,
             lr, beta1, beta2, eps, weight_decay, step)
        Returns (new_param, new_m, new_v). One pass per parameter
        tensor — saves ~5× the bandwidth vs a naked mx ops chain.

  ema_update(ema, param, alpha)
        Returns alpha * ema + (1 - alpha) * param. `mx.compile`-fused
        — same speed as a manual chain, but ergonomically named.

  clip_grad_norm(grads, max_norm, *, eps=1e-6)
        Returns (scale, total_norm) and applies the scale to each
        grad in-place. Single fused reduction across all grad tensors.
"""
from __future__ import annotations

from typing import Sequence

import mlx.core as mx


# ─── AdamW fused step ────────────────────────────────────────────────────────

@mx.compile
def _adamw_inner(param, grad, m, v,
                 lr, beta1, beta2, eps, weight_decay, m_bc, v_bc):
    """Single-pass AdamW. Returns (new_param, new_m, new_v).

    The bias-correction factors `m_bc = 1/(1 - beta1**step)` and
    `v_bc = 1/(1 - beta2**step)` are passed in pre-computed (host-side
    scalars) so the kernel has nothing but elementwise ops to do."""
    new_m = beta1 * m + (1.0 - beta1) * grad
    new_v = beta2 * v + (1.0 - beta2) * grad * grad
    m_hat = new_m * m_bc
    v_hat = new_v * v_bc
    new_param = param - lr * (m_hat / (mx.sqrt(v_hat) + eps) + weight_decay * param)
    return new_param, new_m, new_v


def adamw_step(param: mx.array, grad: mx.array, m: mx.array, v: mx.array,
               *, lr: float, beta1: float = 0.9, beta2: float = 0.999,
               eps: float = 1e-8, weight_decay: float = 0.0,
               step: int) -> tuple[mx.array, mx.array, mx.array]:
    """Fused AdamW parameter update.

    Args:
      param, grad, m, v:  same shape, FP32. m and v are the first and
                          second moment buffers from your optimizer
                          state — pass `mx.zeros_like(param)` at step 1.
      lr, beta1, beta2, eps, weight_decay: standard AdamW hyperparams.
      step: 1-indexed timestep, used for bias correction.

    Returns:
      (new_param, new_m, new_v). Caller is responsible for storing them
      back into their optimizer state dict.
    """
    if step <= 0:
        raise ValueError("step must be >= 1")
    m_bc = 1.0 / (1.0 - beta1 ** step)
    v_bc = 1.0 / (1.0 - beta2 ** step)
    return _adamw_inner(param, grad, m, v,
                        lr, beta1, beta2, eps, weight_decay, m_bc, v_bc)


# ─── EMA update ──────────────────────────────────────────────────────────────

@mx.compile
def _ema_inner(ema, param, alpha):
    return alpha * ema + (1.0 - alpha) * param


def ema_update(ema: mx.array, param: mx.array, alpha: float = 0.999) -> mx.array:
    """new_ema = alpha * ema + (1 - alpha) * param.

    Common in diffusion training to maintain a slow-moving copy of the
    model weights for inference. Caller stores the result.
    """
    return _ema_inner(ema, param, alpha)


# ─── Gradient L2 clipping ────────────────────────────────────────────────────

def clip_grad_norm(grads: Sequence[mx.array], max_norm: float,
                   *, eps: float = 1e-6) -> tuple[mx.array, list[mx.array]]:
    """Global L2 gradient clipping.

    Computes  total_norm = sqrt( sum_g sum(g²) )
              scale      = min(1, max_norm / (total_norm + eps))
    Returns (total_norm_before_clip, scaled_grads).

    Single fused reduction over all grad tensors. The caller assigns
    `scaled_grads` back into their training loop's grad list (or just
    feeds them straight to `adamw_step`).
    """
    if not grads:
        return mx.array(0.0), []

    # Sum of squares per tensor → sum across tensors. mx fuses each
    # local mx.sum(g * g) into a single reduction kernel and the
    # tensor-wise add at the end is a tiny scalar op.
    parts = [mx.sum(g * g) for g in grads]
    total_sq = parts[0]
    for p in parts[1:]:
        total_sq = total_sq + p
    total_norm = mx.sqrt(total_sq)
    scale = mx.minimum(1.0, max_norm / (total_norm + eps))
    scaled = [g * scale for g in grads]
    return total_norm, scaled


__all__ = ["adamw_step", "ema_update", "clip_grad_norm"]
