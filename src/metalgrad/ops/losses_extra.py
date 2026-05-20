"""Extra loss functions used by speech / distillation papers.

  l1_loss(pred, target)             mean |pred - target|
  smooth_l1_loss(pred, target, β)   Huber-like — quadratic |·| ≤ β,
                                    linear outside. Used by data2vec /
                                    data2vec2 for distillation
                                    (default β = 0 == L2; β > 0 == Huber).
  cosine_loss(a, b, dim=-1)         1 - cos_sim(a, b), averaged over
                                    leading dims. Common in
                                    representation distillation
                                    (BYOL, sylber, etc.).

All three use mx.compile-fused forwards and explicit closed-form VJPs
where the math is straightforward. Same pattern as `mse` / `mse.py`.
"""
from __future__ import annotations

import mlx.core as mx

from metalgrad.differentiable import differentiable


# ─── L1 loss ─────────────────────────────────────────────────────────────────

@mx.compile
def _l1_fused(pred, target):
    return mx.mean(mx.abs(pred - target))


@differentiable
def _l1_inner(pred, target):
    return _l1_fused(pred, target)


@_l1_inner.vjp
def _l1_vjp(primals, cotangent, output):
    pred, target = primals
    gy = cotangent
    # d/d_pred mean(|p - t|) = sign(p - t) / N
    diff = pred - target
    sign = mx.where(diff > 0, mx.ones_like(diff),
                    mx.where(diff < 0, -mx.ones_like(diff), mx.zeros_like(diff)))
    grad_pred = gy * sign / float(pred.size)
    return grad_pred, -grad_pred


def l1_loss(pred: mx.array, target: mx.array) -> mx.array:
    if pred.shape != target.shape:
        raise ValueError(f"l1_loss shape mismatch {pred.shape} vs {target.shape}")
    return _l1_inner(pred, target)


# ─── Smooth L1 loss (Huber) ──────────────────────────────────────────────────

def _smooth_l1_inner_fn(pred, target, beta):
    diff = pred - target
    abs_diff = mx.abs(diff)
    quadratic = 0.5 * diff * diff / beta if beta > 0 else 0.5 * diff * diff
    linear = abs_diff - 0.5 * beta if beta > 0 else abs_diff
    return mx.mean(mx.where(abs_diff < beta, quadratic, linear) if beta > 0
                   else 0.5 * diff * diff)


def smooth_l1_loss(pred: mx.array, target: mx.array,
                   beta: float = 1.0) -> mx.array:
    """Huber loss.

      beta = 0:  reduces to 0.5 * (pred - target)^2 mean (L2). data2vec's
                 `loss_beta=0` default.
      beta > 0:  Huber: quadratic when |d| < beta, linear outside.

    Backward flows through the mx.where, so mx.grad handles it natively.
    No @differentiable wrapper (mx ops are already grad-safe).
    """
    if pred.shape != target.shape:
        raise ValueError(f"smooth_l1_loss shape mismatch {pred.shape} vs {target.shape}")
    return _smooth_l1_inner_fn(pred, target, float(beta))


# ─── Cosine similarity loss ──────────────────────────────────────────────────

@mx.compile
def _cosine_loss_fused(a, b):
    # Treat the last axis as the feature dim; flatten leading dims.
    a2 = mx.sum(a * b, axis=-1)
    na = mx.sqrt(mx.sum(a * a, axis=-1) + 1e-8)
    nb = mx.sqrt(mx.sum(b * b, axis=-1) + 1e-8)
    return 1.0 - mx.mean(a2 / (na * nb))


def cosine_loss(a: mx.array, b: mx.array) -> mx.array:
    """1 − mean( cos(a, b) ), computed over the last axis.

    Useful as a distillation target (sylber, BYOL, SimCLR-style).
    Backward via mx.compile + mx native autograd (no custom VJP).
    """
    if a.shape != b.shape:
        raise ValueError(f"cosine_loss shape mismatch {a.shape} vs {b.shape}")
    return _cosine_loss_fused(a, b)


# ─── L2 normalization ────────────────────────────────────────────────────────

@mx.compile
def _l2_norm_fused(x, eps):
    n = mx.sqrt(mx.sum(x * x, axis=-1, keepdims=True) + eps)
    return x / n


def l2_normalize(x: mx.array, eps: float = 1e-12) -> mx.array:
    """Project each row to unit L2 norm along the last axis. Common
    before cosine similarity / contrastive losses."""
    return _l2_norm_fused(x, float(eps))


__all__ = ["l1_loss", "smooth_l1_loss", "cosine_loss", "l2_normalize"]
