"""Mean-squared error loss.

  loss = mean((pred - target)^2)

Used everywhere: regression, diffusion training (denoising objective),
autoencoders. mx has no fused version; the elementwise chain
(`(pred - target) ** 2` → `mean`) materialises an intermediate
`(pred - target) ** 2` tensor.

`mx.compile` fuses the chain into a single kernel — empirically faster
than a hand-written `mx.fast.metal_kernel` for the same op (see the
README §"How the speedups work" for why).

API:
    loss = mse(pred, target)              # scalar
    loss = mse(pred, target, axis=None)   # axis=None: full mean
"""
from __future__ import annotations

import mlx.core as mx

from metalgrad.differentiable import differentiable


@mx.compile
def _mse_fused(pred, target):
    diff = pred - target
    return mx.mean(diff * diff)


@differentiable
def _mse_inner(pred, target):
    return _mse_fused(pred, target)


@_mse_inner.vjp
def _mse_vjp(primals, cotangent, output):
    pred, target = primals
    gy = cotangent
    n_elems = float(pred.size)
    # d/d_pred mean((pred - target)^2) = 2 * (pred - target) / N
    grad_pred = gy * (2.0 / n_elems) * (pred - target)
    grad_target = -grad_pred   # symmetric for the loss; many users
                               # detach target outside, but we provide
                               # it for completeness.
    return grad_pred, grad_target


def mse(pred: mx.array, target: mx.array) -> mx.array:
    """Mean-squared error: mean((pred - target)^2). Scalar output."""
    if pred.shape != target.shape:
        raise ValueError(f"mse: shape mismatch {pred.shape} vs {target.shape}")
    return _mse_inner(pred, target)


__all__ = ["mse"]
