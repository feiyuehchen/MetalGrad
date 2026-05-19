"""Differentiable matmul.

Math:
  y = x @ w               where x: (..., M, K),  w: (K, N) -> (..., M, N)

VJP given upstream gradient gy w.r.t. y:
  gx = gy @ w.T
  gw = x.T @ gy   (summed over leading batch dims)

v0.0.1 forward = `mx.matmul`. This validates the framework end-to-end
on a trivial op; future versions can swap the forward for an
`mx.fast.metal_kernel` GEMM if it beats `mx.matmul` on representative
shapes. The VJP does not need to change.
"""
from __future__ import annotations

import mlx.core as mx

from metalgrad.differentiable import differentiable


@differentiable
def matmul(x: mx.array, w: mx.array) -> mx.array:
    return mx.matmul(x, w)


@matmul.vjp
def _matmul_vjp(primals, cotangent, output):
    x, w = primals
    gy = cotangent
    # gx: gy @ w.T   shape (..., M, K)
    gx = mx.matmul(gy, mx.swapaxes(w, -1, -2))
    # gw: x.T @ gy summed over leading batch dims  shape (K, N)
    # For a 2D x, this is straightforward. For higher-rank x we need to
    # contract over all leading dims.
    if x.ndim == 2:
        gw = mx.matmul(mx.swapaxes(x, -1, -2), gy)
    else:
        # Flatten leading dims into one batch axis, then contract.
        K = x.shape[-1]
        N = gy.shape[-1]
        x_flat = x.reshape(-1, K)
        gy_flat = gy.reshape(-1, N)
        gw = mx.matmul(mx.swapaxes(x_flat, -1, -2), gy_flat)
    return gx, gw


__all__ = ["matmul"]
