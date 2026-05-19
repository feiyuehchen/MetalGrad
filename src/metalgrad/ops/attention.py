"""Attention — thin re-export of `mx.fast.scaled_dot_product_attention`.

mx.fast.SDPA already has a registered VJP (verified bit-exact against
manual SDPA backward); wrapping it with @differentiable adds overhead
without any forward speedup. We do not ship the wrapped version.

When a FlashAttention-style kernel lands in v0.0.x, the wrapper comes
back so the same import path stays.

API:
    y = attention(q, k, v, scale=None, mask=None)
    q, k, v: (B, n_heads, T, head_dim)
"""
from __future__ import annotations

import mlx.core as mx


def attention(q: mx.array, k: mx.array, v: mx.array,
              *, scale: float | None = None,
              mask: mx.array | None = None) -> mx.array:
    if scale is None:
        scale = 1.0 / (q.shape[-1] ** 0.5)
    if mask is None:
        return mx.fast.scaled_dot_product_attention(q, k, v, scale=float(scale))
    return mx.fast.scaled_dot_product_attention(q, k, v, scale=float(scale), mask=mask)


__all__ = ["attention"]
