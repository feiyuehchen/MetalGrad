"""Differentiable scaled-dot-product attention (SDPA).

  y = softmax( Q K^T / scale ) V

with optional additive mask. Shape convention follows
`mx.fast.scaled_dot_product_attention`:

  Q, K, V: (B, n_heads, T, head_dim)
  mask:    None | (B, n_heads, T, T) | broadcastable variant
  scale:   float, typically 1 / sqrt(head_dim)
  output:  (B, n_heads, T, head_dim)

v0.0.1 forward delegates to `mx.fast.scaled_dot_product_attention`,
which MLX has already wired into autograd. The wrapper exists so the
metalgrad.ops surface is complete and so v0.0.2 can swap the forward
for a FlashAttention-style Metal kernel without changing user code or
breaking gradient flow.

Gradient correctness: VJP recomputes through `mx.fast.SDPA` (matches
the forward), so any future change to the forward kernel only needs to
preserve mathematical equivalence with mx.fast.SDPA to keep gradients
correct.
"""
from __future__ import annotations

import mlx.core as mx

from metalgrad.differentiable import differentiable


@differentiable
def _attention_inner(q, k, v, scale, has_mask, mask):
    """All-positional inner op. `has_mask` is a bool flag so we can
    distinguish None from a real array — mx.custom_function does not
    pass through None cleanly across boundaries in all versions."""
    if has_mask:
        return mx.fast.scaled_dot_product_attention(q, k, v, scale=float(scale), mask=mask)
    return mx.fast.scaled_dot_product_attention(q, k, v, scale=float(scale))


@_attention_inner.vjp
def _attention_vjp(primals, cotangent, output):
    q, k, v, scale, has_mask, mask = primals
    gy = cotangent
    s = float(scale)
    use_mask = bool(has_mask)
    m_array = mask if use_mask else None

    def _ref(qq, kk, vv):
        if use_mask:
            return mx.fast.scaled_dot_product_attention(qq, kk, vv, scale=s, mask=m_array)
        return mx.fast.scaled_dot_product_attention(qq, kk, vv, scale=s)

    _, (gq, gk, gv) = mx.vjp(_ref, [q, k, v], [gy])
    # scale: Python float, no gradient
    # has_mask: bool, no gradient
    # mask: not part of the differentiable input set in any model I know
    return gq, gk, gv, None, None, None


def attention(q: mx.array, k: mx.array, v: mx.array,
              *, scale: float | None = None,
              mask: mx.array | None = None) -> mx.array:
    """Differentiable scaled-dot-product attention.

    Args:
      q, k, v: (B, n_heads, T, head_dim).
      scale:   1/sqrt(head_dim) by default.
      mask:    additive mask, optional.

    Returns: same shape as q.
    """
    if scale is None:
        head_dim = q.shape[-1]
        scale = 1.0 / (head_dim ** 0.5)
    has_mask = mask is not None
    # When mask is None we still need to pass *something* positional —
    # the inner op ignores it via has_mask.
    if not has_mask:
        mask = mx.zeros((1,), dtype=q.dtype)
    return _attention_inner(q, k, v, scale, has_mask, mask)


__all__ = ["attention"]
