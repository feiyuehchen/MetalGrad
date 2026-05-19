"""Differentiable Conv1d.

v0.0.1 strategy: forward wraps `mx.conv1d`; VJP delegates to `mx.vjp`
on the same reference forward. This is mathematically identical to
plain `mx.grad(mx.conv1d)`, but it establishes the `@differentiable`
wrapper around a non-trivial op so v0.0.2 can swap the forward for a
fast Metal kernel without touching the VJP.

Args follow MLX convention:
  x: (N, L, C_in)
  w: (C_out, K, C_in / groups)
  b: (C_out,) or None
  stride / padding / dilation / groups: int
"""
from __future__ import annotations

import mlx.core as mx

from metalgrad.differentiable import differentiable


@differentiable
def _conv1d_inner(x: mx.array, w: mx.array, b: mx.array,
                  stride, padding, dilation, groups) -> mx.array:
    """All-positional inner op. mx.custom_function does not accept
    keyword arguments — when called with kwargs it forwards them to the
    VJP, which raises. We expose the kwarg-friendly API as `conv1d`
    below and route through this positional inner op."""
    y = mx.conv1d(x, w, stride=int(stride), padding=int(padding),
                  dilation=int(dilation), groups=int(groups))
    return y + b


@_conv1d_inner.vjp
def _conv1d_vjp(primals, cotangent, output):
    x, w, b, stride, padding, dilation, groups = primals
    gy = cotangent
    s, p, d, g = int(stride), int(padding), int(dilation), int(groups)

    def _ref(xx, ww):
        return mx.conv1d(xx, ww, stride=s, padding=p, dilation=d, groups=g)

    _, (gx, gw) = mx.vjp(_ref, [x, w], [gy])
    # gb: bias broadcasts over (N, L_out). Sum those out.
    gb = mx.sum(gy, axis=tuple(range(gy.ndim - 1)))
    return gx, gw, gb, None, None, None, None


def conv1d(x: mx.array, w: mx.array, b: mx.array,
           stride: int = 1, padding: int = 0,
           dilation: int = 1, groups: int = 1) -> mx.array:
    """Differentiable Conv1d. Kwarg-friendly wrapper around the
    positional `mx.custom_function`-registered inner op.
    """
    return _conv1d_inner(x, w, b, stride, padding, dilation, groups)


__all__ = ["conv1d"]
