"""Differentiable Conv2d.

v0.0.1: forward = `mx.conv2d`; VJP via `mx.vjp`. Same structure as conv1d.

Layout (MLX-native NHWC):
  x: (N, H, W, C_in)
  w: (C_out, KH, KW, C_in / groups)
  b: (C_out,) or None
"""
from __future__ import annotations

import mlx.core as mx

from metalgrad.differentiable import differentiable


@differentiable
def _conv2d_inner(x, w, b, stride, padding, dilation, groups):
    y = mx.conv2d(x, w, stride=int(stride), padding=int(padding),
                  dilation=int(dilation), groups=int(groups))
    return y + b


@_conv2d_inner.vjp
def _conv2d_vjp(primals, cotangent, output):
    x, w, b, stride, padding, dilation, groups = primals
    gy = cotangent
    s, p, d, g = int(stride), int(padding), int(dilation), int(groups)

    def _ref(xx, ww):
        return mx.conv2d(xx, ww, stride=s, padding=p, dilation=d, groups=g)

    _, (gx, gw) = mx.vjp(_ref, [x, w], [gy])
    gb = mx.sum(gy, axis=tuple(range(gy.ndim - 1)))
    return gx, gw, gb, None, None, None, None


def conv2d(x: mx.array, w: mx.array, b: mx.array,
           stride: int = 1, padding: int = 0,
           dilation: int = 1, groups: int = 1) -> mx.array:
    return _conv2d_inner(x, w, b, stride, padding, dilation, groups)


__all__ = ["conv2d"]
