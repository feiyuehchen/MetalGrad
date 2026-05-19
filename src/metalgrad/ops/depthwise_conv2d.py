"""Differentiable depthwise Conv2d.

Depthwise = groups == in_channels == out_channels. The K=7 ConvNeXt
depthwise is the canonical "we win 2x+ over mx" case once a real
Metal kernel lands in v0.0.2; v0.0.1 wraps `mx.conv2d` with groups=C
so the framework is usable for training immediately.

Layout (NHWC):
  x: (N, H, W, C)
  w: (C, KH, KW, 1)
  b: (C,) or None
"""
from __future__ import annotations

import mlx.core as mx

from metalgrad.differentiable import differentiable


@differentiable
def _depthwise_conv2d_inner(x, w, b, stride, padding, dilation):
    C = x.shape[-1]
    y = mx.conv2d(x, w, stride=int(stride), padding=int(padding),
                  dilation=int(dilation), groups=C)
    return y + b


@_depthwise_conv2d_inner.vjp
def _depthwise_conv2d_vjp(primals, cotangent, output):
    x, w, b, stride, padding, dilation = primals
    gy = cotangent
    s, p, d = int(stride), int(padding), int(dilation)
    C = x.shape[-1]

    def _ref(xx, ww):
        return mx.conv2d(xx, ww, stride=s, padding=p, dilation=d, groups=C)

    _, (gx, gw) = mx.vjp(_ref, [x, w], [gy])
    gb = mx.sum(gy, axis=tuple(range(gy.ndim - 1)))
    return gx, gw, gb, None, None, None


def depthwise_conv2d(x: mx.array, w: mx.array, b: mx.array,
                     stride: int = 1, padding: int = 0,
                     dilation: int = 1) -> mx.array:
    return _depthwise_conv2d_inner(x, w, b, stride, padding, dilation)


__all__ = ["depthwise_conv2d"]
