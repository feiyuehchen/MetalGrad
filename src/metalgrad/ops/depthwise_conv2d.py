"""Depthwise Conv2d — thin re-export of `mx.conv2d` with groups=C.

The K=7 ConvNeXt / FCDM depthwise was an attempted fast-kernel target,
but our naive implementation could not beat MPSGraph's depthwise path
(0.32-0.56× of mx on representative shapes). The wrapper has been
stripped so callers do not pay backward overhead while we work on a
proper tiled kernel.

Layout (NHWC):
  x: (N, H, W, C)
  w: (C, KH, KW, 1)
  b: (C,) or None
"""
from __future__ import annotations

import mlx.core as mx


def depthwise_conv2d(x: mx.array, w: mx.array, b: mx.array | None = None,
                     stride: int = 1, padding: int = 0,
                     dilation: int = 1) -> mx.array:
    C = x.shape[-1]
    y = mx.conv2d(x, w, stride=stride, padding=padding,
                  dilation=dilation, groups=C)
    return y + b if b is not None else y


__all__ = ["depthwise_conv2d"]
