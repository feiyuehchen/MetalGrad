"""Conv2d — thin re-export of `mx.conv2d`. See matmul.py rationale.

Layout (NHWC):
  x: (N, H, W, C_in)
  w: (C_out, KH, KW, C_in / groups)
  b: (C_out,) or None
"""
from __future__ import annotations

import mlx.core as mx


def conv2d(x: mx.array, w: mx.array, b: mx.array | None = None,
           stride: int = 1, padding: int = 0,
           dilation: int = 1, groups: int = 1) -> mx.array:
    y = mx.conv2d(x, w, stride=stride, padding=padding,
                  dilation=dilation, groups=groups)
    return y + b if b is not None else y


__all__ = ["conv2d"]
