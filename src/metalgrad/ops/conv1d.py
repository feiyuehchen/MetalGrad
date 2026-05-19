"""Conv1d — thin re-export of `mx.conv1d`.

No faster forward yet; wrapping costs ~2x on backward (measured on
matmul). Same policy as matmul / attention: re-export until we have a
genuine forward speedup.

Layout:
  x: (N, L, C_in)
  w: (C_out, K, C_in / groups)
  b: (C_out,) or None
"""
from __future__ import annotations

import mlx.core as mx


def conv1d(x: mx.array, w: mx.array, b: mx.array | None = None,
           stride: int = 1, padding: int = 0,
           dilation: int = 1, groups: int = 1) -> mx.array:
    y = mx.conv1d(x, w, stride=stride, padding=padding,
                  dilation=dilation, groups=groups)
    return y + b if b is not None else y


__all__ = ["conv1d"]
