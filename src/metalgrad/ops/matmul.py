"""Matmul — thin re-export of `mx.matmul`.

We do not currently offer a faster forward than mx.matmul (which is
MPSGraph-tuned and uses Apple's simdgroup_matrix FP16 MMA hardware via
internal mixed precision). Wrapping it with @differentiable adds
backward-side overhead that we measured at ~2x — net loss for users.

When a real FP32-fast custom GEMM kernel lands in v0.0.x, the wrapper
goes back. Today the import-from-metalgrad story is consistency, not
speed, and we are not willing to ship a wrapper that is slower than the
underlying mx op.

Math:
    y = x @ w
"""
from __future__ import annotations

import mlx.core as mx


def matmul(x: mx.array, w: mx.array) -> mx.array:
    return mx.matmul(x, w)


__all__ = ["matmul"]
