"""Differentiable LayerNorm (over the last axis).

Math:
    mean = mean(x, axis=-1, keepdims=True)
    var  = mean((x - mean)^2, axis=-1, keepdims=True)
    y    = (x - mean) / sqrt(var + eps) * weight + bias

VJP is the standard LayerNorm gradient. We delegate to `mx.vjp` on a
reference implementation written in mx ops, mirroring the conv1d/2d
pattern. v0.0.2 can swap the forward for a fused Metal kernel.
"""
from __future__ import annotations

import mlx.core as mx

from metalgrad.differentiable import differentiable


def _layer_norm_ref(x: mx.array, weight: mx.array, bias: mx.array,
                    eps: float) -> mx.array:
    mean = mx.mean(x, axis=-1, keepdims=True)
    var = mx.mean((x - mean) * (x - mean), axis=-1, keepdims=True)
    norm = (x - mean) * mx.rsqrt(var + eps)
    return norm * weight + bias


@differentiable
def _layer_norm_inner(x, weight, bias, eps):
    return _layer_norm_ref(x, weight, bias, float(eps))


@_layer_norm_inner.vjp
def _layer_norm_vjp(primals, cotangent, output):
    x, weight, bias, eps = primals
    gy = cotangent
    e = float(eps)

    def _ref(xx, ww, bb):
        return _layer_norm_ref(xx, ww, bb, e)

    _, (gx, gw, gb) = mx.vjp(_ref, [x, weight, bias], [gy])
    return gx, gw, gb, None


def layer_norm(x: mx.array, weight: mx.array, bias: mx.array,
               eps: float = 1e-5) -> mx.array:
    return _layer_norm_inner(x, weight, bias, eps)


__all__ = ["layer_norm"]
