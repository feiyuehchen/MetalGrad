"""GLU-variant activations: SwiGLU, GeGLU, SquaredReLU.

These ops show up in modern transformer FFNs (Llama, Mixtral use
SwiGLU; PaLM uses SwiGLU; SquaredReLU appears in Primer / DeepMind
work). They are pure elementwise but combine 2-3 ops that mx executes
as separate graph nodes. Even a single fused kernel saves a memory
pass.

v0.0.1 wraps mx ops with explicit VJP. v0.0.2 forwards stay mx-backed
— elementwise fusion in MLX's lazy graph already amortises most of
the benefit. A real win would require sticking them right onto the
matmul epilogue, which is matmul's job, not the activation's.

API:
    SwiGLU(a, b) = silu(a) * b
    GeGLU(a, b) = gelu(a) * b
    SquaredReLU(x) = max(x, 0) ** 2

The Llama-style FFN block is
    out = down_proj(SwiGLU(gate_proj(x), up_proj(x)))
— we provide the activation here; matmuls go through
`metalgrad.ops.matmul` separately.
"""
from __future__ import annotations

import mlx.core as mx

from metalgrad.differentiable import differentiable


# ─── SwiGLU ──────────────────────────────────────────────────────────────────

def _silu(x):
    return x * mx.sigmoid(x)


@differentiable
def _swiglu_inner(a, b):
    return _silu(a) * b


@_swiglu_inner.vjp
def _swiglu_vjp(primals, cotangent, output):
    a, b = primals
    gy = cotangent
    sig = mx.sigmoid(a)
    # d/da silu(a) = sigmoid(a) + a * sigmoid(a) * (1 - sigmoid(a))
    #              = sigmoid(a) * (1 + a * (1 - sigmoid(a)))
    d_silu = sig * (1 + a * (1 - sig))
    silu_a = sig * a
    ga = gy * b * d_silu
    gb = gy * silu_a
    return ga, gb


def swiglu(a: mx.array, b: mx.array) -> mx.array:
    """SwiGLU activation: silu(a) * b. Used between FFN matmuls in
    Llama, Mixtral, PaLM-style transformers."""
    return _swiglu_inner(a, b)


# ─── GeGLU ───────────────────────────────────────────────────────────────────

def _gelu(x):
    return 0.5 * x * (1 + mx.tanh(0.7978845608 * (x + 0.044715 * x * x * x)))


@differentiable
def _geglu_inner(a, b):
    return _gelu(a) * b


@_geglu_inner.vjp
def _geglu_vjp(primals, cotangent, output):
    a, b = primals
    gy = cotangent
    # Use mx.vjp on the reference, mirroring the conv-op pattern. GeGLU's
    # closed-form VJP through tanh is unwieldy; mx ops handle it.
    def _ref(aa, bb):
        return _gelu(aa) * bb
    _, (ga, gb) = mx.vjp(_ref, [a, b], [gy])
    return ga, gb


def geglu(a: mx.array, b: mx.array) -> mx.array:
    """GeGLU activation: gelu(a) * b."""
    return _geglu_inner(a, b)


# ─── SquaredReLU ─────────────────────────────────────────────────────────────

@differentiable
def _squared_relu_inner(x):
    r = mx.maximum(x, 0)
    return r * r


@_squared_relu_inner.vjp
def _squared_relu_vjp(primals, cotangent, output):
    # mx.custom_function: for single-arg ops primals is the bare array;
    # for multi-arg ops primals is a tuple. Handle both safely.
    x = primals if isinstance(primals, mx.array) else primals[0]
    gy = cotangent
    return gy * 2.0 * mx.maximum(x, 0)


def squared_relu(x: mx.array) -> mx.array:
    """Squared ReLU: max(x, 0) ** 2. Used in Primer-style transformers
    (DeepMind 2021)."""
    return _squared_relu_inner(x)


__all__ = ["swiglu", "geglu", "squared_relu"]
