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


# Empirically, `mx.compile` on an elementwise op chain delivers ~2x
# over unfused mx and beats a hand-rolled `mx.fast.metal_kernel` (which
# pays extra wrapper overhead). For pure elementwise, MLX's own fusion
# is the right tool — we just need to wrap it in @differentiable so
# the user gets autograd plus the fused forward.


# ─── SwiGLU ──────────────────────────────────────────────────────────────────

@mx.compile
def _swiglu_fused(a, b):
    return (a * mx.sigmoid(a)) * b


@differentiable
def _swiglu_inner(a, b):
    return _swiglu_fused(a, b)


@_swiglu_inner.vjp
def _swiglu_vjp(primals, cotangent, output):
    a, b = primals
    gy = cotangent
    sig = mx.sigmoid(a)
    # d/da silu(a) = sigmoid(a) * (1 + a * (1 - sigmoid(a)))
    d_silu = sig * (1 + a * (1 - sig))
    silu_a = sig * a
    ga = gy * b * d_silu
    gb = gy * silu_a
    return ga, gb


def swiglu(a: mx.array, b: mx.array) -> mx.array:
    """SwiGLU activation: silu(a) * b. Used between FFN matmuls in
    Llama, Mixtral, PaLM-style transformers. ~2× over unfused mx."""
    return _swiglu_inner(a, b)


# ─── GeGLU ───────────────────────────────────────────────────────────────────

def _gelu(x):
    return 0.5 * x * (1 + mx.tanh(0.7978845608 * (x + 0.044715 * x * x * x)))


@mx.compile
def _geglu_fused(a, b):
    return _gelu(a) * b


@differentiable
def _geglu_inner(a, b):
    return _geglu_fused(a, b)


@_geglu_inner.vjp
def _geglu_vjp(primals, cotangent, output):
    a, b = primals
    gy = cotangent

    def _ref(aa, bb):
        return _gelu(aa) * bb

    _, (ga, gb) = mx.vjp(_ref, [a, b], [gy])
    return ga, gb


def geglu(a: mx.array, b: mx.array) -> mx.array:
    """GeGLU activation: gelu(a) * b."""
    return _geglu_inner(a, b)


# ─── SquaredReLU ─────────────────────────────────────────────────────────────

@mx.compile
def _squared_relu_fused(x):
    r = mx.maximum(x, 0)
    return r * r


@differentiable
def _squared_relu_inner(x):
    return _squared_relu_fused(x)


@_squared_relu_inner.vjp
def _squared_relu_vjp(primals, cotangent, output):
    x = primals if isinstance(primals, mx.array) else primals[0]
    gy = cotangent
    return gy * 2.0 * mx.maximum(x, 0)


def squared_relu(x: mx.array) -> mx.array:
    """Squared ReLU: max(x, 0) ** 2. Used in Primer-style transformers
    (DeepMind 2021)."""
    return _squared_relu_inner(x)


__all__ = ["swiglu", "geglu", "squared_relu"]
