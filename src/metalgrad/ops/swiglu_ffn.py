"""Llama-style SwiGLU FFN block — fused gate+up matmul.

The Llama FFN computes
    h = silu(x @ W_gate) * (x @ W_up)
    y = h @ W_down

Standard optimisation (used in vLLM, MLX-LM, llama.cpp):

  Stack `W_gate` and `W_up` along the output axis into a single weight
  `W_gate_up` of shape (D_in, 2 * D_ff), then compute one matmul:

      gate_up = x @ W_gate_up                  # (..., 2 * D_ff)
      gate, up = split(gate_up, axis=-1)
      h = silu(gate) * up
      y = h @ W_down

This drops one matmul dispatch per call. The arithmetic cost is the
same, but at small batch / sequence sizes per-call overhead dominates,
so the fused form wins materially. At large batch sizes the gain is
smaller (~5-10%) but never a regression.

This module exposes both forms:

  swiglu_ffn(x, W_gate_up, W_down, ...)         # fused, fast path
  swiglu_ffn_unfused(x, W_gate, W_up, W_down, ...)  # reference, for tests

mx.compile fuses the silu(gate)*up tail, and mx.grad handles backward
through both matmuls and the activation — no @differentiable wrapper
needed since all components are already autograd-safe.
"""
from __future__ import annotations

import mlx.core as mx


@mx.compile
def _silu_mul(gate, up):
    return (gate * mx.sigmoid(gate)) * up


def stack_gate_up(W_gate: mx.array, W_up: mx.array) -> mx.array:
    """One-time helper to stack gate + up weights for the fused path.

    Both inputs are (D_in, D_ff). Returns (D_in, 2 * D_ff).
    Apply this once at model init, NOT inside the FFN call.
    """
    return mx.concatenate([W_gate, W_up], axis=-1)


def swiglu_ffn(x: mx.array, W_gate_up: mx.array, W_down: mx.array,
               b_gate_up: mx.array | None = None,
               b_down: mx.array | None = None) -> mx.array:
    """Llama-style SwiGLU FFN with fused gate+up matmul.

    Args:
      x:          (..., D_in)
      W_gate_up:  (D_in, 2 * D_ff)  — use `stack_gate_up()` to build.
      W_down:     (D_ff, D_in)
      b_gate_up:  optional (2 * D_ff,) bias.
      b_down:     optional (D_in,) bias.

    Returns:  (..., D_in)
    """
    gate_up = x @ W_gate_up
    if b_gate_up is not None:
        gate_up = gate_up + b_gate_up
    D_ff = gate_up.shape[-1] // 2
    gate = gate_up[..., :D_ff]
    up = gate_up[..., D_ff:]
    h = _silu_mul(gate, up)
    y = h @ W_down
    if b_down is not None:
        y = y + b_down
    return y


def swiglu_ffn_unfused(x: mx.array, W_gate: mx.array, W_up: mx.array,
                       W_down: mx.array,
                       b_gate: mx.array | None = None,
                       b_up: mx.array | None = None,
                       b_down: mx.array | None = None) -> mx.array:
    """Reference SwiGLU FFN: separate gate and up matmuls."""
    gate = x @ W_gate
    if b_gate is not None:
        gate = gate + b_gate
    up = x @ W_up
    if b_up is not None:
        up = up + b_up
    h = _silu_mul(gate, up)
    y = h @ W_down
    if b_down is not None:
        y = y + b_down
    return y


__all__ = ["swiglu_ffn", "swiglu_ffn_unfused", "stack_gate_up"]
