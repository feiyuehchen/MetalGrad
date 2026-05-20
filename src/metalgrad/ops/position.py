"""Positional encoding utilities — sinusoidal absolute PE.

Most transformers in the speech-and-syllable family (sylber,
SyllableLM, data2vec, LLVC's transformer decoder) use sinusoidal
absolute PE for at least one of their attention stacks. mx.fast.rope
covers rotary; this module fills the absolute-PE gap.

API:
    pe = sinusoidal_pe(seq_len, d_model)        # (seq_len, d_model)
    x = x + sinusoidal_pe(x.shape[1], x.shape[-1])

The result has no gradient (it's a constant function of seq_len /
d_model), so no @differentiable wrapper is needed.
"""
from __future__ import annotations

import math

import mlx.core as mx


def sinusoidal_pe(seq_len: int, d_model: int,
                  base: float = 10000.0,
                  dtype: mx.Dtype = mx.float32) -> mx.array:
    """Build the standard sinusoidal positional encoding.

      PE[pos, 2i]   = sin(pos / base^(2i/d))
      PE[pos, 2i+1] = cos(pos / base^(2i/d))

    Returns shape (seq_len, d_model).
    """
    half = d_model // 2
    position = mx.arange(seq_len, dtype=dtype).reshape(-1, 1)            # (T, 1)
    div_term = mx.exp(mx.arange(0, half, dtype=dtype) *
                      (-math.log(base) / half))                          # (half,)
    angles = position * div_term[None, :]                                # (T, half)
    pe_even = mx.sin(angles)
    pe_odd = mx.cos(angles)
    # Interleave sin/cos columns: (T, half, 2) → (T, d_model)
    pe = mx.stack([pe_even, pe_odd], axis=-1).reshape(seq_len, d_model)
    return pe


__all__ = ["sinusoidal_pe"]
