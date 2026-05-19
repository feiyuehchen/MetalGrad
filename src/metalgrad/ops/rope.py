"""Rotary position embedding helpers — variants for long-context
extension and modern LLM architectures.

`mx.fast.rope` already ships a fused kernel (≈ 2.9× over manual mx
composition on B=4 H=32 T=512 D=128). It accepts an optional `freqs`
argument: a (dims/2,) array of angular frequencies. Every RoPE
"variant" in the literature is just a different way to compute those
frequencies — the rotation kernel itself is unchanged.

This module supplies the frequency-table builders plus thin wrappers
that call `mx.fast.rope(x, dims, freqs=..., base=None)` for each
variant. Backward flows through `mx.fast.rope` natively.

Variants implemented (all from public papers / model cards):

  - `rope_freqs_standard`      vanilla RoPE (Su et al. 2021)
  - `rope_freqs_linear_pi`     Position Interpolation (Chen et al. 2023)
  - `rope_freqs_ntk_aware`     NTK-aware base rescaling (bloc97 2023)
  - `rope_freqs_yarn`          YaRN (Peng et al. 2023)
  - `rope_freqs_llama3`        Llama 3.1 piecewise scaling (Meta 2024)

And matching thin convenience wrappers:

  rope_standard(x, dims=D)
  rope_linear_pi(x, dims=D, scale=4.0)
  rope_ntk_aware(x, dims=D, scale=4.0)
  rope_yarn(x, dims=D, original_max_pos=4096, scale=4.0, ...)
  rope_llama3(x, dims=D, factor=8.0, low_freq_factor=1.0,
              high_freq_factor=4.0, original_max_pos=8192)
"""
from __future__ import annotations

import math

import mlx.core as mx


# ─── Frequency tables ────────────────────────────────────────────────────────

def rope_freqs_standard(dims: int, base: float = 10000.0,
                        dtype=mx.float32) -> mx.array:
    """Vanilla RoPE: θ_i = base^(-2i / dims) for i in [0, dims/2)."""
    half = dims // 2
    return base ** (-mx.arange(0, half, dtype=dtype) / half)


def rope_freqs_linear_pi(dims: int, scale: float = 4.0,
                         base: float = 10000.0, dtype=mx.float32) -> mx.array:
    """Position Interpolation (Chen et al. 2023): divide every frequency
    by `scale` so positions m become m/scale before the rotation.

    Used to extend context length: with scale=4, the model's original
    max-position effectively covers 4× more tokens. Trains/fine-tunes
    needed to recover quality.
    """
    return rope_freqs_standard(dims, base, dtype) / float(scale)


def rope_freqs_ntk_aware(dims: int, scale: float = 4.0,
                         base: float = 10000.0, dtype=mx.float32) -> mx.array:
    """NTK-aware (bloc97 2023): rescale the *base* rather than the
    positions. Preserves high-frequency components while compressing
    low-frequency ones.

        base' = base * scale^(dims / (dims - 2))
    """
    half = dims // 2
    new_base = base * (float(scale) ** (dims / (dims - 2.0)))
    return new_base ** (-mx.arange(0, half, dtype=dtype) / half)


def rope_freqs_yarn(dims: int,
                    original_max_pos: int = 4096,
                    scale: float = 4.0,
                    beta_fast: float = 32.0,
                    beta_slow: float = 1.0,
                    base: float = 10000.0,
                    dtype=mx.float32) -> mx.array:
    """YaRN (Peng et al. 2023): NTK-by-parts. Frequencies are
    interpolated based on how many full rotations they would have
    completed within the original context length.

      r_i = original_max_pos / (2π · base^(2i/dims))    # rotations
      ramp(i) = linear in r between beta_slow and beta_fast,
                clamped to [0, 1]
      θ_i = (1 - ramp) · θ_pi   +   ramp · θ_standard

    where θ_pi uses the position-interpolation freqs and θ_standard
    uses unmodified frequencies. The result preserves high-frequency
    detail while smoothly extending the low-frequency reach.
    """
    half = dims // 2
    inv = -mx.arange(0, half, dtype=dtype) / half       # exponents
    theta_std = base ** inv                              # standard freqs
    theta_pi = theta_std / float(scale)                  # PI freqs

    # Number of full rotations the i-th frequency completes within the
    # original context.
    rots = float(original_max_pos) / (2.0 * math.pi * (base ** (-inv)))
    ramp = (rots - beta_slow) / (beta_fast - beta_slow)
    ramp = mx.clip(ramp, 0.0, 1.0)

    return (1.0 - ramp) * theta_pi + ramp * theta_std


def rope_freqs_llama3(dims: int,
                      factor: float = 8.0,
                      low_freq_factor: float = 1.0,
                      high_freq_factor: float = 4.0,
                      original_max_pos: int = 8192,
                      base: float = 10000.0,
                      dtype=mx.float32) -> mx.array:
    """Llama 3.1 RoPE scaling (Meta 2024 model card).

    Piecewise scaling based on wavelength:

      wavelength_i = 2π / θ_i
      low_threshold  = original_max_pos / low_freq_factor   # high-wavelength cutoff
      high_threshold = original_max_pos / high_freq_factor  # low-wavelength cutoff

      if wavelength < high_threshold:          # high frequency
          θ' = θ                                # leave alone
      elif wavelength > low_threshold:          # low frequency
          θ' = θ / factor                       # PI-scale
      else:                                     # smooth ramp in between
          smooth = (original_max_pos / wavelength - low_freq_factor)
                   / (high_freq_factor - low_freq_factor)
          θ' = (1 - smooth) · θ/factor + smooth · θ
    """
    half = dims // 2
    theta = base ** (-mx.arange(0, half, dtype=dtype) / half)    # standard freqs
    wavelength = 2.0 * math.pi / theta

    low_thresh = float(original_max_pos) / float(low_freq_factor)
    high_thresh = float(original_max_pos) / float(high_freq_factor)

    # Smooth-ramp factor: 1.0 at high freq (use θ as-is),
    #                    0.0 at low freq (use θ/factor),
    #                    linear in between.
    smooth = (float(original_max_pos) / wavelength - low_freq_factor) / (
        high_freq_factor - low_freq_factor
    )
    smooth = mx.clip(smooth, 0.0, 1.0)
    theta_pi = theta / float(factor)
    return (1.0 - smooth) * theta_pi + smooth * theta


# ─── Convenience wrappers — call mx.fast.rope with the right freqs ───────────

def _rope_with_freqs(x: mx.array, dims: int, freqs: mx.array,
                     offset: int = 0, traditional: bool = False) -> mx.array:
    return mx.fast.rope(x, dims=dims, traditional=traditional,
                        base=None, scale=1.0, offset=offset, freqs=freqs)


def rope_standard(x: mx.array, dims: int, *, base: float = 10000.0,
                  offset: int = 0, traditional: bool = False) -> mx.array:
    return _rope_with_freqs(x, dims,
                            rope_freqs_standard(dims, base, x.dtype),
                            offset, traditional)


def rope_linear_pi(x: mx.array, dims: int, *, scale: float = 4.0,
                   base: float = 10000.0, offset: int = 0,
                   traditional: bool = False) -> mx.array:
    return _rope_with_freqs(x, dims,
                            rope_freqs_linear_pi(dims, scale, base, x.dtype),
                            offset, traditional)


def rope_ntk_aware(x: mx.array, dims: int, *, scale: float = 4.0,
                   base: float = 10000.0, offset: int = 0,
                   traditional: bool = False) -> mx.array:
    return _rope_with_freqs(x, dims,
                            rope_freqs_ntk_aware(dims, scale, base, x.dtype),
                            offset, traditional)


def rope_yarn(x: mx.array, dims: int, *,
              original_max_pos: int = 4096,
              scale: float = 4.0,
              beta_fast: float = 32.0,
              beta_slow: float = 1.0,
              base: float = 10000.0,
              offset: int = 0,
              traditional: bool = False) -> mx.array:
    return _rope_with_freqs(
        x, dims,
        rope_freqs_yarn(dims, original_max_pos, scale, beta_fast, beta_slow,
                        base, x.dtype),
        offset, traditional)


def rope_llama3(x: mx.array, dims: int, *,
                factor: float = 8.0,
                low_freq_factor: float = 1.0,
                high_freq_factor: float = 4.0,
                original_max_pos: int = 8192,
                base: float = 10000.0,
                offset: int = 0,
                traditional: bool = False) -> mx.array:
    return _rope_with_freqs(
        x, dims,
        rope_freqs_llama3(dims, factor, low_freq_factor, high_freq_factor,
                          original_max_pos, base, x.dtype),
        offset, traditional)


__all__ = [
    "rope_freqs_standard", "rope_freqs_linear_pi", "rope_freqs_ntk_aware",
    "rope_freqs_yarn", "rope_freqs_llama3",
    "rope_standard", "rope_linear_pi", "rope_ntk_aware",
    "rope_yarn", "rope_llama3",
]
