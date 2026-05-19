"""Finite-difference gradient checker.

Validates that `mx.grad(fn)` matches numerical gradients within tolerance.
This is the CI gate: every op we ship must pass `gradcheck`.

Usage:

    from metalgrad.testing import gradcheck

    def fn(x, w):
        return mx.sum(my_op(x, w) ** 2)

    gradcheck(fn, [x, w], rtol=1e-3)         # raises AssertionError if off
"""
from __future__ import annotations

from typing import Sequence

import mlx.core as mx
import numpy as np


def _to_np(a) -> np.ndarray:
    return np.array(a, copy=False)


def _set_idx(arr: mx.array, idx: tuple, value: float) -> mx.array:
    """Out-of-place: return a copy of `arr` with arr[idx] = value."""
    np_arr = _to_np(arr).copy()
    np_arr[idx] = value
    return mx.array(np_arr, dtype=arr.dtype)


def _iter_indices(shape: tuple, sample: int | None):
    """Yield index tuples into a tensor of `shape`. If `sample` is set,
    yield at most `sample` random indices (so gradcheck stays cheap on
    big tensors)."""
    flat_n = int(np.prod(shape))
    if sample is None or sample >= flat_n:
        for flat in range(flat_n):
            yield np.unravel_index(flat, shape)
    else:
        rng = np.random.default_rng(0)
        chosen = rng.choice(flat_n, size=sample, replace=False)
        for flat in chosen:
            yield np.unravel_index(int(flat), shape)


def gradcheck(fn,
              inputs: Sequence[mx.array],
              *,
              argnums: Sequence[int] | None = None,
              eps: float = 1e-3,
              rtol: float = 1e-2,
              atol: float = 1e-3,
              sample: int | None = 64,
              verbose: bool = False) -> None:
    """Verify autograd matches finite differences.

    Args:
      fn:       callable mapping `(*inputs) -> scalar mx.array`.
      inputs:   sequence of mx.arrays. Should be float dtype.
      argnums:  which inputs to differentiate w.r.t. Default = all floats.
      eps:      finite-difference step.
      rtol/atol: tolerances. Element passes if
                  |fd - ad| < atol + rtol * |fd|
      sample:   max number of elements to spot-check per input. None = all.
      verbose:  print per-element diagnostics.

    Raises AssertionError on the first failed element with a clear diff.
    """
    inputs = list(inputs)
    if argnums is None:
        argnums = list(range(len(inputs)))
    argnums = tuple(argnums)

    # Force-eval forward once so we hit any kernel-compile errors here,
    # not inside the finite-difference loop.
    fn(*inputs)

    grads = mx.grad(fn, argnums=argnums)(*inputs)
    if not isinstance(grads, tuple):
        grads = (grads,)
    mx.eval(*grads)

    for ai, arg_i in enumerate(argnums):
        ad_grad = grads[ai]
        x = inputs[arg_i]
        ad_np = _to_np(ad_grad)
        if ad_np.shape != x.shape:
            raise AssertionError(
                f"arg {arg_i}: grad shape {ad_np.shape} != input shape {x.shape}"
            )

        for idx in _iter_indices(tuple(x.shape), sample):
            orig = float(_to_np(x)[idx])
            x_plus = list(inputs)
            x_minus = list(inputs)
            x_plus[arg_i] = _set_idx(x, idx, orig + eps)
            x_minus[arg_i] = _set_idx(x, idx, orig - eps)
            f_plus = float(_to_np(fn(*x_plus)))
            f_minus = float(_to_np(fn(*x_minus)))
            fd = (f_plus - f_minus) / (2 * eps)
            ad = float(ad_np[idx])
            tol = atol + rtol * abs(fd)
            if verbose:
                print(f"  arg{arg_i}{list(idx)}: fd={fd:+.5f}  ad={ad:+.5f}  diff={abs(fd-ad):.2e}  tol={tol:.2e}")
            if abs(fd - ad) > tol:
                raise AssertionError(
                    f"gradcheck FAIL at arg {arg_i}, index {idx}: "
                    f"finite_diff={fd:.6f}, autograd={ad:.6f}, "
                    f"|diff|={abs(fd-ad):.2e} > tol={tol:.2e}"
                )

    # Reached here = every sampled element passes.


__all__ = ["gradcheck"]
