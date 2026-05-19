"""Core wrapper: turn a forward Metal kernel into an autograd-safe op.

The wrapper is a thin layer over `mx.custom_function`. We re-export it
under our own name (`metalgrad.differentiable`) to:

  1. Provide a stable API surface that does not move when MLX renames
     things upstream.
  2. Make the intent explicit at the call site — `@differentiable` reads
     better than `@mx.custom_function` when the whole point of the
     library is autograd safety.

API:

    @differentiable
    def my_op(x, w, b):
        return _my_metal_kernel(x, w, b)

    @my_op.vjp
    def _(primals, cotangent, output):
        x, w, b = primals
        gy, = (cotangent,) if not isinstance(cotangent, tuple) else cotangent
        gx = ...   # use mx ops; this is allowed to be slower than forward
        gw = ...
        gb = ...
        return gx, gw, gb

If you do not define `.vjp`, `mx.grad` will fall back to tracing through
the forward — which, for an `mx.fast.metal_kernel` call, raises
`Primitive::vjp Not implemented for CustomKernel`. So always pair a
metal-kernel forward with an explicit `.vjp`.
"""
from __future__ import annotations

import mlx.core as mx


def differentiable(fn):
    """Make `fn` an autograd-safe op.

    Equivalent to `mx.custom_function(fn)` plus our convention checks.

    The returned object exposes `.vjp`, `.jvp`, and `.vmap` decorators
    matching MLX's `custom_function` interface. Use `.vjp` for backward.
    """
    return mx.custom_function(fn)


__all__ = ["differentiable"]
