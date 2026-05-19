"""metalgrad — differentiable Metal kernels for Apple Silicon training.

Public API:

  differentiable(forward_fn)
      Decorator that registers a Metal-kernel forward and a Python
      backward (using `mx.custom_function` under the hood).

  ops.conv1d, ops.conv2d, ops.depthwise_conv2d, ops.matmul, ops.layer_norm
      Pre-built ops: fast Metal forward + correct VJP. Drop-in replacements
      for the MLX equivalents in `mx.grad` contexts.

  testing.gradcheck(fn, inputs, rtol=1e-3)
      Finite-difference gradient checker. CI uses this to guarantee
      backward correctness for every shipped op.
"""

__version__ = "0.0.1"

from metalgrad.differentiable import differentiable

__all__ = ["differentiable"]
