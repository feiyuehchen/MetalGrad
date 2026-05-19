"""Pre-built differentiable ops.

Each module here ships:

  * A forward implementation. v0.1 ops wrap `mx` baselines so we can
    validate the framework end-to-end. Future versions swap individual
    forwards for fast custom Metal kernels — the `.vjp` does not change.
  * An explicit VJP, written in mx ops so it participates in autograd.
  * A gradcheck test in `tests/`.

Import like:

    from metalgrad.ops import matmul, rms_norm
"""
from metalgrad.ops.matmul import matmul
from metalgrad.ops.rms_norm import rms_norm
from metalgrad.ops.conv1d import conv1d
from metalgrad.ops.conv2d import conv2d
from metalgrad.ops.depthwise_conv2d import depthwise_conv2d
from metalgrad.ops.layer_norm import layer_norm

__all__ = [
    "matmul", "rms_norm",
    "conv1d", "conv2d", "depthwise_conv2d",
    "layer_norm",
]
