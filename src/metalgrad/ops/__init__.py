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
from metalgrad.ops.attention import attention
from metalgrad.ops.activations import swiglu, geglu, squared_relu
from metalgrad.ops.swiglu_ffn import swiglu_ffn, swiglu_ffn_unfused, stack_gate_up
from metalgrad.ops.cross_entropy import cross_entropy
from metalgrad.ops.mse import mse
from metalgrad.ops.kl_div import kl_div_logits
from metalgrad.ops.rope import (
    rope_freqs_standard, rope_freqs_linear_pi, rope_freqs_ntk_aware,
    rope_freqs_yarn, rope_freqs_llama3,
    rope_standard, rope_linear_pi, rope_ntk_aware, rope_yarn, rope_llama3,
)

__all__ = [
    "matmul", "rms_norm",
    "conv1d", "conv2d", "depthwise_conv2d",
    "layer_norm",
    "attention",
    "swiglu", "geglu", "squared_relu",
    "swiglu_ffn", "swiglu_ffn_unfused", "stack_gate_up",
    "cross_entropy", "mse", "kl_div_logits",
    "rope_freqs_standard", "rope_freqs_linear_pi", "rope_freqs_ntk_aware",
    "rope_freqs_yarn", "rope_freqs_llama3",
    "rope_standard", "rope_linear_pi", "rope_ntk_aware",
    "rope_yarn", "rope_llama3",
]
