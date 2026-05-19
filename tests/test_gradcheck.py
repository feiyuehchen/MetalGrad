"""Gradcheck tests for every shipped op.

Tolerances are FP32-realistic: finite differences in FP32 have an
absolute noise floor proportional to the loss magnitude. We keep test
shapes small so loss values stay O(1) and the noise floor stays below
the assertion tolerance.

Each op also has a "VJP correctness" check that compares our explicit
backward against differentiating an mx-only reference forward — this
catches mathematical errors in the VJP independent of FP32 noise.
"""
from __future__ import annotations

import mlx.core as mx
import numpy as np

from metalgrad.ops import matmul, rms_norm
from metalgrad.testing import gradcheck


RNG = np.random.default_rng(0)
SCALE = 0.3


def _arr(*shape, seed: int = 0, scale: float = SCALE) -> mx.array:
    rng = np.random.default_rng(seed)
    return mx.array((rng.standard_normal(shape) * scale).astype(np.float32))


# ─── matmul ──────────────────────────────────────────────────────────────────

def test_matmul_gradcheck_2d():
    x = _arr(4, 8, seed=1)
    w = _arr(8, 6, seed=2)
    gradcheck(lambda x, w: mx.sum(matmul(x, w) ** 2),
              [x, w], rtol=5e-2, atol=5e-3, sample=32)


def test_matmul_gradcheck_batched():
    x = _arr(2, 4, 8, seed=3)
    w = _arr(8, 6, seed=4)
    gradcheck(lambda x, w: mx.sum(matmul(x, w) ** 2),
              [x, w], rtol=5e-2, atol=5e-3, sample=32)


def test_matmul_vjp_vs_ref():
    """Our VJP matches mx autograd of mx.matmul bit-for-bit (FP32 rel < 1e-5)."""
    x = _arr(4, 8, seed=5)
    w = _arr(8, 6, seed=6)

    def ours(x, w): return mx.sum(matmul(x, w) ** 2)
    def ref(x, w):  return mx.sum(mx.matmul(x, w) ** 2)

    g_o = mx.grad(ours, argnums=(0, 1))(x, w)
    g_r = mx.grad(ref,  argnums=(0, 1))(x, w)
    mx.eval(*g_o, *g_r)
    assert float(mx.abs(g_o[0] - g_r[0]).max()) / max(float(mx.abs(g_r[0]).max()), 1e-9) < 1e-5
    assert float(mx.abs(g_o[1] - g_r[1]).max()) / max(float(mx.abs(g_r[1]).max()), 1e-9) < 1e-5


# ─── rms_norm ────────────────────────────────────────────────────────────────

def test_rms_norm_gradcheck_small():
    # Small shape keeps loss magnitude O(1) so FP32 finite-diff noise
    # stays under the assertion tolerance.
    x = _arr(1, 2, 32, seed=7)
    w = mx.array(np.ones(32, dtype=np.float32) + 0.1 *
                 np.random.default_rng(8).standard_normal(32).astype(np.float32))
    gradcheck(
        lambda x, w: mx.sum(rms_norm(x, w, 1e-5) ** 2),
        [x, w], argnums=(0, 1), rtol=5e-2, atol=1e-2, sample=32,
    )


def test_rms_norm_vjp_vs_ref():
    """Our VJP matches mx autograd of the same mx forward to FP32 precision."""
    x = _arr(2, 5, 64, seed=9)
    w = mx.array(np.ones(64, dtype=np.float32) + 0.1 *
                 np.random.default_rng(10).standard_normal(64).astype(np.float32))

    def mx_ref(x, w, eps=1e-5):
        s = mx.mean(x * x, axis=-1, keepdims=True) + eps
        return x * mx.rsqrt(s) * w

    def ours(x, w): return mx.sum(rms_norm(x, w, 1e-5) ** 2)
    def ref(x, w):  return mx.sum(mx_ref(x, w) ** 2)

    g_o = mx.grad(ours, argnums=(0, 1))(x, w)
    g_r = mx.grad(ref,  argnums=(0, 1))(x, w)
    mx.eval(*g_o, *g_r)
    rel_x = float(mx.abs(g_o[0] - g_r[0]).max()) / max(float(mx.abs(g_r[0]).max()), 1e-9)
    rel_w = float(mx.abs(g_o[1] - g_r[1]).max()) / max(float(mx.abs(g_r[1]).max()), 1e-9)
    assert rel_x < 1e-5, f"gx rel err {rel_x:.2e}"
    assert rel_w < 1e-5, f"gw rel err {rel_w:.2e}"


if __name__ == "__main__":
    test_matmul_gradcheck_2d()
    test_matmul_gradcheck_batched()
    test_matmul_vjp_vs_ref()
    test_rms_norm_gradcheck_small()
    test_rms_norm_vjp_vs_ref()
    print("ALL PASS")
