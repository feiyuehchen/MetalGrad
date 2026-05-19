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

from metalgrad.ops import matmul, rms_norm, conv1d, conv2d, depthwise_conv2d, layer_norm
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


# ─── conv1d ──────────────────────────────────────────────────────────────────

def test_conv1d_gradcheck():
    x = _arr(1, 8, 4, seed=11)
    w = _arr(6, 3, 4, seed=12)
    b = _arr(6, seed=13)
    gradcheck(
        lambda x, w, b: mx.sum(conv1d(x, w, b, padding=1) ** 2),
        [x, w, b], argnums=(0, 1, 2), rtol=5e-2, atol=1e-2, sample=32,
    )


def test_conv1d_vjp_vs_ref():
    x = _arr(1, 16, 8, seed=14)
    w = _arr(12, 5, 8, seed=15)
    b = _arr(12, seed=16)

    def ours(x, w, b): return mx.sum(conv1d(x, w, b, padding=2) ** 2)
    def ref(x, w, b):  return mx.sum((mx.conv1d(x, w, padding=2) + b) ** 2)
    g_o = mx.grad(ours, argnums=(0, 1, 2))(x, w, b)
    g_r = mx.grad(ref,  argnums=(0, 1, 2))(x, w, b)
    mx.eval(*g_o, *g_r)
    for i, name in enumerate("xwb"):
        rel = float(mx.abs(g_o[i] - g_r[i]).max()) / max(float(mx.abs(g_r[i]).max()), 1e-9)
        assert rel < 1e-5, f"conv1d g{name} rel err {rel:.2e}"


# ─── conv2d ──────────────────────────────────────────────────────────────────

def test_conv2d_gradcheck():
    x = _arr(1, 6, 6, 4, seed=17)
    w = _arr(8, 3, 3, 4, seed=18)
    b = _arr(8, seed=19)
    gradcheck(
        lambda x, w, b: mx.sum(conv2d(x, w, b, padding=1) ** 2),
        [x, w, b], argnums=(0, 1, 2), rtol=5e-2, atol=1e-2, sample=32,
    )


def test_conv2d_vjp_vs_ref():
    x = _arr(2, 8, 8, 4, seed=20)
    w = _arr(6, 3, 3, 4, seed=21)
    b = _arr(6, seed=22)

    def ours(x, w, b): return mx.sum(conv2d(x, w, b, padding=1) ** 2)
    def ref(x, w, b):  return mx.sum((mx.conv2d(x, w, padding=1) + b) ** 2)
    g_o = mx.grad(ours, argnums=(0, 1, 2))(x, w, b)
    g_r = mx.grad(ref,  argnums=(0, 1, 2))(x, w, b)
    mx.eval(*g_o, *g_r)
    for i, name in enumerate("xwb"):
        rel = float(mx.abs(g_o[i] - g_r[i]).max()) / max(float(mx.abs(g_r[i]).max()), 1e-9)
        assert rel < 1e-5, f"conv2d g{name} rel err {rel:.2e}"


# ─── depthwise_conv2d (FCDM-relevant) ────────────────────────────────────────

def test_depthwise_conv2d_gradcheck():
    C = 8
    x = _arr(1, 8, 8, C, seed=23)
    w = _arr(C, 7, 7, 1, seed=24, scale=0.05)
    b = _arr(C, seed=25)
    gradcheck(
        lambda x, w, b: mx.sum(depthwise_conv2d(x, w, b, padding=3) ** 2),
        [x, w, b], argnums=(0, 1, 2), rtol=5e-2, atol=1e-2, sample=32,
    )


def test_depthwise_conv2d_vjp_vs_ref():
    C = 16
    x = _arr(1, 12, 12, C, seed=26)
    w = _arr(C, 7, 7, 1, seed=27, scale=0.05)
    b = _arr(C, seed=28)

    def ours(x, w, b): return mx.sum(depthwise_conv2d(x, w, b, padding=3) ** 2)
    def ref(x, w, b):  return mx.sum((mx.conv2d(x, w, padding=3, groups=C) + b) ** 2)
    g_o = mx.grad(ours, argnums=(0, 1, 2))(x, w, b)
    g_r = mx.grad(ref,  argnums=(0, 1, 2))(x, w, b)
    mx.eval(*g_o, *g_r)
    for i, name in enumerate("xwb"):
        rel = float(mx.abs(g_o[i] - g_r[i]).max()) / max(float(mx.abs(g_r[i]).max()), 1e-9)
        assert rel < 1e-5, f"depthwise_conv2d g{name} rel err {rel:.2e}"


# ─── layer_norm ──────────────────────────────────────────────────────────────

def test_layer_norm_gradcheck():
    x = _arr(1, 2, 32, seed=29)
    w = mx.array(np.ones(32, dtype=np.float32) + 0.1 *
                 np.random.default_rng(30).standard_normal(32).astype(np.float32))
    b = _arr(32, seed=31, scale=0.05)
    gradcheck(
        lambda x, w, b: mx.sum(layer_norm(x, w, b, 1e-5) ** 2),
        [x, w, b], argnums=(0, 1, 2), rtol=5e-2, atol=1e-2, sample=32,
    )


def test_layer_norm_vjp_vs_ref():
    x = _arr(2, 8, 64, seed=32)
    w = mx.array(np.ones(64, dtype=np.float32) + 0.1 *
                 np.random.default_rng(33).standard_normal(64).astype(np.float32))
    b = _arr(64, seed=34, scale=0.05)

    def ref_fwd(x, w, b, eps=1e-5):
        mean = mx.mean(x, axis=-1, keepdims=True)
        var  = mx.mean((x - mean) ** 2, axis=-1, keepdims=True)
        return (x - mean) * mx.rsqrt(var + eps) * w + b

    def ours(x, w, b): return mx.sum(layer_norm(x, w, b, 1e-5) ** 2)
    def ref(x, w, b):  return mx.sum(ref_fwd(x, w, b) ** 2)
    g_o = mx.grad(ours, argnums=(0, 1, 2))(x, w, b)
    g_r = mx.grad(ref,  argnums=(0, 1, 2))(x, w, b)
    mx.eval(*g_o, *g_r)
    for i, name in enumerate("xwb"):
        rel = float(mx.abs(g_o[i] - g_r[i]).max()) / max(float(mx.abs(g_r[i]).max()), 1e-9)
        assert rel < 1e-5, f"layer_norm g{name} rel err {rel:.2e}"


if __name__ == "__main__":
    test_matmul_gradcheck_2d()
    test_matmul_gradcheck_batched()
    test_matmul_vjp_vs_ref()
    test_rms_norm_gradcheck_small()
    test_rms_norm_vjp_vs_ref()
    test_conv1d_gradcheck()
    test_conv1d_vjp_vs_ref()
    test_conv2d_gradcheck()
    test_conv2d_vjp_vs_ref()
    test_depthwise_conv2d_gradcheck()
    test_depthwise_conv2d_vjp_vs_ref()
    test_layer_norm_gradcheck()
    test_layer_norm_vjp_vs_ref()
    print("ALL PASS")
