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

from metalgrad.ops import (
    matmul, rms_norm, conv1d, conv2d, depthwise_conv2d, layer_norm, attention,
    swiglu, geglu, squared_relu, cross_entropy, mse, kl_div_logits,
    rope_standard, rope_llama3, rope_yarn,
)
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


# ─── attention ───────────────────────────────────────────────────────────────

def test_attention_gradcheck_small():
    # Tiny shape so fp32 finite-diff stays clean.
    B, H, T, D = 1, 2, 4, 8
    q = _arr(B, H, T, D, seed=35)
    k = _arr(B, H, T, D, seed=36)
    v = _arr(B, H, T, D, seed=37)
    scale = 1.0 / (D ** 0.5)
    gradcheck(
        lambda q, k, v: mx.sum(attention(q, k, v, scale=scale) ** 2),
        [q, k, v], argnums=(0, 1, 2), rtol=5e-2, atol=1e-2, sample=32,
    )


def test_attention_vjp_vs_ref():
    """Our wrapper VJP matches mx.fast.SDPA's built-in VJP exactly."""
    B, H, T, D = 2, 4, 8, 16
    q = _arr(B, H, T, D, seed=38)
    k = _arr(B, H, T, D, seed=39)
    v = _arr(B, H, T, D, seed=40)
    scale = 1.0 / (D ** 0.5)

    def ours(q, k, v): return mx.sum(attention(q, k, v, scale=scale) ** 2)
    def ref(q, k, v):
        return mx.sum(mx.fast.scaled_dot_product_attention(q, k, v, scale=scale) ** 2)

    g_o = mx.grad(ours, argnums=(0, 1, 2))(q, k, v)
    g_r = mx.grad(ref,  argnums=(0, 1, 2))(q, k, v)
    mx.eval(*g_o, *g_r)
    for i, name in enumerate("qkv"):
        rel = float(mx.abs(g_o[i] - g_r[i]).max()) / max(float(mx.abs(g_r[i]).max()), 1e-9)
        assert rel < 1e-5, f"attention g{name} rel err {rel:.2e}"


def test_attention_with_mask_vjp_vs_ref():
    B, H, T, D = 1, 2, 6, 8
    q = _arr(B, H, T, D, seed=41)
    k = _arr(B, H, T, D, seed=42)
    v = _arr(B, H, T, D, seed=43)
    # Causal mask: -inf above diagonal, 0 elsewhere. Same shape mx expects.
    mask = mx.array(np.where(
        np.tri(T, T, dtype=bool).reshape(1, 1, T, T), 0.0, -1e9
    ).astype(np.float32))
    scale = 1.0 / (D ** 0.5)

    def ours(q, k, v): return mx.sum(attention(q, k, v, scale=scale, mask=mask) ** 2)
    def ref(q, k, v):
        return mx.sum(mx.fast.scaled_dot_product_attention(
            q, k, v, scale=scale, mask=mask) ** 2)

    g_o = mx.grad(ours, argnums=(0, 1, 2))(q, k, v)
    g_r = mx.grad(ref,  argnums=(0, 1, 2))(q, k, v)
    mx.eval(*g_o, *g_r)
    for i, name in enumerate("qkv"):
        rel = float(mx.abs(g_o[i] - g_r[i]).max()) / max(float(mx.abs(g_r[i]).max()), 1e-9)
        assert rel < 1e-5, f"attention(mask) g{name} rel err {rel:.2e}"


# ─── activations ─────────────────────────────────────────────────────────────

def test_swiglu_gradcheck():
    a = _arr(2, 16, seed=50)
    b = _arr(2, 16, seed=51)
    gradcheck(
        lambda a, b: mx.sum(swiglu(a, b) ** 2),
        [a, b], rtol=5e-2, atol=1e-2, sample=32,
    )


def test_swiglu_vjp_vs_ref():
    a = _arr(8, 32, seed=52)
    b = _arr(8, 32, seed=53)

    def ref_fwd(a, b): return (a * mx.sigmoid(a)) * b
    def ours(a, b): return mx.sum(swiglu(a, b) ** 2)
    def ref(a, b):  return mx.sum(ref_fwd(a, b) ** 2)
    g_o = mx.grad(ours, argnums=(0, 1))(a, b)
    g_r = mx.grad(ref,  argnums=(0, 1))(a, b)
    mx.eval(*g_o, *g_r)
    for i, name in enumerate("ab"):
        rel = float(mx.abs(g_o[i] - g_r[i]).max()) / max(float(mx.abs(g_r[i]).max()), 1e-9)
        assert rel < 1e-5, f"swiglu g{name} rel err {rel:.2e}"


def test_geglu_gradcheck():
    a = _arr(2, 16, seed=54)
    b = _arr(2, 16, seed=55)
    gradcheck(
        lambda a, b: mx.sum(geglu(a, b) ** 2),
        [a, b], rtol=5e-2, atol=1e-2, sample=32,
    )


def test_squared_relu_gradcheck():
    x = _arr(4, 16, seed=56)
    gradcheck(
        lambda x: mx.sum(squared_relu(x) ** 2),
        [x], rtol=5e-2, atol=1e-2, sample=32,
    )


def test_squared_relu_vjp_vs_ref():
    x = _arr(4, 16, seed=57)

    def ours(x): return mx.sum(squared_relu(x) ** 2)
    def ref(x):  return mx.sum((mx.maximum(x, 0) ** 2) ** 2)
    g_o = mx.grad(ours)(x)
    g_r = mx.grad(ref)(x)
    mx.eval(g_o, g_r)
    rel = float(mx.abs(g_o - g_r).max()) / max(float(mx.abs(g_r).max()), 1e-9)
    assert rel < 1e-5, f"squared_relu rel err {rel:.2e}"


# ─── cross_entropy ───────────────────────────────────────────────────────────

def test_cross_entropy_value_vs_ref():
    N, V = 32, 128
    rng = np.random.default_rng(60)
    logits = mx.array((rng.standard_normal((N, V)) * 0.5).astype(np.float32))
    labels = mx.array(rng.integers(0, V, N).astype(np.int32))

    def ref(l, lb):
        lse = mx.logsumexp(l, axis=-1)
        lab = mx.take_along_axis(l, lb[:, None], axis=-1).squeeze(-1)
        return mx.mean(lse - lab)

    v_ours = float(cross_entropy(logits, labels))
    v_ref = float(ref(logits, labels))
    assert abs(v_ours - v_ref) < 1e-5, f"value: ours {v_ours}, ref {v_ref}"


def test_cross_entropy_grad_vs_ref():
    """Our grad kernel matches mx.grad of the mx reference to FP32 precision."""
    N, V = 32, 256
    rng = np.random.default_rng(61)
    logits = mx.array((rng.standard_normal((N, V)) * 0.5).astype(np.float32))
    labels = mx.array(rng.integers(0, V, N).astype(np.int32))

    def ref(l, lb):
        lse = mx.logsumexp(l, axis=-1)
        lab = mx.take_along_axis(l, lb[:, None], axis=-1).squeeze(-1)
        return mx.mean(lse - lab)

    g_o = mx.grad(lambda l: cross_entropy(l, labels))(logits)
    g_r = mx.grad(lambda l: ref(l, labels))(logits)
    mx.eval(g_o, g_r)
    rel = float(mx.abs(g_o - g_r).max()) / max(float(mx.abs(g_r).max()), 1e-9)
    assert rel < 1e-5, f"cross_entropy grad rel err {rel:.2e}"


def test_cross_entropy_gradcheck_finite_diff():
    """Per-element finite-diff vs autograd. Small V so FP32 fd has room."""
    N, V = 4, 16
    rng = np.random.default_rng(62)
    logits = mx.array((rng.standard_normal((N, V)) * 0.3).astype(np.float32))
    labels = mx.array(rng.integers(0, V, N).astype(np.int32))

    gradcheck(
        lambda l: cross_entropy(l, labels),
        [logits], argnums=(0,), rtol=5e-2, atol=1e-2, sample=32,
    )


# ─── mse ─────────────────────────────────────────────────────────────────────

def test_mse_value_vs_ref():
    p = _arr(4, 16, seed=63)
    t = _arr(4, 16, seed=64)
    v_ours = float(mse(p, t))
    v_ref = float(mx.mean((p - t) ** 2))
    assert abs(v_ours - v_ref) < 1e-5


def test_mse_grad_vs_ref():
    p = _arr(4, 16, seed=65)
    t = _arr(4, 16, seed=66)
    g_o = mx.grad(lambda p: mse(p, t))(p)
    g_r = mx.grad(lambda p: mx.mean((p - t) ** 2))(p)
    mx.eval(g_o, g_r)
    rel = float(mx.abs(g_o - g_r).max()) / max(float(mx.abs(g_r).max()), 1e-9)
    assert rel < 1e-5, f"mse grad rel err {rel:.2e}"


# ─── kl_div_logits ───────────────────────────────────────────────────────────

def test_kl_div_value_vs_ref():
    N, V = 16, 64
    rng = np.random.default_rng(67)
    p = mx.array((rng.standard_normal((N, V)) * 0.5).astype(np.float32))
    t = mx.array((rng.standard_normal((N, V)) * 0.5).astype(np.float32))

    def kl_ref(p_logits, t_logits):
        log_q = p_logits - mx.logsumexp(p_logits, axis=-1, keepdims=True)
        log_p = t_logits - mx.logsumexp(t_logits, axis=-1, keepdims=True)
        return mx.mean(mx.sum(mx.exp(log_p) * (log_p - log_q), axis=-1))

    v_o = float(kl_div_logits(p, t))
    v_r = float(kl_ref(p, t))
    assert abs(v_o - v_r) < 1e-4, f"kl value: ours {v_o}, ref {v_r}"


def test_kl_div_grad_vs_ref():
    N, V = 16, 64
    rng = np.random.default_rng(68)
    p = mx.array((rng.standard_normal((N, V)) * 0.5).astype(np.float32))
    t = mx.array((rng.standard_normal((N, V)) * 0.5).astype(np.float32))

    def kl_ref(p_logits, t_logits):
        log_q = p_logits - mx.logsumexp(p_logits, axis=-1, keepdims=True)
        log_p = t_logits - mx.logsumexp(t_logits, axis=-1, keepdims=True)
        return mx.mean(mx.sum(mx.exp(log_p) * (log_p - log_q), axis=-1))

    # Grad w.r.t. pred only (target is the teacher, treated as constant)
    g_o = mx.grad(lambda p: kl_div_logits(p, t))(p)
    g_r = mx.grad(lambda p: kl_ref(p, t))(p)
    mx.eval(g_o, g_r)
    rel = float(mx.abs(g_o - g_r).max()) / max(float(mx.abs(g_r).max()), 1e-9)
    assert rel < 1e-4, f"kl_div_logits grad rel err {rel:.2e}"


# ─── rope variants (correctness only; no grad win to test) ───────────────────

def test_rope_variants_finite():
    """Just check the new RoPE variants produce finite outputs of the
    expected shape. The math equivalence to mx.fast.rope with the right
    freqs is by construction."""
    rng = np.random.default_rng(69)
    B, H, T, D = 1, 2, 16, 32
    x = mx.array(rng.standard_normal((B, H, T, D)).astype(np.float32) * 0.3)

    for fn in [
        lambda x: rope_standard(x, D),
        lambda x: rope_llama3(x, D, original_max_pos=64),
        lambda x: rope_yarn(x, D, original_max_pos=64),
    ]:
        y = fn(x)
        mx.eval(y)
        assert y.shape == x.shape
        assert bool(mx.all(mx.isfinite(y))), "non-finite RoPE output"


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
    test_attention_gradcheck_small()
    test_attention_vjp_vs_ref()
    test_attention_with_mask_vjp_vs_ref()
    test_swiglu_gradcheck()
    test_swiglu_vjp_vs_ref()
    test_geglu_gradcheck()
    test_squared_relu_gradcheck()
    test_squared_relu_vjp_vs_ref()
    test_cross_entropy_value_vs_ref()
    test_cross_entropy_grad_vs_ref()
    test_cross_entropy_gradcheck_finite_diff()
    test_mse_value_vs_ref()
    test_mse_grad_vs_ref()
    test_kl_div_value_vs_ref()
    test_kl_div_grad_vs_ref()
    test_rope_variants_finite()
    print("ALL PASS")
