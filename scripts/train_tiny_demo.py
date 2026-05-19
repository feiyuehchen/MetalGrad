"""End-to-end proof: train a tiny ConvNeXt-style model using metalgrad ops.

Hits all of the wrapped ops at once:
  - depthwise_conv2d (K=7, the ConvNeXt depthwise)
  - layer_norm (the per-token norm)
  - conv2d K=1 (pointwise MLP)
  - matmul (classifier head)

Synthetic data: 10-class image classification on 16x16 noise where each
class has a fixed-shift bias added. Trivially learnable; the point is
that mx.grad backpropagates through the full metalgrad stack and the
loss actually drops. If it doesn't, our VJPs are broken.

50 SGD steps. ~20 seconds on M3 Pro.
"""
from __future__ import annotations

import time
import numpy as np
import mlx.core as mx

from metalgrad.ops import depthwise_conv2d, conv2d, layer_norm, matmul


# ─── model ───────────────────────────────────────────────────────────────────

DIM = 32           # channel dim of the ConvNeXt block
NUM_CLASSES = 10
H = W = 16


def _gelu(x):
    return 0.5 * x * (1 + mx.tanh(0.7978845608 * (x + 0.044715 * x * x * x)))


def init_params(seed: int = 0):
    rng = np.random.default_rng(seed)
    f32 = np.float32
    p = {}

    # Stem: Conv2d 3 -> DIM, K=4, stride=4 (16x16 -> 4x4)
    p["stem_w"] = mx.array((rng.standard_normal((DIM, 4, 4, 3)) * 0.1).astype(f32))
    p["stem_b"] = mx.zeros((DIM,))

    # ConvNeXt block (× 1 for the demo)
    p["dw_w"] = mx.array((rng.standard_normal((DIM, 7, 7, 1)) * 0.05).astype(f32))
    p["dw_b"] = mx.zeros((DIM,))
    p["ln_w"] = mx.ones((DIM,))
    p["ln_b"] = mx.zeros((DIM,))
    p["pw1_w"] = mx.array((rng.standard_normal((4 * DIM, 1, 1, DIM)) * 0.1).astype(f32))
    p["pw1_b"] = mx.zeros((4 * DIM,))
    p["pw2_w"] = mx.array((rng.standard_normal((DIM, 1, 1, 4 * DIM)) * 0.1).astype(f32))
    p["pw2_b"] = mx.zeros((DIM,))

    # Classifier head: global-average-pool then matmul.
    p["head_w"] = mx.array((rng.standard_normal((DIM, NUM_CLASSES)) * 0.1).astype(f32))
    p["head_b"] = mx.zeros((NUM_CLASSES,))
    return p


def forward(p, x):
    """x: (N, 16, 16, 3) -> logits (N, 10)."""
    # Stem
    h = conv2d(x, p["stem_w"], p["stem_b"], stride=4, padding=0)         # (N, 4, 4, DIM)

    # ConvNeXt block: dw -> ln -> pw1 -> gelu -> pw2 (+ residual)
    res = h
    h = depthwise_conv2d(h, p["dw_w"], p["dw_b"], padding=3)             # (N, 4, 4, DIM)
    h = layer_norm(h, p["ln_w"], p["ln_b"], 1e-6)                        # over last axis
    h = conv2d(h, p["pw1_w"], p["pw1_b"])                                # (N, 4, 4, 4D)
    h = _gelu(h)
    h = conv2d(h, p["pw2_w"], p["pw2_b"])                                # (N, 4, 4, DIM)
    h = h + res

    # Global average pool over (H, W) — keep channel axis.
    h = mx.mean(h, axis=(1, 2))                                          # (N, DIM)

    # Classifier head.
    logits = matmul(h, p["head_w"]) + p["head_b"]
    return logits


def cross_entropy(logits, labels):
    """Cross entropy with integer labels."""
    log_probs = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
    one_hot = mx.eye(NUM_CLASSES)[labels]
    return -mx.mean(mx.sum(one_hot * log_probs, axis=-1))


def make_batch(batch_size: int, seed: int):
    """Synthetic 10-class task: noise + class-conditional bias pattern."""
    rng = np.random.default_rng(seed)
    labels = rng.integers(0, NUM_CLASSES, size=batch_size).astype(np.int32)
    # Per-class fixed pattern: a small bias in one of 10 spatial cells.
    x = rng.standard_normal((batch_size, H, W, 3)).astype(np.float32) * 0.3
    for i, lbl in enumerate(labels):
        row, col = lbl // 4, lbl % 4   # maps 0-9 to rough quadrants
        x[i, row * 4:(row + 1) * 4, col * 4:(col + 1) * 4, :] += 1.5
    return mx.array(x), mx.array(labels)


# ─── training loop ───────────────────────────────────────────────────────────

def main():
    params = init_params()
    lr = 0.3
    batch_size = 64
    steps = 200

    def loss_fn(params, x, y):
        return cross_entropy(forward(params, x), y)

    grad_fn = mx.value_and_grad(loss_fn)

    print(f"Training {steps} SGD steps, batch {batch_size}, lr {lr}")
    print(f"{'step':>4}  {'loss':>10}  {'iter (ms)':>10}")
    losses = []
    t_iters = []

    for step in range(steps):
        x, y = make_batch(batch_size, seed=step)
        mx.eval(x, y)

        t0 = time.perf_counter()
        loss, grads = grad_fn(params, x, y)
        # SGD update
        params = {k: v - lr * grads[k] for k, v in params.items()}
        mx.eval(loss, *params.values())
        dt = (time.perf_counter() - t0) * 1000

        losses.append(float(loss))
        t_iters.append(dt)
        if step % 20 == 0 or step == steps - 1:
            print(f"{step:>4d}  {float(loss):>10.4f}  {dt:>10.1f}")

    print()
    print(f"Start loss : {losses[0]:.4f}")
    print(f"End loss   : {losses[-1]:.4f}")
    print(f"Reduction  : {losses[0] / losses[-1]:.2f}x")
    print(f"Mean iter  : {sum(t_iters[5:]) / len(t_iters[5:]):.1f} ms (excl. first 5 warmup)")
    # Random-init log-loss for 10 classes is log(10) ≈ 2.30. We need
    # significant drop below that to confirm gradients are flowing.
    assert losses[-1] < 1.5, (
        f"loss only reached {losses[-1]:.3f} after {steps} steps — gradients "
        f"may not be flowing correctly through the metalgrad ops."
    )
    print()
    print("✓ Training works through metalgrad ops end-to-end.")


if __name__ == "__main__":
    main()
