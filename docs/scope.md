# MetalGrad — Scope

## One-line definition

> Differentiable Metal-kernel wrappers for Apple Silicon training,
> targeting CUDA-on-good-GPU-equivalent training experience on MLX (and,
> later, PyTorch-MPS).

## v0.1 IN SCOPE

1. **Core wrapper** — `@differentiable` decorator over `mx.custom_function`.
2. **Five pre-built ops** — each with forward kernel + explicit VJP:
   - `matmul`
   - `rms_norm`
   - `conv1d`
   - `conv2d`
   - `depthwise_conv2d`
   - `layer_norm`
3. **Gradient correctness testing** — `metalgrad.testing.gradcheck`
   (finite-difference vs autograd). CI-enforced.
4. **End-to-end demo** — tiny ConvNeXt classifier trains 50 steps on
   toy data using metalgrad ops; loss decreases.

## v0.2 (planned, not committed)

- PyTorch + MPS backend via `torch.autograd.Function`, sharing the
  same Metal source as the MLX path.
- More ops: `group_norm`, `scaled_dot_product_attention`.
- Faster forwards for ops where v0.1 wrapped mx baseline.

## Hard OUT-OF-SCOPE (permanent)

- ❌ CUDA / Linux / x86. Apple Silicon only.
- ❌ Sharing source with `conv1d_for_apple_silicon`. Independent repo.
- ❌ Higher-order derivatives (`mx.grad(mx.grad(...))`).
- ❌ Forward-mode autodiff (JVP).
- ❌ Optimizers / dataloaders / training loops — this is a kernel
  library, not a framework.
- ❌ Inference-only paths — those belong in sister repos.
- ❌ Non-differentiable ops (quantization, `argmax`).

## Success criteria for v0.1 ship

| | Target |
|---|---|
| Correctness | all ops `gradcheck` passes (rtol 1e-2, atol 1e-2) |
| VJP exactness | every op's VJP matches `mx.grad` of an `mx`-only reference forward to FP32 precision (rel err < 1e-5) |
| Forward speed | ≥ 1.5× over `mx.{op}` baseline on representative shape |
| Backward speed | not slower than `mx.grad` baseline |
| End-to-end | tiny ConvNeXt train loop runs 50 steps, loss monotonically decreases |
| Aspirational CUDA parity | training throughput ≥ 0.3× of a comparable CUDA card (RTX 4070 ~30 TFLOPS vs M3 Pro ~5 TFLOPS hardware bound) |

## Layout

```
MetalGrad/
├── pyproject.toml
├── README.md
├── docs/
│   └── scope.md
├── src/metalgrad/
│   ├── __init__.py
│   ├── differentiable.py        # @differentiable wrapper
│   ├── ops/
│   │   ├── __init__.py
│   │   ├── matmul.py
│   │   ├── rms_norm.py
│   │   ├── conv1d.py           (planned)
│   │   ├── conv2d.py           (planned)
│   │   ├── depthwise_conv2d.py (planned)
│   │   └── layer_norm.py       (planned)
│   └── testing/
│       ├── __init__.py
│       └── gradcheck.py
├── scripts/                    (benches, demos)
└── tests/
    └── test_gradcheck.py
```
