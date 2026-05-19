# MetalGrad

**Differentiable Metal kernels for Apple Silicon training.**

Custom Metal kernels written via `mx.fast.metal_kernel` don't auto-generate
backward passes — calling `mx.grad` on them raises `Primitive::vjp Not
implemented for CustomKernel`. This makes them inference-only, which
blocks training-time use of fast custom kernels.

MetalGrad fixes this. Wrap any forward (a Metal kernel, a `mx.fast.*`
call, or pure mx ops) with `@differentiable`, then attach a VJP written
in mx ops. The result behaves like a regular MLX op under `mx.grad`.

```python
import mlx.core as mx
from metalgrad import differentiable

@differentiable
def my_op(x, w):
    return _my_metal_kernel(x, w)   # whatever forward you want

@my_op.vjp
def _(primals, cotangent, output):
    x, w = primals
    gy = cotangent
    gx = mx.matmul(gy, mx.swapaxes(w, -1, -2))    # backward in mx ops
    gw = mx.matmul(mx.swapaxes(x, -1, -2), gy)
    return gx, gw

# Now my_op works under mx.grad
def loss(x, w): return mx.sum(my_op(x, w) ** 2)
gx, gw = mx.grad(loss, argnums=(0, 1))(x, w)
```

The forward can be fast (custom Metal); the backward can be correct
(plain mx ops). You ship a Metal-kernel forward for inference *and*
training without losing autograd support.

## Pre-built ops (v0.0.1)

```python
from metalgrad.ops import matmul, rms_norm
```

| op | Forward speed vs `mx` | Notes |
|---|---:|---|
| `matmul` | 1.0× | `mx.matmul` is already MPSGraph-tuned |
| **`rms_norm`** | **1.91×** ✓ | fused TG-cooperative SIMD reduction (v0.0.2) |
| `conv1d` | 1.0× | `mx.conv1d` baseline |
| `conv2d` | 1.0× | `mx.conv2d` baseline |
| `depthwise_conv2d` | 1.0× | naive K=7 kernel lost to mx; disabled. Real kernel = v0.0.3. |
| **`layer_norm`** | **2.80×** ✓ | fused two-pass reduction (v0.0.2) |
| `attention` | 1.0× | `mx.fast.scaled_dot_product_attention` |
| `swiglu` / `geglu` / `squared_relu` | 1.0× | new in v0.0.2, mx-backed forward |

Benched on M3 Pro, FP32, batch shape `(4, 512, 1024)` for the norm ops.
All ops pass `gradcheck` with `rel_err < 1e-5` vs the mx reference VJP.

The v0.0.1 ops use mx-based forwards to establish the framework and
pass gradcheck end-to-end. Custom Metal kernels land in v0.0.2 without
changing the VJP — that is the point of the wrapper.

## Testing

Every op ships with `metalgrad.testing.gradcheck`:

```python
from metalgrad.testing import gradcheck
gradcheck(lambda x, w: mx.sum(my_op(x, w) ** 2), [x, w], rtol=5e-2)
```

CI runs `tests/test_gradcheck.py` and refuses to merge anything that
breaks gradcheck.

## Scope

See [`docs/scope.md`](docs/scope.md) for what is in / out / hard-out of
scope. Short version:

- **In:** wrapper utility + a curated set of common ops + gradient
  testing infrastructure. MLX backend, Apple Silicon only.
- **Out:** CUDA, optimizers, dataloaders, higher-order gradients,
  forward-mode AD, copying source from sister repos.

## Install

```bash
git clone git@github.com:feiyuehchen/MetalGrad.git
cd MetalGrad
uv sync
uv run python tests/test_gradcheck.py
```

## Relationship to other repos

- [`conv1d_for_apple_silicon`](https://github.com/feiyuehchen/conv1d_for_apple_silicon)
  — sister project, inference-only Conv1d dispatcher. Source-independent
  from MetalGrad. If/when MetalGrad's training-safe ops are useful for
  conv1d's training path, conv1d will import them — but MetalGrad
  itself does not depend on conv1d.

## License

MIT.
