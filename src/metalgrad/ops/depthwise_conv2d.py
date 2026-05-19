"""Differentiable depthwise Conv2d.

The canonical ConvNeXt / FCDM depthwise: K=7, padding=3, groups=C.
v0.0.1 wrapped `mx.conv2d` (slow on Apple Silicon for this shape — mx
runs at ~2% of GPU FP32 peak). v0.0.2 ships a hand-tuned Metal kernel
in addition to the wrapped mx fallback.

Strategy of the Metal kernel:

  Layout: NHWC, x pre-padded on the host with mx.pad. The kernel reads
  the (K=7) x (K=7) = 49 input values per output position with no
  boundary checks — input is large enough.

  One thread = one output element (n, h_out, w_out, c). Grid is
  (C, W_out, H_out) flattened along the channel axis so adjacent
  threads in a SIMD have consecutive c values — coalesced x reads
  along the innermost NHWC axis.

  K=7 inner loop is unrolled via #pragma. Compiler can keep all 49
  weights in registers; the rest of the budget goes to the input
  loads.

The mx-wrapped fallback path stays for cases the kernel does not
handle (non-K=7, non-square, stride>1, dilation>1, N>1 for now). The
dispatcher picks the kernel only when its precondition is met.

Layout:
  x: (N, H, W, C)
  w: (C, KH, KW, 1)
  b: (C,) or None
"""
from __future__ import annotations

import mlx.core as mx

from metalgrad.differentiable import differentiable


# ─── Metal kernel ────────────────────────────────────────────────────────────

_DWCONV2D_K7_SRC = """
    // Grid: (C, W_out, H_out*N) -- batch fused into z dim.
    // x:    (N, H_OUT, W_OUT, C) — padding=3 handled in-kernel.
    // w:    (C, 7, 7, 1)
    // bias: (C,)
    // y:    (N, H_OUT, W_OUT, C)
    uint c     = thread_position_in_grid.x;
    uint w_out = thread_position_in_grid.y;
    uint hz    = thread_position_in_grid.z;
    if (c >= C || w_out >= W_OUT || hz >= (uint)(H_OUT * N)) return;

    uint n     = hz / (uint)H_OUT;
    uint h_out = hz - n * (uint)H_OUT;

    int h_base = (int)h_out - 3;
    int w_base = (int)w_out - 3;
    int H_IN = (int)H_OUT;
    int W_IN = (int)W_OUT;
    uint batch_off = n * (uint)H_OUT * (uint)W_OUT * (uint)C;

    float acc = 0.0f;
    #pragma clang loop unroll(full)
    for (int kh = 0; kh < 7; ++kh) {
        int h_in = h_base + kh;
        bool h_ok = (h_in >= 0) & (h_in < H_IN);
        #pragma clang loop unroll(full)
        for (int kw = 0; kw < 7; ++kw) {
            int w_in = w_base + kw;
            bool ok = h_ok & (w_in >= 0) & (w_in < W_IN);
            int h_safe = ok ? h_in : 0;
            int w_safe = ok ? w_in : 0;
            float xv = ok ? x[batch_off + ((uint)h_safe * (uint)W_IN + (uint)w_safe) * (uint)C + c] : 0.0f;
            float wv = w[((c * 7u) + (uint)kh) * 7u + (uint)kw];
            acc = fma(xv, wv, acc);
        }
    }
    y[batch_off + (h_out * (uint)W_OUT + w_out) * (uint)C + c] = acc + bias[c];
"""

_dw_k7_kernels: dict = {}


def _get_dw_k7_kernel(C: int, H_out: int, W_out: int):
    """Cache compiled kernels per (C, H_out, W_out). The template params
    bake those constants into the kernel source so the compiler can
    optimise the index math."""
    key = (C, H_out, W_out)
    k = _dw_k7_kernels.get(key)
    if k is None:
        k = mx.fast.metal_kernel(
            name=f"metalgrad_dwconv2d_k7_C{C}_H{H_out}_W{W_out}",
            input_names=["x", "w", "bias"],
            output_names=["y"],
            source=_DWCONV2D_K7_SRC,
        )
        _dw_k7_kernels[key] = k
    return k


def _supports_fast(x_shape, w_shape, stride: int, padding: int,
                    dilation: int) -> bool:
    """Disabled for v0.0.2.

    Our naive Metal kernel cannot beat mx.conv2d's depthwise path on
    representative FCDM/ConvNeXt shapes — mx (MPSGraph) appears to use
    threadgroup-shared input tiles and register tiling that we do not
    yet replicate. Measured: kernel 0.32-0.56x of mx on B>=1 H=32 C>=384
    workloads.

    Forward falls back to mx.conv2d. VJP unchanged (via mx.vjp on the
    same mx forward). A proper tiled kernel is v0.0.3 work; the
    framework + VJP are ready to consume it.
    """
    return False


def _dwconv2d_k7_fast(x: mx.array, w: mx.array, b: mx.array) -> mx.array:
    """Custom Metal forward for K=7 depthwise Conv2d. Input layout NHWC,
    padding=3 handled in-kernel (no mx.pad dispatch)."""
    N, H, W, C = x.shape

    kernel = _get_dw_k7_kernel(C, H, W)
    (y,) = kernel(
        inputs=[x, w, b],
        template=[("C", C), ("H_OUT", H), ("W_OUT", W), ("N", N)],
        grid=(C, W, H * N),
        threadgroup=(min(32, C), 1, 1),
        output_shapes=[(N, H, W, C)],
        output_dtypes=[x.dtype],
    )
    return y


# ─── differentiable op ───────────────────────────────────────────────────────

@differentiable
def _depthwise_conv2d_inner(x, w, b, stride, padding, dilation):
    C = x.shape[-1]
    s, p, d = int(stride), int(padding), int(dilation)
    if _supports_fast(x.shape, w.shape, s, p, d):
        return _dwconv2d_k7_fast(x, w, b)
    # Fallback for unsupported shapes / params.
    y = mx.conv2d(x, w, stride=s, padding=p, dilation=d, groups=C)
    return y + b


@_depthwise_conv2d_inner.vjp
def _depthwise_conv2d_vjp(primals, cotangent, output):
    x, w, b, stride, padding, dilation = primals
    gy = cotangent
    s, p, d = int(stride), int(padding), int(dilation)
    C = x.shape[-1]

    def _ref(xx, ww):
        return mx.conv2d(xx, ww, stride=s, padding=p, dilation=d, groups=C)

    _, (gx, gw) = mx.vjp(_ref, [x, w], [gy])
    gb = mx.sum(gy, axis=tuple(range(gy.ndim - 1)))
    return gx, gw, gb, None, None, None


def depthwise_conv2d(x: mx.array, w: mx.array, b: mx.array,
                     stride: int = 1, padding: int = 0,
                     dilation: int = 1) -> mx.array:
    return _depthwise_conv2d_inner(x, w, b, stride, padding, dilation)


__all__ = ["depthwise_conv2d"]
