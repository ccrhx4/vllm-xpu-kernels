# SPDX-License-Identifier: Apache-2.0
"""Correctness tests for the dense (non-grouped) interleaved SwiGLU/GELU
GEMM+activation fusion op (torch.ops._xpu_C.dense_swiglu_gemm_xe20_interleaved),
the dense-MLP analog of moe_grouped_mm_xe20_interleaved -- see
tests/fused_moe/test_moe_interleaved.py and /work/fusemlp/design.md.

The fused op is compared against the plain reference path (torch.matmul +
silu_and_mul/gelu_and_mul) for a range of M (batch size), including the
Qwen3.6-27B dense-FFN shape (hidden=5120, intermediate=17408).
"""
import pytest
import torch

import vllm_xpu_kernels._xpu_C  # noqa: F401
from tests.utils import seed_everything
from vllm_xpu_kernels.moe_utils import interleave_gate_up_weights_xe20

DEVICE = "xpu"

# (hidden, inter, m)
SHAPES = [
    (256, 512, 1),  # decode-shaped
    (256, 512, 8),
    (256, 512, 40),  # small prefill-shaped batch
    (512, 1024, 256),  # large prefill-shaped batch (config-1 tile)
    (5120, 17408, 1),  # Qwen3.6-27B dense-FFN shape, decode
    (5120, 17408, 32),  # Qwen3.6-27B dense-FFN shape, small prefill
    (5120, 17408, 512),  # Qwen3.6-27B dense-FFN shape, large prefill
]


def _ref_gemm_activation(acts, w13, bias, activation):
    """Reference: plain matmul (pre-transposed [K, N] weight) followed by
    the elementwise silu_and_mul/gelu_and_mul kernel."""
    gemm_out = torch.matmul(acts, w13.t())
    if bias is not None:
        gemm_out = gemm_out + bias
    n = gemm_out.shape[-1]
    ref_out = torch.empty(
        (*gemm_out.shape[:-1], n // 2), dtype=acts.dtype, device=acts.device)
    if activation == "silu":
        torch.ops._C.silu_and_mul(ref_out, gemm_out)
    else:
        torch.ops._C.gelu_and_mul(ref_out, gemm_out)
    return ref_out


def _run_case(hidden, inter, m, activation, with_bias, tile_id_override=-1):
    seed_everything(0)
    k = hidden
    n_full = 2 * inter

    weight = torch.randn(
        n_full, k, dtype=torch.bfloat16, device=DEVICE) * 0.02
    acts = torch.randn(m, k, dtype=torch.bfloat16, device=DEVICE) * 0.02

    bias = None
    if with_bias:
        bias = torch.randn(n_full, dtype=torch.bfloat16, device=DEVICE) * 0.01

    ref_out = _ref_gemm_activation(acts, weight, bias, activation)

    # interleave_gate_up_weights_xe20 expects an [E, N] / [E, N, K] tensor;
    # a dense weight degenerates to the E=1 case via unsqueeze/squeeze.
    weight_interleaved = interleave_gate_up_weights_xe20(
        weight.unsqueeze(0).clone()).squeeze(0)
    bias_interleaved = None
    if bias is not None:
        bias_interleaved = interleave_gate_up_weights_xe20(
            bias.clone().to(torch.float32).unsqueeze(0)).squeeze(0)

    fused_out = torch.empty((m, inter), dtype=torch.bfloat16, device=DEVICE)
    activation_type = 0 if activation == "silu" else 1
    torch.ops._xpu_C.dense_swiglu_gemm_xe20_interleaved(
        output=fused_out,
        activations=acts,
        weight=weight_interleaved,
        bias=bias_interleaved,
        activation_type=activation_type,
        gemm1_alpha=1.702,
        gemm1_limit=7.0,
        tile_id_override=tile_id_override)

    torch.testing.assert_close(
        ref_out.float(), fused_out.float(), atol=2e-2, rtol=2e-2)


@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("activation", ["silu", "gelu"])
def test_dense_mlp_interleaved(shape, activation):
    hidden, inter, m = shape
    with_bias = activation == "gelu"
    _run_case(hidden, inter, m, activation, with_bias)


# Explicit per-tile correctness coverage (tile_id_override 0-8, see
# dense_select_tile()/launch_dense_interleaved() in dense_mlp_interleaved.cpp).
# Tiles 6/7/8 were added when investigating the "wave-count quantization"
# gap at small-N/moderate-M shapes (e.g. TP4 M=256/512/1024); tile 8
# (128x128, SG 4x4x1, swizzled) is now selected by dense_select_tile() for
# those shapes, replacing tile 5 there -- exercised explicitly here since the
# auto-select path alone wouldn't otherwise cover every tile id.
ALL_TILE_IDS = [0, 1, 2, 3, 4, 5, 6, 7, 8]


@pytest.mark.parametrize("tile_id", ALL_TILE_IDS)
def test_dense_mlp_interleaved_all_tiles(tile_id):
    # Qwen3.6-27B TP4-shaped N (hidden=5120, per-rank intermediate=4352) at
    # M=512 -- large enough that every tile id (including the small M-tiles
    # 0-3, which are far from their intended M range here) still produces a
    # numerically valid, correctness-checkable result.
    _run_case(5120, 4352, 512, "silu", with_bias=False,
              tile_id_override=tile_id)


@pytest.mark.parametrize("tp_n", [(1, 34816), (2, 17408), (4, 8704)])
@pytest.mark.parametrize("m", [256, 512, 1024, 2048])
def test_dense_mlp_interleaved_wave_crossover_shapes(tp_n, m):
    # Correctness at the exact (TP, M) crossover shapes where
    # dense_select_tile() switches between tile 5 and tile 8 based on the
    # wave-inflation heuristic -- see the comment above dense_select_tile().
    _, n_full = tp_n
    _run_case(5120, n_full // 2, m, "silu", with_bias=False)


def test_dense_mlp_interleaved_bias_fp32_required():
    hidden, inter = 256, 512
    weight = torch.randn(2 * inter, hidden, dtype=torch.bfloat16, device=DEVICE)
    weight_interleaved = interleave_gate_up_weights_xe20(
        weight.unsqueeze(0)).squeeze(0)
    m = 8
    acts = torch.randn(m, hidden, dtype=torch.bfloat16, device=DEVICE)
    fused_out = torch.empty((m, inter), dtype=torch.bfloat16, device=DEVICE)
    # bf16 bias must be rejected -- the kernel requires fp32.
    bad_bias = torch.randn(2 * inter, dtype=torch.bfloat16, device=DEVICE)
    with pytest.raises(RuntimeError):
        torch.ops._xpu_C.dense_swiglu_gemm_xe20_interleaved(
            output=fused_out,
            activations=acts,
            weight=weight_interleaved,
            bias=bad_bias,
            activation_type=0,
            gemm1_alpha=1.702,
            gemm1_limit=7.0)
