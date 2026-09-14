# SPDX-License-Identifier: Apache-2.0
"""Correctness tests for the interleaved SwiGLU/GELU MoE GEMM1 fusion op
(torch.ops._xpu_C.moe_grouped_mm_xe20_interleaved), ported from
sgl-kernel-xpu per /work/fusemlp/design.md.

The fused op is compared against the existing unfused reference path
(cutlass_grouped_gemm_interface + silu_and_mul/gelu_and_mul) for a range
of avg_m (tokens-per-expert), including uneven per-expert distributions
and both silu (no bias) and gelu (with fp32 bias) configurations.
"""
import pytest
import torch

import vllm_xpu_kernels._xpu_C  # noqa: F401
from tests.utils import seed_everything
from vllm_xpu_kernels.moe_utils import interleave_gate_up_weights_xe20

DEVICE = "xpu"

# (num_experts, hidden, inter, tokens_per_expert)
UNIFORM_SHAPES = [
    (8, 256, 512, 1),  # decode-shaped, avg_m below the config-1 threshold
    (8, 256, 512, 40),  # small prefill-shaped batch
    (8, 512, 1024, 256),  # large prefill-shaped batch (config-1 tile)
]

MINI_PYTEST_PARAMS = {
    "default": {
        "shape": [(8, 256, 512, 8)],
    },
}


def _ref_gemm1_activation(acts, w13, bias, rows_per_expert, num_experts, n,
                          k, activation):
    """Reference: unfused grouped-gemm (pre-transposed [E, K, N] weights)
    followed by the elementwise silu_and_mul/gelu_and_mul kernel."""
    w13_t = w13.transpose(-1, -2).contiguous()
    total_m = acts.shape[0]
    gemm1_out = torch.empty((total_m, n), dtype=acts.dtype, device=acts.device)
    torch.ops._xpu_C.cutlass_grouped_gemm_interface(
        ptr_A=acts,
        ptr_A_scale=None,
        ptr_B=w13_t,
        ptr_B_scale=None,
        ptr_bias=bias,
        ptr_D=gemm1_out,
        rows_per_expert=rows_per_expert,
        N=n,
        K=k,
        num_experts=num_experts)
    ref_out = torch.empty((total_m, n // 2), dtype=acts.dtype, device=acts.device)
    if activation == "silu":
        torch.ops._C.silu_and_mul(ref_out, gemm1_out)
    else:
        torch.ops._C.gelu_and_mul(ref_out, gemm1_out)
    return ref_out


def _run_case(num_experts, hidden, inter, tokens_per_expert, activation,
             with_bias, uneven=False):
    seed_everything(0)
    k = hidden
    n_full = 2 * inter

    w13 = torch.randn(
        num_experts, n_full, k, dtype=torch.bfloat16, device=DEVICE) * 0.02

    if uneven:
        # Skew tokens across experts (still summing to a fixed total),
        # exercising the kernel's per-expert-tile-size handling.
        base = max(tokens_per_expert - 4, 0)
        counts = [
            base + (i % 9) for i in range(num_experts)
        ]
    else:
        counts = [tokens_per_expert] * num_experts
    rows_per_expert = torch.tensor(counts, dtype=torch.int32, device=DEVICE)
    total_m = sum(counts)

    acts = torch.randn(
        total_m, k, dtype=torch.bfloat16, device=DEVICE) * 0.02

    bias = None
    if with_bias:
        # Bias lives in the model's native (bf16) params_dtype on the
        # unfused reference path; the interleaved kernel requires its own
        # fp32 copy (see interleave_gate_up_weights_xe20's docstring / the
        # vllm hook's `.to(torch.float32)` cast), so the two paths
        # intentionally use different bias dtypes here, matching
        # production usage.
        bias = torch.randn(
            num_experts, n_full, dtype=torch.bfloat16, device=DEVICE) * 0.01

    ref_out = _ref_gemm1_activation(acts, w13, bias, rows_per_expert,
                                    num_experts, n_full, k, activation)

    w13_interleaved = interleave_gate_up_weights_xe20(w13.clone())
    bias_interleaved = None
    if bias is not None:
        bias_interleaved = interleave_gate_up_weights_xe20(
            bias.clone().to(torch.float32))

    fused_out = torch.empty(
        (total_m, inter), dtype=torch.bfloat16, device=DEVICE)
    activation_type = 0 if activation == "silu" else 1
    torch.ops._xpu_C.moe_grouped_mm_xe20_interleaved(
        output=fused_out,
        activations=acts,
        weights=w13_interleaved,
        bias=bias_interleaved,
        rows_per_expert=rows_per_expert,
        num_experts=num_experts,
        activation_type=activation_type,
        gemm1_alpha=1.702,
        gemm1_limit=7.0)

    torch.testing.assert_close(
        ref_out.float(), fused_out.float(), atol=2e-2, rtol=2e-2)


@pytest.mark.parametrize("shape", UNIFORM_SHAPES)
@pytest.mark.parametrize("activation", ["silu", "gelu"])
def test_moe_interleaved_uniform(shape, activation):
    num_experts, hidden, inter, tokens_per_expert = shape
    with_bias = activation == "gelu"
    _run_case(num_experts, hidden, inter, tokens_per_expert, activation,
             with_bias, uneven=False)


@pytest.mark.parametrize("shape", UNIFORM_SHAPES)
def test_moe_interleaved_uneven_rows_per_expert(shape):
    num_experts, hidden, inter, tokens_per_expert = shape
    _run_case(num_experts, hidden, inter, tokens_per_expert, "silu", False,
             uneven=True)


def test_moe_interleaved_bias_fp32_required():
    num_experts, hidden, inter = 8, 256, 512
    w13 = torch.randn(
        num_experts, 2 * inter, hidden, dtype=torch.bfloat16, device=DEVICE)
    w13_interleaved = interleave_gate_up_weights_xe20(w13)
    rows_per_expert = torch.full((num_experts, ), 8, dtype=torch.int32,
                                 device=DEVICE)
    acts = torch.randn(
        num_experts * 8, hidden, dtype=torch.bfloat16, device=DEVICE)
    fused_out = torch.empty(
        (num_experts * 8, inter), dtype=torch.bfloat16, device=DEVICE)
    # bf16 bias must be rejected -- the kernel requires fp32.
    bad_bias = torch.randn(
        num_experts, 2 * inter, dtype=torch.bfloat16, device=DEVICE)
    with pytest.raises(RuntimeError):
        torch.ops._xpu_C.moe_grouped_mm_xe20_interleaved(
            output=fused_out,
            activations=acts,
            weights=w13_interleaved,
            bias=bad_bias,
            rows_per_expert=rows_per_expert,
            num_experts=num_experts,
            activation_type=0,
            gemm1_alpha=1.702,
            gemm1_limit=7.0)
