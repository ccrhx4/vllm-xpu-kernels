# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Correctness check for the Triton decode path (`causal_conv1d_update` +
`fused_recurrent_gated_delta_rule_packed_decode`, both imported from the
`vllm` package) used in `benchmark/benchmark_gdn_attn_triton.py`, against
the same `ref_gdn_attention` reference used to validate the native SYCL
`gdn_attention` decode kernel.

Only the decode shape (exactly one new token per sequence, every sequence
continuing an existing state) is covered here -- this mirrors the packed
decode fast path used in production
(`Qwen3NextGatedDeltaNet._forward_core_decode_non_spec`).
"""

import random

import pytest
import torch
from vllm.model_executor.layers.mamba.ops.causal_conv1d import (
    causal_conv1d_update)
from vllm.third_party.flash_linear_attention.ops.fused_recurrent import (
    fused_recurrent_gated_delta_rule_packed_decode)

from tests.gdn_attn.test_gdn_attn import _extract_qkv_b_a_z, ref_gdn_attention
from tests.utils import format_tc

BATCH_SIZE = [1, 8, 32]
NUM_K_HEADS = [16]
NUM_K_DIMS = [128]
NUM_V_HEADS = [32]
NUM_V_DIMS = [128]
WIDTH = [4]
DTYPES = [torch.bfloat16]


def _extract_qkv_b_a(projected_states_qkvz, projected_states_ba, num_k_heads,
                     num_v_heads, head_k_dim, head_v_dim,
                     num_actual_tokens):
    """`gdn_attention`'s native op input layout groups b/a per k-head
    (`[b_1..b_rep, a_1..a_rep]` repeated per k-head), not a flat
    `ba.chunk(2, dim=-1)` -- reuse the exact reference splitting logic from
    `ref_gdn_attention` (`reorder_input=False`, `tp_size=1`) so the Triton
    inputs match what the native SYCL kernel (and its reference) consume."""
    qkv, b, a, _z = _extract_qkv_b_a_z(projected_states_qkvz,
                                       projected_states_ba,
                                       num_actual_tokens, num_k_heads,
                                       num_v_heads, head_k_dim, head_v_dim,
                                       tp_size=1, reorder_input=False)
    return qkv.contiguous(), b.contiguous(), a.contiguous()


@pytest.mark.parametrize("batch_size", BATCH_SIZE)
@pytest.mark.parametrize("num_k_heads", NUM_K_HEADS)
@pytest.mark.parametrize("head_k_dim", NUM_K_DIMS)
@pytest.mark.parametrize("num_v_heads", NUM_V_HEADS)
@pytest.mark.parametrize("head_v_dim", NUM_V_DIMS)
@pytest.mark.parametrize("width", WIDTH)
@pytest.mark.parametrize("dtype", DTYPES, ids=format_tc)
@torch.inference_mode()
def test_gdn_attention_triton_decode(batch_size, num_k_heads, head_k_dim,
                                     num_v_heads, head_v_dim, width, dtype):
    device = "xpu"
    random.seed(42)
    torch.manual_seed(42)
    tp_size = 1

    num_actual_tokens = batch_size
    cache_batch_size = 200

    mixed_qkvz_size = num_k_heads * (2 * head_k_dim +
                                     2 * head_v_dim * num_v_heads //
                                     num_k_heads)
    mixed_ba_size = num_k_heads * (2 * num_v_heads // num_k_heads)
    mixed_qkv_size = num_k_heads * (2 * head_k_dim +
                                    head_v_dim * num_v_heads // num_k_heads)

    projected_states_qkvz = torch.randn((num_actual_tokens, mixed_qkvz_size),
                                        dtype=dtype,
                                        device=device)
    projected_states_ba = torch.randn((num_actual_tokens, mixed_ba_size),
                                      dtype=dtype,
                                      device=device)

    # Decode always continues an existing sequence, so every row has a
    # valid initial conv/ssm state -- the Triton packed-decode kernels have
    # no notion of "no initial state" (unlike the native kernel's
    # has_initial_state flag).
    conv_state = torch.randn((cache_batch_size, width - 1, mixed_qkv_size),
                             dtype=dtype,
                             device=device)
    ref_conv_state = conv_state.clone()
    # Qwen3.5 forces the recurrent/temporal SSM state to fp32 by default
    # (vLLM applies the HF config's `mamba_ssm_dtype` as
    # `cache_config.mamba_ssm_cache_dtype`); the conv state stays bf16.
    ssm_state = torch.randn(
        (cache_batch_size, num_v_heads, head_v_dim, head_k_dim),
        dtype=torch.float32,
        device=device)
    ref_ssm_state = ssm_state.clone()

    conv_weights = torch.randn((mixed_qkv_size, width),
                               dtype=dtype,
                               device=device)
    conv_bias = torch.randn((mixed_qkv_size, ), dtype=dtype, device=device)

    A_log = torch.randn((num_v_heads, ), dtype=torch.float32, device=device)
    dt_bias = torch.randn((num_v_heads, ), dtype=dtype, device=device)

    non_spec_query_start_loc = torch.arange(0,
                                            batch_size + 1,
                                            dtype=torch.int32,
                                            device=device)
    has_initial_state = torch.ones(batch_size, dtype=torch.bool,
                                   device=device)
    # Index 0 is a reserved NULL_BLOCK_ID in vLLM's Triton decode kernels
    # (see vllm.v1.attention.backends.utils.NULL_BLOCK_ID) -- a row mapped
    # to it is silently skipped, so avoid it here to get a real comparison.
    non_spec_state_indices_tensor = torch.tensor(random.sample(
        range(1, cache_batch_size), batch_size),
                                                 device=device,
                                                 dtype=torch.int32)

    # ---- Triton decode path ----
    mixed_qkv, b, a = _extract_qkv_b_a(projected_states_qkvz,
                                       projected_states_ba, num_k_heads,
                                       num_v_heads, head_k_dim, head_v_dim,
                                       num_actual_tokens)
    conv_state_t = conv_state.transpose(1, 2).contiguous()
    out = torch.zeros((num_actual_tokens, 1, num_v_heads, head_v_dim),
                      dtype=dtype,
                      device=device)

    mixed_qkv_out = causal_conv1d_update(
        mixed_qkv,
        conv_state_t,
        conv_weights,
        conv_bias,
        "silu",
        conv_state_indices=non_spec_state_indices_tensor,
        validate_data=True,
    )
    fused_recurrent_gated_delta_rule_packed_decode(
        mixed_qkv=mixed_qkv_out,
        a=a,
        b=b,
        A_log=A_log,
        dt_bias=dt_bias,
        scale=head_k_dim**-0.5,
        initial_state=ssm_state,
        out=out,
        ssm_state_indices=non_spec_state_indices_tensor,
        use_qk_l2norm_in_kernel=True,
    )
    core_attn_out = out.reshape(num_actual_tokens, num_v_heads, head_v_dim)
    conv_state = conv_state_t.transpose(1, 2)

    # ---- reference ----
    ref_core_attn_out = torch.zeros_like(core_attn_out)
    ref_z = torch.empty_like(core_attn_out)
    ref_gdn_attention(
        ref_core_attn_out,
        ref_z,
        projected_states_qkvz,
        projected_states_ba,
        num_k_heads,
        num_v_heads,
        head_k_dim,
        head_v_dim,
        conv_state=ref_conv_state,
        ssm_state=ref_ssm_state,
        conv_weights=conv_weights,
        conv_bias=conv_bias,
        activation="silu",
        A_log=A_log,
        dt_bias=dt_bias,
        num_prefills=0,
        num_decodes=batch_size,
        has_initial_state=has_initial_state,
        non_spec_query_start_loc=non_spec_query_start_loc,
        non_spec_state_indices_tensor=non_spec_state_indices_tensor,
        num_actual_tokens=num_actual_tokens,
        tp_size=tp_size,
        reorder_input=False,
    )

    atol = 5e-2
    rtol = 5e-2
    torch.testing.assert_close(core_attn_out,
                               ref_core_attn_out,
                               atol=atol,
                               rtol=rtol,
                               equal_nan=True)
    for i in range(batch_size):
        state_id = non_spec_state_indices_tensor[i]
        torch.testing.assert_close(conv_state[state_id],
                                   ref_conv_state[state_id],
                                   atol=atol,
                                   rtol=rtol)
        torch.testing.assert_close(ssm_state[state_id],
                                   ref_ssm_state[state_id],
                                   atol=atol,
                                   rtol=rtol)
