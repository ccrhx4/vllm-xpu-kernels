# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# ruff: noqa: E402
"""
Side-by-side SYCL vs. Triton decode benchmark for Gated DeltaNet, with a
fix for a benchmark-pattern bug present in `benchmark_gdn_attn.py` /
`benchmark_gdn_attn_triton.py`: those harnesses build a single fixed
`state_indices` tensor per config and reuse the *exact same* physical
cache rows (conv_state / ssm_state) for all timed iterations.

That is unrepresentative of real serving (where each decode step touches
different requests/rows) and, worse, it lets those rows' content
degenerate/converge over hundreds of repeated in-place updates -- which is
far more compressible (and hence gives an inflated benefit from GPU buffer
compression, e.g. `RenderCompressedBuffersEnabled`) than real traffic.

Fix: `cache_batch_size = max(256, batch_size * 2)` in `make_inputs()`
already allocates more cache rows than a single decode step touches.
We use that headroom to pre-build N = (cache_batch_size - 1) // batch_size
disjoint row-index groups (a random permutation of the cache, excluding
row 0 which is a reserved NULL_BLOCK_ID for the Triton decode kernels)
*once* before timing, then cycle through them (`groups[i % N]`) inside the
timed loop. This is a plain CPU list-index lookup -- no extra GPU kernel
launch and no extra memory beyond N small int32 tensors of size
batch_size -- so it does not distort the timing the way re-randomizing
tensor contents in-loop (e.g. `.normal_()`) would.

This was empirically validated to reproduce the same qualitative results
as the stock harnesses (SYCL is largely compression-insensitive; Triton's
tuned decode path is measurably compression-sensitive), confirming that
sensitivity is a genuine property of the kernels' access pattern and not
an artifact of the stock benchmarks' degenerate row reuse.

Usage:
    python benchmark_gdn_attn_decode_rotated.py

Set `NEOReadDebugKeys=1` and `RenderCompressedBuffersEnabled=0` in the
environment to compare against GPU buffer compression disabled.
"""
# isort: off
import torch

from utils import bootstrap_benchmark_env

bootstrap_benchmark_env(__file__)

from tests.utils import seed_everything
from benchmark_gdn_attn import MODEL_SHAPES, WORKLOADS, make_inputs
from benchmark_gdn_attn_triton import (
    DECODE_WORKLOADS,
    _make_triton_decode_inputs,
)

from vllm.model_executor.layers.mamba.ops.causal_conv1d import (
    causal_conv1d_update,
)
from vllm.third_party.flash_linear_attention.ops.fused_recurrent import (
    fused_recurrent_gated_delta_rule_packed_decode,
)
# isort: on

WARMUP = 50
ITERATIONS = 200


def build_rotating_groups(cache_batch_size, batch_size, device):
    """Disjoint contiguous groups of `batch_size` rows from a random
    permutation of [1, cache_batch_size) (row 0 excluded: it is a reserved
    NULL_BLOCK_ID for the Triton decode kernels)."""
    n_groups = max(1, (cache_batch_size - 1) // batch_size)
    perm = torch.randperm(cache_batch_size - 1) + 1
    perm = perm[:n_groups * batch_size]
    groups = [
        perm[i * batch_size:(i + 1) * batch_size].to(device=device,
                                                       dtype=torch.int32)
        for i in range(n_groups)
    ]
    return groups


def bench_triton(shape_name, workload_name, dtype_str="bf16"):
    shape = next(s for s in MODEL_SHAPES if s.name == shape_name)
    workload = next(w for w in DECODE_WORKLOADS if w.name == workload_name)
    dtype = torch.bfloat16 if dtype_str == "bf16" else torch.float16

    kwargs = make_inputs(shape, workload, dtype)
    tkwargs = _make_triton_decode_inputs(kwargs)
    cache_batch_size = kwargs["conv_state"].shape[0]
    groups = build_rotating_groups(cache_batch_size, workload.batch_size,
                                    tkwargs["state_indices"].device)
    n = len(groups)

    def _run(i):
        state_indices = groups[i % n]
        mixed_qkv_out = causal_conv1d_update(
            tkwargs["mixed_qkv"],
            tkwargs["conv_state"],
            tkwargs["conv_weights"],
            tkwargs["conv_bias"],
            tkwargs["activation"],
            conv_state_indices=state_indices,
            validate_data=False,
        )
        fused_recurrent_gated_delta_rule_packed_decode(
            mixed_qkv=mixed_qkv_out,
            a=tkwargs["a"],
            b=tkwargs["b"],
            A_log=tkwargs["A_log"],
            dt_bias=tkwargs["dt_bias"],
            scale=tkwargs["scale"],
            initial_state=tkwargs["ssm_state"],
            out=tkwargs["out"],
            ssm_state_indices=state_indices,
            use_qk_l2norm_in_kernel=True,
        )

    for i in range(WARMUP):
        _run(i)
    torch.xpu.synchronize()

    start_event = torch.xpu.Event(enable_timing=True)
    end_event = torch.xpu.Event(enable_timing=True)
    start_event.record()
    for i in range(WARMUP, ITERATIONS):
        _run(i)
    end_event.record()
    torch.xpu.synchronize()
    ms = start_event.elapsed_time(end_event) / (ITERATIONS - WARMUP)
    return 1000 * ms  # us


def bench_sycl(shape_name, workload_name, dtype_str="bf16"):
    shape = next(s for s in MODEL_SHAPES if s.name == shape_name)
    workload = next(w for w in WORKLOADS
                     if w.name == workload_name and w.mode == "decode")
    dtype = torch.bfloat16 if dtype_str == "bf16" else torch.float16

    kwargs = make_inputs(shape, workload, dtype)
    cache_batch_size = kwargs["conv_state"].shape[0]
    groups = build_rotating_groups(
        cache_batch_size, workload.batch_size,
        kwargs["non_spec_state_indices_tensor"].device)
    n = len(groups)

    def _run(i):
        state_indices = groups[i % n]
        intermediates = torch.ops._xpu_C.causal_conv1d_non_spec(
            kwargs["z"],
            kwargs["projected_states_qkvz"],
            kwargs["projected_states_ba"],
            kwargs["num_k_heads"],
            kwargs["num_v_heads"],
            kwargs["head_k_dim"],
            kwargs["head_v_dim"],
            conv_state=kwargs["conv_state"],
            conv_weights=kwargs["conv_weights"],
            conv_bias=kwargs["conv_bias"],
            activation=kwargs["activation"],
            num_prefills=kwargs["num_prefills"],
            num_decodes=kwargs["num_decodes"],
            num_spec_decodes=kwargs["num_spec_decodes"],
            has_initial_state=kwargs["has_initial_state"],
            non_spec_query_start_loc=kwargs["non_spec_query_start_loc"],
            non_spec_token_indx=kwargs["non_spec_token_indx"],
            non_spec_state_indices_tensor=state_indices,
            num_actual_tokens=kwargs["num_actual_tokens"],
            tp_size=kwargs["tp_size"],
            reorder_input=kwargs["reorder_input"])
        torch.ops._xpu_C.gated_delta_rule_non_spec(
            kwargs["core_attn_out"],
            *intermediates,
            kwargs["num_v_heads"],
            kwargs["head_v_dim"],
            A_log=kwargs["A_log"],
            dt_bias=kwargs["dt_bias"],
            ssm_state=kwargs["ssm_state"],
            num_prefills=kwargs["num_prefills"],
            num_decodes=kwargs["num_decodes"],
            num_spec_decodes=kwargs["num_spec_decodes"],
            has_initial_state=kwargs["has_initial_state"],
            non_spec_query_start_loc=kwargs["non_spec_query_start_loc"],
            non_spec_token_indx=kwargs["non_spec_token_indx"],
            non_spec_state_indices_tensor=state_indices,
            num_actual_tokens=kwargs["num_actual_tokens"],
            tp_size=kwargs["tp_size"])

    for i in range(WARMUP):
        _run(i)
    torch.xpu.synchronize()

    start_event = torch.xpu.Event(enable_timing=True)
    end_event = torch.xpu.Event(enable_timing=True)
    start_event.record()
    for i in range(WARMUP, ITERATIONS):
        _run(i)
    end_event.record()
    torch.xpu.synchronize()
    ms = start_event.elapsed_time(end_event) / (ITERATIONS - WARMUP)
    return 1000 * ms  # us


if __name__ == "__main__":
    seed_everything(1234)
    torch.set_default_device("xpu")
    torch.xpu.set_device("xpu:0")

    decode_workload_names = [w.name for w in WORKLOADS if w.mode == "decode"]
    configs = [(s.name, w) for s in MODEL_SHAPES for w in decode_workload_names]
    print("shape,workload,sycl_us,triton_us", flush=True)
    for shape_name, workload_name in configs:
        sycl_us = bench_sycl(shape_name, workload_name)
        triton_us = bench_triton(shape_name, workload_name)
        print(f"{shape_name},{workload_name},{sycl_us:.3f},{triton_us:.3f}",
              flush=True)
