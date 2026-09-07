# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# ruff: noqa: E402
"""
Benchmark the Triton decode path for Gated DeltaNet linear attention (used by
Qwen3-Next / Qwen3.5), as an alternative to the native SYCL
`torch.ops._xpu_C.gdn_attention` decode path benchmarked in
`benchmark_gdn_attn.py`.

On CUDA, vLLM's decode fast path
(`Qwen3NextGatedDeltaNet._forward_core_decode_non_spec` in
`vllm/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py`) runs:
    causal_conv1d_update(...)                              (Triton)
    fused_recurrent_gated_delta_rule_packed_decode(...)     (Triton, vendored
                                                              FLA op)
This benchmark drives the same two ops directly (imported from the `vllm`
package) using the same shapes/workloads/input-construction as
`benchmark_gdn_attn.py`'s decode workloads, so the two benchmarks are
directly comparable.

Only decode workloads (`mode == "decode"`, exactly one new token per
sequence) are covered here -- prefill/mix/spec are out of scope for this
comparison.

Output format matches benchmark_gdn_attn.py
(triton.testing.perf_report -> nightly picks up the auto-generated CSV).
"""
# isort: off
import gc

import torch
import triton
import triton.testing

from utils import bootstrap_benchmark_env, ensure_save_path_exists

bootstrap_benchmark_env(__file__)

import vllm_xpu_kernels._xpu_C  # noqa: F401
from benchmark.presets import get_hardware_preset
from tests.utils import parse_args, seed_everything
# Reuse the model shapes, workloads and input construction from the fused
# gdn_attention benchmark so the two stay in lockstep.
from benchmark_gdn_attn import (
    MODEL_SHAPES,
    WORKLOADS,
    estimate_bytes_moved,
    estimate_flops,
    make_inputs,
)
from tests.gdn_attn.test_gdn_attn import _extract_qkv_b_a_z

from vllm.model_executor.layers.mamba.ops.causal_conv1d import (
    causal_conv1d_update,
)
from vllm.third_party.flash_linear_attention.ops.fused_recurrent import (
    fused_recurrent_gated_delta_rule_packed_decode,
)
# isort: on

DEVICE = "xpu"
WARMUP = 50

# This benchmark only makes sense for the native decode path (T=1/seq); the
# Triton packed-decode kernel used here does not cover prefill/mix/spec.
DECODE_WORKLOADS = [w for w in WORKLOADS if w.mode == "decode"]


def clear_xpu_cache():
    torch.xpu.empty_cache()
    torch.xpu.synchronize()
    gc.collect()


# ----------------------------------------------------------------------------
# qkvz/ba splitting -- reuses `_extract_qkv_b_a_z`'s exact reference logic
# (from tests/gdn_attn/test_gdn_attn.py, reorder_input=False) since the
# native op's `projected_states_ba` groups b/a per k-head
# (`[b_1..b_rep, a_1..a_rep]` repeated per k-head), not a flat
# `ba.chunk(2, dim=-1)`.
# ----------------------------------------------------------------------------
def _extract_qkv_b_a(kwargs):
    num_k_heads = kwargs["num_k_heads"]
    num_v_heads = kwargs["num_v_heads"]
    head_k_dim = kwargs["head_k_dim"]
    head_v_dim = kwargs["head_v_dim"]
    tp_size = kwargs["tp_size"]
    num_actual_tokens = kwargs["num_actual_tokens"]

    key_dim = head_k_dim * num_k_heads // tp_size
    value_dim = head_v_dim * num_v_heads // tp_size
    qkv_size = key_dim * 2 + value_dim
    mixed_qkv, b, a, _z = _extract_qkv_b_a_z(
        kwargs["projected_states_qkvz"],
        kwargs["projected_states_ba"],
        num_actual_tokens,
        num_k_heads,
        num_v_heads,
        head_k_dim,
        head_v_dim,
        tp_size,
        reorder_input=False)
    assert mixed_qkv.shape == (num_actual_tokens, qkv_size)
    assert b.shape == (num_actual_tokens, num_v_heads // tp_size)
    return mixed_qkv.contiguous(), b.contiguous(), a.contiguous()


def _make_triton_decode_inputs(kwargs):
    """Adapt the shared `make_inputs()` tensors (native-kernel layout) to the
    layout expected by the Triton decode ops."""
    mixed_qkv, b, a = _extract_qkv_b_a(kwargs)

    # causal_conv1d_update expects conv_state as (..., dim, state_len);
    # make_inputs() builds it as (cache_batch, state_len, dim) (native
    # kernel layout), so transpose to the Triton layout.
    conv_state_t = kwargs["conv_state"].transpose(1, 2).contiguous()

    # Index 0 is a reserved NULL_BLOCK_ID in vLLM's Triton decode kernels
    # (vllm.v1.attention.backends.utils.NULL_BLOCK_ID): a row mapped to it
    # is silently skipped. make_inputs() samples state indices uniformly
    # from [0, cache_batch_size), so remap any 0 to an unused slot to keep
    # every row doing real work (the native kernel has no such reservation).
    state_indices = kwargs["non_spec_state_indices_tensor"].clone()
    zero_mask = state_indices == 0
    if zero_mask.any():
        cache_batch_size = kwargs["conv_state"].shape[0]
        used = set(state_indices.tolist())
        spare = next(i for i in range(1, cache_batch_size) if i not in used)
        state_indices[zero_mask.nonzero(as_tuple=True)[0][0]] = spare

    out = torch.zeros(
        (kwargs["num_actual_tokens"], 1, kwargs["num_v_heads"] //
         kwargs["tp_size"], kwargs["head_v_dim"]),
        dtype=mixed_qkv.dtype,
        device=DEVICE,
    )

    return dict(
        mixed_qkv=mixed_qkv,
        b=b,
        a=a,
        conv_state=conv_state_t,
        conv_weights=kwargs["conv_weights"],
        conv_bias=kwargs["conv_bias"],
        activation=kwargs["activation"],
        state_indices=state_indices,
        A_log=kwargs["A_log"],
        dt_bias=kwargs["dt_bias"],
        ssm_state=kwargs["ssm_state"],
        out=out,
        scale=kwargs["head_k_dim"]**-0.5,
    )


# ----------------------------------------------------------------------------
# Benchmark driver (mirrors benchmark_gdn_attn.py)
# ----------------------------------------------------------------------------
def benchmark_gdn_triton(shape_name, workload_name, dtype_str, provider,
                         iterations):
    shape = next(s for s in MODEL_SHAPES if s.name == shape_name)
    workload = next(w for w in DECODE_WORKLOADS if w.name == workload_name)
    dtype = torch.bfloat16 if dtype_str == "bf16" else torch.float16

    print(f"Running config: shape={shape.name}, workload={workload.name}, "
          f"dtype={dtype_str}, Provider: {provider}", flush=True)
    assert iterations > WARMUP, \
        "Number of iterations should be greater than WARMUP to account " \
        "for warmup"

    kwargs = make_inputs(shape, workload, dtype)
    tkwargs = _make_triton_decode_inputs(kwargs)

    def _run():
        mixed_qkv_out = causal_conv1d_update(
            tkwargs["mixed_qkv"],
            tkwargs["conv_state"],
            tkwargs["conv_weights"],
            tkwargs["conv_bias"],
            tkwargs["activation"],
            conv_state_indices=tkwargs["state_indices"],
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
            ssm_state_indices=tkwargs["state_indices"],
            use_qk_l2norm_in_kernel=True,
        )

    # warmup
    for _ in range(WARMUP):
        _run()
    torch.xpu.synchronize()

    start_event = torch.xpu.Event(enable_timing=True)
    end_event = torch.xpu.Event(enable_timing=True)
    start_event.record()
    for _ in range(WARMUP, iterations):
        _run()
    end_event.record()
    torch.xpu.synchronize()
    ms = start_event.elapsed_time(end_event) / (iterations - WARMUP)

    if provider == "gdn_triton":
        clear_xpu_cache()
        return 1000 * ms  # us
    if provider == "gdn_triton_memBandwidth":
        bytes_moved = estimate_bytes_moved(kwargs)
        clear_xpu_cache()
        return (bytes_moved / 1e9) / (ms / 1000)  # GB/s
    if provider == "gdn_triton_MBU":
        hardware_presets = get_hardware_preset(torch.xpu.get_device_name())
        if hardware_presets is None:
            clear_xpu_cache()
            return float("nan")
        peak_bw = hardware_presets["memory_bandwidth_GBs"]
        bytes_moved = estimate_bytes_moved(kwargs)
        bw = (bytes_moved / 1e9) / (ms / 1000)
        clear_xpu_cache()
        return (bw / peak_bw) * 100
    if provider == "gdn_triton_TFLOPS":
        flops = estimate_flops(kwargs)
        clear_xpu_cache()
        return flops / (ms / 1000) / 1e12
    raise ValueError(f"Unknown provider {provider}")


def get_benchmark(configs, iterations=200):

    @triton.testing.perf_report(
        triton.testing.Benchmark(
            x_names=["shape_name", "workload_name", "dtype_str"],
            x_vals=[tuple(c) for c in configs],
            line_arg="provider",
            line_vals=[
                "gdn_triton",
                "gdn_triton_memBandwidth",
                "gdn_triton_MBU",
                "gdn_triton_TFLOPS",
            ],
            line_names=[
                "GDN_Triton(us)",
                "GDN_Triton_memBandwidth(GB/s)",
                "GDN_Triton_MBU (%)",
                "GDN_Triton_TFLOPS",
            ],
            styles=[("blue", "-"), ("purple", "-"), ("red", "-"),
                    ("green", "-")],
            ylabel="Latency (us)",
            plot_name="gdn-attn-triton-decode",
            args={},
        ))
    def benchmark(shape_name, workload_name, dtype_str, provider):
        return benchmark_gdn_triton(shape_name=shape_name,
                                    workload_name=workload_name,
                                    dtype_str=dtype_str,
                                    provider=provider,
                                    iterations=iterations)

    return benchmark


def gen_perf_configs(dtype_str="bf16"):
    return [(s.name, w.name, dtype_str)
            for s in MODEL_SHAPES for w in DECODE_WORKLOADS]


if __name__ == "__main__":
    args = parse_args()
    seed = 1234
    seed_everything(seed)
    iterations = 200
    torch.set_default_device("xpu")
    torch.xpu.set_device("xpu:0")

    configs = gen_perf_configs("bf16")
    benchmark = get_benchmark(configs, iterations=iterations)
    save_path = ensure_save_path_exists(args.save_path)
    benchmark.run(print_data=True, save_path=save_path)
