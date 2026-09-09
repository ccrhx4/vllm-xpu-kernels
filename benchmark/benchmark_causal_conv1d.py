# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# ruff: noqa: E402
"""
Benchmark `torch.ops._xpu_C.causal_conv1d` (the conv stage of Gated DeltaNet
linear attention, used by Qwen3-Next).

causal_conv1d is the first half of the split gdn_attention op (see
csrc/xpu/gdn_attn/gdn_attn_interface.cpp). It runs the depthwise causal conv1d
(+ activation), writes z, updates conv_state, and returns the intermediate
{q, k, v, b, a} tensors consumed by gated_delta_rule.

Workload modes (shared with benchmark_gdn_attn.py):
    - prefill : every sequence is a prefill chunk (varlen)  -> XE2 chunked path
    - decode  : every sequence is exactly 1 new token       -> native path
    - mix     : mix of prefill + decode                     -> XE2 chunked path
    - spec    : MTP / speculative decode (drafts per seq)   -> native path

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
    _bpe,
    make_inputs,
)
# Corrected decode timing helpers (rotate across fresh state buffers and
# disjoint cache-row groups so results are not inflated by L2 residency of
# a single reused input/state buffer).
from benchmark_gdn_attn_decode_rotated import (
    NUM_STATE_BUFFERS,
    build_rotating_groups,
    measure_rotated,
)
# isort: on

DEVICE = "xpu"


def clear_xpu_cache():
    torch.xpu.empty_cache()
    torch.xpu.synchronize()
    gc.collect()


# ----------------------------------------------------------------------------
# Memory / FLOPs models (conv stage only)
# ----------------------------------------------------------------------------
def estimate_bytes_moved(kwargs):
    """Lower-bound bytes moved by causal_conv1d for BW/MBU reporting.

    The conv stage reads projected_states_{qkvz,ba}, reads/writes conv_state,
    reads conv_weights/bias, and writes z plus the {q,k,v,b,a} intermediates.
    """
    bpe_in = _bpe(kwargs["projected_states_qkvz"].dtype)
    bpe_out = _bpe(kwargs["z"].dtype)

    n_tok = kwargs["num_actual_tokens"]
    nk = kwargs["num_k_heads"] // kwargs["tp_size"]
    nv = kwargs["num_v_heads"] // kwargs["tp_size"]
    hk = kwargs["head_k_dim"]
    hv = kwargs["head_v_dim"]
    width = kwargs["conv_weights"].shape[-1]

    qkvz_per_tok = nk * (2 * hk + 2 * hv * nv // nk)
    ba_per_tok = nk * (2 * nv // nk)
    qkv_per_tok = nk * (2 * hk + hv * nv // nk)
    z_per_tok = nv * hv
    conv_state_len = kwargs["conv_state"].shape[1]
    # {q,k,v} (=qkv_per_tok) plus b, a (2*nv) intermediates written out.
    inter_per_tok = qkv_per_tok + 2 * nv

    batch = max(1,
                kwargs["num_prefills"] + kwargs["num_decodes"]
                + kwargs["num_spec_decodes"])

    bytes_in = n_tok * (qkvz_per_tok + ba_per_tok) * bpe_in
    bytes_z = n_tok * z_per_tok * bpe_out
    bytes_inter = n_tok * inter_per_tok * bpe_out
    bytes_conv_state = batch * conv_state_len * qkv_per_tok * bpe_in * 2
    bytes_weights = qkv_per_tok * width * bpe_in
    # Bias is only read when conv_bias is provided.
    bytes_bias = qkv_per_tok * bpe_in if kwargs["conv_bias"] is not None else 0

    return (bytes_in + bytes_z + bytes_inter
            + bytes_conv_state + bytes_weights + bytes_bias)


def estimate_flops(kwargs):
    """Lower-bound FLOPs for the conv stage (per-token depthwise MAC).

    Conv1d:  2 * n_tok * mixed_qkv_size * width
    """
    n_tok = kwargs["num_actual_tokens"]
    nk = kwargs["num_k_heads"] // kwargs["tp_size"]
    nv = kwargs["num_v_heads"] // kwargs["tp_size"]
    hk = kwargs["head_k_dim"]
    hv = kwargs["head_v_dim"]
    width = kwargs["conv_weights"].shape[-1]

    qkv_per_tok = nk * (2 * hk + hv * nv // nk)
    return 2 * n_tok * qkv_per_tok * width


# ----------------------------------------------------------------------------
# Corrected decode timing (rotated buffers)
# ----------------------------------------------------------------------------
def _measure_conv_decode_rotated(shape, workload, dtype):
    """Rotated-buffer per-iteration timing (us) for decode conv1d.

    The single-buffer loop reuses one conv_state / input set and one set of
    physical cache rows every iteration. On XPU those stay resident in the
    large last-level cache, so DRAM traffic is understated and latency is
    ~1.1-1.3x optimistic (the effect is independent of buffer compression).
    Rotate across NUM_STATE_BUFFERS freshly built datasets and disjoint
    cache-row groups, refreshing the in-place-updated conv_state between
    timed rounds (see measure_rotated). Returns (us_per_iter, kwargs) where
    kwargs is a representative dataset for the byte/FLOP models.
    """
    K = NUM_STATE_BUFFERS
    datasets = [make_inputs(shape, workload, dtype) for _ in range(K)]
    cache_batch_size = datasets[0]["conv_state"].shape[0]
    groups = build_rotating_groups(
        cache_batch_size, workload.batch_size,
        datasets[0]["non_spec_state_indices_tensor"].device)
    n = len(groups)

    def _run(i):
        kw = datasets[i % K]
        state_indices = groups[(i // K) % n]
        torch.ops._xpu_C.causal_conv1d_non_spec(
            kw["z"],
            kw["projected_states_qkvz"],
            kw["projected_states_ba"],
            kw["num_k_heads"],
            kw["num_v_heads"],
            kw["head_k_dim"],
            kw["head_v_dim"],
            conv_state=kw["conv_state"],
            conv_weights=kw["conv_weights"],
            conv_bias=kw["conv_bias"],
            activation=kw["activation"],
            num_prefills=kw["num_prefills"],
            num_decodes=kw["num_decodes"],
            num_spec_decodes=kw["num_spec_decodes"],
            has_initial_state=kw["has_initial_state"],
            non_spec_query_start_loc=kw["non_spec_query_start_loc"],
            non_spec_token_indx=kw["non_spec_token_indx"],
            non_spec_state_indices_tensor=state_indices,
            num_actual_tokens=kw["num_actual_tokens"],
            tp_size=kw["tp_size"],
            reorder_input=kw["reorder_input"])

    mutated = [d["conv_state"] for d in datasets]
    us = measure_rotated(_run, mutated_state=mutated, n_groups=K * n)
    return us, datasets[0]


# ----------------------------------------------------------------------------
# Benchmark driver (mirrors benchmark_gdn_attn.py)
# ----------------------------------------------------------------------------
def benchmark_causal_conv1d(shape_name, workload_name, dtype_str, provider,
                            iterations):
    shape = next(s for s in MODEL_SHAPES if s.name == shape_name)
    workload = next(w for w in WORKLOADS if w.name == workload_name)
    dtype = torch.bfloat16 if dtype_str == "bf16" else torch.float16

    print(f"Running config: shape={shape.name}, workload={workload.name}, "
          f"dtype={dtype_str}, Provider: {provider}", flush=True)
    assert iterations > 5, \
        "Number of iterations should be greater than 5 to account for warmup"

    kwargs = make_inputs(shape, workload, dtype)

    if kwargs["num_spec_decodes"] > 0:
        def _run():
            torch.ops._xpu_C.causal_conv1d_spec(
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
                spec_query_start_loc=kwargs["spec_query_start_loc"],
                spec_token_indx=kwargs["spec_token_indx"],
                spec_state_indices_tensor=kwargs["spec_state_indices_tensor"],
                num_accepted_tokens=kwargs["num_accepted_tokens"],
                num_actual_tokens=kwargs["num_actual_tokens"],
                tp_size=kwargs["tp_size"],
                reorder_input=kwargs["reorder_input"])
    else:
        def _run():
            torch.ops._xpu_C.causal_conv1d_non_spec(
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
                non_spec_state_indices_tensor=kwargs[
                    "non_spec_state_indices_tensor"],
                num_actual_tokens=kwargs["num_actual_tokens"],
                tp_size=kwargs["tp_size"],
                reorder_input=kwargs["reorder_input"])

    # warmup
    for _ in range(5):
        _run()
    torch.xpu.synchronize()

    if workload.mode == "decode":
        # Decode reuses the recurrent conv_state in place; use the rotated
        # method so L2 residency of a single reused buffer does not inflate
        # the result. This also replaces the per-iteration byte model's
        # dataset with the rotated representative dataset.
        us, kwargs = _measure_conv_decode_rotated(shape, workload, dtype)
        ms = us / 1000.0
    else:
        start_event = torch.xpu.Event(enable_timing=True)
        end_event = torch.xpu.Event(enable_timing=True)
        start_event.record()
        for _ in range(5, iterations):
            _run()
        end_event.record()
        torch.xpu.synchronize()
        ms = start_event.elapsed_time(end_event) / (iterations - 5)

    if provider == "conv1d":
        clear_xpu_cache()
        return 1000 * ms  # us
    if provider == "conv1d_memBandwidth":
        bytes_moved = estimate_bytes_moved(kwargs)
        clear_xpu_cache()
        return (bytes_moved / 1e9) / (ms / 1000)  # GB/s
    if provider == "conv1d_MBU":
        hardware_presets = get_hardware_preset(torch.xpu.get_device_name())
        if hardware_presets is None:
            clear_xpu_cache()
            return float("nan")
        peak_bw = hardware_presets["memory_bandwidth_GBs"]
        bytes_moved = estimate_bytes_moved(kwargs)
        bw = (bytes_moved / 1e9) / (ms / 1000)
        clear_xpu_cache()
        return (bw / peak_bw) * 100
    if provider == "conv1d_TFLOPS":
        flops = estimate_flops(kwargs)
        clear_xpu_cache()
        return flops / (ms / 1000) / 1e12
    raise ValueError(f"Unknown provider {provider}")


def get_benchmark(configs, iterations=20):

    @triton.testing.perf_report(
        triton.testing.Benchmark(
            x_names=["shape_name", "workload_name", "dtype_str"],
            x_vals=[tuple(c) for c in configs],
            line_arg="provider",
            line_vals=[
                "conv1d",
                "conv1d_memBandwidth",
                "conv1d_MBU",
                "conv1d_TFLOPS",
            ],
            line_names=[
                "CausalConv1d(us)",
                "CausalConv1d_memBandwidth(GB/s)",
                "CausalConv1d_MBU (%)",
                "CausalConv1d_TFLOPS",
            ],
            styles=[("blue", "-"), ("purple", "-"), ("red", "-"),
                    ("green", "-")],
            ylabel="Latency (us)",
            plot_name="causal-conv1d",
            args={},
        ))
    def benchmark(shape_name, workload_name, dtype_str, provider):
        return benchmark_causal_conv1d(shape_name=shape_name,
                                       workload_name=workload_name,
                                       dtype_str=dtype_str,
                                       provider=provider,
                                       iterations=iterations)

    return benchmark


def gen_perf_configs(dtype_str="bf16"):
    return [(s.name, w.name, dtype_str)
            for s in MODEL_SHAPES for w in WORKLOADS]


if __name__ == "__main__":
    args = parse_args()
    seed = 1234
    seed_everything(seed)
    iterations = 30
    torch.set_default_device("xpu")
    torch.xpu.set_device("xpu:0")

    configs = gen_perf_configs("bf16")
    benchmark = get_benchmark(configs, iterations=iterations)
    save_path = ensure_save_path_exists(args.save_path)
    benchmark.run(print_data=True, save_path=save_path)
