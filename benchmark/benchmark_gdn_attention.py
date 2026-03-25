# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import argparse
import csv
import itertools
import os
import random
import time

import torch

import vllm_xpu_kernels._xpu_C  # noqa: F401

# Keep the same default grid as tests/gdn_attn/test_gdn_attn.py.
NUM_TOKENS = [1, 32, 1024, 8192]
BATCH_SIZE = [32]
NUM_K_HEADS = [16]
NUM_K_DIMS = [128]
NUM_V_HEADS = [32]
NUM_V_DIMS = [128]
WIDTH = [4]
TP_SIZE = [1]
HAS_BIAS = [True, False]
ACTIVATION = ["silu"]
MODE = ["prefill", "decode", "mix_mode"]
REORDER_INPUT = [True, False]
DTYPES = [torch.float16]

MINI_PYTEST_PARAMS = {
    "default": {
        "num_actual_tokens": [32],
    },
}


def simple_random_distribute(n: int, batch_size: int) -> torch.Tensor:
    distribution = torch.ones([batch_size])
    for _ in range(n - batch_size):
        selected_idx = random.randint(0, batch_size - 1)
        distribution[selected_idx] += 1
    return distribution


def prepare_case(
    num_actual_tokens: int,
    batch_size: int,
    num_k_heads: int,
    head_k_dim: int,
    num_v_heads: int,
    head_v_dim: int,
    width: int,
    tp_size: int,
    has_bias: bool,
    activation: str,
    reorder_input: bool,
    mode: str,
    dtype: torch.dtype,
    cache_batch_size: int,
    device: str,
):
    if batch_size > num_actual_tokens:
        batch_size = num_actual_tokens

    if mode == "prefill":
        num_prefills = batch_size
    elif mode == "decode":
        num_prefills = 0
        if batch_size < num_actual_tokens:
            return None
    else:
        num_prefills = random.randint(1, batch_size - 1) if batch_size > 1 else 1

    num_decodes = batch_size - num_prefills

    mixed_qkvz_size = num_k_heads // tp_size * (
        2 * head_k_dim + 2 * head_v_dim * num_v_heads // num_k_heads)
    mixed_ba_size = num_k_heads // tp_size * (2 * num_v_heads // num_k_heads)
    mixed_qkv_size = num_k_heads // tp_size * (
        2 * head_k_dim + head_v_dim * num_v_heads // num_k_heads)

    projected_states_qkvz = torch.randn((num_actual_tokens, mixed_qkvz_size),
                                        dtype=dtype,
                                        device=device)
    projected_states_ba = torch.randn((num_actual_tokens, mixed_ba_size),
                                      dtype=dtype,
                                      device=device)

    conv_state = torch.randn((cache_batch_size, width - 1, mixed_qkv_size),
                             dtype=dtype,
                             device=device)
    ssm_state = torch.randn(
        (cache_batch_size, num_v_heads // tp_size, head_v_dim, head_k_dim),
        dtype=dtype,
        device=device)

    conv_weights = torch.randn((mixed_qkv_size, width),
                               dtype=dtype,
                               device=device)
    conv_bias = None
    if has_bias:
        conv_bias = torch.randn((mixed_qkv_size), dtype=dtype, device=device)

    a_log = torch.randn((num_v_heads // tp_size), dtype=dtype, device=device)
    dt_bias = torch.randn((num_v_heads // tp_size), dtype=dtype, device=device)

    prefill_batches = simple_random_distribute(num_actual_tokens - num_decodes,
                                               batch_size - num_decodes)
    token_batches = torch.cat([torch.ones([num_decodes]), prefill_batches]).to(device)
    perm = torch.randperm(token_batches.size(0)).to(device)
    shuffled_tensor = token_batches[perm]
    non_spec_query_start_loc = torch.cat([
        torch.zeros([1], device=device),
        torch.cumsum(shuffled_tensor, dim=0)
    ]).to(torch.int32)
    has_initial_state = perm >= num_decodes
    non_spec_state_indices_tensor = torch.tensor(random.sample(
        range(cache_batch_size), batch_size),
                                                 device=device,
                                                 dtype=torch.int32)

    core_attn_out = torch.zeros(
        (num_actual_tokens, num_v_heads // tp_size, head_v_dim),
        dtype=dtype,
        device=device,
    )
    z = torch.empty_like(core_attn_out)

    return {
        "core_attn_out": core_attn_out,
        "z": z,
        "projected_states_qkvz": projected_states_qkvz,
        "projected_states_ba": projected_states_ba,
        "num_k_heads": num_k_heads,
        "num_v_heads": num_v_heads,
        "head_k_dim": head_k_dim,
        "head_v_dim": head_v_dim,
        "conv_state": conv_state,
        "ssm_state": ssm_state,
        "conv_weights": conv_weights,
        "conv_bias": conv_bias,
        "activation": activation,
        "A_log": a_log,
        "dt_bias": dt_bias,
        "num_prefills": num_prefills,
        "num_decodes": num_decodes,
        "has_initial_state": has_initial_state,
        "non_spec_query_start_loc": non_spec_query_start_loc,
        "non_spec_state_indices_tensor": non_spec_state_indices_tensor,
        "num_actual_tokens": num_actual_tokens,
        "tp_size": tp_size,
        "reorder_input": reorder_input,
    }


def run_kernel(case_data: dict) -> None:
    torch.ops._xpu_C.gdn_attention(
        case_data["core_attn_out"],
        case_data["z"],
        case_data["projected_states_qkvz"],
        case_data["projected_states_ba"],
        case_data["num_k_heads"],
        case_data["num_v_heads"],
        case_data["head_k_dim"],
        case_data["head_v_dim"],
        conv_state=case_data["conv_state"],
        ssm_state=case_data["ssm_state"],
        conv_weights=case_data["conv_weights"],
        conv_bias=case_data["conv_bias"],
        activation=case_data["activation"],
        A_log=case_data["A_log"],
        dt_bias=case_data["dt_bias"],
        num_prefills=case_data["num_prefills"],
        num_decodes=case_data["num_decodes"],
        has_initial_state=case_data["has_initial_state"],
        non_spec_query_start_loc=case_data["non_spec_query_start_loc"],
        non_spec_state_indices_tensor=case_data["non_spec_state_indices_tensor"],
        num_actual_tokens=case_data["num_actual_tokens"],
        tp_size=case_data["tp_size"],
        reorder_input=case_data["reorder_input"],
    )


def build_xpu_cache_flusher(flush_size_mb: int, device: str):
    # Touching a large temporary tensor helps evict hot lines from smaller
    # on-chip caches (e.g., L1) before each measured kernel invocation.
    num_elems = max(1, (flush_size_mb * 1024 * 1024) // 4)
    flush_buf = torch.empty(num_elems, dtype=torch.float32, device=device)

    def flush_once() -> None:
        flush_buf.add_(1.0)

    return flush_once


def benchmark_case(case_data: dict, num_warmup_iters: int, num_iters: int,
                   flush_once) -> float:
    for _ in range(num_warmup_iters):
        flush_once()
        run_kernel(case_data)
    torch.xpu.synchronize()

    total = 0.0
    for _ in range(num_iters):
        flush_once()
        torch.xpu.synchronize()
        start = time.perf_counter()
        run_kernel(case_data)
        torch.xpu.synchronize()
        end = time.perf_counter()
        total += (end - start)

    return total / num_iters


def write_results_to_csv(csv_path: str, rows: list[list[object]]) -> None:
    parent = os.path.dirname(csv_path)
    if parent:
        os.makedirs(parent, exist_ok=True)

    header = [
        "num_tokens",
        "batch_size",
        "num_k_heads",
        "head_k_dim",
        "num_v_heads",
        "head_v_dim",
        "width",
        "tp_size",
        "has_bias",
        "activation",
        "mode",
        "reorder_input",
        "dtype",
        "latency_us",
    ]

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)


@torch.inference_mode()
def main(args: argparse.Namespace) -> None:
    if args.device != "xpu":
        raise ValueError("This benchmark currently supports only --device xpu")

    torch.set_default_device(args.device)
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    num_tokens = MINI_PYTEST_PARAMS["default"][
        "num_actual_tokens"] if args.mini else NUM_TOKENS

    configs = list(
        itertools.product(
            num_tokens,
            BATCH_SIZE,
            NUM_K_HEADS,
            NUM_K_DIMS,
            NUM_V_HEADS,
            NUM_V_DIMS,
            WIDTH,
            TP_SIZE,
            HAS_BIAS,
            ACTIVATION,
            MODE,
            REORDER_INPUT,
            DTYPES,
        ))

    print("gdn_attention benchmark")
    print(
        f"device={args.device} warmup={args.num_warmup_iters} iters={args.num_iters} seed={args.seed} mini={args.mini}"
    )
    print(f"flush_cache_size_mb={args.flush_cache_size_mb}")
    print("-" * 120)
    print(
        "num_tokens,batch_size,num_k_heads,head_k_dim,num_v_heads,head_v_dim,width,tp_size,has_bias,activation,mode,reorder_input,dtype,latency_us"
    )

    valid_count = 0
    flush_once = build_xpu_cache_flusher(args.flush_cache_size_mb,
                                         args.device)
    result_rows: list[list[object]] = []

    for config in configs:
        (num_actual_tokens, batch_size, num_k_heads, head_k_dim, num_v_heads,
         head_v_dim, width, tp_size, has_bias, activation, mode,
         reorder_input, dtype) = config

        if head_k_dim != head_v_dim:
            continue
        if num_v_heads % num_k_heads != 0:
            continue

        case_data = prepare_case(
            num_actual_tokens=num_actual_tokens,
            batch_size=batch_size,
            num_k_heads=num_k_heads,
            head_k_dim=head_k_dim,
            num_v_heads=num_v_heads,
            head_v_dim=head_v_dim,
            width=width,
            tp_size=tp_size,
            has_bias=has_bias,
            activation=activation,
            reorder_input=reorder_input,
            mode=mode,
            dtype=dtype,
            cache_batch_size=args.cache_batch_size,
            device=args.device,
        )
        if case_data is None:
            continue

        latency_s = benchmark_case(case_data, args.num_warmup_iters,
                       args.num_iters, flush_once)
        latency_us = latency_s * 1e6
        dtype_name = str(dtype)

        row = [
            num_actual_tokens,
            min(batch_size, num_actual_tokens),
            num_k_heads,
            head_k_dim,
            num_v_heads,
            head_v_dim,
            width,
            tp_size,
            has_bias,
            activation,
            mode,
            reorder_input,
            dtype_name,
            round(latency_us, 3),
        ]
        result_rows.append(row)

        print(
            f"{num_actual_tokens},{min(batch_size, num_actual_tokens)},{num_k_heads},{head_k_dim},{num_v_heads},{head_v_dim},{width},{tp_size},{has_bias},{activation},{mode},{reorder_input},{dtype_name},{latency_us:.3f}"
        )
        valid_count += 1

    print("-" * 120)
    if valid_count == 0:
        print("No valid configurations were benchmarked.")
    else:
        print(f"Benchmarked {valid_count} configurations.")
        if args.csv_path:
            write_results_to_csv(args.csv_path, result_rows)
            print(f"Saved CSV results to: {args.csv_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Benchmark torch.ops._xpu_C.gdn_attention (non-pytest).")
    parser.add_argument("--num-warmup-iters", type=int, default=10)
    parser.add_argument("--num-iters", type=int, default=50)
    parser.add_argument("--cache-batch-size", type=int, default=200)
    parser.add_argument(
        "--flush-cache-size-mb",
        type=int,
        default=64,
        help=("Size of XPU buffer (MB) touched between iterations to reduce "
              "cache reuse effects."),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="xpu")
    parser.add_argument(
        "--csv-path",
        type=str,
        default=None,
        help="Optional path to save benchmark results as CSV.",
    )
    parser.add_argument(
        "--mini",
        action="store_true",
        help=("Use mini config (num_actual_tokens=[32]) from test file to "
              "reduce runtime."),
    )

    main(parser.parse_args())
