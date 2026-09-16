# SPDX-License-Identifier: Apache-2.0
# python3 benchmark/benchmark_moe_interleaved.py
#
# Compares vllm-xpu-kernels' two GEMM1 (gate+up projection + SwiGLU/GELU)
# strategies for MoE:
#
#   1. `unfused`      -- existing 2-kernel path: `cutlass_grouped_gemm_interface`
#                        (pre-transposed [E, K, N] weights) writing a full
#                        [M, 2N] gate|up intermediate, followed by a
#                        separate `silu_and_mul`/`gelu_and_mul` kernel. This
#                        is what `XpuFusedMoe._apply_kernel` falls back to
#                        when the interleaved fusion is not eligible/enabled.
#   2. `interleaved`  -- new single-accumulator fused kernel
#                        (`moe_grouped_mm_xe20_interleaved`), ported from
#                        sgl-kernel-xpu per /work/fusemlp/design.md, using
#                        gate/up-interleaved [E, N, K] weights produced by
#                        `interleave_gate_up_weights_xe20()`.
#
# Same shape-sweep methodology as sgl-kernel-xpu's
# benchmark/bench_moe_accum_strategy.py (decode-shaped -> prefill-shaped
# avg_m, plus a few wide-N/deep-K/large-model shapes), so the two reports
# are directly comparable.
import torch
import triton
import triton.testing

import vllm_xpu_kernels._xpu_C  # noqa: F401
from vllm_xpu_kernels.moe_utils import interleave_gate_up_weights_xe20

ACTIVATIONS = {"silu": 0, "gelu": 1}

# (num_experts, K, N, avg_m) -- N is the gate (== up) width; avg_m is the
# average tokens routed to each expert (decode: ~1-8, prefill: ~128-1024).
shape_configs = [
    {
        "name": "qwen3-30b-a3b-decode-m1",
        "num_experts": 128,
        "K": 2048,
        "N": 768,
        "avg_m": 1,
    },
    {
        "name": "qwen3-30b-a3b-decode-m8",
        "num_experts": 128,
        "K": 2048,
        "N": 768,
        "avg_m": 8,
    },
    {
        "name": "qwen3-30b-a3b-mid-m32",
        "num_experts": 128,
        "K": 2048,
        "N": 768,
        "avg_m": 32,
    },
    {
        "name": "qwen3-30b-a3b-mid-m128",
        "num_experts": 128,
        "K": 2048,
        "N": 768,
        "avg_m": 128,
    },
    {
        "name": "qwen3-30b-a3b-prefill-m512",
        "num_experts": 128,
        "K": 2048,
        "N": 768,
        "avg_m": 512,
    },
    # Qwen3.5-35B-A3B (huggingface.co/Qwen/Qwen3.5-35B-A3B). From its
    # text_config: hidden_size=2048, moe_intermediate_size=512,
    # num_experts=256, num_experts_per_tok=8, hidden_act="silu",
    # num_hidden_layers=40. N here is the full (unsharded) gate/up width,
    # matching the qwen3-30b-a3b-* convention above.
    {
        "name": "qwen3.5-35b-a3b-decode-m1",
        "num_experts": 256,
        "K": 2048,
        "N": 512,
        "avg_m": 1,
    },
    {
        "name": "qwen3.5-35b-a3b-decode-m8",
        "num_experts": 256,
        "K": 2048,
        "N": 512,
        "avg_m": 8,
    },
    {
        "name": "qwen3.5-35b-a3b-mid-m32",
        "num_experts": 256,
        "K": 2048,
        "N": 512,
        "avg_m": 32,
    },
    {
        "name": "qwen3.5-35b-a3b-mid-m128",
        "num_experts": 256,
        "K": 2048,
        "N": 512,
        "avg_m": 128,
    },
    {
        "name": "qwen3.5-35b-a3b-prefill-m512",
        "num_experts": 256,
        "K": 2048,
        "N": 512,
        "avg_m": 512,
    },
    {
        "name": "wide-N-8e-k2048-n4096-m1024",
        "num_experts": 8,
        "K": 2048,
        "N": 4096,
        "avg_m": 1024,
    },
    {
        "name": "wide-N-8e-k1024-n8192-m1024",
        "num_experts": 8,
        "K": 1024,
        "N": 8192,
        "avg_m": 1024,
    },
    {
        "name": "deep-K-8e-k7168-n2048-m512",
        "num_experts": 8,
        "K": 7168,
        "N": 2048,
        "avg_m": 512,
    },
    # Qwen3.5-397B-A17B (huggingface.co/Qwen/Qwen3.5-397B-A17B), routed MoE
    # experts, prefill batch 8192 tokens, TP=4. From its text_config:
    #   hidden_size=4096, moe_intermediate_size=1024, num_experts=512,
    #   num_experts_per_tok=10, hidden_act="silu".
    # Under TP=4 the expert intermediate dim is sharded across ranks, so each
    # rank's GEMM1 is K=4096 (hidden, unsharded) x N=1024/4=256 per expert
    # (interleaved gate+up width 2N=512). With top-10 routing over 512
    # experts, 8192 tokens give 8192*10=81920 token-expert pairs, i.e.
    # avg_m = 81920/512 = 160 rows per expert.
    {
        "name": "qwen3.5-397b-a17b-prefill8192-tp4",
        "num_experts": 512,
        "K": 4096,
        "N": 256,
        "avg_m": 160,
    },
]


def _make_row_counts(num_experts, avg_m, device):
    return torch.full((num_experts, ), avg_m, dtype=torch.int32, device=device)


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["shape_id"],
        x_vals=list(range(len(shape_configs))),
        line_arg="strategy",
        line_vals=["unfused", "interleaved"],
        line_names=[
            "Unfused (cutlass_grouped_gemm_interface + *_and_mul)",
            "Single-accumulator interleaved fused (new)",
        ],
        styles=[("red", "-"), ("green", "-")],
        ylabel="ms",
        plot_name="moe-interleaved-vs-unfused-gemm1",
        args={"activation": "silu"},
    ))
def benchmark(shape_id, strategy, activation):
    cfg = shape_configs[shape_id]
    device = "xpu"
    torch.manual_seed(0)
    num_experts, K, N, avg_m = cfg["num_experts"], cfg["K"], cfg["N"], cfg[
        "avg_m"]
    activation_type = ACTIVATIONS[activation]

    rows_per_expert = _make_row_counts(num_experts, avg_m, device)
    total_m = int(rows_per_expert.sum().item())

    activations = torch.empty(
        total_m, K, dtype=torch.bfloat16,
        device=device).normal_(0, 0.02)
    # Loader/original layout: [E, N, K] (out_features-major).
    w_block = torch.empty(
        num_experts, 2 * N, K, dtype=torch.bfloat16,
        device=device).normal_(0, 0.02)

    if strategy == "unfused":
        w_t = w_block.transpose(-1, -2).contiguous()  # [E, K, N]
        gemm1_out = torch.empty(
            total_m, 2 * N, dtype=torch.bfloat16, device=device)
        out = torch.empty(total_m, N, dtype=torch.bfloat16, device=device)
        act_fn = (torch.ops._C.silu_and_mul
                 if activation == "silu" else torch.ops._C.gelu_and_mul)

        def run():
            torch.ops._xpu_C.cutlass_grouped_gemm_interface(
                ptr_A=activations,
                ptr_A_scale=None,
                ptr_B=w_t,
                ptr_B_scale=None,
                ptr_bias=None,
                ptr_D=gemm1_out,
                rows_per_expert=rows_per_expert,
                N=2 * N,
                K=K,
                num_experts=num_experts)
            act_fn(out, gemm1_out)
    else:
        w_interleaved = interleave_gate_up_weights_xe20(w_block)
        out = torch.empty(total_m, N, dtype=torch.bfloat16, device=device)

        def run():
            torch.ops._xpu_C.moe_grouped_mm_xe20_interleaved(
                output=out,
                activations=activations,
                weights=w_interleaved,
                bias=None,
                rows_per_expert=rows_per_expert,
                num_experts=num_experts,
                activation_type=activation_type,
                gemm1_alpha=1.702,
                gemm1_limit=7.0)

    torch.xpu.synchronize()
    ms, _, _ = triton.testing.do_bench(
        run, warmup=10, rep=100, quantiles=[0.5, 0.2, 0.8])
    torch.xpu.empty_cache()
    return ms


if __name__ == "__main__":
    print("Shapes benchmarked:")
    for i, cfg in enumerate(shape_configs):
        print(f"  [{i}] {cfg['name']}: num_experts={cfg['num_experts']} "
             f"K={cfg['K']} N={cfg['N']} avg_m={cfg['avg_m']}")
    benchmark.run(print_data=True)
