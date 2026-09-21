# SPDX-License-Identifier: Apache-2.0
# python3 benchmark/benchmark_dense_mlp_interleaved.py
#
# Compares vllm-xpu-kernels' two dense-FFN gate/up-projection strategies
# for a plain (non-MoE) MLP -- the dense-MLP analog of
# benchmark_moe_interleaved.py:
#
#   1. `unfused`      -- plain 2-kernel path: torch.matmul (against a
#                        [K, N] weight) writing a full [M, 2N] gate|up
#                        intermediate, followed by a separate
#                        silu_and_mul/gelu_and_mul kernel. This is what
#                        Qwen2MoeMLP/Qwen3NextMLP's forward() does today
#                        (gate_up_proj -> act_fn).
#   2. `interleaved`  -- new single-accumulator fused kernel
#                        (`dense_swiglu_gemm_xe20_interleaved`), the
#                        non-grouped analog of moe_grouped_mm_xe20_interleaved,
#                        using the gate/up-interleaved [N, K] weight
#                        produced by `interleave_gate_up_weights_xe20()`
#                        (E=1 degenerate case via unsqueeze/squeeze).
#
# Shapes: Qwen3.6-27B's dense FFN (hidden=5120, intermediate=17408) at
# TP=1/2/4 (intermediate sharded across ranks under TP; hidden is not),
# swept across decode-shaped to prefill-shaped M.
import torch
import triton
import triton.testing

import vllm_xpu_kernels._xpu_C  # noqa: F401
from vllm_xpu_kernels.moe_utils import interleave_gate_up_weights_xe20

ACTIVATIONS = {"silu": 0, "gelu": 1}

# Qwen3.6-27B (Qwen3_5ForConditionalGeneration, text_config
# model_type=qwen3_5_text): hidden_size=5120, intermediate_size=17408,
# hidden_act="silu", dense FFN (no MoE). Under tensor parallelism the
# gate_up_proj is column-parallel (intermediate size sharded across ranks)
# and down_proj is row-parallel (hidden size unsharded) -- so each rank's
# fused GEMM1 is K=5120 (hidden, unsharded) x N=intermediate/TP (gate width;
# interleaved gate+up width 2N).
HIDDEN = 5120
INTERMEDIATE = 17408

M_SWEEP = [1, 8, 32, 128, 512, 2048, 4096]

shape_configs = []
for tp in (1, 2, 4):
    n_per_rank = INTERMEDIATE // tp
    for m in M_SWEEP:
        shape_configs.append({
            "name": f"qwen3.6-27b-dense-tp{tp}-m{m}",
            "K": HIDDEN,
            "N": n_per_rank,
            "M": m,
        })


def _make_activations(m, k, device):
    return torch.empty(m, k, dtype=torch.bfloat16, device=device).normal_(
        0, 0.02)


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["shape_id"],
        x_vals=list(range(len(shape_configs))),
        line_arg="strategy",
        line_vals=["unfused", "interleaved"],
        line_names=[
            "Unfused (matmul + *_and_mul)",
            "Single-accumulator interleaved fused (new)",
        ],
        styles=[("red", "-"), ("green", "-")],
        ylabel="ms",
        plot_name="dense-mlp-interleaved-vs-unfused-gemm1",
        args={"activation": "silu"},
    ))
def benchmark(shape_id, strategy, activation):
    cfg = shape_configs[shape_id]
    device = "xpu"
    torch.manual_seed(0)
    K, N, M = cfg["K"], cfg["N"], cfg["M"]
    activation_type = ACTIVATIONS[activation]

    activations = _make_activations(M, K, device)
    # Loader/original layout: [N, K] (out_features-major), N = 2 * gate_width.
    w_block = torch.empty(
        2 * N, K, dtype=torch.bfloat16, device=device).normal_(0, 0.02)

    if strategy == "unfused":
        w_t = w_block.t().contiguous()  # [K, 2N]
        out = torch.empty(M, N, dtype=torch.bfloat16, device=device)
        act_fn = (torch.ops._C.silu_and_mul
                 if activation == "silu" else torch.ops._C.gelu_and_mul)

        def run():
            gemm1_out = torch.matmul(activations, w_t)
            act_fn(out, gemm1_out)
    else:
        w_interleaved = interleave_gate_up_weights_xe20(
            w_block.unsqueeze(0)).squeeze(0)
        out = torch.empty(M, N, dtype=torch.bfloat16, device=device)

        def run():
            torch.ops._xpu_C.dense_swiglu_gemm_xe20_interleaved(
                output=out,
                activations=activations,
                weight=w_interleaved,
                bias=None,
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
        print(f"  [{i}] {cfg['name']}: K={cfg['K']} N={cfg['N']} "
             f"M={cfg['M']}")
    benchmark.run(print_data=True)
