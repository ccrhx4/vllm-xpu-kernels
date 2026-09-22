# SPDX-License-Identifier: Apache-2.0
# python3 benchmark/sweep_dense_mlp_tile_ids.py
#
# Tile-tuning sweep for the decode-shaped M regime (M in {1, 8, 32, 128}) of
# the dense MLP fusion kernel (dense_swiglu_gemm_xe20_interleaved). Forces
# each of the 6 available tile configs (via the tile_id_override arg) at
# each (TP, M) point and compares against the auto-selected tile
# (dense_select_tile()'s default) and the oneDNN torch.matmul baseline, to
# find whether a different static tile assignment closes the decode-regime
# gap identified in /work/fusemlp/DENSE_MLP_FUSION_RESULTS.md.
#
# Tile-id -> config table (see dense_mlp_interleaved.cpp's
# dense_select_tile()/launch_dense_interleaved() switch):
#   0: 8x128x32,   SG 1x4x1
#   1: 16x128x32,  SG 1x4x1
#   2: 32x128x32,  SG 1x4x1
#   3: 128x128x32, SG 4x2x1
#   4: 256x128x32, SG 8x2x1
#   5: 256x256x32, SG 8x4x1 (Config 1, swizzle=8)
import torch
import triton.testing

import vllm_xpu_kernels._xpu_C  # noqa: F401
from vllm_xpu_kernels.moe_utils import interleave_gate_up_weights_xe20

HIDDEN = 5120
INTERMEDIATE = 17408
M_SWEEP = [1, 8, 32, 128]
TILE_IDS = [0, 1, 2, 3, 4, 5]
TILE_NAMES = {
    0: "8x128",
    1: "16x128",
    2: "32x128",
    3: "128x128",
    4: "256x128",
    5: "256x256(swz8)",
}


def make_inputs(m, k, n_full, device):
    torch.manual_seed(0)
    activations = torch.empty(
        m, k, dtype=torch.bfloat16, device=device).normal_(0, 0.02)
    w_block = torch.empty(
        n_full, k, dtype=torch.bfloat16, device=device).normal_(0, 0.02)
    return activations, w_block


def bench_unfused(activations, w_block, m, n_half):
    w_t = w_block.t().contiguous()
    out = torch.empty(m, n_half, dtype=torch.bfloat16, device="xpu")

    def run():
        gemm1_out = torch.matmul(activations, w_t)
        torch.ops._C.silu_and_mul(out, gemm1_out)

    torch.xpu.synchronize()
    ms, _, _ = triton.testing.do_bench(
        run, warmup=10, rep=100, quantiles=[0.5, 0.2, 0.8])
    return ms


def bench_fused(activations, w_block, m, n_half, tile_id_override):
    w_interleaved = interleave_gate_up_weights_xe20(
        w_block.unsqueeze(0)).squeeze(0)
    out = torch.empty(m, n_half, dtype=torch.bfloat16, device="xpu")

    def run():
        torch.ops._xpu_C.dense_swiglu_gemm_xe20_interleaved(
            output=out,
            activations=activations,
            weight=w_interleaved,
            bias=None,
            activation_type=0,
            gemm1_alpha=1.702,
            gemm1_limit=7.0,
            tile_id_override=tile_id_override)

    torch.xpu.synchronize()
    try:
        ms, _, _ = triton.testing.do_bench(
            run, warmup=10, rep=100, quantiles=[0.5, 0.2, 0.8])
    except Exception as e:  # noqa: BLE001
        return None, str(e)
    return ms, None


def main():
    device = "xpu"
    print(f"{'TP':>3} {'M':>5} {'unfused(ms)':>12} {'auto(ms)':>10} "
          + " ".join(f"{TILE_NAMES[t]:>14}" for t in TILE_IDS))
    for tp in (1, 2, 4):
        n_per_rank = INTERMEDIATE // tp
        for m in M_SWEEP:
            activations, w_block = make_inputs(
                m, HIDDEN, 2 * n_per_rank, device)
            unfused_ms = bench_unfused(activations, w_block, m, n_per_rank)
            auto_ms, _ = bench_fused(
                activations, w_block, m, n_per_rank, tile_id_override=-1)
            row = [f"{tp:>3}", f"{m:>5}", f"{unfused_ms:>12.4f}",
                   f"{auto_ms:>10.4f}"]
            for tile_id in TILE_IDS:
                ms, err = bench_fused(
                    activations, w_block, m, n_per_rank,
                    tile_id_override=tile_id)
                if ms is None:
                    row.append(f"{'ERR':>14}")
                else:
                    row.append(f"{ms:>14.4f}")
            print(" ".join(row))
            torch.xpu.empty_cache()


if __name__ == "__main__":
    main()
