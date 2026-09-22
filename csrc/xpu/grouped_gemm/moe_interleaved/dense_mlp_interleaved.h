#pragma once

#include <torch/all.h>

// Fused, single-accumulator interleaved SwiGLU dense-MLP GEMM op -- the
// plain (non-grouped) analog of moe_grouped_mm_xe20_interleaved
// (moe_grouped_mm_interleaved.h/.cpp), for a dense FFN's gate/up projection
// (no expert routing). See
// csrc/xpu/grouped_gemm/moe_interleaved/xe20/bf16/dense_kernel_interleaved.hpp
// and /work/fusemlp/design.md for the fusion rationale.
//
// Requires the caller to have already interleaved the gate/up weight (and
// bias, if present) columns in 16-wide groups -- see
// vllm_xpu_kernels.moe_utils.interleave_gate_up_weights_xe20 (the same
// helper used by the MoE path; a dense weight is simply the num_experts=1
// case of that per-slice transform).
//
// activations: [M, K] bf16
// weight:      [N, K] bf16, N = full interleaved gate+up width (no expert dim)
// bias:        optional [N] float32, pre-interleaved to match weight
// output:      [M, N / 2] bf16 (SwiGLU-fused gate*up output)
// activation_type: 0 = silu, 1 = gelu (matches MoE::ActivationType)
// tile_id_override: -1 (default) = auto-select via dense_select_tile(); else
// force a specific tile config (0-5, see dense_select_tile()'s table) --
// exposed for op-level tile-tuning/benchmarking only, not meant for
// production callers.
torch::Tensor dense_swiglu_gemm_xe20_interleaved(
    torch::Tensor& output,
    const torch::Tensor& activations,
    const torch::Tensor& weight,
    const c10::optional<at::Tensor>& bias,
    int64_t activation_type,
    double gemm1_alpha,
    double gemm1_limit,
    int64_t tile_id_override = -1);
