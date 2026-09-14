#pragma once

#include <torch/all.h>

// Fused, single-accumulator interleaved SwiGLU MoE grouped-GEMM op, ported
// from sgl-kernel-xpu's moe_grouped_mm_nt_xe20_interleaved
// (src/sycl/GroupGemmXe20Interleaved.cpp). Requires the caller to have
// already interleaved the gate/up weight (and bias, if present) columns in
// 16-wide groups -- see vllm_xpu_kernels.moe_utils.interleave_gate_up_weights_xe20
// (Python port of sgl_kernel.moe.interleave_gate_up_weights_xe20).
//
// activations: [total_m, K] bf16
// weights:     [num_experts, N, K] bf16, N = full interleaved gate+up width
// bias:        optional [num_experts, N] float32, pre-interleaved to match
//              weights
// output:      [total_m, N / 2] bf16 (SwiGLU-fused gate*up output)
// rows_per_expert: [num_experts] int32, number of activation rows routed to
//              each expert (must sum to total_m)
// activation_type: 0 = silu, 1 = gelu (matches MoE::ActivationType)
torch::Tensor moe_grouped_mm_xe20_interleaved(
    torch::Tensor& output,
    const torch::Tensor& activations,
    const torch::Tensor& weights,
    const c10::optional<at::Tensor>& bias,
    const torch::Tensor& rows_per_expert,
    int64_t num_experts,
    int64_t activation_type,
    double gemm1_alpha,
    double gemm1_limit);
