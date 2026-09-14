// SPDX-License-Identifier: Apache-2.0
//
// Host-side op wrapper for the interleaved SwiGLU MoE grouped-GEMM fusion,
// ported from sgl-kernel-xpu's src/sycl/GroupGemmXe20Interleaved.cpp. Only
// two tile configs are enabled here (matching what sgl-kernel-xpu has
// validated for correctness, perf, and register pressure on real Xe20/Arc-B
// hardware): a small/medium-avg_m 256x128 (half 256x64) config, and a
// large-avg_m 256x256 (half 256x128) "Config 1" config from design.md.
#include "csrc/utils.h"
#include "moe_grouped_mm_interleaved.h"

#ifdef VLLM_XPU_ENABLE_XE2
  #include "moe_interleaved_launcher.hpp"

using namespace vllm_xpu::moe_interleaved;
using namespace cute;

namespace {

using Tile_256_128_32 = Shape<_256, _128, _32>;
using Tile_256_64_32 = Shape<_256, _64, _32>;
using Tile_256_256_32 = Shape<_256, _256, _32>;
using SG_8_2_1 = Layout<Shape<_8, _2, _1>, Stride<_2, _1, _0>>;
using SG_8_4_1 = Layout<Shape<_8, _4, _1>, Stride<_4, _1, _0>>;

// `gemm_n` is the FULL interleaved (gate+up) width. Config 1 (the large
// 256x256 tile) is only reachable by the single-accumulator interleaved
// epilogue -- the dual-accumulator (unfused-then-fused) path cannot fit two
// accumulators at this tile size -- so it is only used once avg_m and
// gemm_n are both large enough to keep it compute-bound. See
// /work/fusemlp/design.md and sgl-kernel-xpu's grouped_gemm_dispatch.h for
// the equivalent tile-selection heuristic on the dual-accumulator path.
inline bool interleaved_use_config1(int avg_m, int gemm_n) {
  return avg_m >= 256 && gemm_n >= 256;
}

template <ActivationType ActType, bool WithBias>
void launch_interleaved(
    sycl::queue queue,
    bool use_config1,
    const void* activations,
    const void* weights,
    const void* bias,
    void* outputs,
    int gemm_n,
    int gemm_k,
    const int* rows_per_expert,
    int num_experts,
    int* workspace,
    float gemm1_alpha,
    float gemm1_limit,
    int ld_b) {
  if (use_config1) {
    Xe20MoEGEMMInterleavedLauncher<Tile_256_256_32, Tile_256_128_32, SG_8_4_1, ActType, WithBias>(
        queue,
        activations,
        weights,
        bias,
        outputs,
        gemm_n,
        gemm_k,
        rows_per_expert,
        num_experts,
        workspace,
        gemm1_alpha,
        gemm1_limit,
        ld_b);
  } else {
    Xe20MoEGEMMInterleavedLauncher<Tile_256_128_32, Tile_256_64_32, SG_8_2_1, ActType, WithBias>(
        queue,
        activations,
        weights,
        bias,
        outputs,
        gemm_n,
        gemm_k,
        rows_per_expert,
        num_experts,
        workspace,
        gemm1_alpha,
        gemm1_limit,
        ld_b);
  }
}

}  // namespace
#endif  // VLLM_XPU_ENABLE_XE2

torch::Tensor moe_grouped_mm_xe20_interleaved(
    torch::Tensor& output,
    const torch::Tensor& activations,
    const torch::Tensor& weights,
    const c10::optional<at::Tensor>& bias,
    const torch::Tensor& rows_per_expert,
    int64_t num_experts,
    int64_t activation_type,
    double gemm1_alpha,
    double gemm1_limit) {
#ifndef VLLM_XPU_ENABLE_XE2
  TORCH_CHECK(
      false,
      "moe_grouped_mm_xe20_interleaved requires a build with "
      "VLLM_XPU_ENABLE_XE2=1 (Xe2/Xe20, e.g. Arc B-series).");
#else
  TORCH_CHECK(
      vllm::xpu::is_xe2_arch(),
      "moe_grouped_mm_xe20_interleaved is only implemented for Xe2/Xe20 "
      "hardware (e.g. Arc B-series).");

  int64_t total_m = activations.size(0);
  int64_t gemm_k = activations.size(1);
  int64_t gemm_n = weights.size(1);  // full (interleaved gate+up) width

  TORCH_CHECK(weights.dim() == 3, "weights must be 3D [n_experts, N, K]");
  TORCH_CHECK(weights.size(0) == num_experts, "weights must have num_experts as the first dimension");
  TORCH_CHECK(
      weights.size(0) == rows_per_expert.size(0), "rows_per_expert must have the same size as weights.size(0)");
  TORCH_CHECK(output.size(0) == total_m, "output must have the same number of rows as activations");
  TORCH_CHECK(output.size(1) == gemm_n / 2, "output must have half the number of columns as the interleaved weights");
  TORCH_CHECK(gemm_n % 32 == 0, "interleaved N must be a multiple of 32 (16-wide gate/up pairs)");
  TORCH_CHECK(num_experts % 8 == 0, "num_experts must be a multiple of 8 for the current implementation");
  TORCH_CHECK(
      activation_type == static_cast<int64_t>(MoE::ActivationType::SILU) ||
          activation_type == static_cast<int64_t>(MoE::ActivationType::GELU),
      "moe_grouped_mm_xe20_interleaved only supports silu (0) or gelu (1)");
  TORCH_CHECK(
      activations.scalar_type() == weights.scalar_type(), "activations and weights must have the same data type");
  TORCH_CHECK(
      activations.scalar_type() == at::ScalarType::BFloat16,
      "Only bfloat16 is supported by moe_grouped_mm_xe20_interleaved currently");
  bool with_bias = bias.has_value();
  if (with_bias) {
    TORCH_CHECK(bias->scalar_type() == at::kFloat, "bias must be float32 to match kernel expectations");
    TORCH_CHECK(bias->dim() == 2, "bias must be 2D [n_experts, N] (already interleaved to match weights)");
    TORCH_CHECK(bias->size(0) == num_experts && bias->size(1) == gemm_n, "bias shape mismatch with weight");
  }

  auto stream = at::xpu::getCurrentXPUStream();
  auto queue = stream.queue();
  at::Tensor atomic_buffer = at::empty({1}, activations.options().dtype(at::kInt));
  const void* bias_ptr = with_bias ? bias->data_ptr() : nullptr;
  int ld_b = static_cast<int>(weights.stride(1));
  const int avg_m = num_experts > 0 ? static_cast<int>(total_m / num_experts) : static_cast<int>(total_m);
  const bool use_config1 = interleaved_use_config1(avg_m, static_cast<int>(gemm_n));

  if (activation_type == static_cast<int64_t>(MoE::ActivationType::SILU)) {
    if (with_bias) {
      launch_interleaved<MoE::ActivationType::SILU, true>(
          queue,
          use_config1,
          activations.data_ptr(),
          weights.data_ptr(),
          bias_ptr,
          output.data_ptr(),
          static_cast<int>(gemm_n),
          static_cast<int>(gemm_k),
          rows_per_expert.data_ptr<int>(),
          static_cast<int>(num_experts),
          atomic_buffer.data_ptr<int>(),
          static_cast<float>(gemm1_alpha),
          static_cast<float>(gemm1_limit),
          ld_b);
    } else {
      launch_interleaved<MoE::ActivationType::SILU, false>(
          queue,
          use_config1,
          activations.data_ptr(),
          weights.data_ptr(),
          bias_ptr,
          output.data_ptr(),
          static_cast<int>(gemm_n),
          static_cast<int>(gemm_k),
          rows_per_expert.data_ptr<int>(),
          static_cast<int>(num_experts),
          atomic_buffer.data_ptr<int>(),
          static_cast<float>(gemm1_alpha),
          static_cast<float>(gemm1_limit),
          ld_b);
    }
  } else {
    if (with_bias) {
      launch_interleaved<MoE::ActivationType::GELU, true>(
          queue,
          use_config1,
          activations.data_ptr(),
          weights.data_ptr(),
          bias_ptr,
          output.data_ptr(),
          static_cast<int>(gemm_n),
          static_cast<int>(gemm_k),
          rows_per_expert.data_ptr<int>(),
          static_cast<int>(num_experts),
          atomic_buffer.data_ptr<int>(),
          static_cast<float>(gemm1_alpha),
          static_cast<float>(gemm1_limit),
          ld_b);
    } else {
      launch_interleaved<MoE::ActivationType::GELU, false>(
          queue,
          use_config1,
          activations.data_ptr(),
          weights.data_ptr(),
          bias_ptr,
          output.data_ptr(),
          static_cast<int>(gemm_n),
          static_cast<int>(gemm_k),
          rows_per_expert.data_ptr<int>(),
          static_cast<int>(num_experts),
          atomic_buffer.data_ptr<int>(),
          static_cast<float>(gemm1_alpha),
          static_cast<float>(gemm1_limit),
          ld_b);
    }
  }
  return output;
#endif  // VLLM_XPU_ENABLE_XE2
}
