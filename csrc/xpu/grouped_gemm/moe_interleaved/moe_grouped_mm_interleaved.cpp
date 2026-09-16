// SPDX-License-Identifier: Apache-2.0
//
// Host-side op wrapper for the interleaved SwiGLU MoE grouped-GEMM fusion,
// ported from sgl-kernel-xpu's src/sycl/GroupGemmXe20Interleaved.cpp. Six
// tile configs are enabled here (matching sgl-kernel-xpu's fix for the
// small-avg_m decode regression, see docs/moe_interleaved_fusion.md and
// sgl-kernel-xpu's grouped_gemm_dispatch.h grouped_gemm_select_tile()): four
// small/medium-avg_m tiles (tile ids 0-3), the original "small" 256x128
// (half 256x64) config (tile id 4), and the large-avg_m 256x256 (half
// 256x128) "Config 1" config from design.md (tile id 5).
#include "csrc/utils.h"
#include "moe_grouped_mm_interleaved.h"

#ifdef VLLM_XPU_ENABLE_XE2
  #include "moe_interleaved_launcher.hpp"

using namespace vllm_xpu::moe_interleaved;
using namespace cute;

namespace {

// Small-avg_m tiles, ported from sgl-kernel-xpu's
// GroupGemmXe20Interleaved.cpp fix for the decode-shape regression
// documented in docs/moe_interleaved_fusion.md ("Future work": instantiate
// the interleaved kernel at the remaining small-avg_m tile configs so it is
// tile-matched to the dual-accumulator selector at every avg_m). Each
// TileFull's N is exactly 2x the corresponding dual-accumulator tile's
// (real, un-interleaved) N -- e.g. dual-accum tile 0 is Shape<_8,_64,_32>
// (real N=64); the interleaved equivalent tiles the full un-halved
// (gate+up) accumulator at N=128 and halves it back down to 64 for the
// output-space TileHalf. SGLayout is reused as-is from the matching
// dual-accum tile id.
using Tile_8_128_32 = Shape<_8, _128, _32>;
using Tile_8_64_32 = Shape<_8, _64, _32>;
using Tile_16_128_32 = Shape<_16, _128, _32>;
using Tile_16_64_32 = Shape<_16, _64, _32>;
using Tile_32_128_32 = Shape<_32, _128, _32>;
using Tile_32_64_32 = Shape<_32, _64, _32>;
using Tile_128_128_32 = Shape<_128, _128, _32>;
using Tile_128_64_32 = Shape<_128, _64, _32>;
using Tile_256_128_32 = Shape<_256, _128, _32>;
using Tile_256_64_32 = Shape<_256, _64, _32>;
using Tile_256_256_32 = Shape<_256, _256, _32>;
// Matches sgl-kernel-xpu grouped_gemm_dispatch.h's SGL_MOE_GG_LAYOUT_0/1/2
// (tile ids 0-2).
using SG_1_4_1 = Layout<Shape<_1, _4, _1>, Stride<_4, _1, _0>>;
// Matches SGL_MOE_GG_LAYOUT_3/4 (tile ids 3-4).
using SG_4_2_1 = Layout<Shape<_4, _2, _1>, Stride<_2, _1, _0>>;
using SG_8_2_1 = Layout<Shape<_8, _2, _1>, Stride<_2, _1, _0>>;
using SG_8_4_1 = Layout<Shape<_8, _4, _1>, Stride<_4, _1, _0>>;

// This threshold matches sgl-kernel-xpu's
// kernels/moe/xe20/bf16/grouped_gemm_dispatch.h::kGroupedGemmSmallWeightThreshold.
// vllm-xpu-kernels has no dual-accumulator grouped-GEMM path to share the
// constant with, so it is duplicated here rather than pulled in via an
// unrelated include.
constexpr int64_t kGroupedGemmSmallWeightThreshold = 4096LL * 4096LL;

// Tile selection for the interleaved path, mirroring sgl-kernel-xpu's
// grouped_gemm_dispatch.h::grouped_gemm_select_tile() (the dual-accumulator
// selector) as closely as possible, so both fused paths step down tiles at
// the same avg_m breakpoints. Differences from the dual-accumulator version:
//   - This op is always "fused" (there is no non-fused interleaved variant),
//     so branches that pick between a fused/non-fused tile id there always
//     resolve to this op's single fused choice here.
//   - `gemm_n` is the FULL interleaved (gate+up) width; `real_n = gemm_n / 2`
//     is used for the small-weight/narrow-N checks to match the semantics
//     grouped_gemm_select_tile applies to the un-interleaved N.
//   - Tile id 5 ("Config 1", 256x256 full / 256x128 half, SG 8x4) has no
//     dual-accumulator analog reachable while fused; it is this op's
//     original large-avg_m tile, unchanged from interleaved_use_config1()'s
//     prior avg_m >= 256 && gemm_n >= 256 gate.
//
// Tile ids (TileFull/TileHalf/SGLayout):
//   0: 8x128  / 8x64   / SG_1_4_1   (dual-accum tile 0 analog, real N=64)
//   1: 16x128 / 16x64  / SG_1_4_1   (dual-accum tile 1 analog)
//   2: 32x128 / 32x64  / SG_1_4_1   (dual-accum tile 2 analog)
//   3: 128x128/ 128x64 / SG_4_2_1   (dual-accum tile 3 analog, fused-only)
//   4: 256x128/ 256x64 / SG_8_2_1   (dual-accum tile 5 analog, fused-only;
//                                    this op's original "small" tile)
//   5: 256x256/ 256x128/ SG_8_4_1   (design.md "Config 1"; this op's
//                                    original "large" tile)
inline int interleaved_select_tile(int avg_m, int gemm_k, int gemm_n) {
  const int real_n = gemm_n / 2;
  const bool small_weight =
      static_cast<int64_t>(gemm_k) * real_n <= kGroupedGemmSmallWeightThreshold;
  const bool narrow_k = gemm_k <= 256;
  const bool narrow_n_fused = real_n <= 512;

  if (avg_m >= 256 && gemm_n >= 256) return 5;
  if (avg_m <= 8) return 0;
  if (avg_m <= 16 && small_weight) return 1;
  if (avg_m <= 32 && small_weight) return 2;
  if (avg_m <= 128 && small_weight) return 3;
  if (narrow_k) return 3;
  if (narrow_n_fused) return 3;
  return 4;
}

template <ActivationType ActType, bool WithBias>
void launch_interleaved(
    sycl::queue queue,
    int tile_id,
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
  #define CALL_INTERLEAVED_LAUNCHER(TileFull, TileHalf, SGL) \
    Xe20MoEGEMMInterleavedLauncher<                          \
        TileFull,                                            \
        TileHalf,                                            \
        SGL,                                                 \
        ActType,                                             \
        WithBias>(                                           \
        queue,                                               \
        activations,                                         \
        weights,                                             \
        bias,                                                \
        outputs,                                             \
        gemm_n,                                              \
        gemm_k,                                              \
        rows_per_expert,                                     \
        num_experts,                                         \
        workspace,                                           \
        gemm1_alpha,                                         \
        gemm1_limit,                                         \
        ld_b)

  switch (tile_id) {
    case 0:
      CALL_INTERLEAVED_LAUNCHER(Tile_8_128_32, Tile_8_64_32, SG_1_4_1);
      break;
    case 1:
      CALL_INTERLEAVED_LAUNCHER(Tile_16_128_32, Tile_16_64_32, SG_1_4_1);
      break;
    case 2:
      CALL_INTERLEAVED_LAUNCHER(Tile_32_128_32, Tile_32_64_32, SG_1_4_1);
      break;
    case 3:
      CALL_INTERLEAVED_LAUNCHER(Tile_128_128_32, Tile_128_64_32, SG_4_2_1);
      break;
    case 4:
      CALL_INTERLEAVED_LAUNCHER(Tile_256_128_32, Tile_256_64_32, SG_8_2_1);
      break;
    default:
      CALL_INTERLEAVED_LAUNCHER(Tile_256_256_32, Tile_256_128_32, SG_8_4_1);
      break;
  }
  #undef CALL_INTERLEAVED_LAUNCHER
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
  TORCH_CHECK(
      weights.size(0) == num_experts,
      "weights must have num_experts as the first dimension");
  TORCH_CHECK(
      weights.size(0) == rows_per_expert.size(0),
      "rows_per_expert must have the same size as weights.size(0)");
  TORCH_CHECK(
      output.size(0) == total_m,
      "output must have the same number of rows as activations");
  TORCH_CHECK(
      output.size(1) == gemm_n / 2,
      "output must have half the number of columns as the interleaved weights");
  TORCH_CHECK(
      gemm_n % 32 == 0,
      "interleaved N must be a multiple of 32 (16-wide gate/up pairs)");
  TORCH_CHECK(
      num_experts % 8 == 0,
      "num_experts must be a multiple of 8 for the current implementation");
  TORCH_CHECK(
      activation_type == static_cast<int64_t>(MoE::ActivationType::SILU) ||
          activation_type == static_cast<int64_t>(MoE::ActivationType::GELU),
      "moe_grouped_mm_xe20_interleaved only supports silu (0) or gelu (1)");
  TORCH_CHECK(
      activations.scalar_type() == weights.scalar_type(),
      "activations and weights must have the same data type");
  TORCH_CHECK(
      activations.scalar_type() == at::ScalarType::BFloat16,
      "Only bfloat16 is supported by moe_grouped_mm_xe20_interleaved "
      "currently");
  bool with_bias = bias.has_value();
  if (with_bias) {
    TORCH_CHECK(
        bias->scalar_type() == at::kFloat,
        "bias must be float32 to match kernel expectations");
    TORCH_CHECK(
        bias->dim() == 2,
        "bias must be 2D [n_experts, N] (already interleaved to match "
        "weights)");
    TORCH_CHECK(
        bias->size(0) == num_experts && bias->size(1) == gemm_n,
        "bias shape mismatch with weight");
  }

  auto stream = at::xpu::getCurrentXPUStream();
  auto queue = stream.queue();
  at::Tensor atomic_buffer =
      at::empty({1}, activations.options().dtype(at::kInt));
  const void* bias_ptr = with_bias ? bias->data_ptr() : nullptr;
  int ld_b = static_cast<int>(weights.stride(1));
  const int avg_m = num_experts > 0 ? static_cast<int>(total_m / num_experts)
                                    : static_cast<int>(total_m);
  const int tile_id = interleaved_select_tile(
      avg_m, static_cast<int>(gemm_k), static_cast<int>(gemm_n));

  if (activation_type == static_cast<int64_t>(MoE::ActivationType::SILU)) {
    if (with_bias) {
      launch_interleaved<MoE::ActivationType::SILU, true>(
          queue,
          tile_id,
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
          tile_id,
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
          tile_id,
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
          tile_id,
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
