// SPDX-License-Identifier: Apache-2.0
//
// Host-side op wrapper for the plain (non-grouped) dense-MLP interleaved
// SwiGLU GEMM fusion -- the dense-MLP analog of
// moe_grouped_mm_interleaved.cpp. Reuses the exact same tile-selection
// breakpoints as the MoE path (interleaved_select_tile()), since the
// underlying fused GEMM+SwiGLU math and tile configs are identical; the only
// difference is that `avg_m` here is simply the dense batch's raw row count
// (there is no per-expert averaging).
#include "csrc/utils.h"
#include "dense_mlp_interleaved.h"

#ifdef VLLM_XPU_ENABLE_XE2
  #include "dense_interleaved_launcher.hpp"

using namespace vllm_xpu::dense_mlp_interleaved;
using namespace cute;

namespace {

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
using SG_1_4_1 = Layout<Shape<_1, _4, _1>, Stride<_4, _1, _0>>;
using SG_4_2_1 = Layout<Shape<_4, _2, _1>, Stride<_2, _1, _0>>;
using SG_8_2_1 = Layout<Shape<_8, _2, _1>, Stride<_2, _1, _0>>;
using SG_8_4_1 = Layout<Shape<_8, _4, _1>, Stride<_4, _1, _0>>;

// See moe_grouped_mm_interleaved.cpp's copy of this constant for provenance
// (sgl-kernel-xpu's kGroupedGemmSmallWeightThreshold); duplicated here for
// the same reason (no shared dual-accumulator dense path to pull it from).
constexpr int64_t kGroupedGemmSmallWeightThreshold = 4096LL * 4096LL;

// Identical breakpoints to moe_grouped_mm_interleaved.cpp's
// interleaved_select_tile() -- see that file for the full tile-id table and
// rationale. Here `m` is the dense batch's raw row count (no per-expert
// averaging).
inline int dense_select_tile(int m, int gemm_k, int gemm_n) {
  const int real_n = gemm_n / 2;
  const bool small_weight =
      static_cast<int64_t>(gemm_k) * real_n <= kGroupedGemmSmallWeightThreshold;
  const bool narrow_k = gemm_k <= 256;
  const bool narrow_n_fused = real_n <= 512;

  if (m >= 256 && gemm_n >= 256) return 5;
  if (m <= 8) return 0;
  if (m <= 16 && small_weight) return 1;
  if (m <= 32 && small_weight) return 2;
  if (m <= 128 && small_weight) return 3;
  if (narrow_k) return 3;
  if (narrow_n_fused) return 3;
  return 4;
}

template <ActivationType ActType, bool WithBias>
void launch_dense_interleaved(
    sycl::queue queue,
    int tile_id,
    const void* activations,
    const void* weight,
    const void* bias,
    void* outputs,
    int gemm_m,
    int gemm_n,
    int gemm_k,
    int* workspace,
    float gemm1_alpha,
    float gemm1_limit,
    int ld_b) {
  #define CALL_DENSE_INTERLEAVED_LAUNCHER(TileFull, TileHalf, SGL) \
    Xe20DenseMLPGEMMInterleavedLauncher<                           \
        TileFull,                                                  \
        TileHalf,                                                  \
        SGL,                                                       \
        ActType,                                                   \
        WithBias>(                                                 \
        queue,                                                     \
        activations,                                               \
        weight,                                                    \
        bias,                                                      \
        outputs,                                                   \
        gemm_m,                                                    \
        gemm_n,                                                    \
        gemm_k,                                                    \
        workspace,                                                 \
        gemm1_alpha,                                                \
        gemm1_limit,                                                \
        ld_b)

  // Config-1-only raster swizzle (design.md's swizzle=8): empirically this
  // only helps the large-M/256x256-tile case (case 5, the "default" below)
  // and *hurts* the smaller tile configs (0-4), so it is opted into via an
  // explicit KSwizzleM=8 template arg only on that one instantiation.
  #define CALL_DENSE_INTERLEAVED_LAUNCHER_SWZ(TileFull, TileHalf, SGL, Swz) \
    Xe20DenseMLPGEMMInterleavedLauncher<                                   \
        TileFull,                                                          \
        TileHalf,                                                          \
        SGL,                                                               \
        ActType,                                                           \
        WithBias,                                                          \
        Swz>(                                                              \
        queue,                                                             \
        activations,                                                       \
        weight,                                                            \
        bias,                                                              \
        outputs,                                                           \
        gemm_m,                                                            \
        gemm_n,                                                            \
        gemm_k,                                                            \
        workspace,                                                         \
        gemm1_alpha,                                                        \
        gemm1_limit,                                                        \
        ld_b)

  switch (tile_id) {
    case 0:
      CALL_DENSE_INTERLEAVED_LAUNCHER(Tile_8_128_32, Tile_8_64_32, SG_1_4_1);
      break;
    case 1:
      CALL_DENSE_INTERLEAVED_LAUNCHER(Tile_16_128_32, Tile_16_64_32, SG_1_4_1);
      break;
    case 2:
      CALL_DENSE_INTERLEAVED_LAUNCHER(Tile_32_128_32, Tile_32_64_32, SG_1_4_1);
      break;
    case 3:
      CALL_DENSE_INTERLEAVED_LAUNCHER(Tile_128_128_32, Tile_128_64_32, SG_4_2_1);
      break;
    case 4:
      CALL_DENSE_INTERLEAVED_LAUNCHER(Tile_256_128_32, Tile_256_64_32, SG_8_2_1);
      break;
    default:
      CALL_DENSE_INTERLEAVED_LAUNCHER_SWZ(Tile_256_256_32, Tile_256_128_32, SG_8_4_1, 8);
      break;
  }
  #undef CALL_DENSE_INTERLEAVED_LAUNCHER
  #undef CALL_DENSE_INTERLEAVED_LAUNCHER_SWZ
}

}  // namespace
#endif  // VLLM_XPU_ENABLE_XE2

torch::Tensor dense_swiglu_gemm_xe20_interleaved(
    torch::Tensor& output,
    const torch::Tensor& activations,
    const torch::Tensor& weight,
    const c10::optional<at::Tensor>& bias,
    int64_t activation_type,
    double gemm1_alpha,
    double gemm1_limit) {
#ifndef VLLM_XPU_ENABLE_XE2
  TORCH_CHECK(
      false,
      "dense_swiglu_gemm_xe20_interleaved requires a build with "
      "VLLM_XPU_ENABLE_XE2=1 (Xe2/Xe20, e.g. Arc B-series).");
#else
  TORCH_CHECK(
      vllm::xpu::is_xe2_arch(),
      "dense_swiglu_gemm_xe20_interleaved is only implemented for Xe2/Xe20 "
      "hardware (e.g. Arc B-series).");

  int64_t gemm_m = activations.size(0);
  int64_t gemm_k = activations.size(1);
  int64_t gemm_n = weight.size(0);  // full (interleaved gate+up) width

  TORCH_CHECK(weight.dim() == 2, "weight must be 2D [N, K] (no expert dim)");
  TORCH_CHECK(
      weight.size(1) == gemm_k,
      "weight's K dim must match activations' K dim");
  TORCH_CHECK(
      output.size(0) == gemm_m,
      "output must have the same number of rows as activations");
  TORCH_CHECK(
      output.size(1) == gemm_n / 2,
      "output must have half the number of columns as the interleaved weight");
  TORCH_CHECK(
      gemm_n % 32 == 0,
      "interleaved N must be a multiple of 32 (16-wide gate/up pairs)");
  TORCH_CHECK(
      activation_type == static_cast<int64_t>(MoE::ActivationType::SILU) ||
          activation_type == static_cast<int64_t>(MoE::ActivationType::GELU),
      "dense_swiglu_gemm_xe20_interleaved only supports silu (0) or gelu (1)");
  TORCH_CHECK(
      activations.scalar_type() == weight.scalar_type(),
      "activations and weight must have the same data type");
  TORCH_CHECK(
      activations.scalar_type() == at::ScalarType::BFloat16,
      "Only bfloat16 is supported by dense_swiglu_gemm_xe20_interleaved "
      "currently");
  bool with_bias = bias.has_value();
  if (with_bias) {
    TORCH_CHECK(
        bias->scalar_type() == at::kFloat,
        "bias must be float32 to match kernel expectations");
    TORCH_CHECK(bias->dim() == 1, "bias must be 1D [N] (already interleaved to match weight)");
    TORCH_CHECK(bias->size(0) == gemm_n, "bias shape mismatch with weight");
  }

  auto stream = at::xpu::getCurrentXPUStream();
  auto queue = stream.queue();
  at::Tensor atomic_buffer =
      at::empty({1}, activations.options().dtype(at::kInt));
  const void* bias_ptr = with_bias ? bias->data_ptr() : nullptr;
  int ld_b = static_cast<int>(weight.stride(0));
  const int tile_id = dense_select_tile(
      static_cast<int>(gemm_m), static_cast<int>(gemm_k), static_cast<int>(gemm_n));

  if (activation_type == static_cast<int64_t>(MoE::ActivationType::SILU)) {
    if (with_bias) {
      launch_dense_interleaved<MoE::ActivationType::SILU, true>(
          queue,
          tile_id,
          activations.data_ptr(),
          weight.data_ptr(),
          bias_ptr,
          output.data_ptr(),
          static_cast<int>(gemm_m),
          static_cast<int>(gemm_n),
          static_cast<int>(gemm_k),
          atomic_buffer.data_ptr<int>(),
          static_cast<float>(gemm1_alpha),
          static_cast<float>(gemm1_limit),
          ld_b);
    } else {
      launch_dense_interleaved<MoE::ActivationType::SILU, false>(
          queue,
          tile_id,
          activations.data_ptr(),
          weight.data_ptr(),
          bias_ptr,
          output.data_ptr(),
          static_cast<int>(gemm_m),
          static_cast<int>(gemm_n),
          static_cast<int>(gemm_k),
          atomic_buffer.data_ptr<int>(),
          static_cast<float>(gemm1_alpha),
          static_cast<float>(gemm1_limit),
          ld_b);
    }
  } else {
    if (with_bias) {
      launch_dense_interleaved<MoE::ActivationType::GELU, true>(
          queue,
          tile_id,
          activations.data_ptr(),
          weight.data_ptr(),
          bias_ptr,
          output.data_ptr(),
          static_cast<int>(gemm_m),
          static_cast<int>(gemm_n),
          static_cast<int>(gemm_k),
          atomic_buffer.data_ptr<int>(),
          static_cast<float>(gemm1_alpha),
          static_cast<float>(gemm1_limit),
          ld_b);
    } else {
      launch_dense_interleaved<MoE::ActivationType::GELU, false>(
          queue,
          tile_id,
          activations.data_ptr(),
          weight.data_ptr(),
          bias_ptr,
          output.data_ptr(),
          static_cast<int>(gemm_m),
          static_cast<int>(gemm_n),
          static_cast<int>(gemm_k),
          atomic_buffer.data_ptr<int>(),
          static_cast<float>(gemm1_alpha),
          static_cast<float>(gemm1_limit),
          ld_b);
    }
  }
  return output;
#endif  // VLLM_XPU_ENABLE_XE2
}
