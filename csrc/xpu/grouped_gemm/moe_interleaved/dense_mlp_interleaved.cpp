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
using Tile_128_256_32 = Shape<_128, _256, _32>;
using SG_1_4_1 = Layout<Shape<_1, _4, _1>, Stride<_4, _1, _0>>;
using SG_4_2_1 = Layout<Shape<_4, _2, _1>, Stride<_2, _1, _0>>;
using SG_8_2_1 = Layout<Shape<_8, _2, _1>, Stride<_2, _1, _0>>;
using SG_8_4_1 = Layout<Shape<_8, _4, _1>, Stride<_4, _1, _0>>;
using SG_4_4_1 = Layout<Shape<_4, _4, _1>, Stride<_4, _1, _0>>;

// Tile selection for the dense (non-grouped) path. Originally mirrored
// moe_grouped_mm_interleaved.cpp's interleaved_select_tile() breakpoints
// verbatim, gating the smaller M-matched tiles (1/2/3) behind a
// "small_weight" (K*N) heuristic. That heuristic was calibrated for MoE's
// *averaged* per-expert M (avg_m), where a large per-expert weight combined
// with a small avg_m still implies enough total work across experts to
// justify a big tile. In the dense case there is only one, exact M (no
// per-expert averaging), and a direct tile-id sweep
// (benchmark/sweep_dense_mlp_tile_ids.py) across Qwen3.6-27B's actual
// TP1/2/4 shapes showed the weight-size gate was actively harmful: it forced
// small M (32, 128) into the oversized 256-row tile 4 even though N was
// always large enough to fail "small_weight" at every TP degree, when the
// exactly-M-matched tile (2 or 3) was faster in every case tested (e.g. TP4
// M=128: oversized tile4 was 28% *slower* than oneDNN; the matched tile3 is
// 5.5% *faster*). So tile selection here depends on M alone; only the
// M>=256 Config-1 path additionally checks gemm_n, matching its
// swizzle-tuning scope (see dense_kernel_interleaved.hpp's KSwizzleM).
//
// Within the M>=256 regime, tile 5 (256x256, SG 8x4x1, swizzled) and tile 8
// (128x128, SG 4x4x1, swizzled) trade places depending on shape: total
// kernel time is dominated by "wave count" (ceil(total_tiles / 32), this
// platform has 32 subslices) when total_tiles is small, since a partial
// final wave leaves subslices idle. Tile 5's larger 256x256 tile produces
// far fewer total tiles at small N (e.g. TP4's per-rank N=8704 -> only 34
// N-tiles), so a moderate M can land far from a clean multiple of 32
// ("wave-quantization tail"), while tile 8's smaller 128x128 tile produces
// 4x more, finer-grained tiles that land close to a clean wave boundary
// much sooner. Once M grows large enough that tile 5's own total_tiles is
// already near a wave boundary, tile 5's superior per-tile reuse/arithmetic
// intensity wins outright (confirmed by direct measurement: TP4 M=2048/4096
// clearly favor tile 5 despite tile 8 having equal or better predicted wave
// alignment there -- the wave-count model alone is not sufficient at that
// end of the range). Empirically (TP1/2/4 x M in {256,512,1024,2048,4096}),
// the crossover is well predicted by tile 5's own wave-inflation ratio
// (ceil(total5/32) / (total5/32.0)): pick tile 8 when that ratio is >=1.15,
// else tile 5. This reproduces the measured crossover (TP4: tile8 wins at
// M=256/512/1024, tile5 wins at M=2048/4096; TP1/TP2 M=512 stay on tile5,
// where tile8 was verified to be a no-op/tiny-win, not a regression).
inline int dense_select_tile(int m, int gemm_k, int gemm_n) {
  (void)gemm_k;  // no longer used for tile selection; see comment above.
  if (m >= 256 && gemm_n >= 256) {
    int m_tiles5 = (m + 255) / 256;
    int n_tiles5 = (gemm_n + 255) / 256;
    int total5 = m_tiles5 * n_tiles5;
    constexpr int kSubslices = 32;
    int waves5 = (total5 + kSubslices - 1) / kSubslices;
    double infl5 = (waves5 * static_cast<double>(kSubslices)) / total5;
    return (infl5 >= 1.15) ? 8 : 5;
  }
  if (m <= 8) return 0;
  if (m <= 16) return 1;
  if (m <= 32) return 2;
  if (m <= 128) return 3;
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
      CALL_DENSE_INTERLEAVED_LAUNCHER_SWZ(Tile_128_128_32, Tile_128_64_32, SG_4_2_1, 8);
      break;
    case 4:
      CALL_DENSE_INTERLEAVED_LAUNCHER_SWZ(Tile_256_128_32, Tile_256_64_32, SG_8_2_1, 8);
      break;
    case 6:
      // Experimental: same 256x128 accumulator tile as tile 4, but with a
      // 4x4x1 (16-subgroup) partition instead of 8x2x1 -- same per-subgroup
      // register footprint (256/4 x 128/4 = 64x32, vs tile4's 256/8 x 128/2
      // = 32x64), same total area, different aspect ratio/occupancy profile.
      // Added to probe whether this narrows the TP4 M=512 gap vs. tile 5.
      // Swizzle=8 applied per user request, matching tile 5's Config-1 raster
      // swizzle.
      CALL_DENSE_INTERLEAVED_LAUNCHER_SWZ(Tile_256_128_32, Tile_256_64_32, SG_4_4_1, 8);
      break;
    case 7:
      // 128x256 accumulator, SG 4x4x1: per-SG 128/4 x 256/4 = 32x64 (same
      // 2048 area as above), transposed aspect ratio vs. case 6. Swizzle=8
      // applied per user request.
      CALL_DENSE_INTERLEAVED_LAUNCHER_SWZ(Tile_128_256_32, Tile_128_128_32, SG_4_4_1, 8);
      break;
    case 8:
      // 128x128 accumulator, SG 4x4x1: per-SG 128/4 x 128/4 = 32x32 (1024
      // area -- half the footprint of the other experimental configs),
      // trading tile size (more, smaller tiles) for lower register pressure.
      // Swizzle=8 applied per user request.
      CALL_DENSE_INTERLEAVED_LAUNCHER_SWZ(Tile_128_128_32, Tile_128_64_32, SG_4_4_1, 8);
      break;
    case 9:
      // 128x128 accumulator, SG 8x4x1 (32 subgroups/WG -- same subgroup
      // *count* as tile 5/default, which oneDNN's own unconstrained
      // gemmstone catalog dispatch independently confirms is the
      // occupancy-optimal WG size at this TP4/M=512 shape), but with tile
      // 5's coarse 256x256 accumulator swapped for tile 8's finer 128x128
      // one. Motivation: tile 5 loses to tile 8 at M=512 purely from
      // wave-count quantization tail (256-row tiles don't divide 512/N
      // evenly into a clean multiple of 32 subslices), not from having the
      // "wrong" subgroup count -- so this probes whether combining tile 8's
      // finer granularity with tile 5's wider (8x4) subgroup layout beats
      // both. Per-SG tile: 128/8 x 128/4 = 16x32.
      //
      // RESULT (measured, TP1/2/4 x M in {256,512,1024,2048,4096}): tile 9
      // is uniformly *worse* than both tile 5 and tile 8 at every point
      // (e.g. TP4/M=512: tile9 0.428ms vs tile5 0.392ms vs tile8 0.374ms;
      // gap widens at larger M, e.g. TP1/M=4096: tile9 14.19ms vs tile5
      // 9.12ms). Mechanically swapping in oneDNN's preferred WG8x4 subgroup
      // count while keeping a small 128x128 accumulator shrinks the
      // per-subgroup tile to a very thin 16x32, which cuts per-subgroup
      // data reuse/arithmetic intensity far more than the extra parallelism
      // gains back -- confirming that oneDNN's WG8x4 preference in the
      // gemmstone catalog is tied to its own much larger accompanying
      // unroll (32x48 in the winning catalog entry), not a generically
      // transferable "more subgroups is better" rule. Kept here only as a
      // documented negative result; not reachable via dense_select_tile().
      CALL_DENSE_INTERLEAVED_LAUNCHER_SWZ(Tile_128_128_32, Tile_128_64_32, SG_8_4_1, 8);
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
    double gemm1_limit,
    int64_t tile_id_override) {
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
  const int tile_id = tile_id_override >= 0
      ? static_cast<int>(tile_id_override)
      : dense_select_tile(
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
