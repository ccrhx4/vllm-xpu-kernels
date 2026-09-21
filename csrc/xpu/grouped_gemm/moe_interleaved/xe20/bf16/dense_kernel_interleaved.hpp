/***************************************************************************************************
 * Copyright (C) 2026 Intel Corporation, All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice,
 *this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 *ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 *LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 *CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 *SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 *INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 *CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 *ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 *POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/

// Plain (non-grouped) dense-GEMM driver for the single-accumulator
// interleaved SwiGLU fusion (see moe_mainloop_interleaved.hpp).
//
// This is the dense-MLP analog of moe_kernel_interleaved.hpp's
// MoEGEMMInterleaved: same tile-scheduling loop and same reused
// InterleavedMoEMainloop core (the fused GEMM+SwiGLU math is identical --
// InterleavedMoEMainloop::operator() takes exactly one A/B/D tensor and one
// tile coordinate; it has no "expert"/group concept baked into it at all,
// so it is directly reusable here unmodified). What differs from
// MoEGEMMInterleaved is only the *host-side dispatch*: a dense FFN layer is a
// single, statically-shaped (M,N,K) GEMM, not a set of variable-length
// per-expert row ranges, so there is no `M_per_group` walk, no per-group
// pointer-offset arithmetic, and no `num_experts` parameter at all -- the
// tile grid is a flat (M,N) space computed directly from the one A/B/D
// tensor's shape.
//
// The grid is still launched at a fixed (occupancy-based) work-group count
// and each work-group grid-strides across the flat tile space via the same
// atomic work-stealing counter used by the grouped path (a dense GEMM's
// total tile count can still exceed the number of launched work-groups, so
// the persistent/work-stealing dispatch is still needed -- it's just no
// longer nested inside a per-expert loop).
#pragma once

#include "../common/block_2d_copy_d.hpp"
#include "cute/tensor.hpp"
#include "cutlass/cutlass.h"
#include "cutlass/gemm/gemm.h"
#include "cutlass/kernel_hardware_info.hpp"
#include "cutlass/platform/platform.h"
#include "cutlass/util/packed_stride.hpp"
#include "moe_mainloop_interleaved.hpp"

#pragma clang diagnostic ignored "-Wpass-failed"
#pragma clang diagnostic ignored "-Wdeprecated-declarations"

namespace MoE {
using namespace cute;

template <
    typename TileShape,      // accumulator-space tile (un-halved); e.g. Shape<_256,_256,_32>
    typename TileShapeHalf,  // output-space tile: (get<0>(TileShape), get<1>(TileShape)/2, get<2>(TileShape))
    typename SubgroupLayout,
    typename TensorA,
    typename TensorB,
    typename TensorD,
    typename TensorBias,
    typename TiledMMA,
    typename TiledMMAHalf,
    ActivationType ActType,
    bool WithBias,
    typename ElementA,
    typename ElementB = ElementA,
    typename ElementD = ElementA,
    typename GmemTiledCopyD = void,
    // Raster-swizzle band width (design.md's dispatch "swizzle" knob;
    // Config 1 uses swizzle=8). Default 1 reduces to the original plain
    // row-major raster (m_coord = tile_id / num_n_tiles) -- see operator()
    // below. Only beneficial for large, compute-bound M with the big
    // (256x256) tile; empirically *harmful* for the smaller
    // decode/mid-M tile configs, so callers should only opt into a
    // non-default value for that tile (see dense_select_tile()).
    int KSwizzleM = 1>
class DenseGEMMInterleaved {
 public:
  using TiledCopyA = decltype(make_block_2d_copy_A(TiledMMA{}, TensorA{}));
  using TiledCopyB = decltype(make_block_2d_copy_B(TiledMMA{}, TensorB{}));
  using TiledCopyDHalf = decltype(moe_xe20::make_moe_block_2d_copy_D<GmemTiledCopyD>(TiledMMAHalf{}, TensorD{}));
  using SGPerWG = decltype(product(take<1, 4>(shape(typename TiledMMA::ThrLayoutVMNK{}))));

  constexpr static int Stages = 3;
  using MainloopDispatchPolicy = MoE::XeDefault<Stages>;
  using CollectiveMainloop = InterleavedMoEMainloop<
      MainloopDispatchPolicy,
      TiledCopyA,
      TiledCopyB,
      TiledCopyDHalf,
      TensorA,
      TensorB,
      TensorD,
      TensorBias,
      TiledMMA,
      TiledMMAHalf,
      WithBias,
      ActType>;

  struct Params {
    const ElementA* Activations;
    const ElementB* Weights;  // pre-interleaved gate/up columns, [N, ld_b]
    const float* Bias;        // pre-interleaved to match Weights, [N], only read if WithBias
    ElementD* Outputs;        // half-width output, [M, N/2]
    const int32_t M;
    const int32_t N;  // full (gate+up) width
    const int32_t K;
    int32_t* workspace;
    TiledMMA mma;
    TiledMMAHalf mma_half;
    int32_t ld_b;
    float gemm1_alpha = 1.702f;
    float gemm1_limit = 7.0f;
  };

  auto make_A_tensor(const ElementA* ptr_A, int M, int K) {
    // Params::Activations is const (op inputs are read-only); the mainloop's
    // ATensor/BTensor template params are non-const (matching the MoE path's
    // convention), so const_cast here mirrors make_Bias_tensor below.
    return make_tensor(
        make_gmem_ptr<ElementA>(const_cast<ElementA*>(ptr_A)), make_layout(make_shape(M, K), make_stride(K, _1{})));
  }

  auto make_B_tensor(const ElementB* ptr_B, int N, int K, int ld_b) {
    return make_tensor(
        make_gmem_ptr<ElementB>(const_cast<ElementB*>(ptr_B)),
        make_layout(make_shape(N, K), make_stride(ld_b, _1{})));
  }

  auto make_Bias_tensor(const float* ptr_Bias, int N) {
    if constexpr (WithBias) {
      return make_tensor(make_gmem_ptr<float>(const_cast<float*>(ptr_Bias)), make_layout(make_shape(N), make_stride(_1{})));
    } else {
      return make_tensor(make_gmem_ptr<float>(nullptr), make_layout(make_shape(0), make_stride(_1{})));
    }
  }

  auto make_D_tensor(ElementD* ptr_D, int M, int N) {
    // Output is N/2 wide: gate*up fusion halves the interleaved N-width B.
    return make_tensor(make_gmem_ptr<ElementD>(ptr_D), make_layout(make_shape(M, N / 2), make_stride(N / 2, _1{})));
  }

  void operator()(Params const& params, sycl::nd_item<3> item, int32_t* slm_mem) {
    auto M = params.M;
    auto N = params.N;
    auto K = params.K;
    auto mma = params.mma;
    auto mma_half = params.mma_half;
    auto workspace = params.workspace;

    auto wg_tile = mma.tile_mnk();
    auto wg_tile_m = get<0>(wg_tile);
    auto wg_tile_n = get<1>(wg_tile);

    int group_id = item.get_group_linear_id();
    int32_t thr_id = int32_t(item.get_local_linear_id());

    if (group_id == 0 && thr_id == 0) {
      auto atm = sycl::atomic_ref<
          int,
          sycl::memory_order::relaxed,
          sycl::memory_scope::device,
          sycl::access::address_space::global_space>(workspace[0]);
      atm.store(0);
    }

    // Flat tile grid over the single (M,N) problem -- no expert/group
    // boundaries, no per-group pointer offsets.
    int num_m_tiles = ceil_div(M, wg_tile_m);
    int N_pad = ceil_div(N, wg_tile_n) * wg_tile_n;
    int num_n_tiles = N_pad / wg_tile_n;
    int total_tiles = num_m_tiles * num_n_tiles;
    int group_range = item.get_group_range(1);

    auto A_tensor = make_A_tensor(params.Activations, M, K);
    auto B_tensor = make_B_tensor(params.Weights, N, K, params.ld_b);
    auto D_tensor = make_D_tensor(params.Outputs, M, N);
    auto Bias_tensor = make_Bias_tensor(params.Bias, N);

    // L2/L3-locality raster swizzle (design.md's "swizzle" dispatch knob,
    // Config 1 uses swizzle=8): instead of a plain row-major raster
    // (m_coord = tile_id / num_n_tiles), group tiles into bands of
    // kSwizzleM consecutive M-tiles and sweep N-major *within* each band.
    // Work-groups executing concurrently get consecutive tile_ids, so this
    // keeps them reading the same small band of A-tile rows while they
    // sweep across N -- improving A-tile reuse in L2/L3 across concurrently
    // resident work-groups (the B operand is the one that dominates memory
    // traffic here: N=17408 is far larger than a kSwizzleM-row A band).
    while (group_id < total_tiles) {
      int num_pid_in_band = KSwizzleM * num_n_tiles;
      int band_id = group_id / num_pid_in_band;
      int first_m = band_id * KSwizzleM;
      int band_size_m = min(num_m_tiles - first_m, KSwizzleM);
      int m_coord = first_m + (group_id % band_size_m);
      int n_coord = (group_id % num_pid_in_band) / band_size_m;

      CollectiveMainloop mainloop;
      auto tile_coord = make_coord(m_coord, n_coord, 0);
      mainloop(
          A_tensor,
          B_tensor,
          D_tensor,
          tile_coord,
          mma,
          mma_half,
          thr_id,
          Bias_tensor,
          params.gemm1_alpha,
          params.gemm1_limit);

      if (thr_id == 0) {
        slm_mem[0] = cutlass::atomicAdd(workspace, 1);
      }
      item.barrier(sycl::access::fence_space::local_space);
      group_id = group_range + slm_mem[0];
    }
  };
};
}  // namespace MoE
