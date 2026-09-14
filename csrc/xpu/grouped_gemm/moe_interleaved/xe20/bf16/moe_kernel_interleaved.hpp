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

// Per-expert grouped-GEMM driver for the single-accumulator interleaved SwiGLU
// fusion (see moe_mainloop_interleaved.hpp). Structurally the same
// tile-scheduling loop as moe_kernel.hpp's MoEGEMM, simplified to the one case
// this path supports: a single (already gate/up-interleaved) weight tensor
// per expert, tiled at the *un-halved* width, producing a half-width output.
//
// Weight/bias interleaving (grouping gate/up columns in 16-wide pairs) is a
// one-time, host-side layout transform applied when weights are loaded
// (see python/sgl_kernel/moe.py's `interleave_gate_up_weights`); this kernel
// only consumes the already-interleaved layout.
#pragma once

#include "../common/block_2d_copy_d.hpp"
#include "cute/tensor.hpp"
#include "cutlass/cutlass.h"
#include "cutlass/gemm/gemm.h"
#include "cutlass/gemm/group_array_problem_shape.hpp"
#include "cutlass/gemm/kernel/tile_scheduler.hpp"
#include "cutlass/kernel_hardware_info.hpp"
#include "cutlass/platform/platform.h"
#include "cutlass/util/packed_stride.hpp"
#include "moe_mainloop_interleaved.hpp"

#pragma clang diagnostic ignored "-Wpass-failed"
#pragma clang diagnostic ignored "-Wdeprecated-declarations"

namespace MoE {
using namespace cute;

template <
    typename TileShape,      // accumulator-space tile (un-halved); e.g. Shape<_256,_64,_32>
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
    typename GmemTiledCopyD = void>
class MoEGEMMInterleaved {
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
    const ElementB* Weights;  // pre-interleaved gate/up columns, [num_experts, N, ld_b]
    const float* Bias;        // pre-interleaved to match Weights, [num_experts, N], only read if WithBias
    ElementD* Outputs;        // half-width output, [total_rows, N/2]
    const int32_t* M_per_group;
    const int32_t N;  // full (gate+up) width
    const int32_t K;
    const int32_t num_experts;
    int32_t* workspace;
    TiledMMA mma;
    TiledMMAHalf mma_half;
    int32_t ld_b;
    float gemm1_alpha = 1.702f;
    float gemm1_limit = 7.0f;
  };

  auto make_B_tensor(ElementB* ptr_B, int N, int K, int ld_b) {
    return make_tensor(make_gmem_ptr<ElementB>(ptr_B), make_layout(make_shape(N, K), make_stride(ld_b, _1{})));
  }

  auto make_Bias_tensor(float* ptr_Bias, int N) {
    if constexpr (WithBias) {
      return make_tensor(make_gmem_ptr<float>(ptr_Bias), make_layout(make_shape(N), make_stride(_1{})));
    } else {
      return make_tensor(make_gmem_ptr<float>(nullptr), make_layout(make_shape(0), make_stride(_1{})));
    }
  }

  auto make_D_tensor(ElementD* ptr_D, int pre_rows, int M, int N) {
    // Output is N/2 wide: gate*up fusion halves the interleaved N-width B.
    return make_tensor(
        make_gmem_ptr<ElementD>(ptr_D + pre_rows * N / 2), make_layout(make_shape(M, N / 2), make_stride(N / 2, _1{})));
  }

  void operator()(Params const& params, sycl::nd_item<3> item, int32_t* slm_mem) {
    auto N = params.N;
    auto K = params.K;
    auto M_per_group = params.M_per_group;
    auto num_experts = params.num_experts;
    auto mma = params.mma;
    auto mma_half = params.mma_half;
    auto workspace = params.workspace;

    auto wg_tile = mma.tile_mnk();
    auto wg_tile_m = get<0>(wg_tile);
    auto wg_tile_n = get<1>(wg_tile);

    int group_id = item.get_group_linear_id();
    // Tile over the *full* interleaved N (like the unfused/non-gated path in
    // moe_kernel.hpp), not N/2: each wg_tile_n-wide B chunk decodes to a
    // wg_tile_n/2-wide D chunk (see moe_mainloop_interleaved.hpp).
    int N_pad = ceil_div(N, wg_tile_n) * wg_tile_n;
    int group_m_id = (group_id * wg_tile_n) / N_pad;
    int group_range = item.get_group_range(1);
    int32_t thr_id = int32_t(item.get_local_linear_id());

    if (group_id == 0 && thr_id == 0) {
      auto atm = sycl::atomic_ref<
          int,
          sycl::memory_order::relaxed,
          sycl::memory_scope::device,
          sycl::access::address_space::global_space>(workspace[0]);
      atm.store(0);
    }

    int pre_rows = 0;
    int pre_tiles = 0;
    for (int i = 0; i < num_experts; ++i) {
      int M = M_per_group[i];
      int cumsum_rows_for_experts = M + pre_rows;
      int cumsum_tiles_for_experts = (M + wg_tile_m - 1) / wg_tile_m + pre_tiles;

      if (group_m_id >= cumsum_tiles_for_experts) {
        pre_rows = cumsum_rows_for_experts;
        pre_tiles = cumsum_tiles_for_experts;
        continue;
      }

      int expert_id = i;
      int ld_b = params.ld_b;
      int64_t B_offset = static_cast<int64_t>(expert_id) * static_cast<int64_t>(N) * static_cast<int64_t>(ld_b);
      ElementA* ptr_A_curr_batch = const_cast<ElementA*>(params.Activations) + pre_rows * K;
      ElementB* ptr_B_curr_batch = const_cast<ElementB*>(params.Weights) + B_offset;
      float* ptr_Bias_curr_batch = nullptr;
      if constexpr (WithBias) {
        ptr_Bias_curr_batch = const_cast<float*>(params.Bias) + expert_id * N;
      }

      auto A_tensor =
          make_tensor(make_gmem_ptr<ElementA>(ptr_A_curr_batch), make_layout(make_shape(M, K), make_stride(K, _1{})));
      auto B_tensor = make_B_tensor(ptr_B_curr_batch, N, K, ld_b);
      auto D_tensor = make_D_tensor(params.Outputs, pre_rows, M, N);
      auto Bias_tensor = make_Bias_tensor(ptr_Bias_curr_batch, N);

      while (group_m_id < cumsum_tiles_for_experts) {
        int n_coord = (group_id * wg_tile_n) % N_pad / wg_tile_n;
        int m_coord = (group_m_id - pre_tiles);

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
        group_m_id = (group_id * wg_tile_n) / N_pad;
      }
      pre_rows = cumsum_rows_for_experts;
      pre_tiles = cumsum_tiles_for_experts;
    }
  };
};
}  // namespace MoE
