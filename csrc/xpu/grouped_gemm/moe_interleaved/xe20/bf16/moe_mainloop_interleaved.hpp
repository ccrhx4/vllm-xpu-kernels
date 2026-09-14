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

// Single-accumulator, interleaved-weight SwiGLU fusion mainloop.
//
// This is the design.md-style alternative to the dual-accumulator mainloop in
// moe_mainloop.hpp: instead of two separate B tensors (gate, up) reduced into
// two separate MMA accumulators (tCrC0, tCrC1) and combined in the epilogue,
// the caller pre-interleaves gate/up weight columns in groups of 16 (the Xe
// DPAS atom's native N-width) into a SINGLE B tensor:
//
//   B[:, 32*r      : 32*r + 16] = gate weights for interim indices [16*r, 16*r+16)
//   B[:, 32*r + 16 : 32*r + 32] = up   weights for interim indices [16*r, 16*r+16)
//
// One GEMM (`cute::gemm(mma, tSrA, tSrB, tCrC)`) then reduces the whole
// interleaved B into a single accumulator `tCrC`. Because the DPAS atom's
// native N-width is 16 and each subgroup lane owns exactly one 16-wide
// atom-column across all its M-rows, accumulator fragment slot `2*j` (an
// "atom-group", see add_bias()'s existing per-thread column-stride math in
// moe_mainloop.hpp, which already establishes this indexing empirically) and
// slot `2*j + 1` are always owned by the *same lane* and are 16 physical
// columns apart -- i.e. exactly the gate/up pair for interim index group `j`.
// Combining them needs zero cross-lane shuffling, zero SLM traffic, and only
// 1x the accumulator register footprint (vs. 2x for the dual-accumulator
// mainloop), matching design.md's `ffn_gemm/xe_fused_swiglu_epilogue.hpp`.
//
// The accumulator is tiled at the *un-halved* CTA/SG tile (`TiledMMA`, same as
// an ordinary single-B GEMM over the full 2*I-wide interleaved B), while the
// output `Y` (M x I) is addressed with an independently-halved TiledMMA
// (`TiledMMAHalf`) purely to size the D-store copy atom and its MMA-shaped
// staging fragment -- mirroring `applications/dual_gemm`'s decoupling of
// accumulator-space and output-space addressing in intel/sycl-tla.

#include <cute/tensor.hpp>
#include <cute/util/compat.hpp>
#include <sycl/ext/intel/experimental/grf_size_properties.hpp>
#include <sycl/sycl.hpp>

#include "../common/activation.hpp"
#include "cutlass/kernel_hardware_info.h"
#include "cutlass/platform/platform.h"
#include "cutlass/tensor_ref.h"
#include "moe_mainloop.hpp"  // MoE::ActivationType, MoE::XeDefault<Stages>

#pragma clang diagnostic ignored "-Wpass-failed"
#pragma clang diagnostic ignored "-Wdeprecated-declarations"

namespace MoE {

using namespace cute;

// Only plain gated activations (SILU, GELU) are supported here. SWIGLU_GPT_OSS
// already uses an interleaved *bias* layout today (stride-2), but the
// mainloop still reduces into two accumulators; unifying it onto this
// single-accumulator path is left as follow-up work (see design.md scope
// note). RELU2 is non-gated and has no gate/up pair to combine.
template <
    class DispatchPolicy_,
    class TiledCopyA_,
    class TiledCopyB_,
    class TiledCopyDHalf_,
    class ATensor_,
    class BTensor_,
    class DTensor_,
    class BiasTensor_,
    class TiledMMA_,
    class TiledMMAHalf_,
    bool WithBias,
    ActivationType ActType>
struct InterleavedMoEMainloop {
  static_assert(cutlass::detail::dependent_false<DispatchPolicy_>, "Could not find a mainloop specialization.");
};

template <
    int Stages,
    class TiledCopyA_,
    class TiledCopyB_,
    class TiledCopyDHalf_,
    class ATensor_,
    class BTensor_,
    class DTensor_,
    class BiasTensor_,
    class TiledMMA_,
    class TiledMMAHalf_,
    bool WithBias,
    ActivationType ActType>
struct InterleavedMoEMainloop<
    XeDefault<Stages>,
    TiledCopyA_,
    TiledCopyB_,
    TiledCopyDHalf_,
    ATensor_,
    BTensor_,
    DTensor_,
    BiasTensor_,
    TiledMMA_,
    TiledMMAHalf_,
    WithBias,
    ActType> {
  static_assert(
      ActType == ActivationType::SILU || ActType == ActivationType::GELU,
      "InterleavedMoEMainloop only supports plain gated activations (SILU, GELU)");

  using TiledMMA = TiledMMA_;
  using TiledMMAHalf = TiledMMAHalf_;
  using TiledCopyA = TiledCopyA_;
  using TiledCopyB = TiledCopyB_;
  using TiledCopyDHalf = TiledCopyDHalf_;
  using ATensor = ATensor_;
  using BTensor = BTensor_;  // single interleaved (N,K) tensor
  using DTensor = DTensor_;  // single (M,N/2) tensor
  using BiasTensor = BiasTensor_;
  InterleavedMoEMainloop() {}

  template <typename Coord>
  CUTLASS_DEVICE void operator()(
      ATensor& A,  // (M,K)
      BTensor& B,  // (N,K), interleaved gate/up columns
      DTensor& D,  // (M,N/2)
      Coord blk_coord,
      TiledMMA mma,
      TiledMMAHalf mma_half,
      int thr_id,       // work-item ID
      BiasTensor Bias,  // (N,), interleaved to match B; only read if WithBias
      float gemm1_alpha,
      float gemm1_limit) {
    auto wg_m = get<0>(blk_coord);
    auto wg_n = get<1>(blk_coord);

    /* Create proxy coordinate tensors for A/B/C, and for the halved D space */
    Tensor cA = make_identity_tensor(A.shape());  // (M,K)
    Tensor cB = make_identity_tensor(B.shape());  // (N,K)
    Tensor cD = make_identity_tensor(D.shape());  // (M,N/2)

    /* init mma (accumulator space: full, un-halved tile) */
    auto wg_tile = mma.tile_mnk();
    auto wg_coord = make_coord(wg_m, wg_n, 0);

    /* init mma_half (output space: halved N tile). Same wg_m/wg_n coordinate:
       the wg_n-th chunk of the full-width interleaved B always maps to the
       wg_n-th chunk of the half-width D (see file-level comment). */
    auto wg_tile_half = mma_half.tile_mnk();

    // Synthetic (M,N) coordinate tensor for the *accumulator* space -- there is
    // no real full-width D tensor to key off (unlike moe_mainloop.hpp's
    // unfused path), since D is already the halved output tensor.
    Tensor cAcc = make_identity_tensor(make_shape(shape<0>(A), shape<0>(B)));  // (M,N)

    Tensor gA = local_tile(cA, select<0, 2>(wg_tile), make_coord(wg_m, _));      // (BLK_M,BLK_K,k)
    Tensor gB = local_tile(cB, select<1, 2>(wg_tile), make_coord(wg_n, _));      // (BLK_N,BLK_K,k)
    Tensor gAcc = local_tile(cAcc, wg_tile, wg_coord, Step<_1, _1, X>{});        // (BLK_M,BLK_N)
    Tensor gD_half = local_tile(cD, wg_tile_half, wg_coord, Step<_1, _1, X>{});  // (BLK_M,BLK_N/2)

    /* Create global -> register copies */
    TiledCopyA tiled_copy_a{A};
    TiledCopyB tiled_copy_b{B};
    TiledCopyDHalf tiled_copy_d_half{D};

    /* Slice TiledCopy/TiledMMA operations down to work-item level */
    auto thr_copy_a = tiled_copy_a.get_slice(thr_id);
    auto thr_copy_b = tiled_copy_b.get_slice(thr_id);
    auto thr_copy_d_half = tiled_copy_d_half.get_slice(thr_id);
    auto thr_mma = mma.get_slice(thr_id);
    auto thr_mma_half = mma_half.get_slice(thr_id);

    /* Partition coordinate tensors for copy */
    auto tAgA = thr_copy_a.partition_S(gA);
    auto tBgB = thr_copy_b.partition_S(gB);

    /* Create register fragments for MMA and copies */
    auto tArA = thr_copy_a.partition_sg_fragment_D(gA(_, _, 0));
    auto tSrA = thr_mma.partition_sg_fragment_A(gA(_, _, 0));

    auto tBrB = thr_copy_b.partition_sg_fragment_D(gB(_, _, 0));
    auto tSrB = thr_mma.partition_sg_fragment_B(gB(_, _, 0));

    /* Full-width (un-halved) MMA accumulator: holds interleaved gate/up pairs */
    SubgroupTensor tCrC = thr_mma.partition_sg_fragment_C(gAcc);

    /* Half-width (output-space) MMA-shaped staging fragment and copy-shaped
       fragment + global target, built from mma_half / tiled_copy_d_half --
       i.e. decoupled from the full-width accumulator tiling above. */
    SubgroupTensor tCrPaired = thr_mma_half.partition_sg_fragment_C(gD_half);
    SubgroupTensor tCrD_final = thr_copy_d_half.partition_sg_fragment_S(gD_half);
    Tensor tCgD = thr_copy_d_half.partition_D(gD_half);

    /* Create TiledCopy objects for prefetches */
    auto prefetch_a = make_block_2d_prefetch(tiled_copy_a);
    auto prefetch_b = make_block_2d_prefetch(tiled_copy_b);

    /* Partition global tensors for prefetch */
    auto pAgA = prefetch_a.get_slice(thr_id).partition_S(gA);
    auto pBgB = prefetch_b.get_slice(thr_id).partition_S(gB);

    constexpr SPIRVScope barrier_scope = ScopeWorkgroup;
    int k_start_idx = 0;
    int prefetch_k = k_start_idx;
    const int prefetch_dist = Stages;
    int k_tile_count = ceil_div(shape<1>(A), get<2>(wg_tile));

    // ------
    // Compute -- single GEMM over the interleaved B, single accumulator.
    // ------

    CUTE_UNROLL
    for (; prefetch_k < prefetch_dist; prefetch_k++) {
      prefetch(prefetch_a, pAgA(_, _, _, prefetch_k));
      prefetch(prefetch_b, pBgB(_, _, _, prefetch_k));
    }

    for (int k_tile = k_start_idx; k_tile < k_tile_count; k_tile++, prefetch_k++) {
      barrier_arrive(barrier_scope);

      copy(tiled_copy_a, tAgA(_, _, _, k_tile), tArA);
      copy(tiled_copy_b, tBgB(_, _, _, k_tile), tBrB);

      if (prefetch_k < k_tile_count) {
        prefetch(prefetch_a, pAgA(_, _, _, prefetch_k));
        prefetch(prefetch_b, pBgB(_, _, _, prefetch_k));
      }

      reorder(tArA, tSrA);
      reorder(tBrB, tSrB);

      cute::gemm(mma, tSrA, tSrB, tCrC);
      barrier_wait(barrier_scope);
    }

    // Add bias if needed. Bias is pre-interleaved to match B's column order,
    // so the existing (unmodified) per-thread column-stride math applies
    // directly to the full-width accumulator -- gate columns pick up the
    // gate bias and up columns pick up the up bias automatically.
    if constexpr (WithBias) {
      constexpr int BLK_M = get<0>(wg_tile);
      constexpr int BLK_N = get<1>(wg_tile);
      add_bias<decltype(tCrC), BLK_M, BLK_N>(Bias, tCrC, mma, wg_n, thr_id);
    }

    // Pair up atom-groups 2*sn (gate) and 2*sn+1 (up) of the SAME accumulator
    // -- 16 physical columns apart, owned by the same lane -- and apply the
    // gated activation, writing the halved result into tCrPaired (an
    // MMA-shaped fragment sized for the output-space tile).
    //
    // tCrC's flat layout is `sn*SG_M + sm` (established empirically by
    // add_bias() above / in moe_mainloop.hpp: incrementing the atom-group
    // index `sn` by 1 steps the physical column by exactly 16 for a fixed
    // thread). Consecutive flat indices instead step `sm` (the M-register
    // index) first, so the pairing must walk (sn, sm) explicitly rather than
    // assume adjacent flat indices are a gate/up pair.
    constexpr auto ATOM_M = get<1>(typename TiledMMA::ThrLayoutVMNK{}.shape());
    constexpr auto ATOM_N = get<2>(typename TiledMMA::ThrLayoutVMNK{}.shape());
    constexpr int BLK_M = get<0>(wg_tile);
    constexpr int BLK_N = get<1>(wg_tile);
    constexpr int SG_M = BLK_M / ATOM_M;
    constexpr int SG_N = BLK_N / ATOM_N;
    constexpr int NUM_ATOM_GROUPS = SG_N / 16;
    static_assert(
        NUM_ATOM_GROUPS % 2 == 0,
        "interleaved gate/up fusion requires an even number of 16-wide atom-groups per subgroup tile");
    constexpr int NUM_PAIRS = NUM_ATOM_GROUPS / 2;
    static_assert(
        NUM_PAIRS * SG_M == decltype(tCrPaired.size())::value,
        "tCrPaired size must match NUM_PAIRS * SG_M derived from the accumulator tiling");

    CUTLASS_PRAGMA_UNROLL
    for (int sn = 0; sn < NUM_PAIRS; ++sn) {
      CUTLASS_PRAGMA_UNROLL
      for (int sm = 0; sm < SG_M; ++sm) {
        float gate = tCrC((2 * sn) * SG_M + sm);
        float up = tCrC((2 * sn + 1) * SG_M + sm);
        tCrPaired(sn * SG_M + sm) =
            moe_xe20::apply_fused_activation<static_cast<int>(ActType)>(gate, up, gemm1_alpha, gemm1_limit);
      }
    }

    reorder(tCrPaired, tCrD_final);
    copy(tiled_copy_d_half, tCrD_final, tCgD);
  }

  template <
      typename tCrC_t,  // Using SubgroupTensor requires template args
      int tile_m,
      int tile_n>
  void add_bias(const BiasTensor& Bias, tCrC_t& tCrC, const TiledMMA& mma, int wg_n, int thr_id) {
    // Identical to moe_mainloop.hpp's MoEMainloop::add_bias -- see reference
    // there:
    // https://github.com/vllm-project/vllm-xpu-kernels/blob/c771759e75529b47d959809c96badfb7d5ba8c88/csrc/xpu/grouped_gemm/xe_2/gemm_xe2.hpp#L178C1-L208C4
    static constexpr auto ATOM_M = get<1>(typename TiledMMA::ThrLayoutVMNK{}.shape());
    static constexpr auto ATOM_N = get<2>(typename TiledMMA::ThrLayoutVMNK{}.shape());

    static constexpr int sg_local_range = 16;
    int sg_local_n_coord = (thr_id / sg_local_range) % ATOM_N;
    int sg_local_id = (thr_id % sg_local_range);

    static constexpr auto SG_M = tile_m / ATOM_M;
    static constexpr auto SG_N = tile_n / ATOM_N;

    int n_tile_start = wg_n * tile_n;
    int n_sg_start = sg_local_n_coord * SG_N;

    CUTLASS_PRAGMA_UNROLL
    for (int sn = 0; sn < SG_N / sg_local_range; ++sn) {
      int sg_local_n = sn * sg_local_range + sg_local_id;
      float bias = static_cast<float>(Bias(n_tile_start + n_sg_start + sg_local_n));
      CUTLASS_PRAGMA_UNROLL
      for (int sm = 0; sm < SG_M; ++sm) {
        tCrC(sn * SG_M + sm) += bias;
      }
    }
  }
};

}  // namespace MoE
