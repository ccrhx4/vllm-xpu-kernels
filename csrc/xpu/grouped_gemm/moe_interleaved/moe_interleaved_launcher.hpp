// SPDX-License-Identifier: Apache-2.0
//
// Header-only SYCL launcher for the single-accumulator interleaved SwiGLU
// MoE grouped-GEMM fusion, ported from sgl-kernel-xpu
// (src/sycl/GroupGemmXe20InterleavedLauncherInstance.cpp.in +
// src/sycl/kernels/moe/xe20/bf16/moe_kernel_interleaved.hpp). See
// /work/fusemlp/design.md for the fusion rationale (interleaved gate/up
// weight layout lets a single accumulator hold both partial sums, freeing
// half the register/SLM footprint that the dual-accumulator fused path
// needs, which is what makes activation-fused large tiles reachable).
//
// Unlike the sgl-kernel-xpu original (which splits each tile-config
// instantiation into a separate translation unit via CMake `configure_file`
// for parallel compilation), this is a plain header-only template: the two
// tile configs used by vllm-xpu-kernels are explicitly instantiated by
// calling this function directly from moe_grouped_mm_interleaved.cpp.
#pragma once

#include <ATen/ATen.h>
#include <c10/xpu/XPUStream.h>
#include <torch/all.h>

#include <cute/tensor.hpp>

#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/group_array_problem_shape.hpp"
#include "xpu/grouped_gemm/moe_interleaved/xe20/bf16/moe_kernel_interleaved.hpp"

namespace vllm_xpu {
namespace moe_interleaved {

using namespace cute;
using namespace MoE;

template <typename, typename, typename, typename, typename, typename, typename, ActivationType, bool>
class GemmXe20InterleavedName;

// TileFull is the un-halved (gate+up) accumulator-space tile; TileHalf is the
// same tile with N halved, used only to size the D-side copy/fragment (see
// moe_mainloop_interleaved.hpp for why the two must be decoupled).
template <typename TileFull, typename TileHalf, typename SGLayout, ActivationType ActType, bool WithBias>
void Xe20MoEGEMMInterleavedLauncher(
    sycl::queue q,
    const void* activations,
    const void* weights,
    const void* bias,
    void* outputs,
    const int gemm_n,
    const int gemm_k,
    const int* num_rows_per_expert_device,
    const int num_experts,
    int* workspace,
    float gemm1_alpha,
    float gemm1_limit,
    int ld_b_param) {
  using Element = cutlass::bfloat16_t;

  auto make_dummy_tensor = [&](auto val, auto stride) {
    return make_tensor(make_gmem_ptr(&val), make_layout(repeat<rank_v<decltype(stride)>>(1), stride));
  };
  // Match the exact Shape<int> (rank-1 tuple) form make_Bias_tensor() uses in
  // moe_kernel_interleaved.hpp (make_shape(N) with a runtime int wraps in a
  // rank-1 tuple, not a bare int) so the dummy TensorBias type used here for
  // dispatch matches the type actually constructed at the call site.
  auto make_dummy_bias = [&](auto val) {
    return make_tensor(make_gmem_ptr(&val), make_layout(Shape<int>{}, Stride<_1>{}));
  };
  using StrideA = Stride<int, _1>;
  using StrideB = Stride<int, _1>;
  using StrideD = Stride<int, _1>;
  using TensorA = decltype(make_dummy_tensor(Element{}, StrideA{}));
  using TensorB = decltype(make_dummy_tensor(Element{}, StrideB{}));
  using TensorD = decltype(make_dummy_tensor(Element{}, StrideD{}));
  using TensorBias = decltype(make_dummy_bias(float{}));

  using ElementA_non_CV = cutlass::platform::remove_cv_t<Element>;
  using MMAFull =
      typename TiledMMAHelper<MMA_Atom<XE_DPAS_TT<8, float, ElementA_non_CV>>, Layout<TileFull>, SGLayout>::TiledMMA;
  using MMAHalf =
      typename TiledMMAHelper<MMA_Atom<XE_DPAS_TT<8, float, ElementA_non_CV>>, Layout<TileHalf>, SGLayout>::TiledMMA;
  auto mma = MMAFull{};
  auto mma_half = MMAHalf{};

  int sm_count = cutlass::KernelHardwareInfo::query_device_multiprocessor_count(0);
  auto MaxThreadsPerWorkgroup = size(mma);

  static constexpr int MaxThreadsPerSM = 512;

  TORCH_CHECK(
      MaxThreadsPerSM % MaxThreadsPerWorkgroup == 0, "MaxThreadsPerSM must be divisible by MaxThreadsPerWorkgroup");

  sycl::range<3> local(1, 1, MaxThreadsPerWorkgroup);
  sycl::range<3> global(1, sm_count * MaxThreadsPerSM / MaxThreadsPerWorkgroup, 1);

  namespace syclex = sycl::ext::oneapi::experimental;
  namespace intelex = sycl::ext::intel::experimental;

  // Validated on real hardware (Arc Pro B70, see sgl-kernel-xpu design.md) to
  // compile spill-free even at the default 128-GRF-per-thread budget; request
  // 256 anyway to match the dual-accumulator (unfused) launcher's occupancy
  // trade-off.
  syclex::properties kernel_props{syclex::sub_group_size<16>, intelex::grf_size<256>};

  using Kernel = MoE::MoEGEMMInterleaved<
      TileFull,
      TileHalf,
      SGLayout,
      TensorA,
      TensorB,
      TensorD,
      TensorBias,
      MMAFull,
      MMAHalf,
      ActType,
      WithBias,
      Element>;
  typename Kernel::Params params{
      static_cast<const Element*>(activations),
      static_cast<const Element*>(weights),
      static_cast<const float*>(bias),
      static_cast<Element*>(outputs),
      num_rows_per_expert_device,
      gemm_n,
      gemm_k,
      num_experts,
      workspace,
      mma,
      mma_half,
      static_cast<int32_t>(ld_b_param),
      static_cast<float>(gemm1_alpha),
      static_cast<float>(gemm1_limit),
  };

  q.submit([&](sycl::handler& h) {
    sycl::local_accessor<int32_t, 1> local_mem(sycl::range<1>(1), h);
    h.parallel_for<GemmXe20InterleavedName<TileFull, TileHalf, SGLayout, TensorA, TensorB, TensorD, Element, ActType, WithBias>>(
        sycl::nd_range<3>(global * local, local), kernel_props, [=](sycl::nd_item<3> item) {
          int32_t* slm_mem =
              static_cast<int32_t*>(local_mem.template get_multi_ptr<sycl::access::decorated::no>().get());
          Kernel{}(params, item, slm_mem);
        });
  });
}

}  // namespace moe_interleaved
}  // namespace vllm_xpu
