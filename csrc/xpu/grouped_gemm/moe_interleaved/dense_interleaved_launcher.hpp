// SPDX-License-Identifier: Apache-2.0
//
// Header-only SYCL launcher for the plain (non-grouped) dense-GEMM variant
// of the single-accumulator interleaved SwiGLU fusion -- the dense-MLP
// analog of moe_interleaved_launcher.hpp. See
// csrc/xpu/grouped_gemm/moe_interleaved/xe20/bf16/dense_kernel_interleaved.hpp
// and /work/fusemlp/design.md for the fusion rationale.
#pragma once

#include <ATen/ATen.h>
#include <c10/xpu/XPUStream.h>
#include <torch/all.h>

#include <cute/tensor.hpp>

#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "xpu/grouped_gemm/moe_interleaved/xe20/bf16/dense_kernel_interleaved.hpp"

namespace vllm_xpu {
namespace dense_mlp_interleaved {

using namespace cute;
using namespace MoE;

template <typename, typename, typename, typename, typename, typename, typename, ActivationType, bool>
class DenseGemmXe20InterleavedName;

// TileFull is the un-halved (gate+up) accumulator-space tile; TileHalf is the
// same tile with N halved, used only to size the D-side copy/fragment (see
// moe_mainloop_interleaved.hpp for why the two must be decoupled).
// KSwizzleM: raster-swizzle band width (see dense_kernel_interleaved.hpp's
// DenseGEMMInterleaved KSwizzleM doc comment). Default 1 = original raster.
template <
    typename TileFull,
    typename TileHalf,
    typename SGLayout,
    ActivationType ActType,
    bool WithBias,
    int KSwizzleM = 1>
void Xe20DenseMLPGEMMInterleavedLauncher(
    sycl::queue q,
    const void* activations,
    const void* weights,
    const void* bias,
    void* outputs,
    const int gemm_m,
    const int gemm_n,
    const int gemm_k,
    int* workspace,
    float gemm1_alpha,
    float gemm1_limit,
    int ld_b_param) {
  using Element = cutlass::bfloat16_t;

  auto make_dummy_tensor = [&](auto val, auto stride) {
    return make_tensor(make_gmem_ptr(&val), make_layout(repeat<rank_v<decltype(stride)>>(1), stride));
  };
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

  syclex::properties kernel_props{syclex::sub_group_size<16>, intelex::grf_size<256>};

  using Kernel = MoE::DenseGEMMInterleaved<
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
      Element,
      Element,
      Element,
      void,
      KSwizzleM>;
  typename Kernel::Params params{
      static_cast<const Element*>(activations),
      static_cast<const Element*>(weights),
      static_cast<const float*>(bias),
      static_cast<Element*>(outputs),
      gemm_m,
      gemm_n,
      gemm_k,
      workspace,
      mma,
      mma_half,
      static_cast<int32_t>(ld_b_param),
      static_cast<float>(gemm1_alpha),
      static_cast<float>(gemm1_limit),
  };

  q.submit([&](sycl::handler& h) {
    sycl::local_accessor<int32_t, 1> local_mem(sycl::range<1>(1), h);
    h.parallel_for<DenseGemmXe20InterleavedName<
        TileFull,
        TileHalf,
        SGLayout,
        TensorA,
        TensorB,
        TensorD,
        Element,
        ActType,
        WithBias>>(
        sycl::nd_range<3>(global * local, local), kernel_props, [=](sycl::nd_item<3> item) {
          int32_t* slm_mem =
              static_cast<int32_t*>(local_mem.template get_multi_ptr<sycl::access::decorated::no>().get());
          Kernel{}(params, item, slm_mem);
        });
  });
}

}  // namespace dense_mlp_interleaved
}  // namespace vllm_xpu
