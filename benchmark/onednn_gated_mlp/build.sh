#!/bin/bash
# Rebuild the oneDNN gated_mlp graph-API microbenchmark against the
# oneDNN sources vendored under vllm-xpu-kernels/.deps (built as part of
# the normal `pip install -e .` build). Used as a third comparison point
# in benchmark_dense_mlp_interleaved.py (oneDNN's fused gate+up+down
# `gated_mlp` kernel vs. our own dense_swiglu_gemm_xe20_interleaved).
set -e
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ONEDNN_SRC="${ONEDNN_SRC:-$HERE/../../.deps/onednn-src}"
ONEDNN_BUILD="${ONEDNN_BUILD:-$HERE/../../.deps/onednn-build}"
icpx \
  -I"${ONEDNN_BUILD}/include" -I"${ONEDNN_SRC}/include" \
  -I"${ONEDNN_SRC}/src" -I"${ONEDNN_SRC}/src/../include" \
  -I"$HERE" \
  -isystem /opt/intel/oneapi/compiler/2026.1/include \
  -isystem /opt/intel/oneapi/compiler/2026.1/include/sycl \
  -O2 -std=gnu++20 -fsycl -fsycl-targets=spir64_gen \
  -Xsycl-target-backend=spir64_gen "-device pvc,bmg,bmg-g21-a0,bmg-g31-a0,cri,cri-a0" \
  -Wno-unused-command-line-argument \
  "$HERE/gated_mlp_bench.cpp" -o "$HERE/gated_mlp_bench" \
  "${ONEDNN_BUILD}/src/libdnnl.a" -lze_loader
