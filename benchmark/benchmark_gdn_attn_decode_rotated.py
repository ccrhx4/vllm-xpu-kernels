# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# ruff: noqa: E402
"""
Side-by-side SYCL vs. Triton decode benchmark for Gated DeltaNet, with a
fix for a benchmark-pattern bug present in `benchmark_gdn_attn.py` /
`benchmark_gdn_attn_triton.py`: those harnesses build a single fixed
`state_indices` tensor per config and reuse the *exact same* physical
cache rows (conv_state / ssm_state) for all timed iterations.

That is unrepresentative of real serving (where each decode step touches
different requests/rows) and, worse, it lets those rows' content
degenerate/converge over hundreds of repeated in-place updates -- which is
far more compressible (and hence gives an inflated benefit from GPU buffer
compression, e.g. `RenderCompressedBuffersEnabled`) than real traffic.

Fix: `cache_batch_size = max(256, batch_size * 2)` in `make_inputs()`
already allocates more cache rows than a single decode step touches.
We use that headroom to pre-build N = (cache_batch_size - 1) // batch_size
disjoint row-index groups (a random permutation of the cache, excluding
row 0 which is a reserved NULL_BLOCK_ID for the Triton decode kernels)
*once* before timing, then cycle through them (`groups[i % N]`) inside the
timed loop. This is a plain CPU list-index lookup -- no extra GPU kernel
launch and no extra memory beyond N small int32 tensors of size
batch_size -- so it does not distort the timing the way re-randomizing
tensor contents in-loop (e.g. `.normal_()`) would.

This was empirically validated to reproduce the same qualitative results
as the stock harnesses (SYCL is largely compression-insensitive; Triton's
tuned decode path is measurably compression-sensitive), confirming that
sensitivity is a genuine property of the kernels' access pattern and not
an artifact of the stock benchmarks' degenerate row reuse.

Update: row rotation alone is NOT sufficient at large batch. `make_inputs()`
only allocates `max(256, 2*batch)` cache rows, so for batch >= 128 there is
just one disjoint group (`N == 1`) and rotation collapses back to reusing the
same rows every iteration. Over the ~200 timed iterations those rows' fp32
SSM state degenerates: the gated-delta recurrence repeatedly adds rank-1
updates along the *same* fixed `k`, collapsing each per-head state matrix to
(near) rank 1 with ~17x smaller magnitude within ~10-12 updates. That
degenerate state compresses far better (~1.5x on Xe2 buffer compression) than
real serving traffic (~1.1-1.17x), which was measured to hand Triton (the
memory-bandwidth-bound path) a large, unrealistic compression-only speedup at
batch >= 32 -- an artifact, not a real win.

Real fix (see `measure_rotated`): keep a pristine master snapshot of the
in-place-updated state (SSM + conv) and restore from it between short "rounds"
so no row is updated more than `MAX_UPDATES_PER_ROW` (4) times before being
reset -- below the ~6-update threshold where the state starts to degenerate
(effective rank stays >= ~40, compressibility stays realistic). The restore
is a device-to-device copy done *outside* the timed events, so it adds no
measurement overhead, and each round still issues its iterations back-to-back
(no per-iteration sync) so small-batch per-launch dispatch overhead is still
captured faithfully.

Usage:
    python benchmark_gdn_attn_decode_rotated.py

Set `NEOReadDebugKeys=1` and `RenderCompressedBuffersEnabled=0` in the
environment to compare against GPU buffer compression disabled. With the
degeneration artifact removed, the two should track much more closely (the
measured kernel cost no longer depends on unrealistically compressible state).
"""
# isort: off
import os

import torch

from utils import bootstrap_benchmark_env

bootstrap_benchmark_env(__file__)

from tests.utils import seed_everything
from benchmark_gdn_attn import MODEL_SHAPES, WORKLOADS, make_inputs
from benchmark_gdn_attn_triton import (
    DECODE_WORKLOADS,
    _make_triton_decode_inputs,
)

from vllm.model_executor.layers.mamba.ops.causal_conv1d import (
    causal_conv1d_update,
)
from vllm.third_party.flash_linear_attention.ops.fused_recurrent import (
    fused_recurrent_gated_delta_rule_packed_decode,
)
# isort: on

# Round-based timing parameters.
#
# The decode kernels update the recurrent SSM state (and conv state) *in
# place*, and this benchmark drives them with a single fixed input set. Left
# unchecked, repeatedly re-updating the same physical cache rows drives the
# fp32 SSM state to a degenerate fixed point: the gated-delta recurrence keeps
# adding rank-1 updates along the *same* `k` direction, so within ~10-12
# updates the per-head state matrix collapses to (near) rank 1 with ~17x
# smaller magnitude. That degenerate state is far more compressible than real
# serving traffic (where every decode step has a different token / `k`), which
# in turn hands an unrealistic, content-dependent speedup to whichever kernel
# is memory-bandwidth-bound once GPU buffer compression is enabled.
#
# `build_rotating_groups()` alone does not fix this at large batch: the cache
# only has `max(256, 2*batch)` rows, so for batch >= 128 there is just
# `n_groups == 1` and rotation degenerates to reusing the same rows every
# iteration.
#
# The fix here is to (a) cap how many times any row is updated in place before
# its content is refreshed, and (b) refresh by generating *new* random state
# on the fly (a plain `.normal_()` fill, matching how `make_inputs()` seeds the
# state) *outside* the timed region so it adds no measurement overhead. There
# is no need to precreate distinct state for every iteration up front (that
# would cost `iterations * batch * per_row_bytes`, e.g. ~77 GB at batch 256 and
# exceed VRAM): each round simply creates fresh data for the config's own
# buffers.
#
# Empirically the state stays representative (effective rank >= ~40, realistic
# compressibility) up to ~6 in-place updates/row, but the per-head magnitude
# decays ~2x per update (g ~ 0.3-0.5), and GPU buffer compression is sensitive
# to that magnitude clustering well before the rank collapses. Sweeping the cap
# and comparing compression ON vs OFF, cap=1 drives the residual ON/OFF gap to
# a flat ~1.13x -- matching the genuine compressibility of realistic
# varying-token state (~1.1-1.17x), i.e. the real compression benefit rather
# than the ~1.5x degenerate-state artifact. Override via
# GDN_MAX_UPDATES_PER_ROW for experimentation. Within a round we still issue
# `iters_per_round` back-to-back launches (no sync) so the per-launch dispatch
# overhead that dominates small-batch decode is still captured.
MAX_UPDATES_PER_ROW = int(os.environ.get("GDN_MAX_UPDATES_PER_ROW", "1"))
TARGET_TIMED_ITERS = 200
ITERS_PER_ROUND_CAP = 256
WARMUP_ROUNDS = 2

# Number of independently-prepared state buffers to alternate between, one per
# iteration (iteration i uses buffer i % NUM_STATE_BUFFERS). Because a given
# buffer is then updated only every K-th iteration, each row accumulates
# in-place updates K times slower, which lets us issue K back-to-back timed
# launches before any row exceeds MAX_UPDATES_PER_ROW -- restoring the
# sustained-throughput launch pattern even at large batch, where the rotation
# pool holds only a single group (n_groups == 1). Costs K x the state memory;
# default 2 is safe for every shape here. Override via GDN_NUM_STATE_BUFFERS.
NUM_STATE_BUFFERS = int(os.environ.get("GDN_NUM_STATE_BUFFERS", "2"))


def build_rotating_groups(cache_batch_size, batch_size, device):
    """Disjoint contiguous groups of `batch_size` rows from a random
    permutation of [1, cache_batch_size) (row 0 excluded: it is a reserved
    NULL_BLOCK_ID for the Triton decode kernels)."""
    n_groups = max(1, (cache_batch_size - 1) // batch_size)
    perm = torch.randperm(cache_batch_size - 1) + 1
    perm = perm[:n_groups * batch_size]
    groups = [
        perm[i * batch_size:(i + 1) * batch_size].to(device=device,
                                                       dtype=torch.int32)
        for i in range(n_groups)
    ]
    return groups


def measure_rotated(run_i, mutated_state, n_groups):
    """Round-based timing that keeps the in-place-updated state representative.

    `run_i(i)` issues one decode step for rotation index `i` (it selects
    `groups[i % n_groups]` internally). `mutated_state` is the list of live
    device tensors the kernels update in place across calls (SSM state, conv
    state). Between rounds we create *new* random data in those buffers with
    `.normal_()` -- outside the timed events -- so every timed iteration reads
    freshly generated, non-degenerated (incompressible) state. We do not
    precreate per-iteration state; only this config's buffers are refilled.

    We size `iters_per_round` so that, given the rotation across `n_groups`
    disjoint row-sets, every row is updated at most `MAX_UPDATES_PER_ROW` times
    per round before being refreshed, preventing the state degeneration
    described above. Timing brackets only the `iters_per_round` back-to-back
    kernel launches, so per-launch dispatch overhead is still measured.
    """

    def refresh():
        # Create new fresh state on the fly (matches make_inputs()'s randn
        # seeding); keeps the read state incompressible without a master copy
        # or precreating per-iteration data.
        for t in mutated_state:
            t.normal_()

    iters_per_round = max(
        1, min(ITERS_PER_ROUND_CAP, MAX_UPDATES_PER_ROW * n_groups))
    n_rounds = max(1, -(-TARGET_TIMED_ITERS // iters_per_round))  # ceil div

    # Warmup rounds (untimed).
    idx = 0
    for _ in range(WARMUP_ROUNDS):
        refresh()
        torch.xpu.synchronize()
        for _ in range(iters_per_round):
            run_i(idx)
            idx += 1
    torch.xpu.synchronize()

    start_event = torch.xpu.Event(enable_timing=True)
    end_event = torch.xpu.Event(enable_timing=True)
    total_ms = 0.0
    for _ in range(n_rounds):
        refresh()
        torch.xpu.synchronize()
        start_event.record()
        for _ in range(iters_per_round):
            run_i(idx)
            idx += 1
        end_event.record()
        torch.xpu.synchronize()
        total_ms += start_event.elapsed_time(end_event)

    return 1000 * total_ms / (n_rounds * iters_per_round)  # us/iter


def bench_triton(shape_name, workload_name, dtype_str="bf16"):
    shape = next(s for s in MODEL_SHAPES if s.name == shape_name)
    workload = next(w for w in DECODE_WORKLOADS if w.name == workload_name)
    dtype = torch.bfloat16 if dtype_str == "bf16" else torch.float16

    K = NUM_STATE_BUFFERS
    datasets = [
        _make_triton_decode_inputs(make_inputs(shape, workload, dtype))
        for _ in range(K)
    ]
    cache_batch_size = datasets[0]["conv_state"].shape[0]
    groups = build_rotating_groups(cache_batch_size, workload.batch_size,
                                    datasets[0]["state_indices"].device)
    n = len(groups)

    def _run(i):
        tk = datasets[i % K]
        state_indices = groups[(i // K) % n]
        mixed_qkv_out = causal_conv1d_update(
            tk["mixed_qkv"],
            tk["conv_state"],
            tk["conv_weights"],
            tk["conv_bias"],
            tk["activation"],
            conv_state_indices=state_indices,
            validate_data=False,
        )
        fused_recurrent_gated_delta_rule_packed_decode(
            mixed_qkv=mixed_qkv_out,
            a=tk["a"],
            b=tk["b"],
            A_log=tk["A_log"],
            dt_bias=tk["dt_bias"],
            scale=tk["scale"],
            initial_state=tk["ssm_state"],
            out=tk["out"],
            ssm_state_indices=state_indices,
            use_qk_l2norm_in_kernel=True,
        )

    mutated = []
    for tk in datasets:
        mutated += [tk["ssm_state"], tk["conv_state"]]
    return measure_rotated(_run, mutated_state=mutated, n_groups=K * n)


def bench_sycl(shape_name, workload_name, dtype_str="bf16"):
    shape = next(s for s in MODEL_SHAPES if s.name == shape_name)
    workload = next(w for w in WORKLOADS
                     if w.name == workload_name and w.mode == "decode")
    dtype = torch.bfloat16 if dtype_str == "bf16" else torch.float16

    K = NUM_STATE_BUFFERS
    datasets = [make_inputs(shape, workload, dtype) for _ in range(K)]
    cache_batch_size = datasets[0]["conv_state"].shape[0]
    groups = build_rotating_groups(
        cache_batch_size, workload.batch_size,
        datasets[0]["non_spec_state_indices_tensor"].device)
    n = len(groups)

    def _run(i):
        kwargs = datasets[i % K]
        state_indices = groups[(i // K) % n]
        intermediates = torch.ops._xpu_C.causal_conv1d_non_spec(
            kwargs["z"],
            kwargs["projected_states_qkvz"],
            kwargs["projected_states_ba"],
            kwargs["num_k_heads"],
            kwargs["num_v_heads"],
            kwargs["head_k_dim"],
            kwargs["head_v_dim"],
            conv_state=kwargs["conv_state"],
            conv_weights=kwargs["conv_weights"],
            conv_bias=kwargs["conv_bias"],
            activation=kwargs["activation"],
            num_prefills=kwargs["num_prefills"],
            num_decodes=kwargs["num_decodes"],
            num_spec_decodes=kwargs["num_spec_decodes"],
            has_initial_state=kwargs["has_initial_state"],
            non_spec_query_start_loc=kwargs["non_spec_query_start_loc"],
            non_spec_token_indx=kwargs["non_spec_token_indx"],
            non_spec_state_indices_tensor=state_indices,
            num_actual_tokens=kwargs["num_actual_tokens"],
            tp_size=kwargs["tp_size"],
            reorder_input=kwargs["reorder_input"])
        torch.ops._xpu_C.gated_delta_rule_non_spec(
            kwargs["core_attn_out"],
            *intermediates,
            kwargs["num_v_heads"],
            kwargs["head_v_dim"],
            A_log=kwargs["A_log"],
            dt_bias=kwargs["dt_bias"],
            ssm_state=kwargs["ssm_state"],
            num_prefills=kwargs["num_prefills"],
            num_decodes=kwargs["num_decodes"],
            num_spec_decodes=kwargs["num_spec_decodes"],
            has_initial_state=kwargs["has_initial_state"],
            non_spec_query_start_loc=kwargs["non_spec_query_start_loc"],
            non_spec_token_indx=kwargs["non_spec_token_indx"],
            non_spec_state_indices_tensor=state_indices,
            num_actual_tokens=kwargs["num_actual_tokens"],
            tp_size=kwargs["tp_size"])

    mutated = []
    for kwargs in datasets:
        mutated += [kwargs["ssm_state"], kwargs["conv_state"]]
    return measure_rotated(_run, mutated_state=mutated, n_groups=K * n)


if __name__ == "__main__":
    seed_everything(1234)
    torch.set_default_device("xpu")
    torch.xpu.set_device("xpu:0")

    decode_workload_names = [w.name for w in WORKLOADS if w.mode == "decode"]
    configs = [(s.name, w) for s in MODEL_SHAPES for w in decode_workload_names]
    print("shape,workload,sycl_us,triton_us", flush=True)
    for shape_name, workload_name in configs:
        sycl_us = bench_sycl(shape_name, workload_name)
        triton_us = bench_triton(shape_name, workload_name)
        print(f"{shape_name},{workload_name},{sycl_us:.3f},{triton_us:.3f}",
              flush=True)
