#include <sycl/sycl.hpp>
#include <torch/all.h>

#include "utils.h"
#include "dispatch_utils.h"

#include "causal_conv1d.hpp"
#include "gated_delta_rule.hpp"
#ifdef VLLM_XPU_ENABLE_XE2
  #include "xe_2/chunk_causal_conv1d_xe2.hpp"
  #include "xe_2/chunk_gated_delta_rule_xe2.h"
#endif

void gdn_attention(
    torch::Tensor&
        core_attn_out,  // [total_seqlen, num_v_heads / tp_size, head_v_dim]
    torch::Tensor& z,   // [total_seqlen, num_v_heads / tp_size, head_v_dim]
    const torch::Tensor&
        projected_states_qkvz,  // [total_seqlen, num_k_heads / tp_size * (2 *
                                // head_k_dim + 2 * head_v_dim * num_v_heads /
                                // num_k_heads)]
    const torch::Tensor&
        projected_states_ba,  // [total_seqlen, num_k_heads / tp_size * (2 *
                              // num_v_heads / num_k_heads)]
    const int64_t num_k_heads,
    const int64_t num_v_heads,
    const int64_t head_k_dim,
    const int64_t head_v_dim,
    torch::Tensor&
        conv_state,  // [cache_batch_size, width - 1, num_k_heads / tp_size * (2
                     // * head_k_dim + head_v_dim * num_v_heads / num_k_heads)]
    torch::Tensor& ssm_state,  // [cache_batch_size, num_v_heads / tp_size,
                               // head_v_dim, head_k_dim]
    const torch::Tensor&
        conv_weights,  // [num_k_heads / tp_size * (2 * head_k_dim + head_v_dim
                       // * num_v_heads / num_k_heads), width]
    const std::optional<torch::Tensor>&
        conv_bias,  // [num_k_heads / tp_size * (2 * head_k_dim + head_v_dim *
                    // num_v_heads / num_k_heads)] or None
    const std::string& activation,
    const torch::Tensor& A_log,    // [num_v_heads / tp_size]
    const torch::Tensor& dt_bias,  // [num_v_heads / tp_size]
    const int64_t num_prefills,
    const int64_t num_decodes,
    const std::optional<torch::Tensor>&
        has_initial_state,                               // [batch_size] or None
    const torch::Tensor& non_spec_query_start_loc,       // [batch_size + 1]
    const torch::Tensor& non_spec_state_indices_tensor,  // [batch_size]
    const int64_t num_actual_tokens,
    const int64_t tp_size,
    const bool reorder_input) {
  TORCH_CHECK(
      core_attn_out.is_contiguous(), "core_attn_out must be contiguous");
  TORCH_CHECK(z.is_contiguous(), "z must be contiguous");
  TORCH_CHECK(
      projected_states_qkvz.is_contiguous(),
      "projected_states_qkvz must be contiguous");
  TORCH_CHECK(
      projected_states_ba.is_contiguous(),
      "projected_states_ba must be contiguous");
  TORCH_CHECK(
      conv_state[0].is_contiguous(),
      "conv_state of each batch must be contiguous");
  TORCH_CHECK(
      ssm_state[0].is_contiguous(),
      "ssm_state of each batch must be contiguous");
  TORCH_CHECK(conv_weights.is_contiguous(), "conv_weights must be contiguous");
  TORCH_CHECK(A_log.is_contiguous(), "A_log must be contiguous");
  TORCH_CHECK(dt_bias.is_contiguous(), "dt_bias must be contiguous");
  TORCH_CHECK(
      non_spec_query_start_loc.is_contiguous(),
      "non_spec_query_start_loc must be contiguous");
  TORCH_CHECK(
      non_spec_state_indices_tensor.is_contiguous(),
      "non_spec_state_indices_tensor must be contiguous");

  TORCH_CHECK(
      non_spec_query_start_loc.scalar_type() == at::kInt,
      "non_spec_query_start_loc must be int32");
  TORCH_CHECK(
      non_spec_state_indices_tensor.scalar_type() == at::kInt,
      "non_spec_state_indices_tensor must be int32");
  if (has_initial_state.has_value()) {
    TORCH_CHECK(
        has_initial_state->scalar_type() == at::kBool,
        "has_initial_state must be bool");
  }

  // check core_attn_out shape
  TORCH_CHECK(core_attn_out.size(0) == num_actual_tokens);
  TORCH_CHECK(core_attn_out.size(1) == num_v_heads / tp_size);
  TORCH_CHECK(core_attn_out.size(2) == head_v_dim);

  // check z shape
  TORCH_CHECK(z.size(0) == core_attn_out.size(0));
  TORCH_CHECK(z.size(1) == core_attn_out.size(1));
  TORCH_CHECK(z.size(2) == core_attn_out.size(2));

  // check projected_states_qkvz shape
  TORCH_CHECK(projected_states_qkvz.size(0) == num_actual_tokens);
  TORCH_CHECK(
      projected_states_qkvz.size(1) ==
      num_k_heads / tp_size *
          (2 * head_k_dim + 2 * head_v_dim * num_v_heads / num_k_heads));

  // check projected_states_ba shape
  TORCH_CHECK(projected_states_ba.size(0) == num_actual_tokens);
  TORCH_CHECK(projected_states_ba.size(1) == 2 * num_v_heads / tp_size);

    const int pad_slot_id = -1;

  const int64_t batch_size = non_spec_query_start_loc.size(0) - 1;
  TORCH_CHECK(batch_size >= 0, "batch_size must be non-negative");
  TORCH_CHECK(
      batch_size > 0 || num_actual_tokens == 0,
      "query_start_loc implies empty batch but num_actual_tokens is non-zero");
  TORCH_CHECK(
      non_spec_state_indices_tensor.size(0) >= batch_size,
      "non_spec_state_indices_tensor size must be >= batch_size");
  if (has_initial_state.has_value()) {
    TORCH_CHECK(
        has_initial_state->size(0) >= batch_size,
        "has_initial_state size must be >= batch_size");
  }

  if (batch_size > 0) {
    const int qstart_first = non_spec_query_start_loc[0].item<int>();
    const int qstart_last = non_spec_query_start_loc[batch_size].item<int>();
    TORCH_CHECK(qstart_first == 0, "non_spec_query_start_loc[0] must be 0");
    TORCH_CHECK(
        qstart_last == num_actual_tokens,
        "non_spec_query_start_loc[-1] must equal num_actual_tokens");

    // Enforce monotonic non-decreasing offsets to avoid invalid token span
    // calculations in varlen kernels.
    int prev = qstart_first;
    for (int64_t i = 1; i <= batch_size; ++i) {
      const int curr = non_spec_query_start_loc[i].item<int>();
      TORCH_CHECK(
          curr >= prev,
          "non_spec_query_start_loc must be non-decreasing, but got ",
          prev,
          " then ",
          curr,
          " at index ",
          i);
      prev = curr;
    }

    const int min_state_idx = non_spec_state_indices_tensor.min().item<int>();
    const int max_state_idx = non_spec_state_indices_tensor.max().item<int>();
    TORCH_CHECK(
        min_state_idx > pad_slot_id,
        "state index must be > pad_slot_id (-1), got min=",
        min_state_idx);
    TORCH_CHECK(
        max_state_idx < conv_state.size(0),
        "state index exceeds conv_state slots: max=",
        max_state_idx,
        ", conv_state slots=",
        conv_state.size(0));
    TORCH_CHECK(
        max_state_idx < ssm_state.size(0),
        "state index exceeds ssm_state slots: max=",
        max_state_idx,
        ", ssm_state slots=",
        ssm_state.size(0));
  }

    if (num_prefills == 0 && num_decodes > 0) {
        TORCH_CHECK(
        batch_size >= num_decodes,
        "decode-only path requires batch_size >= num_decodes, but got batch_size=",
        batch_size,
        ", num_decodes=",
        num_decodes);
    TORCH_CHECK(
        num_actual_tokens >= num_decodes,
        "decode-only path requires num_actual_tokens >= num_decodes (may be padded for CUDAGraph), but got num_actual_tokens=",
                num_decodes);
    }

  auto& queue = vllm::xpu::vllmGetQueue();
  auto dtype = projected_states_qkvz.dtype();
  auto device = projected_states_qkvz.device();
  gdn::ActMode act_mode;

  if (activation == "silu") {
    act_mode = gdn::ActMode::silu;
  } else if (activation == "swish") {
    act_mode = gdn::ActMode::swish;
  } else {
    TORCH_CHECK(false);
  }

#define NATIVE_LAUNCHER                                           \
  do {                                                            \
    torch::Tensor q = torch::empty(                               \
        {num_actual_tokens, num_k_heads / tp_size, head_k_dim},   \
        torch::dtype(dtype).device(device).requires_grad(false)); \
    torch::Tensor k = torch::empty(                               \
        {num_actual_tokens, num_k_heads / tp_size, head_k_dim},   \
        torch::dtype(dtype).device(device).requires_grad(false)); \
    torch::Tensor v = torch::empty(                               \
        {num_actual_tokens, num_v_heads / tp_size, head_v_dim},   \
        torch::dtype(dtype).device(device).requires_grad(false)); \
    torch::Tensor b = torch::empty(                               \
        {num_actual_tokens, num_v_heads / tp_size},               \
        torch::dtype(dtype).device(device).requires_grad(false)); \
    torch::Tensor a = torch::empty(                               \
        {num_actual_tokens, num_v_heads / tp_size},               \
        torch::dtype(dtype).device(device).requires_grad(false)); \
    gdn::causal_conv1d(                                           \
        queue,                                                    \
        q,                                                        \
        k,                                                        \
        v,                                                        \
        z,                                                        \
        b,                                                        \
        a,                                                        \
        projected_states_qkvz,                                    \
        projected_states_ba,                                      \
        conv_weights,                                             \
        conv_bias,                                                \
        conv_state,                                               \
        non_spec_query_start_loc,                                 \
        non_spec_state_indices_tensor,                            \
        has_initial_state,                                        \
        act_mode,                                                 \
        pad_slot_id,                                              \
        num_prefills,                                             \
        num_decodes,                                              \
        reorder_input);                                           \
    gdn::gated_delta_rule(                                        \
        queue,                                                    \
        core_attn_out,                                            \
        q,                                                        \
        k,                                                        \
        v,                                                        \
        b,                                                        \
        a,                                                        \
        A_log,                                                    \
        dt_bias,                                                  \
        ssm_state,                                                \
        non_spec_query_start_loc,                                 \
        non_spec_state_indices_tensor,                            \
        has_initial_state,                                        \
        num_prefills,                                             \
        num_decodes);                                             \
  } while (0)

#ifdef VLLM_XPU_ENABLE_XE2
    // chunk_prepare_kernel now correctly handles partial chunks (including
    // decode sequences padded to chunk_size virtual tokens) by zeroing gating
    // contributions for padding positions, so mixed prefill+decode batches are
    // safe to run on the XE2 path.
    if (num_prefills > 0) {
    int batch_size = non_spec_query_start_loc.size(0) - 1;
    int padding_size = batch_size * (gdn::chunk_size_xe2 - 1);

    torch::Tensor q = torch::zeros(
        {num_actual_tokens + padding_size, num_k_heads / tp_size, head_k_dim},
        torch::dtype(dtype).device(device).requires_grad(false));
    torch::Tensor k = torch::zeros(
        {num_actual_tokens + padding_size, num_k_heads / tp_size, head_k_dim},
        torch::dtype(dtype).device(device).requires_grad(false));
    torch::Tensor v = torch::zeros(
        {num_actual_tokens + padding_size, num_v_heads / tp_size, head_v_dim},
        torch::dtype(dtype).device(device).requires_grad(false));
    torch::Tensor b = torch::zeros(
        {num_v_heads / tp_size, num_actual_tokens + padding_size},
        torch::dtype(torch::kFloat32).device(device).requires_grad(false));
    torch::Tensor a = torch::zeros(
        {num_v_heads / tp_size, num_actual_tokens + padding_size},
        torch::dtype(torch::kFloat32).device(device).requires_grad(false));

    gdn::chunk_causal_conv1d_xe2(
        queue,
        q,
        k,
        v,
        z,
        b,
        a,
        projected_states_qkvz,
        projected_states_ba,
        conv_weights,
        conv_bias,
        conv_state,
        non_spec_query_start_loc,
        non_spec_state_indices_tensor,
        has_initial_state,
        act_mode,
        pad_slot_id,
        num_prefills,
        num_decodes,
        reorder_input);

    chunk_gated_delta_rule_xe2(
        queue,
        core_attn_out,
        q,
        k,
        v,
        b,
        a,
        A_log,
        dt_bias,
        ssm_state,
        non_spec_query_start_loc,
        non_spec_state_indices_tensor,
        has_initial_state,
        num_prefills,
        num_decodes);
  } else {
    NATIVE_LAUNCHER;
  }
#else
  NATIVE_LAUNCHER;
#endif
#undef NATIVE_LAUNCHER
}