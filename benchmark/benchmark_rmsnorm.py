# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import argparse
import itertools
import types
from typing import Optional, Union

import torch
import triton
from torch import nn

from tests.utils import check_ipex_availability, parse_args

# Auto-detect device: prefer CUDA if available, fall back to XPU
DEVICE = "cuda" if torch.cuda.is_available() else "xpu"

# Conditionally import the right ops backend
HAS_VLLM_OPS = False
if DEVICE == "xpu":
    try:
        from tests import register_ops as vllm_ops
        HAS_VLLM_OPS = True
    except ImportError:
        pass
else:
    try:
        import vllm._C  # noqa: F401 — registers ops for torch::kCUDA
        vllm_ops = types.SimpleNamespace(
            rms_norm=lambda out, x, weight, eps: torch.ops._C.rms_norm(
                out, x.contiguous(), weight, eps),
            fused_add_rms_norm=lambda x, residual, weight, eps:
                torch.ops._C.fused_add_rms_norm(x, residual, weight, eps),
        )
        HAS_VLLM_OPS = True
    except ImportError:
        pass

# IPEX is only relevant on XPU
HAS_IPEX = check_ipex_availability() if DEVICE == "xpu" else False

if HAS_IPEX:
    import intel_extension_for_pytorch as ipex


class HuggingFaceRMSNorm(nn.Module):

    def __init__(self, hidden_size: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(
        self,
        x: torch.Tensor,
        residual: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        orig_dtype = x.dtype
        x = x.to(torch.float32)
        if residual is not None:
            x = x + residual.to(torch.float32)
            residual = x.to(orig_dtype)

        variance = x.pow(2).mean(dim=-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.variance_epsilon)
        x = x.to(orig_dtype) * self.weight
        if residual is None:
            return x
        else:
            return x, residual


def rmsnorm_naive(
    x: torch.Tensor,
    weight: torch.Tensor,
    residual: Optional[torch.Tensor] = None,
    eps: float = 1e-6,
):
    naive_norm = HuggingFaceRMSNorm(x.shape[-1], eps=eps)
    naive_norm.weight = nn.Parameter(weight)
    naive_norm = naive_norm.to(x.device)

    orig_shape = x.shape
    x = x.view(-1, x.shape[-1])
    if residual is not None:
        residual = residual.view(-1, residual.shape[-1])

    output = naive_norm(x, residual)

    if isinstance(output, tuple):
        output = (output[0].view(orig_shape), output[1].view(orig_shape))
    else:
        output = output.view(orig_shape)
    return output


@torch.compile
def rmsnorm_compile(x: torch.Tensor,
                    weight: torch.Tensor,
                    residual: Optional[torch.Tensor] = None,
                    eps: float = 1e-6):
    """PyTorch-native implementation equivalent to forward()."""
    orig_dtype = x.dtype
    x = x.to(torch.float32)
    if residual is not None:
        x = x + residual.to(torch.float32)
        residual = x.to(orig_dtype)

    x_var = x
    variance = x_var.pow(2).mean(dim=-1, keepdim=True)

    x = x * torch.rsqrt(variance + eps)
    x = x.to(orig_dtype)
    x = x * weight
    if residual is None:
        return x
    else:
        return x, residual


def rmsnorm_vllm(
    x: torch.Tensor,
    weight: torch.Tensor,
    residual: Optional[torch.Tensor] = None,
    eps: float = 1e-6,
):
    orig_shape = x.shape
    x = x.view(-1, x.shape[-1])
    if residual is not None:
        residual = residual.view(-1, residual.shape[-1])

    if residual is not None:
        vllm_ops.fused_add_rms_norm(x, residual, weight, eps)
        output = (x, residual)
    else:
        out = torch.empty_like(x)
        vllm_ops.rms_norm(out, x, weight, eps)
        output = out

    if isinstance(output, tuple):
        output = (output[0].view(orig_shape), output[1].view(orig_shape))
    else:
        output = output.view(orig_shape)
    return output

def rmsnorm_vllm_3d(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
):
    """Call rms_norm on a 3D tensor [tokens, heads, head_dim] without
    flattening, to exercise the NUM_DIMS=3 kernel path (QK norm)."""
    out = torch.empty_like(x)
    vllm_ops.rms_norm(out, x, weight, eps)
    return out


def rmsnorm_ipex(
    x: torch.Tensor,
    weight: torch.Tensor,
    residual: Optional[torch.Tensor] = None,
    eps: float = 1e-6,
):
    """IPEX implementation using ipex.llm.functional.rms_norm"""
    if not HAS_IPEX:
        raise RuntimeError("IPEX is not available")

    orig_shape = x.shape
    x = x.view(-1, x.shape[-1])

    if residual is not None:
        residual = residual.view(-1, residual.shape[-1])
        if hasattr(ipex.llm.functional, 'fused_add_rms_norm'):
            output, residual_out = ipex.llm.functional.fused_add_rms_norm(
                x, residual, weight, eps)
            output = (output.view(orig_shape), residual_out.view(orig_shape))
        else:
            x = x + residual
            output = ipex.llm.functional.rms_norm(x, weight, eps)
            output = (output.view(orig_shape), x.view(orig_shape))
    else:
        output = ipex.llm.functional.rms_norm(x, weight, eps)
        output = output.view(orig_shape)

    return output

def get_qknorm_benchmark(dtype, head_dim):
    """Benchmark rms_norm on 3D tensors [tokens, num_heads, head_dim]
    to reproduce QK-norm performance seen in profiler traces."""

    @triton.testing.perf_report(
        triton.testing.Benchmark(
            x_names=["num_tokens", "num_heads"],
            x_vals=[tuple(_) for _ in qknorm_configs],
            line_arg="provider",
            line_vals=["vllm_3d", "vllm_2d"],
            line_names=["vLLM 3D (real QK-norm)", "vLLM 2D (flattened)"],
            styles=[("green", "-"), ("blue", "--")],
            ylabel="us",
            plot_name=f"qknorm-perf-head_dim{head_dim}",
            args={},
        ))
    def benchmark(num_tokens, num_heads, provider):
        x = torch.randn(num_tokens,
                         num_heads,
                         head_dim,
                         dtype=dtype,
                         device=DEVICE)
        weight = torch.ones(head_dim, dtype=dtype, device=DEVICE)

        quantiles = [0.5, 0.2, 0.8]

        if provider == "vllm_3d":
            ms, min_ms, max_ms = triton.testing.do_bench(
                lambda: rmsnorm_vllm_3d(x.clone(), weight),
                quantiles=quantiles,
            )
        else:  # vllm_2d: flatten to [tokens*heads, head_dim]
            x_2d = x.view(-1, head_dim)
            weight_2d = weight
            ms, min_ms, max_ms = triton.testing.do_bench(
                lambda: rmsnorm_vllm(
                    x_2d.clone(), weight_2d),
                quantiles=quantiles,
            )
        return 1000 * ms, 1000 * max_ms, 1000 * min_ms

    return benchmark



def calculate_diff(batch_size, seq_len, hidden_size, use_residual=True):
    dtype = torch.bfloat16
    x = torch.randn(batch_size,
                    seq_len,
                    hidden_size,
                    dtype=dtype,
                    device=DEVICE)
    weight = torch.ones(hidden_size, dtype=dtype, device=DEVICE)
    residual = torch.randn_like(x) if use_residual else None

    output_naive = rmsnorm_naive(
        x.clone(), weight,
        residual.clone() if residual is not None else None)

    if HAS_VLLM_OPS:
        output_vllm = rmsnorm_vllm(
            x.clone(), weight,
            residual.clone() if residual is not None else None)
        if use_residual:
            output_vllm = output_vllm[0]

    if use_residual:
        output_naive = output_naive[0]

    print(f"Naive output={output_naive}")
    if HAS_VLLM_OPS:
        print(f"vLLM output={output_vllm}")

    if HAS_IPEX:
        try:
            output_ipex = rmsnorm_ipex(
                x.clone(), weight,
                residual.clone() if residual is not None else None)
            if use_residual:
                output_ipex = output_ipex[0]
            print(f"IPEX output={output_ipex}")

            if torch.allclose(output_naive, output_ipex, atol=1e-2, rtol=1e-2):
                print("✅ IPEX implementation matches naive")
            else:
                print("❌ IPEX implementation differs from naive")
        except Exception as e:
            print(f"❌ IPEX implementation failed: {e}")

    if HAS_VLLM_OPS:
        if torch.allclose(output_naive, output_vllm, atol=1e-2, rtol=1e-2):
            print("✅ All implementations match")
        else:
            print("❌ Implementations differ")
    else:
        print("⚠️  vLLM ops not available, skipping vLLM correctness check")


def get_benchmark(use_residual, dtype):

    providers = ["huggingface", "t.compile"]
    provider_names = ["HuggingFace", "t.compile"]
    provider_styles = [("blue", "-"), ("orange", "-")]
    if HAS_VLLM_OPS:
        providers.insert(1, "vllm")
        provider_names.insert(1, "vLLM")
        provider_styles.insert(1, ("green", "-"))
    if HAS_IPEX:
        providers.append("ipex")
        provider_names.append("IPEX")
        provider_styles.append(("red", "-"))

    @triton.testing.perf_report(
        triton.testing.Benchmark(
            x_names=["hidden_size", "batch_size", "seq_len"],
            x_vals=[tuple(_) for _ in configs],
            line_arg="provider",
            line_vals=providers,
            line_names=provider_names,
            styles=provider_styles,
            ylabel="us",
            plot_name=
            f"rmsnorm-perf-{'with' if use_residual else 'without'}-residual",
            args={},
        ))
    def benchmark(hidden_size, batch_size, seq_len, provider):

        x = torch.randn(batch_size,
                        seq_len,
                        hidden_size,
                        dtype=dtype,
                        device=DEVICE)
        weight = torch.ones(hidden_size, dtype=dtype, device=DEVICE)
        residual = torch.randn_like(x) if use_residual else None

        quantiles = [0.5, 0.2, 0.8]

        if provider == "huggingface":
            ms, min_ms, max_ms = triton.testing.do_bench(
                lambda: rmsnorm_naive(
                    x.clone(),
                    weight,
                    residual.clone() if residual is not None else None,
                ),
                quantiles=quantiles,
            )
        elif provider == "t.compile":
            ms, min_ms, max_ms = triton.testing.do_bench(
                lambda: rmsnorm_compile(
                    x.clone(),
                    weight,
                    residual.clone() if residual is not None else None,
                ),
                quantiles=quantiles,
            )
        elif provider == "ipex" and HAS_IPEX:
            ms, min_ms, max_ms = triton.testing.do_bench(
                lambda: rmsnorm_ipex(
                    x.clone(),
                    weight,
                    residual.clone() if residual is not None else None,
                ),
                quantiles=quantiles,
            )
        elif provider == "vllm":
            ms, min_ms, max_ms = triton.testing.do_bench(
                lambda: rmsnorm_vllm(
                    x.clone(),
                    weight,
                    residual.clone() if residual is not None else None,
                ),
                quantiles=quantiles,
            )
        return 1000 * ms, 1000 * max_ms, 1000 * min_ms

    return benchmark


if __name__ == "__main__":

    # Parse --device first and remove it from sys.argv so parse_args() won't
    # reject it as an unrecognized argument.
    import sys
    device_parser = argparse.ArgumentParser(add_help=False)
    device_parser.add_argument(
        "--device",
        type=str,
        choices=["cuda", "xpu", "auto"],
        default="auto",
        help="Device to run benchmarks on (default: auto-detect)",
    )
    device_args, remaining_argv = device_parser.parse_known_args()
    sys.argv = [sys.argv[0]] + remaining_argv

    args = parse_args()

    if device_args.device != "auto":
        DEVICE = device_args.device
        # Re-evaluate ops availability for the selected device
        HAS_VLLM_OPS = False
        if DEVICE == "xpu":
            try:
                from tests import register_ops as vllm_ops  # noqa: F811
                HAS_VLLM_OPS = True
            except ImportError:
                pass
        else:
            try:
                import vllm._C  # noqa: F811, F401
                vllm_ops = types.SimpleNamespace(
                    rms_norm=lambda out, x, weight, eps:
                        torch.ops._C.rms_norm(
                            out, x.contiguous(), weight, eps),
                    fused_add_rms_norm=lambda x, residual, weight, eps:
                        torch.ops._C.fused_add_rms_norm(
                            x, residual, weight, eps),
                )
                HAS_VLLM_OPS = True
            except ImportError:
                pass
        HAS_IPEX = (check_ipex_availability() if DEVICE == "xpu" else False)

    print("Final configuration:")
    print(f"  Device: {DEVICE}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Sequence length: {args.seq_len}")
    print(f"  Hidden size: {args.hidden_size}")
    print(f"  Intermediate size: {args.intermediate_size}")
    print(f"  Number of groups: {args.num_groups}")
    print(f"  Data type: {args.dtype}")
    print(f"  Use residual: {args.use_residual}")

    batch_size_range = [2**i for i in range(0, 7, 2)]
    seq_length_range = [2**i for i in range(6, 10, 1)]
    hidden_size_range = [args.hidden_size]
    configs = list(
        itertools.product(hidden_size_range, batch_size_range,
                          seq_length_range))
    # Prefill cases: batch_size=1, longer sequences
    prefill_seq_lengths = [2048, 3500, 4096, 8192]
    for h in hidden_size_range:
        for s in prefill_seq_lengths:
            configs.append((h, 1, s))
    
    if getattr(args, 'qk_norm', False):
        if not HAS_VLLM_OPS:
            print("ERROR: vllm ops not available, cannot run QK-norm benchmark")
            sys.exit(1)
        # QK-norm benchmark: 3D tensors [tokens, heads, head_dim]
        head_dim = args.head_size
        # Default configs: sweep token counts × head counts
        # Matches Qwen3-32B TP=4: q_heads=16, kv_heads=2, head_dim=128
        token_range = [1, 64, 512, 2048, 3500, 7000, 14000]
        head_range = [2, 8, 10, 16, 32, 64]
        qknorm_configs = list(
            itertools.product(token_range, head_range))

        print(f"\nQK-norm benchmark mode:")
        print(f"  head_dim: {head_dim}")
        print(f"  token_range: {token_range}")
        print(f"  head_range: {head_range}")

        benchmark = get_qknorm_benchmark(args.dtype, head_dim)
        benchmark.run(print_data=True, save_path=args.save_path)

    if HAS_IPEX:
        print("✅ IPEX is available")
        print(f"IPEX version: {ipex.__version__}")
    else:
        print("⚠️  IPEX is not available, skipping IPEX benchmarks")

    # Run correctness test
    calculate_diff(
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        hidden_size=args.hidden_size,
        use_residual=args.use_residual,
    )

    # Get the benchmark function with proper use_residual setting
    benchmark = get_benchmark(args.use_residual, args.dtype)
    # Run performance benchmark
    benchmark.run(print_data=True, save_path=args.save_path)
