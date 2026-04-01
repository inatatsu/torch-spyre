import math
import os
import pytest
import torch

def sdpa(
    qk_t: torch.Tensor,
    v: torch.Tensor,
) -> torch.Tensor:
    return (torch.softmax(qk_t, -1) @ v).clone(memory_format=torch.contiguous_format)

def sdpa_decomposed(
    sub: torch.Tensor,
    v: torch.Tensor,
) -> torch.Tensor:
    exp = sub.exp()
    attn = exp / exp.sum(-1, True)
    attn = attn.squeeze(0).bmm(v.squeeze(0)).unsqueeze(0)
    return attn.clone(memory_format=torch.contiguous_format)

def _test_sdpa(q : torch.Tensor, k : torch.Tensor, v : torch.Tensor):
    scale = 0.0078125
    scaling_factor_q = torch.full_like(q, scale)
    scaling_factor_k = torch.full_like(k, scale)
    q_scaled = q * scaling_factor_q
    k_scaled_t = (k * scaling_factor_k).transpose(-2, -1).clone(memory_format=torch.contiguous_format)
    qk_t = q_scaled @ k_scaled_t
    amax = qk_t.amax(-1, True)
    sub = qk_t - amax
    args = [ sub, v ]
    sdpa = sdpa_decomposed
    out_cpu = sdpa(*args)
    args_spyre = [ x.to("spyre") for x in args ]
    out_spyre = torch.compile(sdpa)(*args_spyre).cpu()
    compare_results(out_cpu, out_spyre)

def sdpa_orig(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scale : float,
) -> torch.Tensor:
    q = q.clone(memory_format=torch.contiguous_format)
    k = k.clone(memory_format=torch.contiguous_format)
    v = v.clone(memory_format=torch.contiguous_format)
    scaling_factor = math.sqrt(scale)
    scaling_factor_q = torch.full_like(q, scaling_factor)
    scaling_factor_k = torch.full_like(k, scaling_factor)
    q = q * scaling_factor_q
    k = k * scaling_factor_k
    k_t = k.transpose(-2, -1).clone(memory_format=torch.contiguous_format)
    attn = torch.softmax(q @ k_t, -1)
    return (attn @ v).transpose(1, 2).clone(memory_format=torch.contiguous_format).transpose(1, 2)

def _test_sdpa_orig(q : torch.Tensor, k : torch.Tensor, v : torch.Tensor):
    args = [ q, k, v, 0.0078125 ]
    out_cpu = sdpa_orig(*args)
    args_spyre = [ x.to("spyre") if torch.is_tensor(x) else x for x in args ]
    out_spyre = torch.compile(sdpa_orig)(*args_spyre).cpu()
    compare_results(out_cpu, out_spyre)

def compare_results(out_cpu : torch.Tensor, out_spyre : torch.Tensor):
    atol = 0.005
    rtol = 0.005
    p_diff = (~torch.isclose(out_cpu, out_spyre, atol=atol, rtol=rtol)).sum().item() / out_cpu.numel() * 100
    if p_diff > 0:
        print(f"diff: {p_diff}%")
    assert torch.allclose(out_cpu, out_spyre, atol=atol, rtol=rtol)

def test_sdpa_ok():
    q = torch.rand(1, 32, 1, 128, dtype=torch.float16)
    k = torch.rand(1, 32, 64, 128, dtype=torch.float16)
    v = torch.rand(1, 32, 64, 128, dtype=torch.float16)
    _test_sdpa(q, k, v)

def test_sdpa_ng():
    q = torch.rand(1, 32, 1, 128, dtype=torch.float16)
    k = torch.rand(1, 32, 65, 128, dtype=torch.float16)
    v = torch.rand(1, 32, 65, 128, dtype=torch.float16)
    _test_sdpa(q, k, v)

if __name__ == "__main__":
    os.environ["SENCORES"] = "1"
    try:
        test_sdpa_ok()
        test_sdpa_ng()
    except Exception as e:
        import traceback
        traceback.print_exc()
