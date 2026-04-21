import math
import pytest
import torch

def sdpa(q : torch.Tensor, k : torch.Tensor, v : torch.Tensor) -> torch.Tensor:
    return k.transpose(-2, -1).clone(memory_format=torch.contiguous_format)

def _test_sdpa(q : torch.Tensor, k : torch.Tensor, v : torch.Tensor):
    scale = math.sqrt(0.125)
    scale_q = torch.full_like(q, scale)
    scale_k = torch.full_like(k, scale)
    q = q * scale_q
    k = k * scale_k
    args = [ q, k, v ]
    out_cpu = sdpa(*args)
    args_spyre = [ x.to("spyre") for x in args ]
    out_spyre = torch.compile(sdpa)(*args_spyre).cpu()
    compare_results(out_cpu, out_spyre)

def test_sdpa_ok():
    seq_len = 1088
    batch_size = 1
    num_heads = 16
    head_dim = 64
    q = torch.rand(batch_size, num_heads, seq_len, head_dim, dtype=torch.float16)
    k = torch.rand(batch_size, num_heads, seq_len, head_dim, dtype=torch.float16)
    v = torch.rand(batch_size, num_heads, seq_len, head_dim, dtype=torch.float16)
    _test_sdpa(q, k, v)

def test_sdpa_ng():
    seq_len = 1064
    batch_size = 1
    num_heads = 16
    head_dim = 64
    q = torch.rand(batch_size, num_heads, seq_len, head_dim, dtype=torch.float16)
    k = torch.rand(batch_size, num_heads, seq_len, head_dim, dtype=torch.float16)
    v = torch.rand(batch_size, num_heads, seq_len, head_dim, dtype=torch.float16)
    _test_sdpa(q, k, v)

def compare_results(out_cpu : torch.Tensor, out_spyre : torch.Tensor):
    atol = 0.005
    rtol = 0.005
    p_diff = (~torch.isclose(out_cpu, out_spyre, atol=atol, rtol=rtol)).sum().item() / out_cpu.numel() * 100
    if p_diff > 0:
        print(f"diff: {p_diff}%")
    assert torch.allclose(out_cpu, out_spyre, atol=atol, rtol=rtol)

if __name__ == "__main__":
    os.environ["SENCORES"] = "1"
    try:
        test_sdpa_ok
        test_sdpa_ng
    except Exception as e:
        import traceback
        traceback.print_exc()
