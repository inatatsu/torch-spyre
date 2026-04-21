import pytest
import torch

@pytest.mark.parametrize("i", [i for i in range(100)])
def test_sdpa(i : int):
    batch_size = 1
    num_heads = 16
    seq_len = 1064
    head_dim = 64
    q = torch.rand(batch_size, num_heads, seq_len, head_dim, dtype=torch.float16)
    k = torch.rand(batch_size, num_heads, seq_len, head_dim, dtype=torch.float16)
    v = torch.rand(batch_size, num_heads, seq_len, head_dim, dtype=torch.float16)
    args = [ q, k, v ]
    kwargs = { "dropout_p": 0.0, "scale": 0.125, "is_causal": False }
    out_cpu = torch.nn.functional.scaled_dot_product_attention(*args, **kwargs)
    args_spyre = [ x.to("spyre") for x in args ]
    out_spyre = torch.nn.functional.scaled_dot_product_attention(*args_spyre, **kwargs).cpu()
    compare_results(out_cpu, out_spyre)

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
        test_sdpa
    except Exception as e:
        import traceback
        traceback.print_exc()
