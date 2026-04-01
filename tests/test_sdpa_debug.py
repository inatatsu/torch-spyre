# Copyright 2025 The Torch-Spyre Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Debug tests to isolate which operation in SDPA decomposition fails with size 65.
Run with: pytest tests/test_sdpa_debug.py -v -s
"""

import math
import torch
import pytest


def create_inputs_on_device(device, dtype=torch.float16):
    """Create test inputs directly on the target device with hardcoded seq_len=65."""
    # Hardcode seq_len=65 to avoid symbolic shapes
    query = torch.rand(1, 1, 1, 128, device=device, dtype=dtype)
    key = torch.rand(1, 1, 65, 128, device=device, dtype=dtype)
    value = torch.rand(1, 1, 65, 128, device=device, dtype=dtype)
    return query, key, value


@pytest.mark.parametrize("device", ["cpu", "spyre"])
def test_sdpa_step1_scaling(device):
    """Test step 1: Scale query and key by sqrt(scale) using compiled function"""
    
    # Create inputs on device
    query, key, _ = create_inputs_on_device(device)
    scale = 0.0078125
    scaling_factor = math.sqrt(scale)
    
    # Define the operation to compile
    def scale_tensors(q, k, scale_val):
        q = q.clone(memory_format=torch.contiguous_format)
        k = k.clone(memory_format=torch.contiguous_format)
        scaling_tensor = torch.full_like(q, scale_val)
        q_scaled = q * scaling_tensor
        k_scaled = k * scaling_tensor
        # Rematerialize outputs
        q_scaled = q_scaled.clone(memory_format=torch.contiguous_format)
        k_scaled = k_scaled.clone(memory_format=torch.contiguous_format)
        return q_scaled, k_scaled
    
    # Compile and run
    if device.startswith("spyre"):
        compiled_fn = torch.compile(scale_tensors, backend="inductor")
    else:
        compiled_fn = scale_tensors
    
    query_scaled, key_scaled = compiled_fn(query, key, scaling_factor)
    
    # Move to CPU for checking
    query_scaled_cpu = query_scaled.cpu()
    key_scaled_cpu = key_scaled.cpu()
    
    # Check results
    assert not torch.isnan(query_scaled_cpu).any(), "NaN in query_scaled"
    assert not torch.isnan(key_scaled_cpu).any(), "NaN in key_scaled"
    print(f"✓ Step 1 (scaling) passed on {device}")


@pytest.mark.parametrize("device", ["cpu", "spyre"])
def test_sdpa_step2_transpose(device):
    """Test step 2: Transpose key using compiled function"""
    
    # Create inputs on device
    _, key, _ = create_inputs_on_device(device)
    scale = 0.0078125
    scaling_factor = math.sqrt(scale)
    
    # Define the operation
    def transpose_key(k, scale_val):
        k = k.clone(memory_format=torch.contiguous_format)
        scaling_tensor = torch.full((1, 1, 1, 128), scale_val, device=k.device, dtype=k.dtype)
        k_scaled = k * scaling_tensor
        k_t = k_scaled.transpose(-2, -1).clone(memory_format=torch.contiguous_format)
        # Rematerialize output
        k_t = k_t.clone(memory_format=torch.contiguous_format)
        return k_t
    
    # Compile and run
    if device.startswith("spyre"):
        compiled_fn = torch.compile(transpose_key, backend="inductor")
    else:
        compiled_fn = transpose_key
    
    key_t = compiled_fn(key, scaling_factor)
    key_t_cpu = key_t.cpu()
    
    # Check results
    assert key_t.shape == (1, 1, 128, 65), f"Wrong shape: {key_t.shape}"
    assert not torch.isnan(key_t_cpu).any(), "NaN in key_t"
    print(f"✓ Step 2 (transpose) passed on {device}")


@pytest.mark.parametrize("device", ["cpu", "spyre"])
def test_sdpa_step3_matmul(device):
    """Test step 3: Matmul query @ key_t using compiled function"""
    
    # Create inputs on device
    query, key, _ = create_inputs_on_device(device)
    scale = 0.0078125
    scaling_factor = math.sqrt(scale)
    
    # Define the operation
    def matmul_qk(q, k, scale_val):
        q = q.clone(memory_format=torch.contiguous_format)
        k = k.clone(memory_format=torch.contiguous_format)
        scaling_tensor = torch.full_like(q, scale_val)
        q_scaled = q * scaling_tensor
        k_scaled = k * scaling_tensor
        k_t = k_scaled.transpose(-2, -1).clone(memory_format=torch.contiguous_format)
        attn = torch.matmul(q_scaled, k_t)
        # Rematerialize output
        attn = attn.clone(memory_format=torch.contiguous_format)
        return attn
    
    # Compile and run
    if device.startswith("spyre"):
        compiled_fn = torch.compile(matmul_qk, backend="inductor")
    else:
        compiled_fn = matmul_qk
    
    attn = compiled_fn(query, key, scaling_factor)
    attn_cpu = attn.cpu()
    
    # Check results
    assert attn.shape == (1, 1, 1, 65), f"Wrong shape: {attn.shape}"
    assert not torch.isnan(attn_cpu).any(), f"NaN in attn"
    assert not torch.isinf(attn_cpu).any(), f"Inf in attn"
    assert attn_cpu.abs().max() < 10.0, f"Attn values too large: {attn_cpu.abs().max()}"
    print(f"✓ Step 3 (matmul) passed")
    print(f"  attn range: [{attn_cpu.min():.4f}, {attn_cpu.max():.4f}]")


@pytest.mark.parametrize("device", ["cpu", "spyre"])
def test_sdpa_step4_softmax(device):
    """Test step 4: Softmax using compiled function"""
    
    # Create inputs on device
    query, key, _ = create_inputs_on_device(device)
    scale = 0.0078125
    scaling_factor = math.sqrt(scale)
    
    # Define the operation
    def matmul_and_softmax(q, k, scale_val):
        q = q.clone(memory_format=torch.contiguous_format)
        k = k.clone(memory_format=torch.contiguous_format)
        scaling_tensor = torch.full_like(q, scale_val)
        q_scaled = q * scaling_tensor
        k_scaled = k * scaling_tensor
        k_t = k_scaled.transpose(-2, -1).clone(memory_format=torch.contiguous_format)
        attn = torch.matmul(q_scaled, k_t)
        attn_softmax = torch.softmax(attn, -1)
        # Rematerialize output
        attn_softmax = attn_softmax.clone(memory_format=torch.contiguous_format)
        return attn_softmax
    
    # Compile and run
    if device.startswith("spyre"):
        compiled_fn = torch.compile(matmul_and_softmax, backend="inductor")
    else:
        compiled_fn = matmul_and_softmax
    
    attn_softmax = compiled_fn(query, key, scaling_factor)
    attn_softmax_cpu = attn_softmax.cpu()
    
    # Check results
    assert attn_softmax.shape == (1, 1, 1, 65), f"Wrong shape: {attn_softmax.shape}"
    assert not torch.isnan(attn_softmax_cpu).any(), f"NaN in attn_softmax"
    
    # Softmax should sum to 1
    sum_val = attn_softmax_cpu.sum(dim=-1)
    assert torch.allclose(sum_val, torch.ones_like(sum_val), rtol=1e-2, atol=1e-2), \
        f"Softmax doesn't sum to 1: {sum_val}"
    
    print(f"✓ Step 4 (softmax) passed")
    print(f"  attn_softmax range: [{attn_softmax_cpu.min():.4f}, {attn_softmax_cpu.max():.4f}]")


@pytest.mark.parametrize("device", ["cpu", "spyre"])
def test_sdpa_step5_final_matmul(device):
    """Test step 5: Final matmul with value using compiled function"""
    
    # Create inputs on device
    query, key, value = create_inputs_on_device(device)
    scale = 0.0078125
    scaling_factor = math.sqrt(scale)
    
    # Define the operation
    def full_sdpa_computation(q, k, v, scale_val):
        q = q.clone(memory_format=torch.contiguous_format)
        k = k.clone(memory_format=torch.contiguous_format)
        v = v.clone(memory_format=torch.contiguous_format)
        scaling_tensor = torch.full_like(q, scale_val)
        q_scaled = q * scaling_tensor
        k_scaled = k * scaling_tensor
        k_t = k_scaled.transpose(-2, -1).clone(memory_format=torch.contiguous_format)
        attn = torch.matmul(q_scaled, k_t)
        attn_softmax = torch.softmax(attn, -1)
        out = torch.matmul(attn_softmax, v)
        # Rematerialize output
        out = out.clone(memory_format=torch.contiguous_format)
        return out
    
    # Compile and run
    if device.startswith("spyre"):
        compiled_fn = torch.compile(full_sdpa_computation, backend="inductor")
    else:
        compiled_fn = full_sdpa_computation
    
    out = compiled_fn(query, key, value, scaling_factor)
    out_cpu = out.cpu()
    
    # Check results
    assert out.shape == (1, 1, 1, 128), f"Wrong shape: {out.shape}"
    assert not torch.isnan(out_cpu).any(), f"NaN in out"
    assert not torch.isinf(out_cpu).any(), f"Inf in out"
    assert out_cpu.abs().max() < 2.0, f"Output values too large: {out_cpu.abs().max()}"
    
    print(f"✓ Step 5 (final matmul) passed")
    print(f"  out range: [{out_cpu.min():.4f}, {out_cpu.max():.4f}]")


if __name__ == "__main__":
    # Run tests manually - only test seq_len=65
    for device in ["cpu", "spyre"]:
        print(f"\n{'='*60}")
        print(f"Testing on {device}")
        print(f"{'='*60}")
        try:
            test_sdpa_step1_scaling(device)
            test_sdpa_step2_transpose(device)
            test_sdpa_step3_matmul(device)
            test_sdpa_step4_softmax(device)
            test_sdpa_step5_final_matmul(device)
        except Exception as e:
            print(f"✗ FAILED: {e}")
            import traceback
            traceback.print_exc()

# Made with Bob
