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
Example: Spyre Flash Attention

This example demonstrates how to use the optimized flash attention
implementation on Spyre devices. Flash attention fuses the batch matrix
multiplication and softmax operations for improved performance.

Usage:
    python examples/flash_attention.py
"""

import sys
import torch
import torch.nn.functional as F

# Initialize Spyre device
DEVICE = torch.device("spyre")
torch.manual_seed(0xAFFE)

print("=" * 70)
print("Spyre Flash Attention Example")
print("=" * 70)

# Configuration
BATCH_SIZE = 2
NUM_HEADS = 8
SEQ_LENGTH = 128
HEAD_DIM = 64

print(f"\nConfiguration:")
print(f"  Batch size: {BATCH_SIZE}")
print(f"  Number of heads: {NUM_HEADS}")
print(f"  Sequence length: {SEQ_LENGTH}")
print(f"  Head dimension: {HEAD_DIM}")

# Create random input tensors
print("\nCreating input tensors...")
Q = torch.randn(BATCH_SIZE, NUM_HEADS, SEQ_LENGTH, HEAD_DIM, dtype=torch.float16)
K = torch.randn(BATCH_SIZE, NUM_HEADS, SEQ_LENGTH, HEAD_DIM, dtype=torch.float16)
V = torch.randn(BATCH_SIZE, NUM_HEADS, SEQ_LENGTH, HEAD_DIM, dtype=torch.float16)

# Compute attention on CPU (reference)
print("\nComputing attention on CPU (reference)...")
scale = 1.0 / (HEAD_DIM**0.5)
attn_scores_cpu = torch.matmul(Q, K.transpose(-2, -1)) * scale
attn_weights_cpu = F.softmax(attn_scores_cpu, dim=-1)
output_cpu = torch.matmul(attn_weights_cpu, V)

# Transfer tensors to Spyre device
print("\nTransferring tensors to Spyre device...")
Q_device = Q.to(DEVICE)
K_device = K.to(DEVICE)
V_device = V.to(DEVICE)

# Method 1: Using torch.nn.functional.scaled_dot_product_attention
print("\n" + "=" * 70)
print("Method 1: Using F.scaled_dot_product_attention (Standard API)")
print("=" * 70)

try:
    # Activate flash attention backend
    print("\nActivating flash attention backend...")
    torch.nn.attention.activate_flash_attention_impl("fa_spyre")
    print("✓ Flash attention backend 'fa_spyre' activated")

    # Compile the attention function
    print("\nCompiling attention function...")
    compiled_sdpa = torch.compile(
        lambda q, k, v: F.scaled_dot_product_attention(q, k, v, scale=scale)
    )

    # Run compiled attention
    print("Running compiled flash attention...")
    output_flash = compiled_sdpa(Q_device, K_device, V_device).cpu()

    # Compare results
    max_diff = torch.abs(output_cpu - output_flash).max()
    print(f"\n✓ Flash attention completed")
    print(f"  Max difference vs CPU: {max_diff:.6f}")

except Exception as e:
    print(f"\n✗ Error with Method 1: {e}")
    print("  This may occur if torch.nn.attention API is not available")

sys.exit(0)

# Method 2: Direct activation via torch_spyre
print("\n" + "=" * 70)
print("Method 2: Using torch_spyre.attention directly")
print("=" * 70)

try:
    import torch_spyre.attention

    # Activate flash attention
    print("\nActivating flash attention...")
    torch_spyre.attention.flash_attention.activate_flash_attention()
    print("✓ Flash attention activated")

    # Define attention function
    def attention_forward(q, k, v, scale_factor):
        scores = torch.matmul(q, k.transpose(-2, -1)) * scale_factor
        weights = F.softmax(scores, dim=-1)
        return torch.matmul(weights, v)

    # Compile and run
    print("\nCompiling attention function...")
    compiled_attn = torch.compile(attention_forward)

    print("Running compiled flash attention...")
    output_direct = compiled_attn(Q_device, K_device, V_device, scale).cpu()

    # Compare results
    max_diff = torch.abs(output_cpu - output_direct).max()
    print(f"\n✓ Flash attention completed")
    print(f"  Max difference vs CPU: {max_diff:.6f}")

    # Deactivate flash attention
    torch_spyre.attention.flash_attention.deactivate_flash_attention()
    print("\n✓ Flash attention deactivated")

except Exception as e:
    print(f"\n✗ Error with Method 2: {e}")

# Method 3: Using the fused kernel directly
print("\n" + "=" * 70)
print("Method 3: Using fused kernel directly (Advanced)")
print("=" * 70)

try:
    # Check if fused op is available
    if hasattr(torch.ops.spyre, "fused_attention_bmm_softmax"):
        print("\n✓ Fused attention kernel available")

        # Prepare inputs
        K_T = K_device.transpose(-2, -1).contiguous()

        # Call fused kernel
        print("Running fused BMM + Softmax kernel...")
        attn_weights_fused = torch.ops.spyre.fused_attention_bmm_softmax(
            Q_device, K_T, scale
        )
        output_fused = torch.matmul(attn_weights_fused, V_device).cpu()

        # Compare results
        max_diff = torch.abs(output_cpu - output_fused).max()
        print(f"\n✓ Fused kernel completed")
        print(f"  Max difference vs CPU: {max_diff:.6f}")
    else:
        print("\n✗ Fused attention kernel not available")
        print("  The kernel may not be registered yet")

except Exception as e:
    print(f"\n✗ Error with Method 3: {e}")

print("\n" + "=" * 70)
print("Example completed!")
print("=" * 70)
print("\nKey Benefits of Flash Attention:")
print("  • Fused BMM + Softmax reduces memory bandwidth")
print("  • Tiled computation enables larger sequence lengths")
print("  • Optimized for Spyre hardware architecture")
print("  • Compatible with standard PyTorch APIs")

# Made with Bob
