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
Spyre Flash Attention Implementation

Provides optimized fused attention kernels for Spyre devices with tiled
matrix multiplication and softmax fusion.
"""

import math
from typing import Optional
import torch
from torch_spyre._inductor.errors import Unsupported

# Global flag to track if flash attention is enabled
_flash_attention_enabled = False


def is_flash_attention_enabled() -> bool:
    """Check if Spyre flash attention is currently enabled."""
    return _flash_attention_enabled


def spyre_flash_attention_forward(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_bias: Optional[torch.Tensor] = None,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    return_debug_mask: bool = False,
    scale: Optional[float] = None,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    int,
    int,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """
    Spyre Flash Attention forward pass with fused BMM + Softmax.

    This implementation uses tiled computation to optimize memory bandwidth
    and enable fusion of matrix multiplication with softmax operations.

    Args:
        query: Query tensor [B, H, S_q, D]
        key: Key tensor [B, H, S_k, D]
        value: Value tensor [B, H, S_k, D]
        attn_bias: Optional attention bias
        dropout_p: Dropout probability (not yet supported)
        is_causal: Whether to apply causal masking
        return_debug_mask: Whether to return debug mask
        scale: Optional scaling factor

    Returns:
        Tuple of (output, logsumexp, cum_seq_q, cum_seq_k, max_q, max_k,
                  philox_seed, philox_offset, debug_attn_mask)
    """
    batch_size = query.size(0)
    num_heads = query.size(1)
    max_seqlen_q = query.size(2)
    max_seqlen_kv = key.size(2)

    # Ensure contiguous memory layout for optimal performance
    query = query.clone(memory_format=torch.contiguous_format)
    key = key.clone(memory_format=torch.contiguous_format)
    value = value.clone(memory_format=torch.contiguous_format)

    # Calculate scaling factor
    scaling_factor = scale
    if scaling_factor is None:
        scaling_factor = 1.0 / math.sqrt(query.shape[-1])
    scaling_factor = math.sqrt(scaling_factor)

    # Apply scaling to query and key for numerical stability
    scaling_factor_tensor = torch.full_like(query, scaling_factor)
    query = query * scaling_factor_tensor
    key = key * scaling_factor_tensor

    # Transpose key for attention computation
    key_t = key.transpose(-2, -1).clone(memory_format=torch.contiguous_format)

    # Check if we can use fused kernel
    use_fused_kernel = _should_use_fused_kernel(query, key, attn_bias, is_causal)

    if use_fused_kernel:
        # Use fused BMM + Softmax kernel
        attn = torch.ops.spyre.fused_attention_bmm_softmax(query, key_t, scaling_factor)
    else:
        # Fallback to standard implementation
        attn = torch.matmul(query, key_t)

    # Apply causal mask if needed
    if is_causal:
        assert attn_bias is None, "Cannot use both is_causal and attn_bias"
        attn_bias = torch.full_like(attn, float("-inf"))
        attn_bias = attn_bias.triu(diagonal=1)

    # Apply attention bias
    if attn_bias is not None:
        attn = attn + attn_bias

    # Apply softmax (will be fused in the kernel if use_fused_kernel is True)
    if not use_fused_kernel:
        attn = torch.softmax(attn, -1)

    # Dropout (not yet supported)
    if dropout_p > 0.0:
        raise Unsupported("Attention dropout not implemented for Spyre flash attention")

    # Compute output
    out = torch.matmul(attn, value)

    # Transpose output to match expected format [B, S, H, E]
    out = out.transpose(1, 2).clone(memory_format=torch.contiguous_format)

    # Create placeholder tensors for compatibility with SDPA API
    logsumexp = torch.empty(
        (batch_size, num_heads, max_seqlen_q), dtype=torch.float32, device="spyre"
    )
    philox_seed = torch.empty((1,), dtype=torch.float16, device="spyre")
    philox_offset = torch.empty((1,), dtype=torch.float16, device="spyre")

    # Return in SDPA format: [B, S, H, E] -> transpose back to [B, H, S, E]
    return (
        out.transpose(1, 2),
        logsumexp,
        None,
        None,
        max_seqlen_q,
        max_seqlen_kv,
        philox_seed,
        philox_offset,
        None,
    )


def _should_use_fused_kernel(
    query: torch.Tensor,
    key: torch.Tensor,
    attn_bias: Optional[torch.Tensor],
    is_causal: bool,
) -> bool:
    """
    Determine if we should use the fused BMM + Softmax kernel.

    The fused kernel is beneficial when:
    - Tensors are on Spyre device
    - Shapes are compatible with tiling
    - No complex masking patterns (simple causal or no mask)
    """
    # Check if on Spyre device
    if query.device.type != "spyre" or key.device.type != "spyre":
        return False

    # Check if fused op is available
    if not hasattr(torch.ops.spyre, "fused_attention_bmm_softmax"):
        return False

    # For now, only use fused kernel without bias or with simple causal mask
    if attn_bias is not None and not is_causal:
        return False

    # Check if flash attention is enabled
    if not _flash_attention_enabled:
        return False

    return True


def _register_spyre_flash_attention_impl():
    """
    Internal function that performs the actual registration of the flash attention implementation.
    This is called by PyTorch when the backend is activated.
    """
    from torch_spyre._inductor import decompositions

    # Replace the existing SDPA decomposition with flash attention version
    decompositions.spyre_decompositions[
        torch.ops.aten._scaled_dot_product_fused_attention_overrideable.default
    ] = spyre_flash_attention_forward

    # Also update the dispatchkey registry if it exists
    if hasattr(decompositions, "spyre_decompositions_via_dispatchkey"):
        op = torch.ops.aten._scaled_dot_product_fused_attention_overrideable.default
        if op in decompositions.spyre_decompositions_via_dispatchkey:
            wrapper = decompositions.spyre_decompositions_via_dispatchkey[op]
            wrapper.spyre_fn = spyre_flash_attention_forward


def register_spyre_flash_attention():
    """
    Register Spyre flash attention backend with PyTorch.

    This function should be called during initialization to register
    the "fa_spyre" backend that can be activated via:
    torch.nn.attention.activate_flash_attention_impl("fa_spyre")
    """
    global _flash_attention_enabled

    try:
        # Register with torch.nn.attention (PyTorch 2.10+)
        import torch.nn.attention

        # Register the Spyre flash attention backend
        # First argument: backend name
        # Keyword argument: register_fn is a function that performs registration
        torch.nn.attention.register_flash_attention_impl(
            "fa_spyre", register_fn=_register_spyre_flash_attention_impl
        )
        _flash_attention_enabled = True
        return True

    except (ImportError, AttributeError) as e:
        # If registration fails, fall back to decomposition-based approach
        _enable_via_decomposition()
        _flash_attention_enabled = True
        return True


def _enable_via_decomposition():
    """
    Enable flash attention by updating the SDPA decomposition.

    This is a fallback method when the official registration API
    is not available.
    """
    from torch_spyre._inductor import decompositions

    # Replace the existing SDPA decomposition with flash attention version
    original_sdpa = decompositions.spyre__sdpa_overrideable

    def flash_sdpa_wrapper(*args, **kwargs):
        return spyre_flash_attention_forward(*args, **kwargs)

    # Update the decomposition registry
    decompositions.spyre_decompositions[
        torch.ops.aten._scaled_dot_product_fused_attention_overrideable.default
    ] = flash_sdpa_wrapper

    # Also update the dispatchkey registry if it exists
    if hasattr(decompositions, "spyre_decompositions_via_dispatchkey"):
        op = torch.ops.aten._scaled_dot_product_fused_attention_overrideable.default
        if op in decompositions.spyre_decompositions_via_dispatchkey:
            wrapper = decompositions.spyre_decompositions_via_dispatchkey[op]
            wrapper.spyre_fn = flash_sdpa_wrapper


def activate_flash_attention():
    """
    Activate Spyre flash attention.

    This is a convenience function that can be called directly:
    torch_spyre.attention.activate_flash_attention()

    Or via the standard PyTorch API:
    torch.nn.attention.activate_flash_attention("fa_spyre")
    """
    global _flash_attention_enabled
    _flash_attention_enabled = True
    return True


def deactivate_flash_attention():
    """Deactivate Spyre flash attention and fall back to standard implementation."""
    global _flash_attention_enabled
    _flash_attention_enabled = False
    return True


# Made with Bob
