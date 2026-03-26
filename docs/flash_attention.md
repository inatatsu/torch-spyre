# Flash Attention on Spyre

This document describes the flash attention implementation for Spyre devices, which provides optimized fused kernels for scaled dot-product attention.

## Overview

Flash Attention is an optimized attention mechanism that fuses batch matrix multiplication (BMM) and softmax operations into a single kernel. This fusion provides several benefits:

- **Reduced Memory Bandwidth**: Eliminates intermediate materialization of attention scores
- **Tiled Computation**: Enables processing of larger sequence lengths through memory-efficient tiling
- **Hardware Optimization**: Leverages Spyre's LX memory for stationary data and streaming computation
- **Numerical Stability**: Implements safe softmax with partial computation

## Architecture

The Spyre flash attention implementation consists of several components:

### 1. Custom Operator (`torch.ops.spyre.fused_attention_bmm_softmax`)

A custom PyTorch operator that represents the fused BMM + Softmax computation:

```python
torch.ops.spyre.fused_attention_bmm_softmax(
    query: Tensor,           # [B, H, S_q, D]
    key_transposed: Tensor,  # [B, H, D, S_k]
    scale: float             # Scaling factor
) -> Tensor                  # [B, H, S_q, S_k]
```

**Note**: The Spyre flash attention implementation is automatically registered during torch-spyre initialization and can be activated via `torch.nn.attention.activate_flash_attention_impl("fa_spyre")`.

### 2. Lowering to IR

The custom operator is lowered to `SpyreReduction` IR with:
- Reduction type: `"fused_bmm_softmax"`
- Fusion metadata in `op_info`
- Proper dimension mapping for tiled computation

### 3. SuperDSC Code Generation

The IR is code-generated into a SuperDSC JSON structure with:

```json
{
  "fused_attention_bmm_softmax": {
    "sdscFoldProps_": [
      {"size": 8, "label": "Q_tile"},
      {"size": 8, "label": "K_tile"},
      {"size": 8, "label": "seq_tile"}
    ],
    "dscs_": [{
      "computeOp_": "FUSED_BMM_SOFTMAX",
      "fusedOps_": [
        {
          "opType": "BatchMatMulV2",
          "opFunc": "MACC",
          "inputs": ["Q_tiled", "K_tiled"],
          "output": "attention_scores_partial"
        },
        {
          "opType": "Softmax",
          "opFunc": "EXP",
          "inputs": ["attention_scores_partial"],
          "output": "attention_weights_partial",
          "axis": -1,
          "partialCompute": true
        }
      ],
      "memoryStrategy_": {
        "LX": {
          "stationary": ["K_tiled", "partial_max", "partial_sum"],
          "streaming": ["Q_tiled"],
          "accumulate": ["attention_scores_partial"]
        }
      }
    }]
  }
}
```

## Usage

### Method 1: Standard PyTorch API

Use the standard `torch.nn.attention` API to activate flash attention:

```python
import torch
import torch.nn.functional as F

# Activate flash attention backend
torch.nn.attention.activate_flash_attention_impl("fa_spyre")

# Use standard SDPA
device = torch.device("spyre")
Q = torch.randn(2, 8, 128, 64, device=device, dtype=torch.float16)
K = torch.randn(2, 8, 128, 64, device=device, dtype=torch.float16)
V = torch.randn(2, 8, 128, 64, device=device, dtype=torch.float16)

# This will use flash attention automatically
output = F.scaled_dot_product_attention(Q, K, V)
```

### Method 2: Direct Activation

Activate flash attention directly through torch_spyre:

```python
import torch
import torch_spyre.attention

# Activate flash attention
torch_spyre.attention.flash_attention.activate_flash_attention()

# Your attention code here
def attention(q, k, v, scale):
    scores = torch.matmul(q, k.transpose(-2, -1)) * scale
    weights = torch.softmax(scores, dim=-1)
    return torch.matmul(weights, v)

# Compile and run
compiled_attn = torch.compile(attention)
output = compiled_attn(Q, K, V, 0.125)

# Deactivate when done
torch_spyre.attention.flash_attention.deactivate_flash_attention()
```

### Method 3: Direct Kernel Usage (Advanced)

For advanced users, the fused kernel can be called directly:

```python
import torch

device = torch.device("spyre")
Q = torch.randn(2, 8, 128, 64, device=device, dtype=torch.float16)
K = torch.randn(2, 8, 128, 64, device=device, dtype=torch.float16)
V = torch.randn(2, 8, 128, 64, device=device, dtype=torch.float16)

# Transpose key for attention computation
K_T = K.transpose(-2, -1).contiguous()

# Call fused kernel
scale = 1.0 / (64 ** 0.5)
attn_weights = torch.ops.spyre.fused_attention_bmm_softmax(Q, K_T, scale)

# Complete attention with value matmul
output = torch.matmul(attn_weights, V)
```

## Configuration

### Tile Sizes

The default tile sizes are optimized for typical workloads:

- `Q_TILE = 8`: Query tile size
- `K_TILE = 8`: Key tile size  
- `SEQ_TILE = 8`: Sequence tile size

These can be adjusted in `torch_spyre/_inductor/codegen/compute_ops.py` based on:
- Available LX memory
- Sequence length characteristics
- Hardware configuration

### Memory Strategy

The flash attention kernel uses a specific memory strategy:

- **Stationary in LX**: Key tiles, partial max, partial sum
- **Streaming**: Query tiles
- **Accumulation**: Attention score partials

This strategy minimizes memory bandwidth by:
1. Loading key tiles once and reusing them
2. Streaming query tiles through the computation
3. Accumulating partial results in LX memory

## Performance Considerations

### When to Use Flash Attention

Flash attention is beneficial when:

- Sequence lengths are moderate to large (>64)
- Batch size and number of heads allow for parallelization
- Memory bandwidth is a bottleneck
- Standard attention patterns (no complex masking)

### Limitations

Current limitations:

- Dropout is not yet supported
- Complex attention bias patterns may fall back to standard implementation
- Causal masking is supported but may not be fully fused

## Implementation Details

### Files Modified/Created

1. **`torch_spyre/attention/`** - New module for flash attention
   - `__init__.py` - Module initialization
   - `flash_attention.py` - Flash attention implementation and registration

2. **`torch_spyre/_inductor/customops.py`** - Added fused attention custom op

3. **`torch_spyre/_inductor/lowering.py`** - Added lowering for fused attention

4. **`torch_spyre/_inductor/codegen/superdsc.py`** - Added codegen dispatch

5. **`torch_spyre/_inductor/codegen/compute_ops.py`** - Added SuperDSC generation

6. **`torch_spyre/__init__.py`** - Registers flash attention on initialization

7. **`examples/flash_attention.py`** - Example usage

### Integration Points

The flash attention integrates with PyTorch at multiple levels:

1. **Decomposition Level**: Replaces SDPA decomposition when activated
2. **Lowering Level**: Custom lowering to SpyreReduction IR
3. **Codegen Level**: Generates fused SuperDSC kernels
4. **Runtime Level**: Executes on Spyre hardware

## Testing

Run the example to verify flash attention:

```bash
python examples/flash_attention.py
```

Expected output:
- Successful activation of flash attention backend
- Numerical accuracy within tolerance of CPU reference
- Performance improvements for larger sequence lengths

## Future Enhancements

Planned improvements:

1. **Dropout Support**: Add fused dropout in attention weights
2. **GQA Optimization**: Optimize for grouped-query attention patterns
3. **Dynamic Tiling**: Automatically adjust tile sizes based on input shapes
4. **Multi-Core**: Distribute computation across multiple Spyre cores
5. **Backward Pass**: Implement fused backward pass for training

## References

- [Flash Attention Paper](https://arxiv.org/abs/2205.14135)
- [Flash Attention 2](https://arxiv.org/abs/2307.08691)
- PyTorch SDPA Documentation
- Spyre Architecture Documentation

## Support

For issues or questions:
- Check the example: `examples/flash_attention.py`
- Review this documentation
- File an issue on the torch-spyre repository