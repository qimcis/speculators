"""Utilities for FP8 scaled quantization of hidden states.

One float32 scale per vector along the last dimension: per token and layer
for a ``[seq_len, num_layers, hidden_size]`` hidden-states tensor. Layers differ
in magnitude by orders, so a scale shared across them leaves the quieter ones
with few significant bits. Payloads saved with one scale per token still
dequantize: their scales broadcast the same way.

Quantize: scale = amax / FP8_MAX; fp8 = (tensor / scale).to(fp8_dtype)
Dequantize: restored = fp8.to(target_dtype) * scale
"""

from __future__ import annotations

import torch

SCALES_KEY = "hidden_states_scales"


def _fp8_dtype() -> torch.dtype:
    """Resolve the FP8 dtype lazily (not all torch builds expose it)."""
    dtype = getattr(torch, "float8_e4m3fn", None)
    if dtype is None:
        raise RuntimeError(
            "This torch build does not expose torch.float8_e4m3fn; "
            "FP8 hidden-states quantization is unavailable."
        )
    return dtype


def quantize_tensor_to_fp8(
    tensor: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize a tensor to FP8 with one scale per vector of the last dimension.

    Args:
        tensor: Input tensor whose first dimension is the token/sequence
            dimension, e.g. ``[seq_len, num_layers, hidden_size]``, in any
            float dtype.

    Returns:
        Tuple of ``(fp8_tensor, scale)`` where ``scale`` has the shape of
        ``tensor`` with the last dimension 1 (e.g. ``[seq_len, num_layers, 1]``),
        so it broadcasts directly against ``tensor``.
    """
    fp8_dtype = _fp8_dtype()
    fp8_max = torch.finfo(fp8_dtype).max
    fp32 = tensor.float()
    amax = fp32.abs().amax(dim=-1, keepdim=True)
    scale = (amax / fp8_max).clamp(min=1e-12)
    fp8_tensor = (fp32 / scale).to(fp8_dtype)
    return fp8_tensor, scale


def dequantize_fp8_tensor(
    fp8_tensor: torch.Tensor,
    scale: torch.Tensor,
    dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Dequantize an FP8 tensor (with its scales) back to ``dtype``."""
    return fp8_tensor.to(dtype) * scale.to(dtype)
