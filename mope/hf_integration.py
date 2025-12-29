"""Hugging Face integration helpers for attaching MoPE to transformer models.

Supports models with ``model.layers`` blocks (e.g., Qwen/LLaMA-style) or
``transformer.h`` (GPT-style). Replaces a block's MLP/FFN with a TorchMoPELayer
wrapper for end-to-end training and evaluation.
"""
from __future__ import annotations

from typing import Iterable, List, Optional

import torch
import torch.nn as nn

from .torch_mope import TorchMoPELayer
from .pipeline import PIPELINE_REGISTRY


def _get_blocks(model) -> List[object]:
    # Try common HF structures
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return list(model.model.layers)
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return list(model.transformer.h)
    raise AttributeError("Unsupported model structure: expected model.layers or transformer.h")


def attach_mope_to_hf_model(
    model,
    *,
    layer_indices: Iterable[int],
    hidden_size: Optional[int] = None,
    expert_names: Optional[Iterable[str]] = None,
    gate_json: Optional[dict] = None,
    use_adapters: bool = True,
    alpha: float = 1.0,
) -> List[TorchMoPELayer]:
    """Attach MoPE to selected layers by replacing their MLP/FFN.

    Returns the list of TorchMoPELayer instances (one per attached layer).
    """

    blocks = _get_blocks(model)
    hs = int(hidden_size or getattr(getattr(model, "config", object()), "hidden_size", None) or 0)
    if hs <= 0:
        raise ValueError("hidden_size not found; provide explicitly")
    experts = list(expert_names or PIPELINE_REGISTRY.keys())

    class MoPEFFN(nn.Module):
        def __init__(self, mope_layer: TorchMoPELayer) -> None:
            super().__init__()
            self.mope = mope_layer

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            # x: [B, T, H] or [*, H]
            n = x.shape[:-1]
            H = x.shape[-1]
            flat = x.reshape(-1, H)
            # Prompt is optional; leave empty at inference time
            out_rows = [self.mope(row, "") for row in flat]
            out = torch.stack(out_rows, dim=0).reshape(*n, H)
            return out

    mope_layers: List[TorchMoPELayer] = []
    for idx in layer_indices:
        if not (0 <= idx < len(blocks)):
            raise IndexError(f"layer index {idx} out of range; model has {len(blocks)} blocks")
        block = blocks[idx]
        # Detect common FFN attribute names
        target_attr: Optional[str] = None
        if hasattr(block, "mlp"):
            target_attr = "mlp"
        elif hasattr(block, "ffn"):
            target_attr = "ffn"
        elif hasattr(block, "feed_forward"):
            target_attr = "feed_forward"
        if target_attr is None:
            raise AttributeError(f"block[{idx}] has no FFN attribute among ['mlp','ffn','feed_forward']")
        mope = TorchMoPELayer(hidden_size=hs, expert_names=experts, use_adapters=use_adapters)
        if gate_json is not None:
            mope.load_gate_json(gate_json)
        mope.set_alpha(alpha)
        # Preserve device/dtype
        mope.to(next(model.parameters()).device)
        setattr(block, target_attr, MoPEFFN(mope))
        mope_layers.append(mope)

    return mope_layers
