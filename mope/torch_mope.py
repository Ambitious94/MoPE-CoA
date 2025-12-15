"""Torch-based MoPE gate and layer for nanoGPT integration.

Provides TorchSimpleGate and TorchMoPELayer that mirror the python list-based
SimpleGate/MoPELayer but implemented with torch.nn.Module for end-to-end training.

Notes:
- External tool steps should be disabled in training to avoid IO.
- Vectorizer uses a simple hash-based encoding to produce residual updates.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from .pipeline import PIPELINE_REGISTRY, PipelineExpert


@dataclass
class TorchGateConfig:
    hidden_size: int
    num_experts: int
    temperature: float = 1.0


class TorchSimpleGate(nn.Module):
    def __init__(self, config: TorchGateConfig) -> None:
        super().__init__()
        self.config = config
        self.weight = nn.Parameter(torch.empty(config.hidden_size, config.num_experts))
        self.bias = nn.Parameter(torch.zeros(config.num_experts))
        nn.init.normal_(self.weight, mean=0.0, std=0.1)

    def logits(self, hidden_state: torch.Tensor) -> torch.Tensor:
        # hidden_state: [H]
        return hidden_state @ self.weight + self.bias  # [E]

    def probs(self, hidden_state: torch.Tensor) -> torch.Tensor:
        logits = self.logits(hidden_state) / max(self.config.temperature, 1e-8)
        return F.softmax(logits, dim=-1)

    def route(self, hidden_state: torch.Tensor) -> torch.Tensor:
        # returns expert index tensor []
        p = self.probs(hidden_state)
        return torch.argmax(p, dim=-1)


class TorchHashVectorizer(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim

    def encode(self, text: str) -> torch.Tensor:
        # hash-based deterministic pseudo-embedding
        seed = abs(hash(text)) % (2**32)
        rng = torch.Generator(device='cpu')
        rng.manual_seed(seed)
        return torch.normal(mean=0.0, std=1.0, size=(self.dim,), generator=rng)

    def residual_add(self, hidden: torch.Tensor, update: torch.Tensor) -> torch.Tensor:
        return hidden + update


class TorchMoPELayer(nn.Module):
    def __init__(self, hidden_size: int, expert_names: Sequence[str], pipelines: dict[str, PipelineExpert] | None = None, temperature: float = 1.0) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.expert_names = list(expert_names)
        self.pipelines = pipelines or PIPELINE_REGISTRY
        self.vectorizer = TorchHashVectorizer(dim=hidden_size)
        self.gate = TorchSimpleGate(TorchGateConfig(hidden_size=hidden_size, num_experts=len(self.expert_names), temperature=temperature))

    @torch.no_grad()
    def load_gate_json(self, gate_json: dict) -> None:
        w = torch.tensor(gate_json.get("weight"), dtype=torch.float32)
        b = torch.tensor(gate_json.get("bias"), dtype=torch.float32)
        if w.shape != self.gate.weight.shape:
            raise ValueError(f"gate weight shape mismatch: {w.shape} vs {tuple(self.gate.weight.shape)}")
        if b.shape != self.gate.bias.shape:
            raise ValueError(f"gate bias shape mismatch: {b.shape} vs {tuple(self.gate.bias.shape)}")
        self.gate.weight.copy_(w)
        self.gate.bias.copy_(b)

    def forward(self, hidden_state: torch.Tensor, prompt: str) -> torch.Tensor:
        # hidden_state: [H]
        # choose expert but avoid heavy IO during training
        _ = self.gate.route(hidden_state)
        # produce residual from answer text via hash vectorizer (placeholder)
        # In training, disable external tool usage: use prompt as source to stabilize
        update = self.vectorizer.encode(prompt)
        return self.vectorizer.residual_add(hidden_state, update)

    def compute_gate_ce(self, hidden_batch: torch.Tensor, label_indices: torch.Tensor) -> torch.Tensor:
        """Compute cross-entropy loss over gate probs for supervised routing.

        hidden_batch: [N, H]
        label_indices: [N] long tensor with expert indices
        Returns: scalar loss tensor
        """
        logits = hidden_batch @ self.gate.weight + self.gate.bias  # [N, E]
        logits = logits / max(self.gate.config.temperature, 1e-8)
        loss = F.cross_entropy(logits, label_indices)
        return loss
