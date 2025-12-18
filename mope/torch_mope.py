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
        # ensure update matches hidden's device and dtype to avoid device/dtype mismatch
        if update.device != hidden.device or update.dtype != hidden.dtype:
            update = update.to(device=hidden.device, dtype=hidden.dtype)
        return hidden + update


class TorchMoPELayer(nn.Module):
    def __init__(self, hidden_size: int, expert_names: Sequence[str], pipelines: dict[str, PipelineExpert] | None = None, temperature: float = 1.0, use_adapters: bool = True) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.expert_names = list(expert_names)
        self.pipelines = pipelines or PIPELINE_REGISTRY
        self.vectorizer = TorchHashVectorizer(dim=hidden_size)
        self.gate = TorchSimpleGate(TorchGateConfig(hidden_size=hidden_size, num_experts=len(self.expert_names), temperature=temperature))
        self.use_adapters = bool(use_adapters)
        # RL-related controls
        self.stochastic: bool = False  # if True (and training), sample expert; else use argmax
        self._forced_expert: int | None = None  # when set, force using this expert once
        # Trainable per-expert adapters to produce residual updates from hidden_state
        # Initialize last layer to zeros so initial behavior is near identity
        if self.use_adapters:
            adapters: list[nn.Module] = []
            for _ in self.expert_names:
                mlp = nn.Sequential(
                    nn.Linear(hidden_size, 4 * hidden_size),
                    nn.GELU(),
                    nn.Linear(4 * hidden_size, hidden_size),
                )
                nn.init.zeros_(mlp[2].weight)
                nn.init.zeros_(mlp[2].bias)
                adapters.append(mlp)
            self.expert_adapters = nn.ModuleList(adapters)
        else:
            self.expert_adapters = nn.ModuleList()

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

    def set_forced_expert(self, expert_idx: int | None) -> None:
        """Force the next forward() call to use the given expert index.

        Pass None to clear and resume normal selection (argmax or sampling).
        """
        self._forced_expert = int(expert_idx) if expert_idx is not None else None

    def forward(self, hidden_state: torch.Tensor, prompt: str) -> torch.Tensor:
        # hidden_state: [H]
        # choose expert index
        probs = self.gate.probs(hidden_state)
        if self._forced_expert is not None:
            expert_idx = int(self._forced_expert)
            self._forced_expert = None  # consume once
        elif self.stochastic and self.training:
            expert_idx = int(torch.multinomial(probs, num_samples=1).item())
        else:
            expert_idx = int(torch.argmax(probs, dim=-1).item())
        expert_name = self.expert_names[expert_idx] if 0 <= expert_idx < len(self.expert_names) else "unknown"
        # produce residual via trainable expert adapter; fallback vectorizer can be used for ablation
        if self.use_adapters and 0 <= expert_idx < len(self.expert_adapters):
            update = self.expert_adapters[expert_idx](hidden_state)
        else:
            update = self.vectorizer.encode(f"[{expert_name}] {prompt or ''}")
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
