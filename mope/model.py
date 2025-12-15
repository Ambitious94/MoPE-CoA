"""Minimal MoPE-Transformer scaffold that stacks MoPE layers."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Iterable, List

from .mope_layer import MoPELayer, MoPELayerConfig
from .pipeline import PIPELINE_REGISTRY


def _gaussian_vector(seed: int, dim: int) -> List[float]:
    rng = random.Random(seed)
    return [rng.gauss(0.0, 1.0) for _ in range(dim)]


@dataclass
class TransformerConfig:
    vocab_size: int
    hidden_size: int
    num_layers: int


class MoPETransformer:
    """Tiny Transformer-like container that swaps MoPE layers for FFNs."""

    def __init__(self, config: TransformerConfig) -> None:
        self.config = config
        expert_names = list(PIPELINE_REGISTRY.keys())
        layer_config = MoPELayerConfig(hidden_size=config.hidden_size, expert_names=expert_names)
        self.layers: List[MoPELayer] = [MoPELayer(layer_config) for _ in range(config.num_layers)]

    def encode_prompt(self, prompt: str) -> List[float]:
        seed = abs(hash(prompt)) % (2**32)
        return _gaussian_vector(seed, self.config.hidden_size)

    def forward(self, prompt: str) -> dict:
        hidden = self.encode_prompt(prompt)
        traces = []
        for layer in self.layers:
            output = layer.forward(hidden, prompt)
            hidden = output["hidden_state"]
            traces.append({"pipeline": output["pipeline"], "trace": output["trace"]})
        return {
            "final_state": hidden,
            "layer_traces": traces,
        }

    def batch_forward(self, prompts: Iterable[str]):
        return [self.forward(prompt) for prompt in prompts]
