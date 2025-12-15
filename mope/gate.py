"""Gate modules for selecting a pipeline expert."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Dict, List, Sequence


@dataclass
class GateConfig:
    hidden_size: int
    num_experts: int
    temperature: float = 1.0

    def __post_init__(self) -> None:
        if self.temperature <= 0:
            raise ValueError("temperature must be positive")


def _softmax(values: List[float]) -> List[float]:
    shifted = [v - max(values) for v in values]
    exp_values = [math.exp(v) for v in shifted]
    denom = sum(exp_values)
    return [v / denom for v in exp_values]


class SimpleGate:
    """A lightweight linear gate that selects an expert id.

    The implementation intentionally avoids deep learning frameworks so the
    prototype remains dependency-light. Weight matrices are initialized with a
    deterministic random seed for reproducibility.
    """

    def __init__(self, config: GateConfig, seed: int = 7) -> None:
        self.config = config
        rng = random.Random(seed)
        self.weight = [
            [rng.gauss(0.0, 0.1) for _ in range(config.num_experts)]
            for _ in range(config.hidden_size)
        ]
        self.bias = [0.0 for _ in range(config.num_experts)]

    def logits(self, hidden_state: Sequence[float]) -> List[float]:
        if len(hidden_state) != self.config.hidden_size:
            raise ValueError("hidden state has wrong size")
        logits = []
        for j in range(self.config.num_experts):
            dot = sum(hidden_state[i] * self.weight[i][j] for i in range(self.config.hidden_size))
            logits.append(dot + self.bias[j])
        return logits

    def probs(self, hidden_state: Sequence[float]) -> List[float]:
        raw_logits = [v / self.config.temperature for v in self.logits(hidden_state)]
        return _softmax(raw_logits)

    def select(self, hidden_state: Sequence[float]) -> Dict[str, object]:
        probabilities = self.probs(hidden_state)
        expert_id = int(max(range(len(probabilities)), key=probabilities.__getitem__))
        entropy = -sum(p * math.log(p + 1e-12) for p in probabilities)
        return {
            "expert_id": expert_id,
            "probs": probabilities,
            "entropy": entropy,
        }

    def route(self, hidden_state: Sequence[float], expert_names: List[str]) -> str:
        selection = self.select(hidden_state)
        expert_index = selection["expert_id"]
        if expert_index >= len(expert_names):
            raise IndexError("gate selected expert outside provided names")
        return expert_names[expert_index]
