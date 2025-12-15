"""MoPE layer that swaps in pipeline experts for a Transformer FFN."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence

from .gate import GateConfig, SimpleGate
from .pipeline import PIPELINE_REGISTRY, PipelineExpert, PipelineOutput
from .vectorizer import HashVectorizer


@dataclass
class MoPELayerConfig:
    hidden_size: int
    expert_names: Sequence[str]
    temperature: float = 1.0

    def gate_config(self) -> GateConfig:
        return GateConfig(
            hidden_size=self.hidden_size,
            num_experts=len(self.expert_names),
            temperature=self.temperature,
        )


class MoPELayer:
    """A minimal MoPE layer intended to replace a Transformer FFN block."""

    def __init__(
        self,
        config: MoPELayerConfig,
        pipelines: Dict[str, PipelineExpert] | None = None,
        vectorizer: HashVectorizer | None = None,
    ) -> None:
        self.config = config
        self.pipelines = pipelines or PIPELINE_REGISTRY
        self.vectorizer = vectorizer or HashVectorizer(dim=config.hidden_size)
        self.gate = SimpleGate(config.gate_config())

    def forward(self, hidden_state: Sequence[float], prompt: str) -> Dict[str, object]:
        hidden = list(hidden_state)
        route_name = self.gate.route(hidden, list(self.config.expert_names))
        pipeline = self.pipelines[route_name]
        pipeline_output: PipelineOutput = pipeline.run(prompt)
        vector_update = self.vectorizer.encode(pipeline_output.answer)
        updated = self.vectorizer.residual_add(hidden, vector_update)
        return {
            "hidden_state": updated,
            "pipeline": pipeline.name,
            "trace": pipeline_output.trace,
            "probs": self.gate.probs(hidden),
        }

    def batch_forward(self, hidden_states: Iterable[Sequence[float]], prompts: Iterable[str]):
        return [self.forward(h, p) for h, p in zip(hidden_states, prompts)]
