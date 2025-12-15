import sys
from pathlib import Path

# Allow tests to import the local package without installation
sys.path.append(str(Path(__file__).resolve().parents[1]))

from mope import (
    HashVectorizer,
    MoPELayer,
    MoPELayerConfig,
    MoPETransformer,
    PIPELINE_REGISTRY,
)
from mope.model import TransformerConfig


def test_gate_routes_to_valid_pipeline():
    hidden_size = 8
    expert_names = list(PIPELINE_REGISTRY.keys())
    config = MoPELayerConfig(hidden_size=hidden_size, expert_names=expert_names, temperature=0.7)
    layer = MoPELayer(config)
    hidden_state = [1.0 for _ in range(hidden_size)]
    output = layer.forward(hidden_state, prompt="who discovered penicillin")
    assert output["pipeline"] in expert_names
    assert len(output["hidden_state"]) == hidden_size


def test_vectorizer_residual():
    vec = HashVectorizer(dim=4)
    hidden = [0.5, 0.5, 0.5, 0.5]
    update = [0.1, 0.0, 0.0, 0.0]
    result = vec.residual_add(hidden, update)
    assert result == [0.6, 0.5, 0.5, 0.5]


def test_transformer_traces_all_layers():
    model = MoPETransformer(config=TransformerConfig(hidden_size=6, num_layers=2, vocab_size=10))
    output = model.forward("verify if water boils at 100c")
    assert len(output["layer_traces"]) == 2
    for trace in output["layer_traces"]:
        assert "pipeline" in trace and "trace" in trace
        assert len(trace["trace"]) > 0
