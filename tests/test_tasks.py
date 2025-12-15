import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from mope import (
    DocumentStore,
    MoPELayer,
    MoPELayerConfig,
    SearchQAFactCheckingSystem,
    attach_mope_to_nanogpt,
    build_task_pipelines,
    make_mock_nanogpt,
)


def test_task_engine_answers_with_evidence():
    store = DocumentStore()
    store.add("doc1", "Water boils at 100 degrees Celsius at sea level.")
    store.add("doc2", "Altitude lowers the boiling point of water.")

    system = SearchQAFactCheckingSystem(store)
    result = system.answer("When does water boil?", top_k=2)

    assert "boil" in result["answer"].lower()
    assert result["verdict"]["status"] in {"supported", "no_evidence"}
    assert result["trace"]


def test_task_pipeline_executes_and_routes():
    store = DocumentStore()
    store.add("boiling", "Water boils at 100 degrees Celsius.")
    system = SearchQAFactCheckingSystem(store)
    pipelines = build_task_pipelines(system)

    config = MoPELayerConfig(hidden_size=6, expert_names=list(pipelines.keys()))
    layer = MoPELayer(config, pipelines=pipelines)

    output = layer.forward([0.1 for _ in range(config.hidden_size)], prompt="When does water boil?")

    assert output["pipeline"] in pipelines
    assert any("retrieval" in step.lower() or "verdict" in step.lower() for step in output["trace"])


def test_nanogpt_adapter_replaces_mlp():
    mock_model, hidden_size = make_mock_nanogpt(num_layers=1, hidden_size=5)
    store = DocumentStore()
    store.add("boiling", "Water boils at 100 C.")
    system = SearchQAFactCheckingSystem(store)
    pipelines = build_task_pipelines(system)

    attach_mope_to_nanogpt(
        mock_model,
        hidden_size=hidden_size,
        layer_indices=[0],
        pipelines=pipelines,
        prompt_provider=lambda: "When does water boil?",
    )

    updated = mock_model.transformer.h[0].mlp([0.0] * hidden_size)
    assert isinstance(updated, list)
    assert len(updated) == hidden_size
