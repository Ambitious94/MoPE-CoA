import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from mope import DocumentStore, MoPELayer, MoPELayerConfig, build_retrieval_pipelines


def test_document_store_retrieves_relevant_snippet():
    store = DocumentStore()
    store.add("doc1", "Water boils at 100 degrees Celsius at sea level.")
    store.add("doc2", "The freezing point of water is 0 degrees Celsius.")

    hits = store.search("water boils", top_k=1)
    assert hits[0]["doc_id"] == "doc1"
    assert "Water boils" in hits[0]["snippet"]


def test_retrieval_pipeline_executes_in_layer():
    store = DocumentStore()
    store.add_many(
        [
            ("boiling", "Water boils at 100C."),
            ("coffee", "Coffee brewing can be done at lower temperatures."),
        ]
    )

    retrieval_pipelines = build_retrieval_pipelines(store)
    expert_names = list(retrieval_pipelines.keys())
    config = MoPELayerConfig(hidden_size=8, expert_names=expert_names)
    layer = MoPELayer(config, pipelines=retrieval_pipelines)

    hidden_state = [0.2 for _ in range(config.hidden_size)]
    output = layer.forward(hidden_state, prompt="When does water boil?")

    assert output["pipeline"] in expert_names
    assert any("boil" in step.lower() for step in output["trace"])
