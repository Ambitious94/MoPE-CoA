"""Mixture of Pipeline Experts (MoPE) prototype package.

This package contains lightweight, dependency-minimal components for experimenting
with a Transformer layer where the FFN is replaced by a mixture of executable
pipelines. The focus is on clean interfaces and inspectable traces rather than
state-of-the-art performance.
"""

from .gate import GateConfig, SimpleGate
from .pipeline import PipelineExpert, PipelineOutput, PIPELINE_REGISTRY, register_task_experts
from .vectorizer import HashVectorizer
from .mope_layer import MoPELayer, MoPELayerConfig
from .model import MoPETransformer
from .retrieval import DocumentStore, EvidenceReader, build_retrieval_pipelines
from .task_engine import SearchQAFactCheckingSystem, build_task_pipelines
from .nanogpt_integration import attach_mope_to_nanogpt, make_mock_nanogpt
from .data_prep import (
    parse_coa_output,
    map_steps_to_pipeline_name,
    preprocess_record,
    build_route_batch_from_structured,
)

__all__ = [
    "GateConfig",
    "SimpleGate",
    "PipelineExpert",
    "PipelineOutput",
    "PIPELINE_REGISTRY",
    "register_task_experts",
    "HashVectorizer",
    "MoPELayer",
    "MoPELayerConfig",
    "MoPETransformer",
    "DocumentStore",
    "EvidenceReader",
    "build_retrieval_pipelines",
    "SearchQAFactCheckingSystem",
    "build_task_pipelines",
    "attach_mope_to_nanogpt",
    "make_mock_nanogpt",
    "parse_coa_output",
    "map_steps_to_pipeline_name",
    "preprocess_record",
    "build_route_batch_from_structured",
]
