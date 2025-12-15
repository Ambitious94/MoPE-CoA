"""Smoke test for TorchMoPE integration.

Runs a tiny forward pass replacing a dummy block's FFN with TorchMoPELayer,
optionally loading gate.json. Verifies shape consistency and prints summary.
"""
from __future__ import annotations

import argparse
import json
from types import SimpleNamespace

import torch

from mope.nanogpt_integration import replace_ffn_with_torch_mope


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="torch_mope_smoke")
    parser.add_argument("--hidden-size", type=int, default=8)
    parser.add_argument("--seq-len", type=int, default=4)
    parser.add_argument("--gate-json", type=str, default="gate.json")
    args = parser.parse_args(argv)

    # Dummy block with mlp attribute
    class DummyBlock:
        def __init__(self):
            self.mlp = torch.nn.Identity()

    block = DummyBlock()

    expert_names = [
        "think-plan-websearch-observe-reflect-suggest-check-answer",
        "plan-websearch-crawl-observe-reflect-answer",
    ]

    # Load gate.json if exists
    gate_json = None
    try:
        with open(args.gate_json, "r", encoding="utf-8") as f:
            gate_json = json.load(f)
            print("Loaded gate.json")
    except Exception:
        print("gate.json not found or failed to load; continuing with random init")

    mope_layer = replace_ffn_with_torch_mope(block, hidden_size=args.hidden_size, expert_names=expert_names, gate_json=gate_json)

    # Build dummy input and prompt
    x = torch.randn(1, args.seq_len, args.hidden_size)
    prompt = "Test prompt for TorchMoPE smoke"

    # Forward through replaced mlp
    y = block.mlp(x, prompt=prompt)
    assert y.shape == x.shape, "output shape mismatch"

    print("Smoke OK:", {
        "hidden_size": args.hidden_size,
        "seq_len": args.seq_len,
        "output_shape": tuple(y.shape),
    })
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
