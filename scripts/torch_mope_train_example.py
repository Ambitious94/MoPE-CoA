"""Minimal example: train TorchMoPE gate with supervised routing loss.

This script builds a small synthetic batch of hidden states + labels and
optimizes the gate parameters end-to-end with cross-entropy.
"""
from __future__ import annotations

import argparse
import json
from typing import List

import torch

from mope.torch_mope import TorchMoPELayer


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="torch_mope_train_example")
    parser.add_argument("--hidden-size", type=int, default=8)
    parser.add_argument("--num-experts", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--gate-json", type=str, default="gate.json")
    args = parser.parse_args(argv)

    expert_names: List[str] = [
        "think-plan-websearch-observe-reflect-suggest-check-answer",
        "plan-websearch-crawl-observe-reflect-answer",
    ][: args.num_experts]

    layer = TorchMoPELayer(hidden_size=args.hidden_size, expert_names=expert_names)
    # Optional: load pre-trained gate
    try:
        with open(args.gate_json, "r", encoding="utf-8") as f:
            layer.load_gate_json(json.load(f))
            print("Loaded gate.json for init")
    except Exception:
        print("gate.json not loaded; using random init")

    # Build a tiny synthetic dataset
    N = 128
    H = args.hidden_size
    X = torch.randn(N, H)
    # Make first half label 0, second half label 1
    y = torch.cat([torch.zeros(N // 2, dtype=torch.long), torch.ones(N - N // 2, dtype=torch.long)])
    prompts = ["sample %d" % i for i in range(N)]

    opt = torch.optim.SGD(layer.parameters(), lr=args.lr)

    for epoch in range(args.epochs):
        opt.zero_grad()
        loss_gate = layer.compute_gate_ce(X, y)
        loss_gate.backward()
        opt.step()

        with torch.no_grad():
            logits = X @ layer.gate.weight + layer.gate.bias
            preds = torch.argmax(logits, dim=-1)
            acc = (preds == y).float().mean().item()
        print(f"epoch {epoch+1}/{args.epochs} gate_ce={loss_gate.item():.4f} acc={acc:.4f}")

    print("Training complete. Gate params updated.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
