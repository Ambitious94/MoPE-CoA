"""Small script to prepare data and run supervised routing training (SFT) for MoPE gate.

Usage:
    python -m scripts.sft_route_train --input data.jsonl --hidden-size 8 --epochs 10 --lr 0.03 --out gate.json

This script is intentionally dependency-light and uses the project's preprocessing
and training utilities.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import List

from mope import (
    preprocess_record,
    build_route_batch_from_structured,
    PIPELINE_REGISTRY,
)
from mope.mope_layer import MoPELayer, MoPELayerConfig
from mope.training import train_gate_supervised


def load_records(path: Path) -> List[dict]:
    """Load JSONL or JSON array records.

    - If file ends with .jsonl/.ndjson: parse line-by-line
    - Else: try JSON parse of entire file; expect list or dict with key like 'data' or 'records'
    - Fallback: line-by-line best-effort
    """
    suffix = path.suffix.lower()
    if suffix in {".jsonl", ".ndjson"}:
        items: List[dict] = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                items.append(json.loads(line))
        return items

    text = path.read_text(encoding="utf-8")
    try:
        obj = json.loads(text)
        if isinstance(obj, list):
            return obj
        if isinstance(obj, dict):
            for key in ("data", "records", "items"):
                if key in obj and isinstance(obj[key], list):
                    return obj[key]  # type: ignore[return-value]
            # Single dict record
            return [obj]
    except json.JSONDecodeError:
        # Fallback to line-by-line
        items: List[dict] = []
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                items.append(json.loads(line))
            except Exception:
                # skip unparsable lines
                continue
        return items
    return []


def evaluate_routing_accuracy(layer: MoPELayer, batch) -> float:
    expert_names = list(layer.config.expert_names)
    correct = 0
    total = 0
    for h, label in zip(batch.hidden_states, batch.labels):
        probs = layer.gate.probs(list(h))
        pred_idx = int(max(range(len(probs)), key=probs.__getitem__))
        pred_name = expert_names[pred_idx]
        if pred_name == label:
            correct += 1
        total += 1
    return correct / max(total, 1)


def average_prediction_entropy(layer: MoPELayer, batch) -> float:
    import math
    total = 0.0
    count = 0
    for h in batch.hidden_states:
        probs = layer.gate.probs(list(h))
        ent = -sum(p * math.log(max(p, 1e-12)) for p in probs)
        total += ent
        count += 1
    return total / max(count, 1)


def save_gate(layer: MoPELayer, path: Path) -> None:
    gate = layer.gate
    out = {
        "weight": gate.weight,
        "bias": gate.bias,
        "config": {
            "hidden_size": gate.config.hidden_size,
            "num_experts": gate.config.num_experts,
            "temperature": gate.config.temperature,
        },
    }
    path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="sft_route_train")
    parser.add_argument("--input", type=str, required=True, help="Path to input JSON/JSONL file")
    parser.add_argument("--hidden-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=0.03)
    parser.add_argument("--out", type=str, default="gate.json", help="Where to save gate weights")
    parser.add_argument("--drop-observation", action="store_true", default=True, help="Drop observation content from learning (recommended)")
    parser.add_argument("--batch-size", type=int, default=32, help="Minibatch size for gate training")
    parser.add_argument("--no-shuffle", action="store_true", help="Disable per-epoch shuffling")
    args = parser.parse_args(argv)

    in_path = Path(args.input)
    if not in_path.exists():
        print("Input file not found:", in_path)
        return 2

    raw = load_records(in_path)
    print(f"Loaded {len(raw)} raw records")

    structured = [preprocess_record(r, hidden_size=args.hidden_size, with_instruction=False, drop_observation_content=args.drop_observation) for r in raw]
    # filter out records where no pipeline_label
    structured = [s for s in structured if (s.get("supervision") or {}).get("pipeline_label")]
    print(f"Structured samples with labels: {len(structured)}")

    batch = build_route_batch_from_structured(structured, hidden_size=args.hidden_size)
    print(f"Built RouteBatch size: {len(batch.prompts)}")
    if not batch.prompts:
        print("No training samples available after preprocessing. Exiting.")
        return 1

    expert_names = [
        "think-plan-websearch-observe-reflect-suggest-check-answer",
        "plan-websearch-crawl-observe-reflect-answer",
    ]
    layer = MoPELayer(MoPELayerConfig(hidden_size=args.hidden_size, expert_names=expert_names), pipelines=PIPELINE_REGISTRY)

    # Training loop (minibatch + optional shuffling)
    import random
    num_samples = len(batch.prompts)
    bs = max(1, min(args.batch_size, num_samples))
    for epoch in range(max(1, args.epochs)):
        indices = list(range(num_samples))
        if not args.no_shuffle:
            random.shuffle(indices)

        total_loss = 0.0
        total_count = 0

        for start in range(0, num_samples, bs):
            idx = indices[start:start + bs]
            sub_prompts = [batch.prompts[i] for i in idx]
            sub_hidden = [batch.hidden_states[i] for i in idx]
            sub_labels = [batch.labels[i] for i in idx]
            sub_batch = type(batch)(prompts=sub_prompts, hidden_states=sub_hidden, labels=sub_labels)

            batch_loss = train_gate_supervised(layer, sub_batch, lr=args.lr)
            total_loss += batch_loss * max(len(sub_prompts), 1)
            total_count += max(len(sub_prompts), 1)

        avg_loss = total_loss / max(total_count, 1)
        acc = evaluate_routing_accuracy(layer, batch)
        ent = average_prediction_entropy(layer, batch)
        print(f"epoch {epoch+1}/{args.epochs} loss={avg_loss:.4f} acc={acc:.4f} entropy={ent:.4f}")

    # Save gate weights
    out_path = Path(args.out)
    save_gate(layer, out_path)
    print("Saved gate to", out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
