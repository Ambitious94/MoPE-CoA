"""SFT for MoPE gate integrated with nanoGPT hidden states (PyTorch).

This script:
- Loads a nanoGPT `GPT` model (from the adjacent nanoGPT repo or installed module)
- Replaces one Block's FFN with TorchMoPE (gate is trainable)
- Builds a small supervised routing dataset from a JSON/JSONL file
- Runs a training loop that collects the pre-FFN hidden vectors (ln_2 output)
  and optimizes the MoPE gate with cross-entropy on expert labels

Notes:
- We DO NOT train the GPT; only the MoPE gate parameters are updated.
- External tools are disabled during training; TorchMoPE uses hash vectorizer.
- Tokenization is byte-level fallback (no extra deps). For better results you
  can wire in a proper tokenizer.

Example (Windows PowerShell):

  python -m scripts.nanogpt_mope_sft ^
    --input data/WebAgentSFTDataset.json ^
    --epochs 20 --batch-size 64 --lr 0.05 ^
    --layer-idx 0 --hidden-size 512 ^
    --out gate.json

Optionally init from prior gate:

  python -m scripts.nanogpt_mope_sft --input data.jsonl --gate-json gate.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import torch
import torch.nn as nn

# Local imports from MoPE
from mope.pipeline import PIPELINE_REGISTRY
from mope.data_prep import preprocess_record, build_route_batch_from_structured
from mope.nanogpt_integration import replace_ffn_with_torch_mope


def _maybe_add_repo_to_syspath(nanogpt_root: Path) -> None:
    # Ensure we can import nanoGPT's GPT implementation
    if str(nanogpt_root) not in sys.path:
        sys.path.insert(0, str(nanogpt_root))


def _load_records(path: Path) -> List[dict]:
    # Lightweight JSON/JSONL loader
    items: List[dict] = []
    if not path.exists():
        return items
    suffix = path.suffix.lower()
    if suffix in {".jsonl", ".ndjson"}:
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    items.append(json.loads(line))
                except Exception:
                    continue
        return items
    # try JSON array / dict
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(obj, list):
            return obj
        if isinstance(obj, dict):
            for k in ("data", "records", "items"):
                if k in obj and isinstance(obj[k], list):
                    return obj[k]  # type: ignore[return-value]
            return [obj]
    except Exception:
        pass
    return items


def _byte_encode(texts: List[str], block_size: int, vocab_size: int) -> torch.Tensor:
    # Simple byte-level encoding into [B, T] with truncation/padding
    # vocab_size should be >= 256 for raw bytes; if smaller, mod it
    B = len(texts)
    T = block_size
    x = torch.zeros(B, T, dtype=torch.long)
    for i, s in enumerate(texts):
        bs = s.encode("utf-8", errors="ignore")
        ids = [(b % vocab_size) for b in bs[:T]]
        if len(ids) > 0:
            x[i, : len(ids)] = torch.tensor(ids, dtype=torch.long)
    return x


def _build_dataset(input_path: Path, hidden_size: int) -> Tuple[List[str], List[int], List[str]]:
    raw = _load_records(input_path)
    structured = [
        preprocess_record(r, hidden_size=hidden_size, with_instruction=False, drop_observation_content=True)
        for r in raw
    ]
    structured = [s for s in structured if (s.get("supervision") or {}).get("pipeline_label")]
    batch = build_route_batch_from_structured(structured, hidden_size=hidden_size)
    # map labels to expert names from registry order to keep consistent
    expert_names = list(PIPELINE_REGISTRY.keys())
    name_to_idx = {n: i for i, n in enumerate(expert_names)}
    label_idx: List[int] = [name_to_idx.get(lbl, -1) for lbl in batch.labels]
    # filter out unknown labels (shouldn't happen if registry stable)
    keep: List[int] = [i for i, y in enumerate(label_idx) if y >= 0]
    prompts = [batch.prompts[i] for i in keep]
    labels = [label_idx[i] for i in keep]
    return prompts, labels, expert_names


def main(argv: Iterable[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="nanogpt_mope_sft")
    p.add_argument("--input", type=str, required=True, help="Path to JSON/JSONL with CoA-style outputs")
    p.add_argument("--nanogpt-root", type=str, default=None, help="Path to nanoGPT repo root (where model.py lives)")
    p.add_argument("--layer-idx", type=int, default=0, help="Which Block index to replace and extract features from")
    p.add_argument("--hidden-size", type=int, default=512, help="Model hidden size; must match GPT n_embd")
    p.add_argument("--vocab-size", type=int, default=256, help="Tokenizer vocab size (byte-level fallback)")
    p.add_argument("--block-size", type=int, default=256, help="Context length")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=0.05)
    p.add_argument("--gate-json", type=str, default=None, help="Optional gate.json to init weights")
    p.add_argument("--out", type=str, default="gate.json", help="Where to save trained gate")
    args = p.parse_args(list(argv) if argv is not None else None)

    input_path = Path(args.input)
    if not input_path.exists():
        print("Input file not found:", input_path)
        return 2

    # Make sure nanoGPT can be imported
    if args.nanogpt_root:
        _maybe_add_repo_to_syspath(Path(args.nanogpt_root))

    try:
        from model import GPT, GPTConfig  # nanoGPT's model.py
    except Exception as e:
        print("Failed to import nanoGPT model; ensure --nanogpt-root points to nanoGPT repo")
        print("Error:", e)
        return 3

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Build a small GPT from scratch (no pretraining) for feature extraction
    cfg = GPTConfig(
        block_size=int(args.block_size),
        vocab_size=int(args.vocab_size),
        n_layer=max(args.layer_idx + 1, 1),
        n_head=8,
        n_embd=int(args.hidden_size),
        dropout=0.0,
        bias=True,
    )
    gpt = GPT(cfg).to(device)
    # freeze GPT parameters (we only train the MoPE gate)
    for p_ in gpt.parameters():
        p_.requires_grad_(False)

    # Build dataset (prompts + labels mapped to expert indices)
    prompts, labels, expert_names = _build_dataset(input_path, hidden_size=args.hidden_size)
    if len(prompts) == 0:
        print("No labeled samples found after preprocessing. Exiting.")
        return 1

    # Replace selected Block's FFN with TorchMoPE
    block = gpt.transformer.h[args.layer_idx]
    gate_init = None
    if args.gate_json and Path(args.gate_json).exists():
        try:
            gate_init = json.loads(Path(args.gate_json).read_text(encoding="utf-8"))
            print("Loaded gate init from", args.gate_json)
        except Exception as e:
            print("Failed to load gate json:", e)
    mope_layer = replace_ffn_with_torch_mope(block, hidden_size=args.hidden_size, expert_names=expert_names, gate_json=gate_init)
    mope_layer = mope_layer.to(device)

    # Optimizer over MoPE gate only
    opt = torch.optim.SGD(mope_layer.parameters(), lr=float(args.lr))

    # Register a forward hook on ln_2 to capture pre-FFN features
    captured: List[torch.Tensor] = []

    def hook_ln2(_module, _inp, out):
        # out is ln_2(x): shape [B, T, H]
        captured.append(out.detach())

    hook_handle = block.ln_2.register_forward_hook(hook_ln2)

    def run_epoch(texts: List[str], y: List[int], batch_size: int) -> Tuple[float, float]:
        gpt.eval()
        mope_layer.train()
        total_loss = 0.0
        total_count = 0
        correct = 0
        with torch.enable_grad():
            for start in range(0, len(texts), batch_size):
                end = min(start + batch_size, len(texts))
                batch_txt = texts[start:end]
                batch_y = torch.tensor(y[start:end], dtype=torch.long, device=device)

                # Build token tensor
                idx = _byte_encode(batch_txt, block_size=args.block_size, vocab_size=args.vocab_size).to(device)

                # Forward through GPT to collect ln_2 output
                captured.clear()
                _ = gpt(idx)  # logits, loss (ignored)
                if not captured:
                    raise RuntimeError("ln_2 hook did not capture features; check layer index")
                feats = captured[-1]  # [B, T, H]
                X = feats[:, -1, :]  # last position representation [B, H]

                opt.zero_grad()
                loss = mope_layer.compute_gate_ce(X, batch_y)
                loss.backward()
                opt.step()

                total_loss += float(loss.item()) * X.size(0)
                total_count += X.size(0)

                with torch.no_grad():
                    logits = X @ mope_layer.gate.weight + mope_layer.gate.bias
                    pred = torch.argmax(logits, dim=-1)
                    correct += int((pred == batch_y).sum().item())

        avg_loss = total_loss / max(total_count, 1)
        acc = correct / max(total_count, 1)
        return avg_loss, acc

    for epoch in range(int(args.epochs)):
        loss, acc = run_epoch(prompts, labels, int(args.batch_size))
        print(f"epoch {epoch+1}/{args.epochs} gate_ce={loss:.4f} acc={acc:.4f}")

    # Save gate weights to json
    out = {
        "weight": mope_layer.gate.weight.detach().cpu().tolist(),
        "bias": mope_layer.gate.bias.detach().cpu().tolist(),
        "config": {
            "hidden_size": int(args.hidden_size),
            "num_experts": len(expert_names),
            "temperature": 1.0,
        },
        "experts": expert_names,
    }
    out_path = Path(args.out)
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print("Saved gate to", out_path)

    # cleanup
    hook_handle.remove()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
