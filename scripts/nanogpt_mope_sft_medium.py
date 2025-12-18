"""Medium-scale MoPE SFT with nanoGPT (GPT-2 weights + tokenizer).

Features:
- Loads nanoGPT's GPT with GPT-2 pretrained weights (requires transformers)
- Uses GPT-2 tokenizer (pad to block_size; pad token = eos)
- Replaces one Block's FFN with TorchMoPE; trains MoPE gate
- Optional partial unfreezing of last N GPT blocks
- Joint loss: LM loss (from GPT) + routing CE loss (from MoPE gate)
- AMP (mixed precision) and gradient accumulation
- Cosine LR with warmup

Example (Windows PowerShell):

  python -m scripts.nanogpt_mope_sft_medium `
    --input "data/WebAgentSFTDataset.json" `
    --nanogpt-root "e:/Edge Download/nanoGPT-master" `
    --model-type gpt2 `
    --layer-idx 0 `
    --unfreeze-last 1 `
    --epochs 3 `
    --batch-size 8 `
    --grad-accum-steps 4 `
    --lr 5e-5 `
    --weight-decay 0.01 `
    --warmup-steps 100 `
    --amp `
    --out gate.json
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from mope.data_prep import preprocess_record, build_route_batch_from_structured
from mope.pipeline import PIPELINE_REGISTRY
from mope.nanogpt_integration import replace_ffn_with_torch_mope


def _maybe_add_repo_to_syspath(root: Path) -> None:
    p = str(root)
    if p not in sys.path:
        sys.path.insert(0, p)


def _load_records(path: Path) -> List[dict]:
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


@dataclass
class Sample:
    text: str
    label_idx: int


class RouteDataset(Dataset[Sample]):
    def __init__(self, samples: List[Sample]) -> None:
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Sample:
        return self.samples[idx]


def _build_dataset(input_path: Path, hidden_size: int) -> Tuple[List[Sample], List[str]]:
    raw = _load_records(input_path)
    structured = [
        preprocess_record(r, hidden_size=hidden_size, with_instruction=False, drop_observation_content=True)
        for r in raw
    ]
    structured = [s for s in structured if (s.get("supervision") or {}).get("pipeline_label")]
    batch = build_route_batch_from_structured(structured, hidden_size=hidden_size)
    expert_names = list(PIPELINE_REGISTRY.keys())
    name_to_idx = {n: i for i, n in enumerate(expert_names)}
    samples: List[Sample] = []
    for prompt, lbl in zip(batch.prompts, batch.labels):
        if lbl not in name_to_idx:
            continue
        samples.append(Sample(text=prompt, label_idx=name_to_idx[lbl]))
    return samples, expert_names


def main(argv: Iterable[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="nanogpt_mope_sft_medium")
    ap.add_argument("--input", type=str, required=True)
    ap.add_argument("--nanogpt-root", type=str, required=True)
    ap.add_argument("--model-type", type=str, default="gpt2", choices=["gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"])
    ap.add_argument("--layer-idx", type=int, default=0)
    ap.add_argument("--hidden-size", type=int, default=None, help="Override n_embd; default from model-type")
    ap.add_argument("--block-size", type=int, default=512)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--grad-accum-steps", type=int, default=1)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--warmup-steps", type=int, default=100)
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--unfreeze-last", type=int, default=0, help="Unfreeze last N GPT blocks (0 keeps GPT frozen)")
    ap.add_argument("--lm-weight", type=float, default=1.0)
    ap.add_argument("--route-weight", type=float, default=1.0)
    ap.add_argument("--gate-json", type=str, default=None)
    ap.add_argument("--use-adapters", action="store_true", help="Use trainable expert adapters inside MoPE (default)")
    ap.add_argument("--no-adapters", dest="use_adapters", action="store_false", help="Disable adapters; use vectorizer-only residuals")
    ap.set_defaults(use_adapters=True)
    ap.add_argument("--out", type=str, default="gate.json")
    args = ap.parse_args(list(argv) if argv is not None else None)

    input_path = Path(args.input)
    if not input_path.exists():
        print("Input not found:", input_path)
        return 2

    # Import nanoGPT and transformers
    _maybe_add_repo_to_syspath(Path(args.nanogpt_root))
    try:
        from model import GPT  # type: ignore
    except Exception as e:
        print("Failed to import nanoGPT model.py from --nanogpt-root:", e)
        return 3
    try:
        from transformers import GPT2TokenizerFast
    except Exception as e:
        print("transformers is required: pip install transformers")
        print("Error:", e)
        return 4

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load pretrained GPT-2 weights into nanoGPT GPT
    gpt = GPT.from_pretrained(args.model_type)
    if args.hidden_size:
        if getattr(gpt.config, "n_embd", None) != int(args.hidden_size):
            print("Warning: --hidden-size differs from pretrained; using pretrained n_embd=", gpt.config.n_embd)
    # Adjust block size if needed
    if args.block_size < gpt.config.block_size:
        gpt.crop_block_size(args.block_size)
    gpt.to(device)

    # Freeze by default
    for p in gpt.parameters():
        p.requires_grad_(False)
    # Optionally unfreeze last N blocks
    if args.unfreeze_last > 0:
        n = len(gpt.transformer.h)
        start = max(0, n - int(args.unfreeze_last))
        for i in range(start, n):
            for p in gpt.transformer.h[i].parameters():
                p.requires_grad_(True)
        # keep embeddings and ln_f/lm_head frozen by default

    # Build dataset
    hidden_size = int(getattr(gpt.config, "n_embd", 768))
    samples, expert_names = _build_dataset(input_path, hidden_size=hidden_size)
    if len(samples) == 0:
        print("No labeled samples found. Exiting.")
        return 1

    # Tokenizer
    tok = GPT2TokenizerFast.from_pretrained("gpt2")
    tok.pad_token = tok.eos_token

    def collate(batch: List[Sample]) -> Dict[str, torch.Tensor]:
        texts = [s.text for s in batch]
        enc = tok(
            texts,
            padding="max_length",
            truncation=True,
            max_length=gpt.config.block_size,
            return_tensors="pt",
        )
        input_ids = enc["input_ids"]  # [B,T]
        attn = enc["attention_mask"]  # [B,T]
        # targets: ignore pads by setting to -1
        targets = input_ids.clone()
        targets[attn == 0] = -1
        labels = torch.tensor([s.label_idx for s in batch], dtype=torch.long)
        return {"input_ids": input_ids, "attention_mask": attn, "targets": targets, "route_labels": labels}

    # Build DataLoader
    random.shuffle(samples)
    train_loader = DataLoader(RouteDataset(samples), batch_size=int(args.batch_size), shuffle=True, collate_fn=collate)

    # Replace FFN with TorchMoPE on selected layer
    block = gpt.transformer.h[int(args.layer_idx)]
    gate_init = None
    if args.gate_json and Path(args.gate_json).exists():
        try:
            gate_init = json.loads(Path(args.gate_json).read_text(encoding="utf-8"))
            print("Loaded gate init from", args.gate_json)
        except Exception as e:
            print("Failed to load gate json:", e)
    mope_layer = replace_ffn_with_torch_mope(
        block,
        hidden_size=hidden_size,
        expert_names=expert_names,
        gate_json=gate_init,
        use_adapters=bool(args.use_adapters),
    )
    mope_layer.to(device)

    # Optimizer: MoPE + optionally unfrozen GPT parameters (deduplicated)
    params: List[torch.nn.Parameter] = []
    seen: set[int] = set()
    def add_param(p: torch.nn.Parameter) -> None:
        if p.requires_grad and id(p) not in seen:
            params.append(p)
            seen.add(id(p))

    for p in mope_layer.parameters():
        add_param(p)
    if args.unfreeze_last > 0:
        for p in gpt.parameters():
            add_param(p)
    optimizer = torch.optim.AdamW(params, lr=float(args.lr), weight_decay=float(args.weight_decay))

    # Scheduler: warmup + cosine
    total_steps = max(1, math.ceil(len(train_loader) / max(1, int(args.grad_accum_steps))) * int(args.epochs))

    def lr_lambda(step: int) -> float:
        if step < int(args.warmup_steps):
            return float(step + 1) / float(max(1, int(args.warmup_steps)))
        progress = (step - int(args.warmup_steps)) / max(1, total_steps - int(args.warmup_steps))
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # Hook to capture pre-FFN features (ln_2 output)
    captured: List[torch.Tensor] = []

    def hook_ln2(_module, _inp, out):
        captured.append(out)

    h = block.ln_2.register_forward_hook(hook_ln2)

    # AMP: use torch.amp API
    scaler = torch.amp.GradScaler(
        device_type=("cuda" if device.type == "cuda" else "cpu"),
        enabled=bool(args.amp) and device.type == "cuda",
    )

    gpt.train()  # we'll compute LM loss; some params may be frozen
    for epoch in range(int(args.epochs)):
        total_lm = 0.0
        total_route = 0.0
        total_n = 0
        step_idx = 0
        optimizer.zero_grad(set_to_none=True)

        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            attn = batch["attention_mask"].to(device)
            targets = batch["targets"].to(device)
            y_route = batch["route_labels"].to(device)

            with torch.amp.autocast(device_type=device.type, enabled=bool(args.amp) and device.type == "cuda"):
                captured.clear()
                logits, lm_loss = gpt(input_ids, targets=targets)
                if not captured:
                    raise RuntimeError("ln_2 hook did not capture features; check layer index")
                feats = captured[-1]  # [B,T,H]
                # pick last non-pad position per sample
                lengths = attn.sum(dim=1)  # [B]
                idx_last = (lengths - 1).clamp(min=0).unsqueeze(1).unsqueeze(2).expand(-1, 1, feats.size(-1))
                X = feats.gather(dim=1, index=idx_last).squeeze(1)  # [B,H]
                route_loss = mope_layer.compute_gate_ce(X, y_route)
                loss = float(args.lm_weight) * lm_loss + float(args.route_weight) * route_loss

            scaler.scale(loss).backward()

            total_lm += float(lm_loss.item()) * input_ids.size(0)
            total_route += float(route_loss.item()) * input_ids.size(0)
            total_n += int(input_ids.size(0))

            if (step_idx + 1) % int(args.grad_accum_steps) == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
            step_idx += 1

        avg_lm = total_lm / max(1, total_n)
        avg_route = total_route / max(1, total_n)
        print(f"epoch {epoch+1}/{args.epochs} lm_loss={avg_lm:.4f} route_ce={avg_route:.4f} lr={scheduler.get_last_lr()[0]:.2e}")

    # Save trained gate
    out = {
        "weight": mope_layer.gate.weight.detach().cpu().tolist(),
        "bias": mope_layer.gate.bias.detach().cpu().tolist(),
        "config": {
            "hidden_size": hidden_size,
            "num_experts": len(expert_names),
            "temperature": 1.0,
        },
        "experts": expert_names,
        "meta": {
            "model_type": args.model_type,
            "layer_idx": int(args.layer_idx),
        },
    }
    Path(args.out).write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print("Saved gate to", args.out)

    h.remove()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
