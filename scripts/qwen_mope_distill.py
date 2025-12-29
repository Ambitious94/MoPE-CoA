"""Distill teacher Qwen to student Qwen+MoPE by KL over logits (optional MSE over hidden).

Usage (Windows PowerShell):

  python -m scripts.qwen_mope_distill `
    --teacher Qwen/Qwen2.5-3B-Instruct `
    --student Qwen/Qwen2.5-3B-Instruct `
    --layers 30,31 `
    --alpha 0.01 `
    --max-steps 200 `
    --lr 1e-4

Dataset: Simple text lines; for a quick start, you can pass --text "path/to/text.txt".
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Iterable, List

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from mope.hf_integration import attach_mope_to_hf_model
from mope.pipeline import PIPELINE_REGISTRY


def _read_lines(p: Path, limit: int | None = None) -> List[str]:
    if not p.exists():
        return []
    lines = [ln.strip() for ln in p.read_text(encoding="utf-8", errors="ignore").splitlines() if ln.strip()]
    return lines[: limit] if limit else lines


def main(argv: Iterable[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="qwen_mope_distill")
    ap.add_argument("--teacher", type=str, default="Qwen/Qwen2.5-3B-Instruct")
    ap.add_argument("--student", type=str, default="Qwen/Qwen2.5-3B-Instruct")
    ap.add_argument("--layers", type=str, default="30,31")
    ap.add_argument("--alpha", type=float, default=0.01)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--max-steps", type=int, default=200)
    ap.add_argument("--text", type=str, default=None, help="Path to plain text file (one prompt per line)")
    ap.add_argument("--dtype", type=str, default="float16", choices=["float16", "bfloat16", "float32"]) 
    ap.add_argument("--device-map", type=str, default="auto")
    ap.add_argument("--gate-json", type=str, default=None)
    ap.add_argument("--train-adapters", action="store_true", help="Also train expert adapters; default only trains gate")
    args = ap.parse_args(list(argv) if argv is not None else None)

    dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[args.dtype]
    print(f"Loading teacher={args.teacher} student={args.student} dtype={args.dtype}")
    tok = AutoTokenizer.from_pretrained(args.teacher)
    teacher = AutoModelForCausalLM.from_pretrained(args.teacher, torch_dtype=dtype, device_map=args.device_map)
    student = AutoModelForCausalLM.from_pretrained(args.student, torch_dtype=dtype, device_map=args.device_map)

    hidden_size = int(getattr(getattr(student, "config", object()), "hidden_size", 0) or 0)
    if hidden_size <= 0:
        print("Could not determine hidden_size from student.config.hidden_size")
        return 2
    layer_indices = [int(x.strip()) for x in str(args.layers).split(",") if x.strip()]
    gate_init = None
    if args.gate_json and os.path.exists(args.gate_json):
        try:
            gate_init = json.loads(open(args.gate_json, "r", encoding="utf-8").read())
            print("Loaded gate init from", args.gate_json)
        except Exception as e:
            print("Failed to load gate json:", e)

    experts = list(PIPELINE_REGISTRY.keys())
    mope_layers = attach_mope_to_hf_model(student, layer_indices=layer_indices, hidden_size=hidden_size, expert_names=experts, gate_json=gate_init, use_adapters=bool(args.train_adapters), alpha=float(args.alpha))
    # Freeze base student except MoPE params
    for p in student.parameters():
        p.requires_grad_(False)
    train_params = []
    for ml in mope_layers:
        train_params.append(ml.gate.weight)
        train_params.append(ml.gate.bias)
        if bool(args.train_adapters):
            train_params += list(ml.expert_adapters.parameters())
    opt = torch.optim.Adam(train_params, lr=float(args.lr))

    # Simple dataset
    prompts = _read_lines(Path(str(args.text))) if args.text else [
        "Tell me a short joke.",
        "Summarize the benefits of exercise.",
        "What is the capital of France?",
        "Explain quantum computing briefly.",
    ]
    if not prompts:
        print("No prompts; provide --text")
        return 3

    def _encode_with_answer_prefix(text: str, max_len: int) -> torch.Tensor:
        ans = "Answer:"
        toks_ans = tok(ans, padding=False, truncation=False, return_tensors="pt")["input_ids"]
        ans_len = int(toks_ans.size(1))
        keep_len = max(1, max_len - ans_len)
        toks_text = tok(text, padding=False, truncation=True, max_length=keep_len, return_tensors="pt")["input_ids"]
        ids = torch.cat([toks_text, toks_ans], dim=1)
        if ids.size(1) > max_len:
            ids = ids[:, -max_len:]
        return ids.to(student.device)

    student.train()
    teacher.eval()
    steps = 0
    max_steps = int(args.max_steps)
    while steps < max_steps:
        text = prompts[steps % len(prompts)]
        input_ids = _encode_with_answer_prefix(text, int(student.config.max_position_embeddings or 2048))
        with torch.no_grad():
            tl = teacher(input_ids).logits[:, -1, :]
            tprob = F.softmax(tl, dim=-1)

        sl = student(input_ids).logits[:, -1, :]
        slogprob = F.log_softmax(sl, dim=-1)
        loss = F.kl_div(slogprob, tprob, reduction="batchmean")
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        steps += 1
        if steps % 10 == 0:
            print({"step": steps, "loss": float(loss.detach().cpu().item())})

    print("Finished distillation", {"steps": steps})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
