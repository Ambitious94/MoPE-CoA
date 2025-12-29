"""Smoke test: attach MoPE to Qwen2.5-3B-Instruct via HF and generate text.

Usage (Windows PowerShell):

  python -m scripts.qwen_mope_smoke `
    --model Qwen/Qwen2.5-3B-Instruct `
    --layers 0 `
    --max-new-tokens 32

Note: 3B model requires significant VRAM. Consider using torch_dtype=float16
and device_map="auto". For CPU-only, inference will be very slow.
"""
from __future__ import annotations

import argparse
import json
import os
from typing import Iterable

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from mope.hf_integration import attach_mope_to_hf_model
from mope.pipeline import PIPELINE_REGISTRY


def main(argv: Iterable[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="qwen_mope_smoke")
    ap.add_argument("--model", type=str, default="Qwen/Qwen2.5-3B-Instruct")
    ap.add_argument("--layers", type=str, default="0", help="Comma-separated layer indices to attach MoPE")
    ap.add_argument("--max-new-tokens", type=int, default=32)
    ap.add_argument("--use-adapters", action="store_true")
    ap.add_argument("--no-adapters", dest="use_adapters", action="store_false")
    ap.set_defaults(use_adapters=True)
    ap.add_argument("--gate-json", type=str, default=None)
    ap.add_argument("--device-map", type=str, default="auto")
    ap.add_argument("--dtype", type=str, default="float16", choices=["float16", "bfloat16", "float32"]) 
    ap.add_argument("--alpha", type=float, default=1.0)
    args = ap.parse_args(list(argv) if argv is not None else None)

    dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[args.dtype]
    print(f"Loading {args.model} with dtype={args.dtype}, device_map={args.device_map}")
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dtype, device_map=args.device_map)

    hidden_size = int(getattr(getattr(model, "config", object()), "hidden_size", 0) or 0)
    if hidden_size <= 0:
        print("Could not determine hidden_size from model.config.hidden_size")
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
    mope_layers = attach_mope_to_hf_model(model, layer_indices=layer_indices, hidden_size=hidden_size, expert_names=experts, gate_json=gate_init, use_adapters=bool(args.use_adapters), alpha=float(args.alpha))
    print(f"Attached MoPE to layers {layer_indices} (hidden_size={hidden_size}, experts={len(experts)}, alpha={args.alpha})")

    prompt = "Question: Who wrote 'Pride and Prejudice'?\nContext: \nAnswer:"
    enc = tok(prompt, return_tensors="pt")
    input_ids = enc["input_ids"].to(model.device)
    with torch.inference_mode():
        out = model.generate(input_ids, max_new_tokens=int(args.max_new_tokens), do_sample=False)
    text = tok.decode(out[0], skip_special_tokens=True)
    print("Output:\n", text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
