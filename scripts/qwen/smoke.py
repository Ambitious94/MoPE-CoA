"""Quick smoke test for attaching MoPE to Qwen HF models.

Example (Windows PowerShell):
  python -m scripts.qwen.smoke --model Qwen/Qwen2.5-3B-Instruct --layers 30,31 --alpha 0.01
"""
from __future__ import annotations

import argparse
import json
import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from mope.hf_integration import attach_mope_to_hf_model
from mope.pipeline import PIPELINE_REGISTRY


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, default="Qwen/Qwen2.5-3B-Instruct")
    ap.add_argument("--layers", type=str, default="34,35")
    ap.add_argument("--alpha", type=float, default=0.01)
    ap.add_argument("--gate-json", type=str, default=None)
    ap.add_argument("--dtype", type=str, default="float16", choices=["float16", "bfloat16", "float32"]) 
    ap.add_argument("--device-map", type=str, default="auto")
    args = ap.parse_args()

    dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[args.dtype]
    is_abs = os.path.isabs(args.model) or args.model.startswith(".") or args.model.startswith("/")
    if is_abs and not os.path.isdir(args.model):
        print("Local model directory not found:", args.model)
        return 2
    is_local = os.path.isdir(args.model)
    print(("Loading local model from" if is_local else "Loading from Hub:"), args.model)
    tok = AutoTokenizer.from_pretrained(args.model, use_fast=True, local_files_only=is_local)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=dtype,
        device_map=args.device_map,
        local_files_only=is_local,
    )

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
            print("Failed to load gate json:")
            print(e)

    experts = list(PIPELINE_REGISTRY.keys())
    _ = attach_mope_to_hf_model(model, layer_indices=layer_indices, hidden_size=hidden_size, expert_names=experts, gate_json=gate_init, use_adapters=False, alpha=float(args.alpha))

    prompt = "You are MoPE-augmented Qwen. Answer briefly.\nQuestion: What is the capital of France?\nAnswer:"
    toks = tok(prompt, return_tensors="pt").to(model.device)
    out = model.generate(**toks, max_new_tokens=32, do_sample=False)
    text = tok.decode(out[0], skip_special_tokens=True)
    print("=== Output ===")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
