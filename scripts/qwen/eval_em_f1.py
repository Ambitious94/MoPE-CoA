"""Evaluate EM/F1 on a simple QA dataset using a local HF Qwen model.

Dataset format:
- JSONL: one object per line with fields {"question": str, "answer": str}
- or JSON: array of such objects

Prompts:
- "Question: <q>\nAnswer:"; we decode only the generated continuation.

Usage (Linux/bash):
  python -m scripts.qwen.eval_em_f1 \
    --model /models/qwen2.5-3b-instruct \
    --data /path/hotpot_qa_dev.jsonl \
    --max-new 64 \
    --dtype bfloat16 --device-map auto
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def _load_items(p: Path) -> List[dict]:
    if not p.exists():
        return []
    if p.suffix.lower() in {".jsonl", ".ndjson"}:
        items = []
        for ln in p.read_text(encoding="utf-8", errors="ignore").splitlines():
            ln = ln.strip()
            if not ln:
                continue
            try:
                items.append(json.loads(ln))
            except Exception:
                continue
        return items
    try:
        obj = json.loads(p.read_text(encoding="utf-8"))
        return obj if isinstance(obj, list) else [obj]
    except Exception:
        return []


def _norm(s: str) -> str:
    import re
    s = s.lower().strip()
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"[^a-z0-9\s]", "", s)
    return s.strip()


def _em(a: str, b: str) -> float:
    return 1.0 if _norm(a) == _norm(b) else 0.0


def _f1(pred: str, gold: str) -> float:
    pa = _norm(pred).split()
    ga = _norm(gold).split()
    if not pa and not ga:
        return 1.0
    if not pa or not ga:
        return 0.0
    common = 0
    from collections import Counter
    cp = Counter(pa)
    cg = Counter(ga)
    for w in cp:
        common += min(cp[w], cg.get(w, 0))
    if common == 0:
        return 0.0
    prec = common / max(1, len(pa))
    rec = common / max(1, len(ga))
    return 2 * prec * rec / max(1e-9, prec + rec)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, required=True)
    ap.add_argument("--data", type=str, required=True)
    ap.add_argument("--max-new", type=int, default=64)
    ap.add_argument("--dtype", type=str, default="bfloat16", choices=["float16","bfloat16","float32"]) 
    ap.add_argument("--device-map", type=str, default="auto")
    args = ap.parse_args()

    dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[args.dtype]
    tok = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    if tok.pad_token_id is None and tok.eos_token_id is not None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dtype, device_map=args.device_map)

    items = _load_items(Path(args.data))
    if not items:
        print("No data items loaded.")
        return 2

    em_sum = 0.0
    f1_sum = 0.0
    n = 0

    for it in items:
        q = str(it.get("question", "")).strip()
        gold = str(it.get("answer", "")).strip()
        prompt = f"Question: {q}\nAnswer:"
        enc = tok(prompt, return_tensors="pt")
        enc = {k: v.to(model.device) for k, v in enc.items()}
        out = model.generate(**enc, max_new_tokens=int(args.max_new), do_sample=False)
        text = tok.decode(out[0], skip_special_tokens=True)
        # Slice out only generated continuation after the prompt
        gen = text[len(prompt):].strip()
        em_sum += _em(gen, gold)
        f1_sum += _f1(gen, gold)
        n += 1

    print({"EM": round(em_sum / max(1, n), 4), "F1": round(f1_sum / max(1, n), 4), "N": n})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
