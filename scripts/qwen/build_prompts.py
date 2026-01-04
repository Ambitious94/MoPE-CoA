"""Build prompts.txt for distillation from local JSON/JSONL datasets.

Input format:
- JSONL/NDJSON: one object per line, default question key 'question'
- JSON: array of objects or single object with fields

Output format:
- Plain text where each example is two lines:
  "Question: <q>\nAnswer:\n"

Usage (Linux):
  python -m scripts.qwen.build_prompts \
    --input /data/hotpot_qa_dev.jsonl \
    --output /data/prompts.txt \
    --question-key question
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Iterable, List, Dict, Any


def _load_items(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    suf = path.suffix.lower()
    if suf in {".jsonl", ".ndjson"}:
        items: List[Dict[str, Any]] = []
        for ln in path.read_text(encoding="utf-8", errors="ignore").splitlines():
            ln = ln.strip()
            if not ln:
                continue
            try:
                obj = json.loads(ln)
                if isinstance(obj, dict):
                    items.append(obj)
            except Exception:
                continue
        return items
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(obj, list):
            return [x for x in obj if isinstance(x, dict)]
        if isinstance(obj, dict):
            return [obj]
    except Exception:
        pass
    return []


def main(argv: Iterable[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="build_prompts")
    ap.add_argument("--input", type=str, nargs="+", help="One or more JSON/JSONL files with questions")
    ap.add_argument("--output", type=str, required=True)
    ap.add_argument("--question-key", type=str, default="question")
    ap.add_argument("--shuffle", action="store_true")
    ap.add_argument("--max-examples", type=int, default=0, help="Limit number of prompts (0=no limit)")
    ap.add_argument("--prefix", type=str, default="Question: ")
    ap.add_argument("--suffix", type=str, default="Answer:")
    args = ap.parse_args(list(argv) if argv is not None else None)

    inputs = [Path(p) for p in args.input]
    items: List[Dict[str, Any]] = []
    for p in inputs:
        items += _load_items(p)
    if not items:
        print("No input items loaded.")
        return 2
    if args.shuffle:
        random.shuffle(items)
    out_lines: List[str] = []
    qk = str(args.question_key)
    for it in items:
        q = str(it.get(qk, "")).strip()
        if not q:
            continue
        out_lines.append(f"{args.prefix}{q}\n{args.suffix}\n")
        if args.max_examples and len(out_lines) >= int(args.max_examples):
            break
    if not out_lines:
        print("No questions found under key:", qk)
        return 3
    Path(args.output).write_text("".join(out_lines), encoding="utf-8")
    print({"written": len(out_lines), "output": str(args.output)})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
