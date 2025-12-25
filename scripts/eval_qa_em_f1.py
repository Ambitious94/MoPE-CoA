"""Evaluation of SFT/MoPE on QA datasets (e.g., HotpotQA) using EM/F1.

Loads nanoGPT GPT with a selected block replaced by TorchMoPELayer, optionally
initialized from gate.json. Generates answers for each question and computes
Exact Match (EM) and token-level F1 versus ground-truth.

Supports generic JSON datasets:
- JSON array of objects
- Top-level object with one of: data / records / items (list)
Field mapping:
- --input-key: question text field (default: question)
- --target-key: answer text field (default: answer)

Example (Windows PowerShell):

  python -m scripts.eval_qa_em_f1 `
    --data "path/to/hotpot_train_v1.1.json" `
    --nanogpt-root "e:/Edge Download/nanoGPT-master" `
    --model-type gpt2 `
    --layer-idx 0 `
    --gate-json gate.json `
    --max-new-tokens 64 `
    --temperature 0.8 `
    --top-k 50
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Tuple, Optional
import hashlib
import time

import torch
from mope.nanogpt_integration import replace_ffn_with_torch_mope
from mope.pipeline import PIPELINE_REGISTRY
from mope.tools import serpapi_search, serpapi_search_raw, jina_crawl


def _maybe_add_repo_to_syspath(root: Path) -> None:
    p = str(root)
    if p not in sys.path:
        sys.path.insert(0, p)


def _load_rows(path: Path) -> List[dict]:
    if not path.exists():
        return []
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return []
    if isinstance(obj, list):
        return obj
    if isinstance(obj, dict):
        for k in ("data", "records", "items"):
            v = obj.get(k)
            if isinstance(v, list):
                return v
        return [obj]
    return []


def _cache_path(root: Path, key: str) -> Path:
    h = hashlib.sha1(key.encode('utf-8')).hexdigest()
    return root / 'search' / f"{h}.json"


def _cache_get(root: Path, key: str) -> Optional[str]:
    p = _cache_path(root, key)
    if p.exists():
        try:
            obj = json.loads(p.read_text(encoding='utf-8'))
            return str(obj.get('result') or '')
        except Exception:
            return None
    return None


def _cache_put(root: Path, key: str, result: str) -> None:
    p = _cache_path(root, key)
    p.parent.mkdir(parents=True, exist_ok=True)
    payload = {"query": key, "result": result, "ts": int(time.time())}
    p.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')


def _normalize_text(s: str) -> str:
    s = (s or "").lower()
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = re.sub(r"[^a-z0-9 ]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _f1_score(pred: str, gold: str) -> float:
    p_tokens = _normalize_text(pred).split()
    g_tokens = _normalize_text(gold).split()
    if len(p_tokens) == 0 and len(g_tokens) == 0:
        return 1.0
    if len(p_tokens) == 0 or len(g_tokens) == 0:
        return 0.0
    common: Dict[str, int] = {}
    for t in p_tokens:
        common[t] = common.get(t, 0) + 1
    match = 0
    for t in g_tokens:
        if common.get(t, 0) > 0:
            match += 1
            common[t] -= 1
    if match == 0:
        return 0.0
    precision = match / len(p_tokens)
    recall = match / len(g_tokens)
    return 2 * precision * recall / (precision + recall)


def _exact_match(pred: str, gold: str) -> float:
    return 1.0 if _normalize_text(pred) == _normalize_text(gold) else 0.0


def generate_answer(gpt, block, mope_layer, tok, question_ids: torch.Tensor, max_new_tokens: int, temperature: float, top_k: Optional[int]) -> str:
    gpt.eval()
    captured: List[torch.Tensor] = []

    def hook_ln2(_m, _i, out):
        captured.append(out)

    h = block.ln_2.register_forward_hook(hook_ln2)

    idx = question_ids.clone()
    start_len = int(idx.shape[1])
    for _ in range(int(max_new_tokens)):
        idx_cond = idx if idx.size(1) <= gpt.config.block_size else idx[:, -gpt.config.block_size:]
        captured.clear()
        _ = gpt(idx_cond)
        if not captured:
            raise RuntimeError("ln_2 hook did not capture features")
        # Greedy routing: argmax expert per step
        feats = captured[-1]
        X = feats[:, -1, :]
        logits_gate = X @ mope_layer.gate.weight + mope_layer.gate.bias
        probs_gate = torch.softmax(logits_gate, dim=-1)
        action = int(torch.argmax(probs_gate, dim=-1)[0].item())
        mope_layer.set_forced_expert(action)
        # Next token sampling
        logits2, _ = gpt(idx_cond)
        logits2 = logits2[:, -1, :] / max(temperature, 1e-8)
        if top_k is not None:
            v, _ = torch.topk(logits2, min(top_k, logits2.size(-1)))
            logits2[logits2 < v[:, [-1]]] = -float("Inf")
        probs2 = torch.softmax(logits2, dim=-1)
        idx_next = torch.multinomial(probs2, num_samples=1)
        idx = torch.cat((idx, idx_next), dim=1)

    h.remove()
    # Decode only the generated continuation, not the original prompt
    gen_tokens = idx[0, start_len:]
    return tok.decode(gen_tokens.tolist(), skip_special_tokens=True)


def main(argv: Iterable[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="eval_qa_em_f1")
    ap.add_argument("--data", type=str, required=True, help="Path to JSON dataset (array or object with data/records/items)")
    ap.add_argument("--nanogpt-root", type=str, required=True)
    ap.add_argument("--model-type", type=str, default="gpt2", choices=["gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"])
    ap.add_argument("--layer-idx", type=int, default=0)
    ap.add_argument("--gate-json", type=str, default=None, help="Optional gate.json to initialize MoPE gate")
    ap.add_argument("--max-new-tokens", type=int, default=64)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top-k", type=int, default=None)
    ap.add_argument("--input-key", type=str, default="question")
    ap.add_argument("--target-key", type=str, default="answer")
    ap.add_argument("--limit", type=int, default=None, help="Evaluate first N samples")
    ap.add_argument("--use-adapters", action="store_true", help="Use trainable expert adapters inside MoPE (default)")
    ap.add_argument("--no-adapters", dest="use_adapters", action="store_false", help="Disable adapters; use vectorizer-only residuals")
    ap.set_defaults(use_adapters=True)
    # Retrieval options
    ap.add_argument("--use-tools", action="store_true", help="Enable retrieval to build context (SerpAPI)")
    ap.add_argument("--search-backend", type=str, default="serpapi", choices=["serpapi", "jina", "none"], help="Retrieval backend (search). Use serpapi when composing with crawl.")
    ap.add_argument("--serpapi-key", type=str, default=None, help="SerpAPI key (or use env SERPAPI_API_KEY)")
    ap.add_argument("--search-topk", type=int, default=3, help="Top-k search results to summarize")
    ap.add_argument("--cache-dir", type=str, default=".mope_cache", help="Directory to cache retrieval results")
    ap.add_argument("--tools-no-inject", action="store_true", help="Run tools (search/crawl) but do NOT inject context into the prompt")
    ap.add_argument("--jina-key", type=str, default=None, help="Jina API key (or use env JINA_API_KEY)")
    ap.add_argument("--use-crawl", action="store_true", help="Enable crawl step after search to fetch page content via Jina")
    ap.add_argument("--crawl-topk", type=int, default=1, help="Crawl top-K URLs from search results")
    ap.add_argument("--dump-eval", type=str, default=None, help="Optional JSONL path to dump per-sample results and tool summaries")
    args = ap.parse_args(list(argv) if argv is not None else None)

    data_path = Path(args.data)
    if not data_path.exists():
        print("Data not found:", data_path)
        return 2

    _maybe_add_repo_to_syspath(Path(args.nanogpt_root))
    try:
        from model import GPT
    except Exception as e:
        print("Failed to import nanoGPT model.py from --nanogpt-root:", e)
        return 3
    try:
        from transformers import GPT2TokenizerFast
    except Exception as e:
        print("transformers required: pip install transformers")
        print("Error:", e)
        return 4

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load GPT-2 weights
    gpt = GPT.from_pretrained(args.model_type)
    gpt.to(device)

    # Replace FFN with TorchMoPE
    block = gpt.transformer.h[int(args.layer_idx)]
    expert_names = list(PIPELINE_REGISTRY.keys())
    gate_init = None
    if args.gate_json and Path(args.gate_json).exists():
        try:
            gate_init = json.loads(Path(args.gate_json).read_text(encoding="utf-8"))
            print("Loaded gate init from", args.gate_json)
        except Exception as e:
            print("Failed to load gate json:", e)
    mope_layer = replace_ffn_with_torch_mope(block, hidden_size=int(getattr(gpt.config, "n_embd", 768)), expert_names=expert_names, gate_json=gate_init, use_adapters=bool(args.use_adapters))
    mope_layer.to(device)

    # Tokenizer
    tok = GPT2TokenizerFast.from_pretrained("gpt2")
    tok.pad_token = tok.eos_token

    rows = _load_rows(data_path)
    if args.limit:
        rows = rows[: int(args.limit)]
    if not rows:
        print("No rows to evaluate")
        return 1

    total_em = 0.0
    total_f1 = 0.0
    total_n = 0

    cache_root = Path(str(args.cache_dir))
    dump_fp = None
    if args.dump_eval:
        try:
            dump_path = Path(str(args.dump_eval))
            dump_path.parent.mkdir(parents=True, exist_ok=True)
            dump_fp = open(dump_path, "w", encoding="utf-8")
        except Exception as e:
            print("Failed to open dump file:", e)
            dump_fp = None
    for r in rows:
        if not isinstance(r, dict):
            continue
        q = str(r.get(args.input_key) or "").strip()
        a = str(r.get(args.target_key) or "").strip()
        if not q:
            continue
        # Optionally retrieve context via tools (cached)
        inp_text = q
        summary = None
        summary_injected = False
        cache_hit = False
        backend = args.search_backend
        urls: List[str] = []
        crawl_summaries: List[str] = []
        if bool(args.use_tools):
            # 1) Search phase
            summary = _cache_get(cache_root, f"search:{q}")
            if summary is None:
                try:
                    if args.search_backend == "serpapi":
                        key = args.serpapi_key or os.getenv("SERPAPI_API_KEY")
                        if key:
                            summary = serpapi_search(q, api_key=key, top_k=int(args.search_topk))
                            # Also gather URLs for optional crawl
                            results = serpapi_search_raw(q, api_key=key, top_k=int(args.search_topk))
                            urls = [r.get("link") for r in results if r.get("link")]
                    elif args.search_backend == "jina":
                        key = args.jina_key or os.getenv("JINA_API_KEY")
                        if key:
                            summary = jina_crawl(q, api_key=key, top_k=1)
                except Exception as e:
                    summary = f"tool_error: {e}"
                _cache_put(cache_root, f"search:{q}", summary or "")
            else:
                cache_hit = True
                # On cache hit for search, try to also load cached url list if present (not stored separately -> recompute lazily on next miss)

            # 2) Crawl phase (optional, only if we have URLs)
            if bool(args.use_crawl) and urls:
                topk = max(1, int(args.crawl_topk))
                for url in urls[:topk]:
                    cached = _cache_get(cache_root, f"crawl:{url}")
                    if cached is None:
                        try:
                            key = args.jina_key or os.getenv("JINA_API_KEY")
                            if key:
                                cached = jina_crawl(url, api_key=key, top_k=1)
                        except Exception as e:
                            cached = f"tool_error: {e}"
                        _cache_put(cache_root, f"crawl:{url}", cached or "")
                    if cached:
                        crawl_summaries.append(cached)
            # Only inject if allowed and summary looks valid
            if (not bool(args.tools_no_inject)):
                context_parts: List[str] = []
                if summary and not str(summary).startswith("tool.web_search") and not str(summary).startswith("tool_error"):
                    context_parts.append(summary)
                for cs in crawl_summaries:
                    if cs and not str(cs).startswith("tool_error"):
                        context_parts.append(cs)
                if context_parts:
                    # Concatenate with separator and trim
                    combined = " \n---\n ".join(context_parts)
                    inp_text = f"Question: {q}\nContext: {combined}\nAnswer:"
                    summary_injected = True
                summary_injected = True
        enc = tok(inp_text, padding=False, truncation=True, max_length=gpt.config.block_size, return_tensors="pt")
        input_ids = enc["input_ids"].to(device)
        pred = generate_answer(gpt, block, mope_layer, tok, input_ids, max_new_tokens=int(args.max_new_tokens), temperature=float(args.temperature), top_k=args.top_k)
        if a:
            total_em += _exact_match(pred, a)
            total_f1 += _f1_score(pred, a)
        total_n += 1
        if dump_fp is not None:
            try:
                row = {
                    "question": q,
                    "gold": a,
                    "pred": pred,
                    "tools_enabled": bool(args.use_tools),
                    "backend": backend,
                    "cache_hit": cache_hit,
                    "summary_injected": summary_injected,
                    "summary": (summary[:500] + ("..." if summary and len(summary) > 500 else "")) if isinstance(summary, str) else None,
                    "urls": urls[:5],
                    "crawl_summaries": [ (cs[:300] + ("..." if len(cs) > 300 else "")) for cs in crawl_summaries[:3] ] if crawl_summaries else [],
                }
                dump_fp.write(json.dumps(row, ensure_ascii=False) + "\n")
            except Exception:
                pass
        if total_n % 50 == 0:
            print(f"Processed {total_n} samples...")

    if total_n == 0:
        print("No valid samples evaluated")
        return 5

    em = total_em / total_n
    f1 = total_f1 / total_n if total_f1 > 0 else 0.0
    print(json.dumps({
        "samples": total_n,
        "em": round(em, 4),
        "f1": round(f1, 4),
        "config": {
            "model_type": args.model_type,
            "layer_idx": int(args.layer_idx),
            "use_adapters": bool(args.use_adapters),
            "max_new_tokens": int(args.max_new_tokens),
            "temperature": float(args.temperature),
            "top_k": args.top_k,
        },
    }, ensure_ascii=False, indent=2))

    if dump_fp is not None:
        try:
            dump_fp.close()
        except Exception:
            pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
