"""RL training (REINFORCE) for MoPE gate with LLM-as-Judge (0/1 reward).

This script:
- Loads nanoGPT GPT (optionally pretrained GPT-2) and replaces one Block's FFN with TorchMoPELayer
- Runs rollout episodes that generate text for prompts, uses an LLM judge to score 0/1
- Updates only the MoPE gate parameters using REINFORCE with a moving baseline

Judge backends:
- openai: requires `pip install openai` and env OPENAI_API_KEY; set --judge-model (e.g., gpt-4o-mini)
- mock: a trivial keyword-based judge (no external deps), useful for smoke testing

Example (Windows PowerShell):

  python -m scripts.rl_gate_reinforce `
    --input "data/WebAgentSFTDataset.json" `
    --nanogpt-root "e:/Edge Download/nanoGPT-master" `
    --model-type gpt2 `
    --layer-idx 0 `
    --episodes 50 `
    --max-new-tokens 64 `
    --judge openai `
    --judge-model gpt-4o-mini `
    --lr 5e-5 --amp --out gate.json
"""
from __future__ import annotations

import argparse
import importlib
import json
import math
import os
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Tuple, Dict, Optional

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

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
    target: str | None


class Judge:
    def score(self, prompt: str, answer: str) -> int:
        raise NotImplementedError


class MockJudge(Judge):
    def __init__(self, positive_keywords: Optional[List[str]] = None) -> None:
        self.positive_keywords = positive_keywords or ["answer", "final", "result", "yes", "correct"]

    def score(self, prompt: str, answer: str) -> int:
        a = (answer or "").lower()
        return 1 if any(k in a for k in self.positive_keywords) else 0


class OpenAIJudge(Judge):
    def __init__(self, model: str, api_key: Optional[str] = None) -> None:
        try:
            import openai  # type: ignore
        except Exception as e:
            raise RuntimeError("openai package not installed. pip install openai") from e
        self.openai = importlib.import_module("openai")
        self.client = self.openai.OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))
        self.model = model

    def score(self, prompt: str, answer: str) -> int:
        system = (
            "You are a strict evaluator. Given a user question and a model's answer, "
            "return only a single digit 1 if the answer is correct/sufficient, otherwise 0. "
            "No explanation, no extra text."
        )
        user = (
            f"Question:\n{prompt}\n\nAnswer:\n{answer}\n\n"
            "Respond with 1 for correct, 0 for incorrect."
        )
        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                temperature=0.0,
                max_tokens=1,
            )
            txt = (resp.choices[0].message.content or "").strip()
            return 1 if txt.startswith("1") else 0
        except Exception:
            return 0


def _build_dataset(input_path: Path, hidden_size: int) -> List[Sample]:
    raw = _load_records(input_path)
    structured = [
        preprocess_record(r, hidden_size=hidden_size, with_instruction=False, drop_observation_content=True)
        for r in raw
    ]
    # Use inputs as prompts; take parsed target_answer when available
    samples: List[Sample] = []
    for r in structured:
        q = (r.get("input") or "").strip()
        t = (r.get("target_answer") or "").strip() or None
        if q:
            samples.append(Sample(text=q, target=t))
    return samples


def generate_with_mope(
    gpt,
    block,
    mope_layer,
    input_ids: torch.Tensor,
    max_new_tokens: int,
    temperature: float,
    top_k: Optional[int],
    device: torch.device,
) -> Tuple[str, List[int], torch.Tensor]:
    """Generate text with per-step gating decisions at the selected block.

    Returns: (decoded_text, action_indices, log_probs_tensor)
    """
    from transformers import GPT2TokenizerFast
    tok = GPT2TokenizerFast.from_pretrained("gpt2")

    gpt.eval()
    actions: List[int] = []
    log_probs: List[torch.Tensor] = []

    captured: List[torch.Tensor] = []
    def hook_ln2(_m, _i, out):
        captured.append(out)
    h = block.ln_2.register_forward_hook(hook_ln2)

    idx = input_ids.clone()
    for _ in range(int(max_new_tokens)):
        # crop context
        idx_cond = idx if idx.size(1) <= gpt.config.block_size else idx[:, -gpt.config.block_size:]
        # 1) feature pass to get ln_2
        captured.clear()
        _ = gpt(idx_cond)
        if not captured:
            raise RuntimeError("ln_2 hook did not capture features")
        feats = captured[-1]  # [B,T,H]
        X = feats[:, -1, :]  # [B,H]
        # policy over experts
        logits = X @ mope_layer.gate.weight + mope_layer.gate.bias  # [B,E]
        probs = torch.softmax(logits, dim=-1)
        dist = torch.distributions.Categorical(probs=probs)
        action = int(dist.sample().item())
        logp = dist.log_prob(torch.tensor([action], device=probs.device)).squeeze(0)
        actions.append(action)
        log_probs.append(logp)
        # 2) force expert and do actual forward for logits
        mope_layer.set_forced_expert(action)
        logits2, _ = gpt(idx_cond)
        logits2 = logits2[:, -1, :] / max(temperature, 1e-8)
        if top_k is not None:
            v, _ = torch.topk(logits2, min(top_k, logits2.size(-1)))
            logits2[logits2 < v[:, [-1]]] = -float("Inf")
        probs2 = torch.softmax(logits2, dim=-1)
        idx_next = torch.multinomial(probs2, num_samples=1)
        idx = torch.cat((idx, idx_next), dim=1)

    h.remove()
    text = tok.decode(idx[0].tolist(), skip_special_tokens=True)
    return text, actions, torch.stack(log_probs, dim=0)


def main(argv: Iterable[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="rl_gate_reinforce")
    ap.add_argument("--input", type=str, required=True)
    ap.add_argument("--nanogpt-root", type=str, required=True)
    ap.add_argument("--model-type", type=str, default="gpt2", choices=["gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"])
    ap.add_argument("--layer-idx", type=int, default=0)
    ap.add_argument("--block-size", type=int, default=512)
    ap.add_argument("--episodes", type=int, default=20)
    ap.add_argument("--max-new-tokens", type=int, default=64)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top-k", type=int, default=None)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--weight-decay", type=float, default=0.0)
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--aux-lm-weight", type=float, default=0.5, help="Weight for auxiliary LM loss to train expert adapters")
    ap.add_argument("--unfreeze-last", type=int, default=0)
    ap.add_argument("--judge", type=str, default="mock", choices=["mock", "openai"])
    ap.add_argument("--judge-model", type=str, default="gpt-4o-mini")
    ap.add_argument("--use-adapters", action="store_true", help="Use trainable expert adapters inside MoPE (default)")
    ap.add_argument("--no-adapters", dest="use_adapters", action="store_false", help="Disable adapters; use vectorizer-only residuals")
    ap.set_defaults(use_adapters=True)
    ap.add_argument("--out", type=str, default="gate.json")
    args = ap.parse_args(list(argv) if argv is not None else None)

    input_path = Path(args.input)
    if not input_path.exists():
        print("Input not found:", input_path)
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
        print("transformers is required: pip install transformers")
        print("Error:", e)
        return 4

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load GPT-2 weights
    gpt = GPT.from_pretrained(args.model_type)
    if args.block_size < gpt.config.block_size:
        gpt.crop_block_size(int(args.block_size))
    gpt.to(device)

    # Freeze GPT by default
    for p in gpt.parameters():
        p.requires_grad_(False)
    if args.unfreeze_last > 0:
        n = len(gpt.transformer.h)
        start = max(0, n - int(args.unfreeze_last))
        for i in range(start, n):
            for p in gpt.transformer.h[i].parameters():
                p.requires_grad_(True)

    # Dataset
    hidden_size = int(getattr(gpt.config, "n_embd", 768))
    samples = _build_dataset(input_path, hidden_size=hidden_size)
    if len(samples) == 0:
        print("No prompts found. Exiting.")
        return 1

    # Tokenizer
    tok = GPT2TokenizerFast.from_pretrained("gpt2")
    tok.pad_token = tok.eos_token

    # Replace FFN with TorchMoPE at selected layer
    block = gpt.transformer.h[int(args.layer_idx)]
    expert_names = list(PIPELINE_REGISTRY.keys())
    mope_layer = replace_ffn_with_torch_mope(block, hidden_size=hidden_size, expert_names=expert_names, gate_json=None, use_adapters=bool(args.use_adapters))
    mope_layer.to(device)
    mope_layer.train()

    # Optimizer over gate + (optional) expert adapters
    if bool(args.use_adapters):
        optimizer = torch.optim.AdamW(mope_layer.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))
    else:
        optimizer = torch.optim.AdamW(mope_layer.gate.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))

    # AMP scaler
    scaler = torch.amp.GradScaler(
        device_type=("cuda" if device.type == "cuda" else "cpu"),
        enabled=bool(args.amp) and device.type == "cuda",
    )

    # Judge
    if args.judge == "openai":
        judge: Judge = OpenAIJudge(model=args.judge_model)
    else:
        judge = MockJudge()

    # Moving baseline for REINFORCE
    baseline: float = 0.0
    beta: float = 0.9  # EMA

    gpt.train(False)

    for ep in range(int(args.episodes)):
        sample = random.choice(samples)
        prompt = sample.text
        enc = tok(prompt, padding=False, truncation=True, max_length=gpt.config.block_size, return_tensors="pt")
        input_ids = enc["input_ids"].to(device)

        # rollout with per-step gating decisions
        with torch.amp.autocast(device_type=device.type, enabled=bool(args.amp) and device.type == "cuda"):
            text, actions, log_probs = generate_with_mope(
                gpt=gpt,
                block=block,
                mope_layer=mope_layer,
                input_ids=input_ids,
                max_new_tokens=int(args.max_new_tokens),
                temperature=float(args.temperature),
                top_k=args.top_k,
                device=device,
            )
        # Judge 0/1
        R = judge.score(prompt, text)
        # REINFORCE loss: -(R - b) * sum(logpi)
        advantage = float(R) - baseline
        loss = -(log_probs.sum() * advantage)

        # Auxiliary LM loss on target answer (if available): trains expert adapters via differentiable path
        if sample.target:
            # Build concatenated input = prompt + target; compute CE only on target tokens
            pair = tok(
                prompt + " " + sample.target,
                padding=False,
                truncation=True,
                max_length=gpt.config.block_size,
                return_tensors="pt",
            )
            ids = pair["input_ids"][0]
            # find split point between prompt and target in token space (approximate via separate encodings)
            ids_prompt = tok(prompt, padding=False, truncation=True, max_length=gpt.config.block_size, return_tensors="pt")["input_ids"][0]
            t_len = min(ids.shape[0], gpt.config.block_size)
            p_len = min(ids_prompt.shape[0], t_len)
            inp = ids[:t_len-1].unsqueeze(0).to(device)  # all but last as input
            tgt = ids[1:t_len].clone()
            # mask out prompt part in targets
            mask = torch.ones_like(tgt, dtype=torch.long)
            mask[: max(0, p_len-1)] = 0
            targets_ce = tgt.to(device)
            targets_ce[mask == 0] = -1
            with torch.amp.autocast(device_type=device.type, enabled=bool(args.amp) and device.type == "cuda"):
                _logits, lm_loss = gpt(inp, targets=targets_ce)
            loss = loss + float(args.aux_lm_weight) * lm_loss

        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # update baseline
        baseline = beta * baseline + (1 - beta) * float(R)

        print(f"ep {ep+1}/{args.episodes} R={R} adv={advantage:.3f} actions={actions[:8]}... logp_sum={float(log_probs.sum().item()):.3f}")

    # Save gate
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
            "rl": "reinforce_llm_as_judge",
            "judge": args.judge,
            "model_type": args.model_type,
            "layer_idx": int(args.layer_idx),
        },
    }
    Path(args.out).write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print("Saved gate to", args.out)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
