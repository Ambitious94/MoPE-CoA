from __future__ import annotations

import re
from typing import Any, Dict, List, Sequence, Tuple

from .model import MoPETransformer
from .training import RouteBatch

_TAGS = [
    "think",
    "plan",
    "web_search",
    "crawl_page",
    "observation",
    "reflection",
    "suggested_answer",
    "double_check",
    "answer",
]

# Precompile regex to capture tags in order, non-greedy, dot matches newline
_TAG_RE = re.compile(
    r"<(?P<tag>" + "|".join(_TAGS) + r")>(?P<content>.*?)</(?P=tag)>", re.IGNORECASE | re.DOTALL
)


def _norm(s: str) -> str:
    return (s or "").strip()


def parse_coa_output(output: str) -> Tuple[List[Dict[str, Any]], str]:
    """Parse CoA-style tagged output into ordered step list and target answer.

    Returns (steps, target_answer).
    Each step is a dict with keys similar to the requested schema.
    """
    steps: List[Dict[str, Any]] = []
    answer_text: str = ""

    if not output:
        return steps, answer_text

    idx = 1
    for m in _TAG_RE.finditer(output):
        tag = m.group("tag").lower()
        content = _norm(m.group("content"))
        if tag == "web_search":
            steps.append(
                {
                    "step_id": idx,
                    "stage": "tool",
                    "expert_type": "tool_web_search",
                    "tool_name": "web_search",
                    "tool_args": content,
                }
            )
        elif tag == "crawl_page":
            steps.append(
                {
                    "step_id": idx,
                    "stage": "tool",
                    "expert_type": "tool_crawl_page",
                    "tool_name": "crawl_page",
                    "tool_args": content,
                }
            )
        elif tag == "observation":
            steps.append(
                {
                    "step_id": idx,
                    "stage": "observation",
                    "expert_type": "observation",
                    "observation": content,
                    "observation_mask": True,
                }
            )
        elif tag == "double_check":
            # try extract score like "score:4" or "The score this time is:4"
            m_score = re.search(r"(score\s*[:=]\s*|is\s*:)(\d+)", content, re.IGNORECASE)
            score = int(m_score.group(2)) if m_score else None
            step: Dict[str, Any] = {
                "step_id": idx,
                "stage": "double_check",
                "expert_type": "double_check",
                "content": content,
            }
            if score is not None:
                step["score"] = score
            steps.append(step)
        elif tag == "answer":
            steps.append(
                {
                    "step_id": idx,
                    "stage": "answer",
                    "expert_type": "answer",
                    "content": content,
                }
            )
            answer_text = content
        elif tag == "suggested_answer":
            steps.append(
                {
                    "step_id": idx,
                    "stage": "suggested_answer",
                    "expert_type": "suggested_answer",
                    "content": content,
                }
            )
        elif tag == "reflection":
            steps.append(
                {
                    "step_id": idx,
                    "stage": "reflection",
                    "expert_type": "reflection",
                    "content": content,
                }
            )
        elif tag == "plan":
            steps.append(
                {
                    "step_id": idx,
                    "stage": "plan",
                    "expert_type": "plan",
                    "content": content,
                }
            )
        elif tag == "think":
            steps.append(
                {
                    "step_id": idx,
                    "stage": "think",
                    "expert_type": "think",
                    "content": content,
                }
            )
        idx += 1

    return steps, answer_text


def map_steps_to_pipeline_name(steps: Sequence[Dict[str, Any]]) -> str | None:
    """Map parsed steps to one of the configured pipeline names.

    Rule: if any crawl_page tool exists -> use crawl pipeline, else use think-plan-websearch pipeline.
    """
    has_crawl = any(s.get("expert_type") == "tool_crawl_page" for s in steps)
    if has_crawl:
        return "plan-websearch-crawl-observe-reflect-answer"
    return "think-plan-websearch-observe-reflect-suggest-check-answer" if steps else None


def preprocess_record(
    raw: Dict[str, Any], *, hidden_size: int = 8, with_instruction: bool = False, drop_observation_content: bool = True
) -> Dict[str, Any]:
    """Convert one raw record into the structured schema with pipeline steps and supervision.

    Returns structured dict with id/domain/input/target_answer/pipeline/supervision/meta
    """
    rid = str(raw.get("id") or raw.get("_id") or "")
    input_text = _norm(str(raw.get("input") or ""))
    output_text = str(raw.get("output") or "")
    steps, target_answer = parse_coa_output(output_text)
    # Optionally drop observation text from learning payload
    if drop_observation_content:
        for s in steps:
            if s.get("expert_type") == "observation":
                # Keep the fact that observation happened, but omit content
                s["observation"] = None
                s["observation_mask"] = True
    pipeline_name = map_steps_to_pipeline_name(steps)

    meta_tools = []
    if any(s.get("expert_type") == "tool_web_search" for s in steps):
        meta_tools.append("web_search")
    if any(s.get("expert_type") == "tool_crawl_page" for s in steps):
        meta_tools.append("crawl_page")

    structured: Dict[str, Any] = {
        "id": rid or None,
        "domain": raw.get("domain"),
        "is_train": bool(raw.get("is_train", False)),
        "instruction": raw.get("instruction") if with_instruction else None,
        "input": input_text,
        "target_answer": target_answer,
        "pipeline": steps,
        "supervision": {
            "pipeline_label": pipeline_name,
            "step_counts": {
                "think": sum(1 for s in steps if s.get("expert_type") == "think"),
                "plan": sum(1 for s in steps if s.get("expert_type") == "plan"),
                "tool_web_search": sum(1 for s in steps if s.get("expert_type") == "tool_web_search"),
                "tool_crawl_page": sum(1 for s in steps if s.get("expert_type") == "tool_crawl_page"),
                "observation": sum(1 for s in steps if s.get("expert_type") == "observation"),
                "reflection": sum(1 for s in steps if s.get("expert_type") == "reflection"),
                "suggested_answer": sum(1 for s in steps if s.get("expert_type") == "suggested_answer"),
                "double_check": sum(1 for s in steps if s.get("expert_type") == "double_check"),
                "answer": sum(1 for s in steps if s.get("expert_type") == "answer"),
            },
            "learning_masks": {
                # Explicitly state that observation content is excluded from learning
                "observation_content": False if drop_observation_content else True,
            },
        },
        "meta": {
            "source": "CoA-distilled",
            "tools": meta_tools,
        },
    }
    return structured


def build_route_batch_from_structured(records: Sequence[Dict[str, Any]], *, hidden_size: int = 8) -> RouteBatch:
    """Create a RouteBatch for supervised gate training from structured records.

    Uses MoPETransformer.encode_prompt to generate hidden states if none provided.
    """
    enc = MoPETransformer(
        config=type("C", (), {"hidden_size": hidden_size, "num_layers": 1, "vocab_size": 10})()
    )
    prompts: List[str] = []
    labels: List[str] = []
    hidden_states: List[Sequence[float]] = []

    for r in records:
        q = _norm(str(r.get("input") or ""))
        sup = r.get("supervision") or {}
        label = sup.get("pipeline_label")
        if not q or not label:
            continue
        h = enc.encode_prompt(q)
        prompts.append(q)
        labels.append(label)
        hidden_states.append(h)

    return RouteBatch(prompts=prompts, hidden_states=hidden_states, labels=labels)
