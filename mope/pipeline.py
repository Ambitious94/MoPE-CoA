"""Pipeline experts and atomic reasoning tools used by the MoPE layer."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Sequence
import os

# Atomic tool signatures
PipelineStep = Callable[[str], str]


def planner(prompt: str) -> str:
    return f"plan: break down '{prompt}' into searchable claims"


def search(prompt: str) -> str:
    return f"search results for '{prompt}': [doc1, doc2]"


def reader(prompt: str) -> str:
    return f"read snippets for '{prompt}' and extract key facts"


def verifier(prompt: str) -> str:
    return f"verify facts for '{prompt}' using consensus checks"


def mult_search(prompt: str) -> str:
    return f"multi-search '{prompt}' across sources"


def compare(prompt: str) -> str:
    return f"compare findings for '{prompt}' and resolve conflicts"

# --- Task-oriented expert steps requested ---

def think(prompt: str) -> str:
    return f"think: consider intent and constraints for '{prompt}'"

def plan(prompt: str) -> str:
    return f"plan: propose steps (search -> read -> reflect -> check) for '{prompt}'"

def tool_web_search(prompt: str) -> str:
    # Use SerpAPI when available; otherwise fallback to stub output.
    from .tools import serpapi_search
    api_key = os.getenv("SERPAPI_API_KEY")
    if api_key:
        return serpapi_search(prompt, api_key=api_key, top_k=3)
    return f"tool.web_search: query '{prompt}' and collect top snippets [stub]"

def tool_crawl_page(prompt: str) -> str:
    # Use Jina crawl when available; otherwise fallback to stub output.
    from .tools import jina_crawl
    api_key = os.getenv("JINA_API_KEY")
    if api_key:
        return jina_crawl(prompt, api_key=api_key, top_k=1)
    return f"tool.crawl_page: fetch full page(s) for '{prompt}' [stub]"

def observation(prompt: str) -> str:
    return f"observation: summarize retrieved evidence for '{prompt}'"

def reflection(prompt: str) -> str:
    return f"reflection: assess consistency and gaps for '{prompt}'"

def suggested_answer(prompt: str) -> str:
    return f"suggested_answer: draft concise answer for '{prompt}'"

def double_check(prompt: str) -> str:
    return f"double_check: re-verify key facts for '{prompt}'"

def final_answer(prompt: str) -> str:
    return f"answer: deliver final response for '{prompt}'"


@dataclass
class PipelineOutput:
    """Result of running a pipeline expert."""

    answer: str
    trace: List[str] = field(default_factory=list)

    def as_dict(self) -> Dict[str, str]:
        return {"answer": self.answer, "trace": "\n".join(self.trace)}


@dataclass
class PipelineExpert:
    """A chain of atomic steps representing a search/QA strategy."""

    name: str
    steps: Sequence[PipelineStep]
    description: str

    def run(self, prompt: str) -> PipelineOutput:
        trace: List[str] = []
        intermediate = prompt
        for step in self.steps:
            intermediate = step(intermediate)
            trace.append(intermediate)
        return PipelineOutput(answer=intermediate, trace=trace)


PIPELINE_REGISTRY: Dict[str, PipelineExpert] = {
    # Task-oriented pipelines based on the requested expert steps only
    "think-plan-websearch-observe-reflect-suggest-check-answer": PipelineExpert(
        name="think-plan-websearch-observe-reflect-suggest-check-answer",
        steps=[
            think,
            plan,
            tool_web_search,
            observation,
            reflection,
            suggested_answer,
            double_check,
            final_answer,
        ],
        description="Deliberate thinking, plan, web search, observation, reflection, suggestion, double-check, and final answer.",
    ),
    "plan-websearch-crawl-observe-reflect-answer": PipelineExpert(
        name="plan-websearch-crawl-observe-reflect-answer",
        steps=[
            plan,
            tool_web_search,
            tool_crawl_page,
            observation,
            reflection,
            final_answer,
        ],
        description="Plan, web search, crawl page, observe evidence, reflect, and answer.",
    ),
}


def register_task_experts(system) -> Dict[str, PipelineExpert]:
    """Register task-oriented experts backed by the real retrieval/QA system.

    This augments the global PIPELINE_REGISTRY with experts that call
    SearchQAFactCheckingSystem for retrieve/read/verify/answer steps.
    Returns a dict of the newly registered experts.
    """

    def plan(prompt: str) -> str:
        return f"plan: decompose '{prompt}' into claims for retrieval"

    def retrieve(prompt: str) -> str:
        hits = system.search(prompt, top_k=3)
        formatted = "; ".join(f"{h['doc_id']}({h.get('score', 0):.2f})" for h in hits)
        return f"retrieval hits: {formatted}" if hits else f"no hits for '{prompt}'"

    def read(prompt: str) -> str:
        hits = system.search(prompt, top_k=3)
        summary = system.read(prompt, hits)
        return f"reader summary: {summary}"

    def verify(prompt: str) -> str:
        verdict = system.fact_checker.check(prompt)
        return f"verdict: {verdict['status']} (score={verdict.get('score', 0):.2f})"

    def answer(prompt: str) -> str:
        result = system.answer(prompt, top_k=3)
        return f"answer: {result['answer']} | verdict: {result['verdict']['status']}"

    experts = {
        "task-plan-retrieve-read-verify": PipelineExpert(
            name="task-plan-retrieve-read-verify",
            steps=[plan, retrieve, read, verify],
            description="Real system: plan, retrieve, read evidence, verify facts.",
        ),
        "task-retrieve-read-answer": PipelineExpert(
            name="task-retrieve-read-answer",
            steps=[retrieve, read, answer],
            description="Real system: fast retrieve+read with answer synthesis.",
        ),
    }

    # Merge into global registry (non-destructive for existing keys)
    for k, v in experts.items():
        PIPELINE_REGISTRY[k] = v
    return experts
