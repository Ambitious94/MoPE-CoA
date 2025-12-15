"""Task-oriented search / QA / fact-checking helpers.

These utilities build on the retrieval primitives to produce grounded answers
and verdicts that can be plugged into MoPE pipelines or used standalone.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence

from .pipeline import PipelineExpert
from .retrieval import DocumentStore, EvidenceReader


@dataclass
class Answer:
    """Structured answer with supporting evidence."""

    text: str
    evidence: List[Dict[str, object]]


class AnswerSynthesizer:
    """Produces a concise answer string from retrieved evidence."""

    def synthesize(self, question: str, hits: List[Dict[str, object]]) -> Answer:
        if not hits:
            return Answer(text=f"no answer found for '{question}'", evidence=[])
        best = max(hits, key=lambda h: h.get("score", 0.0))
        evidence_text = best.get("snippet") or best.get("doc_id", "")
        answer_text = evidence_text if evidence_text else f"insufficient evidence for '{question}'"
        return Answer(text=answer_text, evidence=hits)


class FactChecker:
    """Lightweight fact checker that classifies claims against retrieved evidence."""

    def __init__(self, store: DocumentStore, min_hits: int = 1) -> None:
        self.store = store
        self.min_hits = min_hits

    def check(self, claim: str) -> Dict[str, object]:
        hits = self.store.search(claim, top_k=max(self.min_hits, 5))
        # lexical support score: max token overlap ratio across snippets
        claim_tokens = [t for t in claim.lower().split() if t]
        def overlap_ratio(text: str) -> float:
            low = text.lower()
            if not claim_tokens:
                return 0.0
            matched = sum(1 for t in claim_tokens if t in low)
            return matched / max(len(claim_tokens), 1)

        best_score = 0.0
        for h in hits:
            best_score = max(best_score, overlap_ratio(str(h.get("snippet", ""))))

        supported = best_score >= 0.3 or len(hits) >= self.min_hits
        status = "supported" if supported else "no_evidence"
        rationale = "; ".join(f"{h['doc_id']}: {h['snippet']}" for h in hits) if hits else "searched but found nothing"
        return {
            "claim": claim,
            "status": status,
            "score": round(best_score, 3),
            "evidence": hits,
            "rationale": rationale,
        }


class SearchQAFactCheckingSystem:
    """End-to-end engine for search, QA, and fact-checking tasks."""

    def __init__(self, store: DocumentStore) -> None:
        self.store = store
        self.reader = EvidenceReader(store)
        self.synthesizer = AnswerSynthesizer()
        self.fact_checker = FactChecker(store)

    def search(self, query: str, top_k: int = 3) -> List[Dict[str, object]]:
        return self.store.search(query, top_k=top_k)

    def read(self, query: str, hits: Sequence[Dict[str, object]]) -> str:
        return self.reader.read(query, list(hits))

    def answer(self, question: str, top_k: int = 3) -> Dict[str, object]:
        hits = self.search(question, top_k=top_k)
        synthesis = self.synthesizer.synthesize(question, hits)
        reading = self.read(question, hits)
        verdict = self.fact_checker.check(synthesis.text)
        trace = [
            f"search for '{question}' yielded {len(hits)} hits",
            f"reader summary: {reading}",
            f"verdict: {verdict['status']} based on {len(verdict['evidence'])} evidence entries",
        ]
        return {
            "question": question,
            "answer": synthesis.text,
            "evidence": hits,
            "reader_summary": reading,
            "verdict": verdict,
            "trace": trace,
        }


def format_hits(hits: Sequence[Dict[str, object]]) -> str:
    return "; ".join(f"{h['doc_id']} (score={h.get('score', 0):.2f})" for h in hits)


def build_task_pipelines(system: SearchQAFactCheckingSystem):
    """Builds pipelines that perform retrieval, reading, and fact-checking."""

    def plan(prompt: str) -> str:
        return f"plan: decompose '{prompt}' into claims for retrieval"

    def retrieve(prompt: str) -> str:
        hits = system.search(prompt, top_k=3)
        return f"retrieval hits: {format_hits(hits)}" if hits else f"no hits for '{prompt}'"

    def read(prompt: str) -> str:
        hits = system.search(prompt, top_k=3)
        summary = system.read(prompt, hits)
        return f"reader summary: {summary}"

    def verify(prompt: str) -> str:
        verdict = system.fact_checker.check(prompt)
        return f"verdict for '{prompt}': {verdict['status']} ({verdict['rationale']})"

    def answer(prompt: str) -> str:
        result = system.answer(prompt, top_k=3)
        return f"answer: {result['answer']} | verdict: {result['verdict']['status']}"

    return {
        "plan-retrieve-read-verify": PipelineExpert(
            name="plan-retrieve-read-verify",
            steps=[plan, retrieve, read, verify],
            description="Planful retrieval with reading and evidence-backed verification.",
        ),
        "retrieve-read-answer": PipelineExpert(
            name="retrieve-read-answer",
            steps=[retrieve, read, answer],
            description="Fast retrieval, reading, and answer synthesis.",
        ),
    }
