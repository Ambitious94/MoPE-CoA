"""Vectorization utilities for converting pipeline text outputs back to vectors."""

from __future__ import annotations

import hashlib
import math
from typing import Iterable, List


def _zeros(dim: int) -> List[float]:
    return [0.0 for _ in range(dim)]


class HashVectorizer:
    """Deterministic bag-of-words hashing vectorizer."""

    def __init__(self, dim: int = 64):
        self.dim = dim

    def _token_hash(self, token: str) -> int:
        digest = hashlib.sha256(token.encode("utf-8")).hexdigest()
        return int(digest, 16) % self.dim

    def encode(self, text: str) -> List[float]:
        tokens = text.lower().split()
        vec = _zeros(self.dim)
        for tok in tokens:
            vec[self._token_hash(tok)] += 1.0
        norm = math.sqrt(sum(v * v for v in vec))
        return vec if norm == 0 else [v / norm for v in vec]

    def batch_encode(self, texts: Iterable[str]) -> List[List[float]]:
        return [self.encode(t) for t in texts]

    def residual_add(self, hidden: List[float], update: List[float]) -> List[float]:
        if len(hidden) != len(update):
            raise ValueError("hidden and update must have same shape for residual")
        return [h + u for h, u in zip(hidden, update)]
