from __future__ import annotations

import hashlib
import math
import re
from typing import Protocol


_TOKEN_RE = re.compile(r"[a-z0-9']+")


class EmbeddingProvider(Protocol):
    model_name: str

    def embed(self, texts: list[str]) -> list[list[float]]: ...


class HashEmbeddingProvider:
    """Deterministic offline embedding used only for tests/plumbing.

    It is not presented as a semantic-quality baseline. The real assessment
    path uses OpenAI embeddings when configured.
    """

    model_name = "hashing-v1"

    def __init__(self, dimensions: int = 256):
        self.dimensions = dimensions

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [self._one(text) for text in texts]

    def _one(self, text: str) -> list[float]:
        vec = [0.0] * self.dimensions
        for token in _TOKEN_RE.findall(text.lower()):
            digest = hashlib.blake2b(token.encode(), digest_size=8).digest()
            idx = int.from_bytes(digest, "big") % self.dimensions
            sign = 1.0 if digest[0] % 2 == 0 else -1.0
            vec[idx] += sign
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]


class OpenAIEmbeddingProvider:
    def __init__(self, model_name: str = "text-embedding-3-small"):
        try:
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover - exercised only with optional dep
            raise RuntimeError("Install companion-memory-core[all] for OpenAI embeddings") from exc
        self.model_name = model_name
        self.client = OpenAI()

    def embed(self, texts: list[str]) -> list[list[float]]:
        response = self.client.embeddings.create(model=self.model_name, input=texts)
        return [item.embedding for item in response.data]


def cosine_similarity(a: list[float], b: list[float]) -> float:
    if len(a) != len(b) or not a:
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)
