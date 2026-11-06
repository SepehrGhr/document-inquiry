"""Deterministic adapters for tests and offline development.

They implement the same contracts as the OpenRouter adapters without network
access, so the default test suite never depends on credentials or model
availability.
"""

import hashlib
import math
import re

from rag.contracts import AnswerResult, RetrievedPassage

_TOKEN = re.compile(r"\w+", re.UNICODE)


def _tokens(text: str) -> list[str]:
    return [token.lower() for token in _TOKEN.findall(text)]


class HashingEmbedder:
    """Bag-of-words hashing embedder: similar text yields similar vectors."""

    def __init__(self, dimension: int = 64):
        self.dimension = dimension

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self.embed_query(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        vector = [0.0] * self.dimension
        for token in _tokens(text):
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            index = int.from_bytes(digest[:4], "big") % self.dimension
            vector[index] += 1.0
        norm = math.sqrt(sum(value * value for value in vector))
        if norm:
            vector = [value / norm for value in vector]
        return vector


class EchoAnswerGenerator:
    """Cites the best passage and echoes its first sentence."""

    provider = "fake"
    model = "echo"

    def generate(self, question: str, passages: list[RetrievedPassage]) -> AnswerResult:
        if not passages:
            return AnswerResult(
                text="",
                cited_chunk_ids=(),
                abstained=True,
                abstention_reason="No evidence was retrieved.",
                provider=self.provider,
                model=self.model,
            )
        best = passages[0]
        sentence = best.text.strip().split(". ")[0]
        return AnswerResult(
            text=f"{sentence} [S1]",
            cited_chunk_ids=(best.chunk_id,),
            provider=self.provider,
            model=self.model,
        )
