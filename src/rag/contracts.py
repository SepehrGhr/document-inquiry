"""Application-owned types and interfaces for the RAG pipeline.

Nothing in this module imports Django, LangChain, or a provider SDK. Adapters
translate between these types and their framework-specific objects, so the
retrieval and generation stack can be replaced without touching the API or
database layers.
"""

from dataclasses import dataclass, field
from typing import Protocol


@dataclass(frozen=True)
class Chunk:
    """A deterministic slice of one immutable document revision."""

    ordinal: int
    text: str
    start: int
    end: int


@dataclass(frozen=True)
class RetrievedPassage:
    """One piece of evidence returned by retrieval, with citation metadata."""

    chunk_id: str
    document_id: str
    document_title: str
    revision_id: str
    revision_sequence: int
    ordinal: int
    text: str
    score: float
    rank: int
    start: int | None = None
    end: int | None = None


@dataclass(frozen=True)
class AnswerResult:
    """The generator's output before citation validation and persistence."""

    text: str
    cited_chunk_ids: tuple[str, ...]
    abstained: bool = False
    abstention_reason: str = ""
    provider: str = ""
    model: str = ""
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    latency_ms: int | None = None
    metadata: dict = field(default_factory=dict)


class Chunker(Protocol):
    version: str

    def split(self, text: str) -> list[Chunk]: ...


class Embedder(Protocol):
    """Turns text into vectors. All vectors must share one dimension."""

    dimension: int

    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...


class AnswerGenerator(Protocol):
    """Answers a question strictly from the supplied evidence passages."""

    def generate(self, question: str, passages: list[RetrievedPassage]) -> AnswerResult: ...


class Retriever(Protocol):
    def retrieve(self, text: str, *, limit: int) -> list[RetrievedPassage]: ...


class Reranker(Protocol):
    def rerank(
        self, question: str, passages: list[RetrievedPassage], *, limit: int
    ) -> list[RetrievedPassage]: ...
