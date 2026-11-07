"""Running the evaluation.

The retrieval pass embeds each question and searches the active index without
calling the answer model, so retrieval changes can be measured cheaply. The
answer pass (added later) goes through the real `services.ask()` path.
"""

import logging
import time
from dataclasses import dataclass, field

from django.conf import settings

from rag import factory
from rag.evaluation.dataset import Corpus, EvalQuestion
from rag.evaluation.metrics import RetrievalScore
from rag.models import IndexBuild

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RetrievalResult:
    question: EvalQuestion
    retrieved_documents: tuple[str, ...]
    retrieved_titles: tuple[str, ...]
    latency_ms: int

    @property
    def score(self) -> RetrievalScore:
        return RetrievalScore(
            retrieved=self.retrieved_documents, expected=self.question.expected_sources
        )


@dataclass
class RetrievalRun:
    build: IndexBuild
    retrieval_mode: str = "dense"
    candidate_k: int = 20
    rank_constant: int = 60
    reranker: str = "none"
    rerank_model: str = ""
    rerank_candidate_k: int = 20
    results: list[RetrievalResult] = field(default_factory=list)
    errors: list[tuple[str, str]] = field(default_factory=list)


def run_retrieval(
    corpus: Corpus,
    build: IndexBuild,
    *,
    top_k: int,
    mode: str = "dense",
    candidate_k: int = 20,
    rank_constant: int = 60,
    reranker: str = "none",
    rerank_candidate_k: int = 20,
    limit: int = 0,
    delay: float = 0.0,
    progress=None,
) -> RetrievalRun:
    """Search the index for every question and record which documents came back."""
    retriever = factory.build_retriever(
        build,
        mode=mode,
        candidate_k=candidate_k,
        rank_constant=rank_constant,
        reranker=reranker,
        rerank_candidate_k=rerank_candidate_k,
    )
    questions = corpus.questions[:limit] if limit else corpus.questions
    rerank_pool = max(rerank_candidate_k, top_k) if reranker != "none" else 0
    run = RetrievalRun(
        build=build,
        retrieval_mode=mode,
        candidate_k=max(candidate_k, top_k, rerank_pool),
        rank_constant=rank_constant,
        reranker=reranker,
        rerank_model=settings.RAG["RERANK_MODEL"] if reranker != "none" else "",
        rerank_candidate_k=rerank_pool,
    )

    for index, question in enumerate(questions, start=1):
        started = time.monotonic()
        try:
            passages = retriever.retrieve(question.question, limit=top_k)
        except Exception as exc:
            logger.warning("Retrieval failed for %s: %s", question.id, exc)
            run.errors.append((question.id, str(exc)))
            continue
        latency_ms = int((time.monotonic() - started) * 1000)

        run.results.append(
            RetrievalResult(
                question=question,
                retrieved_documents=_document_files(passages, corpus),
                retrieved_titles=tuple(dict.fromkeys(p.document_title for p in passages)),
                latency_ms=latency_ms,
            )
        )
        if progress is not None:
            progress(index, len(questions), question, run.results[-1])
        if delay:
            time.sleep(delay)

    return run


def _document_files(passages, corpus: Corpus) -> tuple[str, ...]:
    """Map retrieved passages to manifest file paths, in rank order, deduplicated."""
    by_title = {entry.title: entry.file for entry in corpus.documents}
    files = (by_title.get(passage.document_title) for passage in passages)
    return tuple(dict.fromkeys(file for file in files if file))
