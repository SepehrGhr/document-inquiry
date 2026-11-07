"""Running the evaluation.

The retrieval pass embeds each question and searches the active index without
calling the answer model, so retrieval changes can be measured cheaply. The
answer pass (added later) goes through the real `services.ask()` path.
"""

import hashlib
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

from django.conf import settings
from django.db.models import Sum
from django.db.models.functions import Length

from documents.ingestion.docx import parse_docx
from rag import factory
from rag.adapters.chunk_queries import chunks_for_build
from rag.contracts import RetrievedPassage
from rag.evaluation.dataset import Corpus, CorpusError, EvalQuestion
from rag.evaluation.evidence import context_prefix
from rag.evaluation.metrics import RetrievalScore
from rag.models import IndexBuild

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RetrievalResult:
    question: EvalQuestion
    retrieved_documents: tuple[str, ...]
    retrieved_titles: tuple[str, ...]
    latency_ms: int
    passages: tuple[RetrievedPassage, ...] = ()
    retrieved_count: int = 0

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
    corpus: Corpus | None = None
    split: str = "all"
    context_chars: int | None = None
    requested_ids: tuple[str, ...] = ()
    provenance: dict = field(default_factory=dict)
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
    split: str = "all",
    context_chars: int | None = None,
    limit: int = 0,
    delay: float = 0.0,
    progress=None,
) -> RetrievalRun:
    """Search the index for selected questions, retaining raw passage positions."""
    if top_k < 1 or limit < 0 or delay < 0 or (context_chars is not None and context_chars < 1):
        raise CorpusError("top_k/context_chars must be positive; limit/delay cannot be negative.")
    questions = select_questions(corpus, split=split, limit=limit)
    if corpus.evidence:
        _validate_evidence_build(corpus, build)
    retriever = factory.build_retriever(
        build,
        mode=mode,
        candidate_k=candidate_k,
        rank_constant=rank_constant,
        reranker=reranker,
        rerank_candidate_k=rerank_candidate_k,
    )
    rerank_pool = max(rerank_candidate_k, top_k) if reranker != "none" else 0
    run = RetrievalRun(
        build=build,
        retrieval_mode=mode,
        candidate_k=max(candidate_k, top_k, rerank_pool),
        rank_constant=rank_constant,
        reranker=reranker,
        rerank_model=settings.RAG["RERANK_MODEL"] if reranker != "none" else "",
        rerank_candidate_k=rerank_pool,
        corpus=corpus,
        split=split,
        context_chars=context_chars,
        requested_ids=tuple(q.id for q in questions),
        provenance=_provenance(corpus, build),
    )

    for index, question in enumerate(questions, start=1):
        started = time.monotonic()
        try:
            retrieved = retriever.retrieve(question.question, limit=top_k)
            passages = context_prefix(retrieved, context_chars)
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
                passages=tuple(passages),
                retrieved_count=len(retrieved),
            )
        )
        if progress is not None:
            progress(index, len(questions), question, run.results[-1])
        if delay:
            time.sleep(delay)

    return run


def select_questions(corpus, *, split="all", limit=0):
    if split == "all":
        questions = corpus.questions
    else:
        if split not in corpus.splits:
            raise CorpusError(f"Split {split!r} is not defined in splits.json.")
        ids = set(corpus.splits[split])
        questions = tuple(q for q in corpus.questions if q.id in ids)
    if not questions:
        raise CorpusError(f"Split {split!r} has no questions.")
    return questions[:limit] if limit else questions


def _validate_evidence_build(corpus, build):
    """Refuse misleading offset scores against stale or differently parsed text."""
    referenced = {s.file for annotation in corpus.evidence.values() for s in annotation.spans}
    documents = {d.title: d for d in corpus.documents if d.file in referenced}
    rows = list(
        chunks_for_build(build)
        .filter(document__title__in=documents)
        .values(
            "document_id",
            "document__title",
            "revision_id",
            "revision__text",
            "text",
            "start",
            "end",
        )
    )
    by_title = {}
    for row in rows:
        by_title.setdefault(row["document__title"], []).append(row)
    for title, document in documents.items():
        chunks = by_title.get(title, [])
        if not chunks or len({r["document_id"] for r in chunks}) != 1:
            raise CorpusError(f"Evidence document {title!r} is missing or ambiguous in this build.")
        with (corpus.root / document.file).open("rb") as stream:
            text = parse_docx(stream).text
        if any(r["revision__text"] != text for r in chunks):
            raise CorpusError(
                f"Indexed revision for {document.file} differs from the corpus. Reload/rebuild it."
            )
        if any(
            r["start"] < 0 or r["end"] > len(text) or text[r["start"] : r["end"]] != r["text"]
            for r in chunks
        ):
            raise CorpusError(
                f"Chunk offsets for {document.file} do not match source text. "
                "Rebuild with corrected offsets."
            )


def _provenance(corpus, build):
    hashes = {}
    if corpus.root is not None:
        for name in ("manifest.json", "questions.jsonl", "evidence.jsonl", "splits.json"):
            path = corpus.root / name
            if path.is_file():
                hashes[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    chunks = chunks_for_build(build)
    inventory = list(
        chunks.values("document_id", "document__title", "revision_id", "revision__content_sha256")
        .distinct()
        .order_by("document_id")
    )
    source = hashlib.sha256()
    source_root = Path(__file__).resolve().parents[1]
    for path in sorted(source_root.rglob("*.py")):
        source.update(str(path.relative_to(source_root)).encode())
        source.update(path.read_bytes())
    return {
        "source_sha256": source.hexdigest(),
        "corpus_hashes": hashes,
        "index_chunks": chunks.count(),
        "index_characters": chunks.aggregate(chars=Sum(Length("text")))["chars"] or 0,
        "index_documents": [{k: str(v) for k, v in row.items()} for row in inventory],
    }


def _document_files(passages, corpus: Corpus) -> tuple[str, ...]:
    """Map retrieved passages to manifest file paths, in rank order, deduplicated."""
    by_title = {entry.title: entry.file for entry in corpus.documents}
    files = (by_title.get(passage.document_title) for passage in passages)
    return tuple(dict.fromkeys(file for file in files if file))
