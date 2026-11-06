"""Question answering: retrieve, generate, validate citations, persist history."""

import logging
import time

from django.conf import settings
from django.db import transaction

from rag import factory
from rag.adapters import pgvector_index
from rag.citations import validate_citations
from rag.contracts import RetrievedPassage
from rag.exceptions import IndexNotReady, InvalidQuestion, ProviderError
from rag.models import Answer, AnswerSource, Question

logger = logging.getLogger(__name__)

MAX_QUESTION_CHARS = 2000


def ask(*, text, actor) -> Answer:
    text = (text or "").strip()
    if not text:
        raise InvalidQuestion("A question is required.")
    if len(text) > MAX_QUESTION_CHARS:
        raise InvalidQuestion(f"The question must be at most {MAX_QUESTION_CHARS} characters.")

    build = pgvector_index.active_build()
    if build is None:
        raise IndexNotReady("No index is active yet. Index documents and try again.")

    question = Question.objects.create(text=text, asked_by=actor)
    started = time.monotonic()

    try:
        passages = _retrieve(build, text)
        result = factory.build_answer_generator().generate(text, passages)
    except ProviderError as exc:
        logger.warning("Answer generation failed: %s", exc)
        return _persist_failure(question, build, str(exc))

    latency_ms = int((time.monotonic() - started) * 1000)
    return _persist_answer(question, build, passages, result, latency_ms)


def _retrieve(build, text) -> list[RetrievedPassage]:
    embedder = factory.build_embedder(build.profile)
    vector = embedder.embed_query(text)
    return pgvector_index.search(build, vector, settings.RAG["TOP_K"])


def _persist_answer(question, build, passages, result, latency_ms) -> Answer:
    valid, invalid = validate_citations(result.text, len(passages))
    abstained = result.abstained
    reason = result.abstention_reason
    if not abstained and not valid:
        # An answer with no supported citation is not grounded; treat it as abstention.
        abstained = True
        reason = "The answer did not cite any supplied evidence."

    with transaction.atomic():
        answer = Answer.objects.create(
            question=question,
            status=Answer.Status.ABSTAINED if abstained else Answer.Status.ANSWERED,
            text="" if abstained else result.text,
            reason=reason,
            build=build,
            provider=result.provider,
            model=result.model,
            prompt_tokens=result.prompt_tokens,
            completion_tokens=result.completion_tokens,
            latency_ms=result.latency_ms or latency_ms,
        )
        cited_ids = set(result.cited_chunk_ids)
        AnswerSource.objects.bulk_create(
            [
                AnswerSource(
                    answer=answer,
                    chunk_id=passage.chunk_id,
                    document_id=passage.document_id,
                    revision_id=passage.revision_id,
                    document_title=passage.document_title,
                    revision_sequence=passage.revision_sequence,
                    ordinal=passage.ordinal,
                    rank=passage.rank,
                    score=passage.score,
                    cited=passage.chunk_id in cited_ids,
                    text=passage.text,
                )
                for passage in passages
            ]
        )
    if invalid:
        logger.info("Dropped %d unsupported citation(s) for answer %s", len(invalid), answer.id)
    return answer


def _persist_failure(question, build, message) -> Answer:
    return Answer.objects.create(
        question=question,
        status=Answer.Status.FAILED,
        reason=message[:2000],
        build=build,
    )
