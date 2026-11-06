"""Loading the evaluation corpus into the document store.

Documents are created through `documents.services`, so they follow the same
validation and revision rules as any other document. Loading is idempotent by
title: a document that already exists with the same content is left alone, and
one whose content changed gets a new revision.
"""

import logging
from dataclasses import dataclass

from documents import selectors, services
from documents.ingestion.docx import parse_docx
from documents.models import Document
from rag.evaluation.dataset import Corpus

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class LoadResult:
    created: int
    updated: int
    unchanged: int
    removed: int

    def __str__(self):
        return (
            f"created={self.created} updated={self.updated} "
            f"unchanged={self.unchanged} removed={self.removed}"
        )


def load_corpus(corpus: Corpus, *, actor, reset: bool = False) -> LoadResult:
    """Create or update every corpus document. Returns counts of what changed."""
    removed = remove_corpus(corpus, actor=actor) if reset else 0
    created = updated = unchanged = 0

    for entry in corpus.documents:
        text = _read_docx(corpus.root / entry.file)
        existing = selectors.active_documents().filter(title=entry.title).first()
        if existing is None:
            services.create_document(
                title=entry.title,
                description=entry.notes,
                content=services.ContentInput(text=text),
                actor=actor,
            )
            created += 1
            continue

        current = selectors.current_revision(existing)
        if current is not None and current.text == text:
            unchanged += 1
            continue
        services.replace_content(
            document_id=existing.pk,
            content=services.ContentInput(text=text),
            expected_revision=current.sequence if current else 0,
            actor=actor,
        )
        updated += 1

    return LoadResult(created=created, updated=updated, unchanged=unchanged, removed=removed)


def remove_corpus(corpus: Corpus, *, actor) -> int:
    """Logically delete corpus documents, matched by title. Returns the count."""
    titles = {entry.title for entry in corpus.documents}
    removed = 0
    for document in selectors.active_documents().filter(title__in=titles):
        services.delete_document(document_id=document.pk, actor=actor)
        removed += 1
    return removed


def corpus_documents(corpus: Corpus) -> list[Document]:
    """The active documents that belong to this corpus, in manifest order."""
    by_title = {document.title: document for document in selectors.active_documents()}
    return [by_title[entry.title] for entry in corpus.documents if entry.title in by_title]


def _read_docx(path) -> str:
    with open(path, "rb") as handle:
        return parse_docx(handle).text
