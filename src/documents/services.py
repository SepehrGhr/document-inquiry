"""Document lifecycle operations.

Every write path (REST API and Django Admin) goes through these functions so
validation, revision rules, and locking cannot diverge between interfaces.
"""

import hashlib
import logging
from dataclasses import dataclass

from django.db import transaction
from django.utils import timezone

from documents.exceptions import (
    DocumentNotFound,
    InvalidDocumentInput,
    RevisionConflict,
)
from documents.ingestion.docx import parse_docx
from documents.ingestion.types import ParsedDocument
from documents.models import Document, DocumentRevision

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ContentInput:
    """Exactly one of text or docx_file must be provided."""

    text: str | None = None
    docx_file: object | None = None


def create_document(*, title, description, content: ContentInput, actor) -> Document:
    title = _clean_title(title)
    parsed, upload = _prepare_content(content)

    with transaction.atomic():
        document = Document.objects.create(
            title=title, description=description or "", created_by=actor
        )
        _store_revision(document=document, sequence=1, parsed=parsed, upload=upload, actor=actor)
    return document


def update_metadata(*, document_id, title=None, description=None, actor) -> Document:
    with transaction.atomic():
        document = _lock_active_document(document_id)
        if title is not None:
            document.title = _clean_title(title)
        if description is not None:
            document.description = description
        document.save(update_fields=["title", "description", "updated_at"])
    return document


def replace_content(*, document_id, content: ContentInput, expected_revision, actor):
    parsed, upload = _prepare_content(content)

    with transaction.atomic():
        document = _lock_active_document(document_id)
        current = document.revisions.order_by("-sequence").first()
        if current is None or current.sequence != expected_revision:
            raise RevisionConflict(
                f"Document has changed; current revision is {current.sequence if current else 0}."
            )
        revision = _store_revision(
            document=document,
            sequence=current.sequence + 1,
            parsed=parsed,
            upload=upload,
            actor=actor,
        )
        Document.objects.filter(pk=document.pk).update(updated_at=timezone.now())
    return revision


def delete_document(*, document_id, actor) -> None:
    with transaction.atomic():
        document = _lock_active_document(document_id)
        document.deleted_at = timezone.now()
        document.deleted_by = actor
        document.save(update_fields=["deleted_at", "deleted_by", "updated_at"])


def _lock_active_document(document_id) -> Document:
    document = Document.objects.select_for_update().filter(pk=document_id).first()
    if document is None or document.deleted_at is not None:
        raise DocumentNotFound("Document not found.")
    return document


def _clean_title(title) -> str:
    title = (title or "").strip()
    if not title:
        raise InvalidDocumentInput("A title is required.")
    if len(title) > 255:
        raise InvalidDocumentInput("The title must be at most 255 characters.")
    return title


def _prepare_content(content: ContentInput):
    """Validate and parse content before any database lock is taken."""
    has_text = content.text is not None
    has_file = content.docx_file is not None
    if has_text == has_file:
        raise InvalidDocumentInput("Provide exactly one of text or a DOCX file.")

    if has_text:
        if not content.text.strip():
            raise InvalidDocumentInput("The document text is empty.")
        return ParsedDocument(text=content.text, parser_version=""), None

    upload = content.docx_file
    if not upload.name.lower().endswith(".docx"):
        raise InvalidDocumentInput("Only .docx files are supported.")
    return parse_docx(upload), upload


def _store_revision(*, document, sequence, parsed, upload, actor) -> DocumentRevision:
    file_name = ""
    file_size = None
    if upload is not None:
        upload.seek(0, 2)
        file_size = upload.tell()
        upload.seek(0)
        upload.name = _safe_filename(upload.name)

    revision = DocumentRevision(
        document=document,
        sequence=sequence,
        text=parsed.text,
        source=DocumentRevision.Source.DOCX if upload else DocumentRevision.Source.TEXT,
        original_filename=upload.name if upload else "",
        file_size=file_size,
        content_sha256=hashlib.sha256(parsed.text.encode("utf-8")).hexdigest(),
        parser_version=parsed.parser_version,
        created_by=actor,
    )
    if upload is not None:
        revision.file.save(upload.name, upload, save=False)
        file_name = revision.file.name
    try:
        revision.save()
    except Exception:
        if file_name:
            _delete_quietly(revision.file)
        raise
    return revision


def _safe_filename(name: str) -> str:
    return name.replace("\\", "/").rsplit("/", 1)[-1][:255] or "document.docx"


def _delete_quietly(file_field) -> None:
    try:
        file_field.storage.delete(file_field.name)
    except Exception:
        logger.exception("Failed to remove orphaned upload %s", file_field.name)
