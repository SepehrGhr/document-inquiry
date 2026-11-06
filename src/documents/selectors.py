"""Read-side queries for active documents."""

from django.db.models import OuterRef, Subquery

from documents.models import Document, DocumentRevision


def active_documents():
    return Document.objects.filter(deleted_at__isnull=True)


def with_current_revision(queryset=None):
    """Annotate each document with its current revision number and source."""
    queryset = queryset if queryset is not None else active_documents()
    latest = DocumentRevision.objects.filter(document=OuterRef("pk")).order_by("-sequence")
    return queryset.annotate(
        current_revision=Subquery(latest.values("sequence")[:1]),
        current_source=Subquery(latest.values("source")[:1]),
    )


def get_active_document(document_id) -> Document | None:
    return active_documents().filter(pk=document_id).first()


def current_revision(document: Document) -> DocumentRevision | None:
    return document.revisions.order_by("-sequence").first()


def revisions(document: Document):
    return document.revisions.order_by("-sequence")
