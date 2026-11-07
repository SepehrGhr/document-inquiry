import uuid

from django.conf import settings
from django.db import models
from django.db.models import Q


class Document(models.Model):
    """A document's stable identity and metadata.

    Content lives in append-only revisions; the current content is the revision
    with the highest sequence. Deletion is logical so history stays intact.
    """

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    title = models.CharField(max_length=255)
    description = models.TextField(blank=True, default="")
    created_by = models.ForeignKey(
        settings.AUTH_USER_MODEL, null=True, on_delete=models.SET_NULL, related_name="+"
    )
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    deleted_at = models.DateTimeField(null=True, blank=True)
    deleted_by = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        null=True,
        blank=True,
        on_delete=models.SET_NULL,
        related_name="+",
    )

    class Meta:
        ordering = ["-created_at", "-id"]

    def __str__(self):
        return self.title


class DocumentRevision(models.Model):
    """An immutable snapshot of a document's full text and its source."""

    class Source(models.TextChoices):
        TEXT = "text", "Plain text"
        DOCX = "docx", "DOCX upload"

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    document = models.ForeignKey(Document, on_delete=models.CASCADE, related_name="revisions")
    sequence = models.PositiveIntegerField()
    text = models.TextField()
    source = models.CharField(max_length=8, choices=Source.choices)
    file = models.FileField(upload_to="documents/%Y/%m/", blank=True)
    original_filename = models.CharField(max_length=255, blank=True, default="")
    file_size = models.PositiveBigIntegerField(null=True, blank=True)
    content_sha256 = models.CharField(max_length=64)
    parser_version = models.CharField(max_length=32, blank=True, default="")
    created_by = models.ForeignKey(
        settings.AUTH_USER_MODEL, null=True, on_delete=models.SET_NULL, related_name="+"
    )
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["document", "-sequence"]
        constraints = [
            models.UniqueConstraint(
                fields=["document", "sequence"], name="unique_revision_sequence"
            ),
            models.CheckConstraint(condition=Q(sequence__gte=1), name="revision_sequence_positive"),
            models.CheckConstraint(
                condition=(
                    Q(source="text", file="", file_size__isnull=True)
                    | Q(source="docx", file__gt="", file_size__isnull=False)
                ),
                name="revision_source_matches_file",
            ),
        ]

    def __str__(self):
        return f"{self.document.title} (revision {self.sequence})"
