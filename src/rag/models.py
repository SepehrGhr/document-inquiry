import uuid

from django.conf import settings
from django.db import models
from pgvector.django import VectorField

from documents.models import Document, DocumentRevision


class IndexProfile(models.Model):
    """The complete identity of an embedding/index configuration.

    Profiles are immutable once created. Changing the model, dimension, metric,
    normalization, parser, or chunker means creating a new profile and building
    a new index — never editing this row.
    """

    class Metric(models.TextChoices):
        COSINE = "cosine", "Cosine"
        L2 = "l2", "Euclidean"
        INNER_PRODUCT = "ip", "Inner product"

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    name = models.CharField(max_length=100, unique=True)
    provider = models.CharField(max_length=50)
    model_id = models.CharField(max_length=200)
    model_revision = models.CharField(max_length=100, blank=True, default="")
    dimension = models.PositiveIntegerField()
    metric = models.CharField(max_length=10, choices=Metric.choices, default=Metric.COSINE)
    normalize = models.BooleanField(default=True)
    parser_version = models.CharField(max_length=32, blank=True, default="")
    chunker_version = models.CharField(max_length=32)
    chunk_size = models.PositiveIntegerField()
    chunk_overlap = models.PositiveIntegerField()
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["-created_at", "-id"]
        constraints = [
            models.UniqueConstraint(
                fields=[
                    "provider",
                    "model_id",
                    "model_revision",
                    "dimension",
                    "metric",
                    "normalize",
                    "parser_version",
                    "chunker_version",
                    "chunk_size",
                    "chunk_overlap",
                ],
                name="unique_index_profile_identity",
            ),
            models.CheckConstraint(
                condition=models.Q(chunk_overlap__lt=models.F("chunk_size")),
                name="index_profile_overlap_below_size",
            ),
        ]

    def __str__(self):
        return self.name


class IndexBuild(models.Model):
    """A snapshot of which immutable revisions are searchable.

    A build becomes active only after every revision it covers has chunks and
    embeddings. At most one build is active at a time; retrieval always filters
    through the active build's revision set.
    """

    class Status(models.TextChoices):
        PENDING = "pending", "Pending"
        RUNNING = "running", "Running"
        COMPLETED = "completed", "Completed"
        FAILED = "failed", "Failed"

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    profile = models.ForeignKey(IndexProfile, on_delete=models.PROTECT, related_name="builds")
    status = models.CharField(max_length=10, choices=Status.choices, default=Status.PENDING)
    is_active = models.BooleanField(default=False)
    error = models.TextField(blank=True, default="")
    chunk_count = models.PositiveIntegerField(default=0)
    created_at = models.DateTimeField(auto_now_add=True)
    started_at = models.DateTimeField(null=True, blank=True)
    completed_at = models.DateTimeField(null=True, blank=True)

    class Meta:
        ordering = ["-created_at", "-id"]
        constraints = [
            models.UniqueConstraint(
                fields=["is_active"],
                condition=models.Q(is_active=True),
                name="single_active_index_build",
            ),
        ]

    def __str__(self):
        return f"{self.profile.name} build {self.id} ({self.status})"


class IndexBuildRevision(models.Model):
    """One immutable revision covered by a build."""

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    build = models.ForeignKey(IndexBuild, on_delete=models.CASCADE, related_name="revisions")
    document = models.ForeignKey(Document, on_delete=models.CASCADE, related_name="+")
    revision = models.ForeignKey(DocumentRevision, on_delete=models.CASCADE, related_name="+")

    class Meta:
        constraints = [
            models.UniqueConstraint(fields=["build", "revision"], name="unique_build_revision"),
            models.UniqueConstraint(fields=["build", "document"], name="unique_build_document"),
        ]

    def __str__(self):
        return f"{self.document_id} r{self.revision.sequence}"


class IndexedChunk(models.Model):
    """A chunk of one immutable revision, with its embedding.

    Chunks are keyed by profile and revision, not by build, so a new build can
    reuse vectors for revisions that did not change instead of re-embedding the
    whole corpus on every document write.
    """

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    profile = models.ForeignKey(IndexProfile, on_delete=models.CASCADE, related_name="chunks")
    document = models.ForeignKey(Document, on_delete=models.CASCADE, related_name="+")
    revision = models.ForeignKey(DocumentRevision, on_delete=models.CASCADE, related_name="+")
    ordinal = models.PositiveIntegerField()
    text = models.TextField()
    start = models.PositiveIntegerField()
    end = models.PositiveIntegerField()
    content_sha256 = models.CharField(max_length=64)
    embedding = VectorField(dimensions=None)

    class Meta:
        ordering = ["document", "ordinal"]
        constraints = [
            models.UniqueConstraint(
                fields=["profile", "revision", "ordinal"], name="unique_chunk_per_profile_revision"
            ),
        ]

    def __str__(self):
        return f"{self.document_id} #{self.ordinal}"


class Question(models.Model):
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    text = models.TextField()
    asked_by = models.ForeignKey(
        settings.AUTH_USER_MODEL, null=True, on_delete=models.SET_NULL, related_name="+"
    )
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["-created_at", "-id"]

    def __str__(self):
        return self.text[:80]


class Answer(models.Model):
    """An append-only answer attempt for a question."""

    class Status(models.TextChoices):
        ANSWERED = "answered", "Answered"
        ABSTAINED = "abstained", "Abstained"
        FAILED = "failed", "Failed"

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    question = models.ForeignKey(Question, on_delete=models.CASCADE, related_name="answers")
    status = models.CharField(max_length=10, choices=Status.choices)
    text = models.TextField(blank=True, default="")
    reason = models.TextField(blank=True, default="")
    build = models.ForeignKey(
        IndexBuild, null=True, on_delete=models.SET_NULL, related_name="answers"
    )
    provider = models.CharField(max_length=50, blank=True, default="")
    model = models.CharField(max_length=200, blank=True, default="")
    prompt_tokens = models.PositiveIntegerField(null=True, blank=True)
    completion_tokens = models.PositiveIntegerField(null=True, blank=True)
    latency_ms = models.PositiveIntegerField(null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["-created_at", "-id"]

    def __str__(self):
        return f"{self.question_id} ({self.status})"


class AnswerSource(models.Model):
    """A passage the answer was allowed to cite, kept for auditability."""

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    answer = models.ForeignKey(Answer, on_delete=models.CASCADE, related_name="sources")
    chunk = models.ForeignKey(IndexedChunk, null=True, on_delete=models.SET_NULL, related_name="+")
    document = models.ForeignKey(Document, null=True, on_delete=models.SET_NULL, related_name="+")
    revision = models.ForeignKey(
        DocumentRevision, null=True, on_delete=models.SET_NULL, related_name="+"
    )
    document_title = models.CharField(max_length=255)
    revision_sequence = models.PositiveIntegerField()
    ordinal = models.PositiveIntegerField()
    rank = models.PositiveIntegerField()
    score = models.FloatField()
    cited = models.BooleanField(default=False)
    text = models.TextField()

    class Meta:
        ordering = ["answer", "rank"]

    def __str__(self):
        return f"{self.document_title} #{self.ordinal}"
