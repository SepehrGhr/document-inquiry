"""Index building.

A build is a snapshot of which immutable revisions are searchable. Chunks are
stored per profile and revision, so a new build reuses vectors for revisions
that did not change and only embeds new ones. A build is activated only after
every revision it covers has chunks and embeddings.
"""

import hashlib
import logging

from django.db import transaction
from django.db.models import OuterRef, Subquery
from django.utils import timezone

from documents.models import Document, DocumentRevision
from rag import factory
from rag.models import IndexBuild, IndexBuildRevision, IndexedChunk, IndexProfile
from rag.search_text import normalize_search_text

logger = logging.getLogger(__name__)

EMBED_BATCH_SIZE = 32


def latest_revision_ids() -> list:
    """The latest revision ID of every active document."""
    latest = DocumentRevision.objects.filter(document=OuterRef("pk")).order_by("-sequence")
    return list(
        Document.objects.filter(deleted_at__isnull=True)
        .annotate(latest_id=Subquery(latest.values("id")[:1]))
        .values_list("latest_id", flat=True)
    )


def create_build(profile: IndexProfile) -> IndexBuild:
    """Create a pending build covering the current revision of every active document."""
    with transaction.atomic():
        build = IndexBuild.objects.create(profile=profile)
        revision_ids = [rid for rid in latest_revision_ids() if rid is not None]
        revisions = DocumentRevision.objects.filter(id__in=revision_ids)
        IndexBuildRevision.objects.bulk_create(
            [
                IndexBuildRevision(build=build, document_id=r.document_id, revision=r)
                for r in revisions
            ]
        )
    return build


def run_build(build_id, *, activate=True) -> IndexBuild:
    """Embed missing chunks; optionally activate a still-current completed build."""
    build = IndexBuild.objects.select_related("profile").get(pk=build_id)
    if build.status == IndexBuild.Status.COMPLETED:
        return build

    build.status = IndexBuild.Status.RUNNING
    build.started_at = timezone.now()
    build.error = ""
    build.save(update_fields=["status", "started_at", "error"])

    try:
        chunk_count = _index_revisions(build)
    except Exception as exc:
        logger.exception("Index build %s failed", build_id)
        build.status = IndexBuild.Status.FAILED
        build.error = str(exc)[:2000]
        build.save(update_fields=["status", "error"])
        raise

    with transaction.atomic():
        build = IndexBuild.objects.select_for_update().get(pk=build_id)
        build.status = IndexBuild.Status.COMPLETED
        build.chunk_count = chunk_count
        build.completed_at = timezone.now()
        build.save(update_fields=["status", "chunk_count", "completed_at"])
        if activate:
            _activate_if_current(build)
    return build


def _index_revisions(build: IndexBuild) -> int:
    profile = build.profile
    chunker = factory.build_chunker(profile)
    embedder = factory.build_embedder(profile)

    pending = []
    for link in build.revisions.select_related("revision").order_by("revision__document_id"):
        revision = link.revision
        if IndexedChunk.objects.filter(profile=profile, revision=revision).exists():
            continue
        pending.append((link, chunker.split(revision.text)))

    for start in range(0, len(pending), EMBED_BATCH_SIZE):
        batch = pending[start : start + EMBED_BATCH_SIZE]
        texts = [chunk.text for _, chunks in batch for chunk in chunks]
        if not texts:
            continue
        vectors = embedder.embed_documents(texts)
        rows = []
        cursor = 0
        for link, chunks in batch:
            for chunk in chunks:
                rows.append(
                    IndexedChunk(
                        profile=profile,
                        document_id=link.document_id,
                        revision=link.revision,
                        ordinal=chunk.ordinal,
                        text=chunk.text,
                        search_text=normalize_search_text(chunk.text),
                        start=chunk.start,
                        end=chunk.end,
                        content_sha256=hashlib.sha256(chunk.text.encode("utf-8")).hexdigest(),
                        embedding=vectors[cursor],
                    )
                )
                cursor += 1
        IndexedChunk.objects.bulk_create(rows, ignore_conflicts=True)

    return IndexedChunk.objects.filter(
        profile=profile, revision__in=build.revisions.values("revision_id")
    ).count()


def _activate_if_current(build: IndexBuild) -> None:
    """Activate only if the build still covers every current revision."""
    current = {rid for rid in latest_revision_ids() if rid is not None}
    covered = set(build.revisions.values_list("revision_id", flat=True))
    if current != covered:
        logger.info("Build %s is stale; not activating.", build.id)
        return
    IndexBuild.objects.filter(is_active=True).update(is_active=False)
    build.is_active = True
    build.save(update_fields=["is_active"])


def ensure_active_build(profile: IndexProfile) -> IndexBuild:
    """Return the active build, creating and running one synchronously if needed."""
    active = IndexBuild.objects.filter(
        is_active=True, status=IndexBuild.Status.COMPLETED, profile=profile
    ).first()
    if active is not None:
        return active
    build = create_build(profile)
    return run_build(build.id)
