"""Schedules indexing after a document revision is committed.

Called from `documents.services` so the REST API and Admin share one trigger.
"""

import logging

from django.conf import settings
from django.db import transaction

from rag import indexing
from rag.models import IndexBuild, IndexProfile

logger = logging.getLogger(__name__)


def schedule_indexing() -> None:
    """Queue a build for the active profile once the current transaction commits."""
    if not settings.RAG["AUTO_INDEX"]:
        return
    transaction.on_commit(_enqueue_build)


def _enqueue_build() -> None:
    profile = IndexProfile.objects.filter(name=settings.RAG["ACTIVE_PROFILE"]).first()
    if profile is None:
        logger.warning(
            "No index profile named %r; skipping indexing.", settings.RAG["ACTIVE_PROFILE"]
        )
        return
    if IndexBuild.objects.filter(
        profile=profile, status__in=[IndexBuild.Status.PENDING, IndexBuild.Status.RUNNING]
    ).exists():
        # A build already in flight will pick up the new revision when it runs.
        return
    build = indexing.create_build(profile)
    if not build.revisions.exists():
        build.delete()
        return
    from rag.tasks import run_index_build

    run_index_build.delay(str(build.id))
