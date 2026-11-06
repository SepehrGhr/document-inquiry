"""Celery tasks for background indexing."""

import logging

from celery import shared_task

from rag import indexing

logger = logging.getLogger(__name__)


@shared_task(bind=True, max_retries=3, default_retry_delay=30)
def run_index_build(self, build_id):
    """Embed and activate one build. Safe to retry: completed builds are no-ops."""
    try:
        build = indexing.run_build(build_id)
    except Exception as exc:
        logger.exception("Index build %s failed", build_id)
        raise self.retry(exc=exc) from exc
    logger.info("Index build %s finished with status %s", build.id, build.status)
    return str(build.id)
