import os
import tempfile

from .base import *  # noqa: F403

DEBUG = False
DATABASES["default"]["PORT"] = os.environ.get("TEST_POSTGRES_PORT", "5433")  # noqa: F405
ALLOWED_HOSTS = ["testserver"]
MEDIA_ROOT = tempfile.mkdtemp(prefix="documents-test-media-")
PASSWORD_HASHERS = ["django.contrib.auth.hashers.MD5PasswordHasher"]

# Tests never touch Redis or a provider; indexing runs inline with fake adapters.
CELERY_TASK_ALWAYS_EAGER = True
CELERY_TASK_EAGER_PROPAGATES = True
RAG["AUTO_INDEX"] = False  # noqa: F405
RAG["ANSWER_PROVIDER"] = "fake"  # noqa: F405
RAG["EMBEDDING_CACHE"] = False  # noqa: F405
RAG["RERANKER"] = "none"  # noqa: F405
