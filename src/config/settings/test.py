import os
import tempfile

from .base import *  # noqa: F403

DEBUG = False
DATABASES["default"]["PORT"] = os.environ.get("TEST_POSTGRES_PORT", "5433")  # noqa: F405
ALLOWED_HOSTS = ["testserver"]
MEDIA_ROOT = tempfile.mkdtemp(prefix="documents-test-media-")
PASSWORD_HASHERS = ["django.contrib.auth.hashers.MD5PasswordHasher"]
