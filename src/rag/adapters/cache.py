"""A disk cache for embeddings.

Free embedding models are rate limited, and evaluation re-embeds the same
questions on every run. The cache is keyed by model and exact text, so a
re-run only pays for text it has never seen. It is a plain directory of JSON
files: no database, no dependency, safe to delete at any time.
"""

import hashlib
import json
import logging
import os
import tempfile
from pathlib import Path

logger = logging.getLogger(__name__)


class EmbeddingCache:
    def __init__(self, root):
        self.root = Path(root)

    def _path(self, model: str, text: str) -> Path:
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
        model_key = hashlib.sha256(model.encode("utf-8")).hexdigest()[:16]
        return self.root / model_key / digest[:2] / f"{digest}.json"

    def get(self, model: str, text: str) -> list[float] | None:
        path = self._path(model, text)
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return None
        except (OSError, json.JSONDecodeError):
            logger.warning("Ignoring unreadable embedding cache entry %s", path)
            return None

    def put(self, model: str, text: str, vector: list[float]) -> None:
        path = self._path(model, text)
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            # Write then rename so a crash never leaves a half-written entry.
            handle, temporary = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
            with os.fdopen(handle, "w", encoding="utf-8") as stream:
                json.dump(vector, stream)
            os.replace(temporary, path)
        except OSError:
            logger.warning("Could not write embedding cache entry %s", path)


class CachedEmbedder:
    """Wraps an embedder, serving repeated text from the cache."""

    def __init__(self, embedder, cache: EmbeddingCache, model: str):
        self._embedder = embedder
        self._cache = cache
        self._model = model
        self.dimension = embedder.dimension

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        missing = [text for text in texts if self._cache.get(self._model, text) is None]
        if missing:
            for text, vector in zip(missing, self._embedder.embed_documents(missing), strict=True):
                self._cache.put(self._model, text, vector)
        return [self._cache.get(self._model, text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        cached = self._cache.get(self._model, text)
        if cached is not None:
            return cached
        vector = self._embedder.embed_query(text)
        self._cache.put(self._model, text, vector)
        return vector
