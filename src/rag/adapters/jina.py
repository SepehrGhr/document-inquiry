"""Jina embeddings adapter.

Jina's API is OpenAI-shaped but takes a `task` parameter that tells the model
whether the text is a document or a query. Using the right task for each side
is the main reason to prefer this adapter over a generic OpenAI-compatible one.
"""

import httpx

from rag.exceptions import ProviderError

DOCUMENT_TASK = "retrieval.passage"
QUERY_TASK = "retrieval.query"


class JinaEmbedder:
    def __init__(self, *, api_key, base_url, model, dimension, timeout=60):
        self.dimension = dimension
        self._model = model
        self._client = httpx.Client(
            base_url=base_url.rstrip("/"),
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=timeout,
        )

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._embed(texts, DOCUMENT_TASK)

    def embed_query(self, text: str) -> list[float]:
        return self._embed([text], QUERY_TASK)[0]

    def _embed(self, texts: list[str], task: str) -> list[list[float]]:
        try:
            response = self._client.post(
                "/embeddings",
                json={"model": self._model, "input": texts, "task": task},
            )
            response.raise_for_status()
            data = response.json()["data"]
        except (httpx.HTTPError, KeyError, ValueError) as exc:
            raise ProviderError(f"Embedding request failed: {exc}") from exc

        vectors = [item["embedding"] for item in sorted(data, key=lambda item: item["index"])]
        if len(vectors) != len(texts):
            raise ProviderError(
                f"Embedding model returned {len(vectors)} vectors for {len(texts)} inputs."
            )
        for vector in vectors:
            if len(vector) != self.dimension:
                raise ProviderError(
                    f"Embedding model returned {len(vector)} dimensions, "
                    f"but the index profile expects {self.dimension}."
                )
        return vectors
