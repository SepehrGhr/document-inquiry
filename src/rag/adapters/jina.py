"""Jina embedding and reranking adapters.

Embedding requests use task-specific query/passage representations. Reranking
scores the original candidate texts and maps results back by their input index.
"""

import math
from dataclasses import replace

import httpx

from rag.contracts import RetrievedPassage
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


class JinaReranker:
    def __init__(self, *, api_key, base_url, model, timeout=60):
        self._api_key = api_key
        self._url = f"{base_url.rstrip('/')}/rerank"
        self._model = model
        self._timeout = timeout

    def rerank(
        self, question: str, passages: list[RetrievedPassage], *, limit: int
    ) -> list[RetrievedPassage]:
        if limit < 1:
            raise ValueError("limit must be positive.")
        if not passages:
            return []
        top_n = min(limit, len(passages))
        try:
            # A one-shot request closes its client after use; retrievers are built
            # per API request, so retaining an unclosed connection pool would leak.
            response = httpx.post(
                self._url,
                headers={"Authorization": f"Bearer {self._api_key}"},
                json={
                    "model": self._model,
                    "query": question,
                    "documents": [passage.text for passage in passages],
                    "top_n": top_n,
                    "return_documents": False,
                },
                timeout=self._timeout,
            )
            response.raise_for_status()
            results = response.json()["results"]
            if not isinstance(results, list) or len(results) != top_n:
                raise ValueError("Unexpected number of reranking results.")
            seen = set()
            ranked = []
            for rank, result in enumerate(results, start=1):
                index, score = result["index"], result["relevance_score"]
                if type(index) is not int or not 0 <= index < len(passages) or index in seen:
                    raise ValueError("Invalid or duplicate reranking index.")
                if type(score) not in (int, float) or not math.isfinite(score):
                    raise ValueError("Invalid reranking score.")
                seen.add(index)
                ranked.append(replace(passages[index], rank=rank, score=float(score)))
            return ranked
        except (httpx.HTTPError, KeyError, TypeError, ValueError) as exc:
            raise ProviderError(f"Reranking request failed: {exc}") from exc
