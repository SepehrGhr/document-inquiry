import httpx
import pytest

from rag.adapters import jina
from rag.exceptions import ProviderError


def _embedder(handler, dimension=3):
    embedder = jina.JinaEmbedder(
        api_key="k", base_url="https://example.invalid/v1", model="m", dimension=dimension
    )
    embedder._client = httpx.Client(
        base_url="https://example.invalid/v1",
        transport=httpx.MockTransport(handler),
    )
    return embedder


def _responder(vectors, captured):
    def handler(request):
        captured.append(request)
        data = [{"index": index, "embedding": vector} for index, vector in enumerate(vectors)]
        return httpx.Response(200, json={"data": data})

    return handler


def test_documents_use_the_passage_task():
    captured = []
    embedder = _embedder(_responder([[1.0, 2.0, 3.0]], captured))

    assert embedder.embed_documents(["a"]) == [[1.0, 2.0, 3.0]]
    assert b'"task":"retrieval.passage"' in captured[0].content


def test_query_uses_the_query_task():
    captured = []
    embedder = _embedder(_responder([[1.0, 2.0, 3.0]], captured))

    assert embedder.embed_query("q") == [1.0, 2.0, 3.0]
    assert b'"task":"retrieval.query"' in captured[0].content


def test_vectors_are_returned_in_input_order():
    captured = []
    embedder = _embedder(_responder([[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]], captured))

    assert embedder.embed_documents(["a", "b"]) == [[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]]


def test_wrong_dimension_is_rejected():
    embedder = _embedder(_responder([[1.0, 2.0]], []))

    with pytest.raises(ProviderError, match="dimensions"):
        embedder.embed_query("q")


def test_http_error_is_wrapped():
    def handler(request):
        return httpx.Response(429, json={"error": "rate limited"})

    embedder = _embedder(handler)

    with pytest.raises(ProviderError, match="Embedding request failed"):
        embedder.embed_query("q")
