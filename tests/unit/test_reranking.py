import json
from dataclasses import replace
from types import SimpleNamespace

import httpx
import pytest
from django.core.exceptions import ImproperlyConfigured

from rag import factory
from rag.adapters import jina, lexical, pgvector_index
from rag.contracts import RetrievedPassage
from rag.exceptions import ProviderError


def passage(index):
    return RetrievedPassage(
        str(index),
        f"doc-{index}",
        f"Title {index}",
        f"rev-{index}",
        2,
        index,
        f"متن دقیق {index}",
        1.0 / (index + 1),
        index + 1,
    )


def mock_jina(monkeypatch, handler):
    def post(url, **kwargs):
        with httpx.Client(transport=httpx.MockTransport(handler)) as client:
            return client.post(url, **kwargs)

    monkeypatch.setattr(jina.httpx, "post", post)


def reranker():
    return jina.JinaReranker(
        api_key="secret", base_url="https://example.invalid/v1/", model="rerank-model", timeout=10
    )


def test_jina_maps_ranked_indices_to_original_evidence(monkeypatch):
    captured = []

    def handler(request):
        captured.append(request)
        return httpx.Response(
            200,
            json={
                "results": [
                    {"index": 2, "relevance_score": 0.9},
                    {"index": 0, "relevance_score": 0.5},
                ]
            },
        )

    mock_jina(monkeypatch, handler)
    passages = [passage(i) for i in range(3)]
    results = reranker().rerank("پرسش", passages, limit=2)

    assert results == [
        replace(passages[2], rank=1, score=0.9),
        replace(passages[0], rank=2, score=0.5),
    ]
    assert passages[2].rank == 3  # The input evidence is not mutated.
    request = captured[0]
    assert str(request.url) == "https://example.invalid/v1/rerank"
    assert request.headers["Authorization"] == "Bearer secret"
    assert json.loads(request.content) == {
        "model": "rerank-model",
        "query": "پرسش",
        "documents": [p.text for p in passages],
        "top_n": 2,
        "return_documents": False,
    }


def test_jina_skips_empty_candidates_and_caps_top_n(monkeypatch):
    def handler(request):
        assert json.loads(request.content)["top_n"] == 1
        return httpx.Response(200, json={"results": [{"index": 0, "relevance_score": 0.8}]})

    mock_jina(monkeypatch, handler)
    model = reranker()
    assert model.rerank("q", [], limit=5) == []
    assert model.rerank("q", [passage(0)], limit=5)[0].score == 0.8


@pytest.mark.parametrize("failure", ["rate_limit", "timeout", "invalid_json"])
def test_jina_wraps_provider_failures(monkeypatch, failure):
    def handler(request):
        if failure == "timeout":
            raise httpx.ReadTimeout("timed out", request=request)
        if failure == "invalid_json":
            return httpx.Response(200, text="not json")
        return httpx.Response(429, json={"error": "rate limited"})

    mock_jina(monkeypatch, handler)
    with pytest.raises(ProviderError, match="Reranking request failed"):
        reranker().rerank("q", [passage(0)], limit=1)


@pytest.mark.parametrize(
    "results",
    [
        [{"index": 0, "relevance_score": 0.9}, {"index": 0, "relevance_score": 0.8}],
        [{"index": -1, "relevance_score": 0.9}, {"index": 1, "relevance_score": 0.8}],
        [{"index": 2, "relevance_score": 0.9}, {"index": 1, "relevance_score": 0.8}],
        [{"index": 0, "relevance_score": "bad"}, {"index": 1, "relevance_score": 0.8}],
        [{"index": 0, "relevance_score": 0.9}],
    ],
)
def test_jina_rejects_results_that_cannot_be_safely_mapped(monkeypatch, results):
    mock_jina(monkeypatch, lambda request: httpx.Response(200, json={"results": results}))
    with pytest.raises(ProviderError):
        reranker().rerank("q", [passage(0), passage(1)], limit=2)


@pytest.mark.parametrize("mode", ["dense", "lexical", "hybrid_rrf"])
@pytest.mark.parametrize("limit, expected_pool", [(2, 7), (9, 9)])
def test_factory_reranks_larger_shortlist_before_final_truncation(
    monkeypatch, settings, mode, limit, expected_pool
):
    settings.RAG = {
        **settings.RAG,
        "RERANKER": "jina",
        "JINA_API_KEY": "k",
        "RERANK_CANDIDATES": 7,
        "RETRIEVAL_CANDIDATES": 3,
    }
    candidates = [passage(i) for i in range(expected_pool)]
    searched = []

    def search(build, query, depth):
        searched.append(depth)
        return candidates[:depth]

    def handler(request):
        payload = json.loads(request.content)
        assert len(payload["documents"]) == expected_pool
        assert payload["top_n"] == limit
        indices = list(reversed(range(expected_pool)))[:limit]
        return httpx.Response(
            200,
            json={
                "results": [
                    {"index": index, "relevance_score": 1.0 / rank}
                    for rank, index in enumerate(indices, start=1)
                ]
            },
        )

    monkeypatch.setattr(pgvector_index, "search", search)
    monkeypatch.setattr(lexical, "search", search)
    mock_jina(monkeypatch, handler)
    build = SimpleNamespace(profile=SimpleNamespace(provider="fake", dimension=8))
    results = factory.build_retriever(build, mode=mode).retrieve("q", limit=limit)

    assert searched == ([expected_pool] * (2 if mode == "hybrid_rrf" else 1))
    assert len(results) == limit
    assert results[0].chunk_id == str(expected_pool - 1)
    assert [p.rank for p in results] == list(range(1, limit + 1))


def test_disabled_reranking_does_not_require_jina_key(monkeypatch, settings):
    settings.RAG = {**settings.RAG, "RERANKER": "jina", "JINA_API_KEY": ""}
    monkeypatch.setattr(lexical, "search", lambda build, text, limit: [passage(0)])
    build = SimpleNamespace()
    assert factory.build_retriever(build, mode="lexical", reranker="none").retrieve("q", limit=1)
    with pytest.raises(ImproperlyConfigured, match="JINA_KEY"):
        factory.build_retriever(build, mode="lexical", reranker="jina")
