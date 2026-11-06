from types import SimpleNamespace

import pytest

from rag.adapters import openrouter
from rag.contracts import RetrievedPassage
from rag.exceptions import ProviderError


def _passage(chunk_id, text):
    return RetrievedPassage(
        chunk_id=chunk_id,
        document_id="d",
        document_title="Doc",
        revision_id="r",
        revision_sequence=1,
        ordinal=0,
        text=text,
        score=1.0,
        rank=1,
    )


def _generator(monkeypatch, content):
    generator = openrouter.OpenRouterAnswerGenerator(
        api_key="x", base_url="https://example.invalid", model="m"
    )
    monkeypatch.setattr(
        type(generator._client),
        "invoke",
        lambda self, messages: SimpleNamespace(content=content, usage_metadata={"input_tokens": 3}),
    )
    return generator


def test_generator_maps_citations_to_supplied_passages(monkeypatch):
    generator = _generator(monkeypatch, "Refunds take 14 days [S1].")

    result = generator.generate("How long?", [_passage("c1", "14 days"), _passage("c2", "other")])

    assert not result.abstained
    assert result.cited_chunk_ids == ("c1",)
    assert result.prompt_tokens == 3


def test_generator_abstains_on_insufficient_evidence(monkeypatch):
    generator = _generator(monkeypatch, "INSUFFICIENT_EVIDENCE")

    result = generator.generate("How long?", [_passage("c1", "14 days")])

    assert result.abstained
    assert result.text == ""


def test_generator_abstains_without_evidence(monkeypatch):
    generator = _generator(monkeypatch, "should not be called")

    result = generator.generate("How long?", [])

    assert result.abstained


def test_generator_wraps_provider_failures(monkeypatch):
    generator = _generator(monkeypatch, "")

    def boom(self, messages):
        raise RuntimeError("rate limited")

    monkeypatch.setattr(type(generator._client), "invoke", boom)

    with pytest.raises(ProviderError):
        generator.generate("How long?", [_passage("c1", "14 days")])


def test_embedder_rejects_wrong_dimension(monkeypatch):
    embedder = openrouter.OpenRouterEmbedder(
        api_key="x", base_url="https://example.invalid", model="m", dimension=4
    )
    monkeypatch.setattr(type(embedder._client), "embed_query", lambda self, text: [0.1, 0.2])

    with pytest.raises(ProviderError):
        embedder.embed_query("hello")
