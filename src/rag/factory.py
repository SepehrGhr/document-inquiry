"""The single composition point for the RAG pipeline.

Everything else depends on the contracts in `rag.contracts`; only this module
knows which concrete adapters are in use.
"""

from django.conf import settings
from django.core.exceptions import ImproperlyConfigured

from rag.adapters.cache import CachedEmbedder, EmbeddingCache
from rag.adapters.fakes import EchoAnswerGenerator, HashingEmbedder
from rag.adapters.jina import JinaEmbedder
from rag.adapters.openrouter import OpenRouterAnswerGenerator, OpenRouterEmbedder
from rag.chunking import CharacterChunker


def build_chunker(profile) -> CharacterChunker:
    return CharacterChunker(chunk_size=profile.chunk_size, overlap=profile.chunk_overlap)


def build_embedder(profile):
    embedder = _build_uncached_embedder(profile)
    if not settings.RAG["EMBEDDING_CACHE"]:
        return embedder
    return CachedEmbedder(
        embedder, EmbeddingCache(settings.RAG["EMBEDDING_CACHE_DIR"]), profile.model_id
    )


def _build_uncached_embedder(profile):
    if profile.provider == "fake":
        return HashingEmbedder(dimension=profile.dimension)
    if profile.provider == "openrouter":
        config = settings.RAG
        if not config["OPENROUTER_API_KEY"]:
            raise ImproperlyConfigured("OPENROUTER_KEY is not set.")
        return OpenRouterEmbedder(
            api_key=config["OPENROUTER_API_KEY"],
            base_url=config["OPENROUTER_BASE_URL"],
            model=profile.model_id,
            dimension=profile.dimension,
            timeout=config["EMBEDDING_TIMEOUT"],
        )
    if profile.provider == "jina":
        config = settings.RAG
        if not config["JINA_API_KEY"]:
            raise ImproperlyConfigured("JINA_KEY is not set.")
        return JinaEmbedder(
            api_key=config["JINA_API_KEY"],
            base_url=config["JINA_BASE_URL"],
            model=profile.model_id,
            dimension=profile.dimension,
            timeout=config["EMBEDDING_TIMEOUT"],
        )
    raise ImproperlyConfigured(f"Unknown embedding provider {profile.provider!r}.")


def build_answer_generator():
    config = settings.RAG
    if config["ANSWER_PROVIDER"] == "fake":
        return EchoAnswerGenerator()
    if config["ANSWER_PROVIDER"] == "openrouter":
        if not config["OPENROUTER_API_KEY"]:
            raise ImproperlyConfigured("OPENROUTER_KEY is not set.")
        return OpenRouterAnswerGenerator(
            api_key=config["OPENROUTER_API_KEY"],
            base_url=config["OPENROUTER_BASE_URL"],
            model=config["ANSWER_MODEL"],
            timeout=config["ANSWER_TIMEOUT"],
        )
    raise ImproperlyConfigured(f"Unknown answer provider {config['ANSWER_PROVIDER']!r}.")
