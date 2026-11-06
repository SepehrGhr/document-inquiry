from rag.adapters.cache import CachedEmbedder, EmbeddingCache


class CountingEmbedder:
    dimension = 3

    def __init__(self):
        self.calls = []

    def embed_documents(self, texts):
        self.calls.append(("documents", list(texts)))
        return [[float(len(text))] * 3 for text in texts]

    def embed_query(self, text):
        self.calls.append(("query", text))
        return [float(len(text))] * 3


def test_query_is_served_from_cache_on_second_call(tmp_path):
    inner = CountingEmbedder()
    embedder = CachedEmbedder(inner, EmbeddingCache(tmp_path), "model-a")

    first = embedder.embed_query("hello")
    second = embedder.embed_query("hello")

    assert first == second == [5.0, 5.0, 5.0]
    assert inner.calls == [("query", "hello")]


def test_documents_only_embed_the_missing_texts(tmp_path):
    inner = CountingEmbedder()
    embedder = CachedEmbedder(inner, EmbeddingCache(tmp_path), "model-a")
    embedder.embed_query("hello")

    vectors = embedder.embed_documents(["hello", "world"])

    assert vectors == [[5.0, 5.0, 5.0], [5.0, 5.0, 5.0]]
    assert inner.calls == [("query", "hello"), ("documents", ["world"])]


def test_cache_is_scoped_by_model(tmp_path):
    cache = EmbeddingCache(tmp_path)
    cache.put("model-a", "hello", [1.0])

    assert cache.get("model-a", "hello") == [1.0]
    assert cache.get("model-b", "hello") is None


def test_corrupt_entry_is_treated_as_a_miss(tmp_path):
    cache = EmbeddingCache(tmp_path)
    cache.put("model-a", "hello", [1.0])
    path = cache._path("model-a", "hello")
    path.write_text("{not json", encoding="utf-8")

    assert cache.get("model-a", "hello") is None
