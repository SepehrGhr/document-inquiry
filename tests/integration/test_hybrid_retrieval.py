import importlib
import json
from types import SimpleNamespace

import httpx
import pytest
from django.apps import apps
from django.core.management import call_command
from django.db import connection

from documents import services
from rag import factory, indexing
from rag.adapters import lexical
from rag.evaluation.dataset import load_corpus
from rag.evaluation.report import retrieval_report
from rag.evaluation.runner import run_retrieval
from rag.models import IndexedChunk, IndexProfile


@pytest.fixture
def profile(db):
    return IndexProfile.objects.create(
        name="hybrid",
        provider="fake",
        model_id="hashing",
        dimension=64,
        chunker_version="char-1",
        chunk_size=1200,
        chunk_overlap=200,
    )


def create(text, actor, title="Spec"):
    return services.create_document(
        title=title, description="", content=services.ContentInput(text=text), actor=actor
    )


def build(profile):
    return indexing.run_build(indexing.create_build(profile).id)


@pytest.mark.django_db
def test_lexical_search_normalizes_persian_and_distinguishes_product_codes(profile, staff_user):
    text = "آجر كاوه می‌رود اندازه ۲۴۰ KAV-240P"
    expected = create(text, staff_user)
    create("KAV-240 facade brick", staff_user, title="Near duplicate")
    indexed = build(profile)

    for query in ("کاوه", "می رود", "٢٤٠", "KAV-240P", "what is KAV-240P"):
        passages = lexical.search(indexed, query, 5)
        assert [p.document_id for p in passages] == [str(expected.id)]
        assert passages[0].text == text
    assert lexical.search(indexed, "!!!", 5) == []
    assert lexical.search(indexed, "nonexistent", 5) == []
    assert lexical.search(indexed, "what is the", 5) == []


@pytest.mark.django_db
def test_lexical_ranks_stored_lexemes_without_restemming_them(profile, staff_user):
    # The simple vector stores "refunds" literally; an English reparse would
    # turn it into "refund" and give the simple-config query zero rank.
    create("refunds", staff_user, title="One match")
    repeated = create("refunds refunds refunds", staff_user, title="Three matches")
    indexed = build(profile)

    with connection.cursor() as cursor:
        cursor.execute("SET LOCAL default_text_search_config = 'english'")
    passages = lexical.search(indexed, "refunds", 5)

    assert passages[0].document_id == str(repeated.id)
    assert passages[0].score > passages[1].score > 0


@pytest.mark.django_db
def test_lexical_search_scopes_profile_revisions_and_deleted_documents(profile, staff_user):
    changed = create("oldterm", staff_user)
    deleted = create("newterm", staff_user, title="Deleted")
    build(profile)
    services.replace_content(
        document_id=changed.id,
        content=services.ContentInput(text="newterm"),
        expected_revision=1,
        actor=staff_user,
    )
    indexed = build(profile)
    services.delete_document(document_id=deleted.id, actor=staff_user)
    other = IndexProfile.objects.create(
        name="other",
        provider="fake",
        model_id="other",
        dimension=64,
        chunker_version="char-1",
        chunk_size=1200,
        chunk_overlap=200,
    )
    chunk = IndexedChunk.objects.get(profile=profile, document=changed, revision__sequence=2)
    IndexedChunk.objects.create(
        profile=other,
        document=changed,
        revision=chunk.revision,
        ordinal=0,
        text="newterm",
        search_text="newterm",
        start=0,
        end=7,
        embedding=[1.0] * 64,
        content_sha256=chunk.content_sha256,
    )

    assert lexical.search(indexed, "oldterm", 5) == []
    found = lexical.search(indexed, "newterm", 5)
    assert [p.chunk_id for p in found] == [str(chunk.id)]
    assert found[0].revision_sequence == 2


@pytest.mark.django_db
def test_existing_chunks_are_backfilled_without_changing_source_or_vectors(profile, staff_user):
    create("كَاوه ۲۴۰", staff_user)
    indexed = build(profile)
    chunk = IndexedChunk.objects.get()
    original_vector = chunk.embedding.copy()
    IndexedChunk.objects.update(search_text="")
    migration = importlib.import_module("rag.migrations.0002_indexedchunk_search_text_and_more")
    migration.populate_search_text(apps, SimpleNamespace(connection=connection))

    chunk.refresh_from_db()
    assert chunk.search_text == "کاوه 240"
    assert chunk.text == "كَاوه ۲۴۰"
    assert list(chunk.embedding) == list(original_vector)
    assert lexical.search(indexed, "کاوه", 1)[0].chunk_id == str(chunk.id)


@pytest.mark.django_db
def test_hybrid_evaluation_uses_shared_retriever_and_records_fusion_config(profile, staff_user):
    corpus = load_corpus("tests/fixtures/corpus")
    create("Navardeh industrial town Sanasa Brick Works", staff_user, corpus.documents[0].title)
    indexed = build(profile)
    run = run_retrieval(corpus, indexed, top_k=5, limit=1, mode="hybrid_rrf", candidate_k=10)
    report = retrieval_report(run, top_k=5)

    assert not run.errors
    assert report["overall"]["recall@1"] == 1.0
    assert report["metadata"]["retrieval_mode"] == "hybrid_rrf"
    assert report["metadata"]["candidate_k"] == 10
    assert report["metadata"]["rrf_k"] == 60
    assert report["metadata"]["lexical"]["ranking"] == "ts_rank_cd"
    # Lexical-only retrieval must work without an embedding provider.
    assert factory.build_retriever(indexed, mode="lexical").retrieve("Sanasa", limit=1)


@pytest.mark.django_db
@pytest.mark.parametrize("mode", ["dense", "hybrid_rrf"])
def test_evaluation_command_opts_into_reranking_and_records_actual_depths(
    profile, staff_user, settings, monkeypatch, tmp_path, mode
):
    corpus = load_corpus("tests/fixtures/corpus")
    create("Navardeh industrial town Sanasa Brick Works", staff_user, corpus.documents[0].title)
    build(profile)
    settings.RAG = {**settings.RAG, "RERANKER": "jina", "JINA_API_KEY": "test-key"}
    calls = []

    def post(url, **kwargs):
        calls.append(kwargs["json"])
        return httpx.Response(
            200,
            request=httpx.Request("POST", url),
            json={"results": [{"index": 0, "relevance_score": 0.9}]},
        )

    monkeypatch.setattr("rag.adapters.jina.httpx.post", post)
    call_command(
        "evaluate_rag",
        profile=profile.name,
        retrieval_only=True,
        top_k=1,
        limit=1,
        mode=mode,
        candidate_k=2,
        reranker="jina",
        rerank_candidate_k=7,
        output=str(tmp_path / "reranked"),
    )
    report = json.loads((tmp_path / "reranked/retrieval.json").read_text())
    assert not report["errors"]
    assert report["overall"]["recall@1"] == 1.0
    assert report["metadata"]["reranking"] == {
        "provider": "jina",
        "model": settings.RAG["RERANK_MODEL"],
        "candidate_k": 7,
    }
    if mode == "hybrid_rrf":
        assert report["metadata"]["candidate_k"] == 7
    assert "Reranking: jina" in (tmp_path / "reranked/retrieval.md").read_text()
    assert len(calls) == 1
    assert calls[0]["top_n"] == 1

    # Even with Jina enabled in API settings, a baseline command makes no rerank call.
    call_command(
        "evaluate_rag",
        profile=profile.name,
        retrieval_only=True,
        top_k=1,
        limit=1,
        mode=mode,
        output=str(tmp_path / "baseline"),
    )
    baseline = json.loads((tmp_path / "baseline/retrieval.json").read_text())
    assert baseline["metadata"]["reranking"] == {"provider": "none"}
    assert len(calls) == 1
