import pytest

from documents import services
from rag import indexing
from rag.adapters import pgvector_index
from rag.adapters.fakes import HashingEmbedder
from rag.models import IndexBuild, IndexedChunk, IndexProfile


@pytest.fixture
def profile(db):
    return IndexProfile.objects.create(
        name="default",
        provider="fake",
        model_id="hashing",
        dimension=64,
        chunker_version="char-1",
        chunk_size=200,
        chunk_overlap=20,
    )


def _text(value):
    return services.ContentInput(text=value)


def _build(profile):
    return indexing.run_build(indexing.create_build(profile).id)


@pytest.mark.django_db
def test_build_indexes_current_revision_and_activates(staff_user, profile):
    document = services.create_document(
        title="Doc", description="", content=_text("alpha beta gamma"), actor=staff_user
    )

    build = _build(profile)

    assert build.status == IndexBuild.Status.COMPLETED
    assert build.is_active
    chunk = IndexedChunk.objects.get(profile=profile)
    assert chunk.revision == document.revisions.get()
    assert chunk.text == "alpha beta gamma"


@pytest.mark.django_db
def test_new_build_reuses_unchanged_vectors_and_drops_old_revision(staff_user, profile):
    document = services.create_document(
        title="Doc", description="", content=_text("first version"), actor=staff_user
    )
    first = _build(profile)
    first_chunk_id = IndexedChunk.objects.get().id

    services.replace_content(
        document_id=document.pk,
        content=_text("second version"),
        expected_revision=1,
        actor=staff_user,
    )
    second = _build(profile)

    assert second.is_active and not IndexBuild.objects.get(pk=first.pk).is_active
    assert IndexedChunk.objects.get(revision__sequence=1).id == first_chunk_id  # reused
    passages = pgvector_index.search(second, HashingEmbedder(64).embed_query("second"), 5)
    assert [p.text for p in passages] == ["second version"]


@pytest.mark.django_db
def test_deleted_documents_are_not_retrieved(staff_user, profile):
    document = services.create_document(
        title="Doc", description="", content=_text("secret content"), actor=staff_user
    )
    build = _build(profile)
    services.delete_document(document_id=document.pk, actor=staff_user)

    assert pgvector_index.search(build, HashingEmbedder(64).embed_query("secret"), 5) == []


@pytest.mark.django_db
def test_stale_build_is_not_activated(staff_user, profile):
    services.create_document(title="A", description="", content=_text("a"), actor=staff_user)
    build = indexing.create_build(profile)
    services.create_document(title="B", description="", content=_text("b"), actor=staff_user)

    build = indexing.run_build(build.id)

    assert build.status == IndexBuild.Status.COMPLETED
    assert not build.is_active


@pytest.mark.django_db
def test_retrying_a_completed_build_does_not_duplicate_chunks(staff_user, profile):
    services.create_document(title="A", description="", content=_text("a"), actor=staff_user)
    build = _build(profile)

    indexing.run_build(build.id)

    assert IndexedChunk.objects.count() == 1
