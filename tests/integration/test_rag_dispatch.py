import pytest
from django.db import transaction

from documents import services
from rag import dispatch
from rag.models import IndexBuild, IndexProfile


@pytest.fixture
def profile(db, settings):
    settings.RAG = {**settings.RAG, "AUTO_INDEX": True, "ACTIVE_PROFILE": "default"}
    return IndexProfile.objects.create(
        name="default",
        provider="fake",
        model_id="hashing",
        dimension=64,
        chunker_version="char-1",
        chunk_size=200,
        chunk_overlap=20,
    )


@pytest.mark.django_db(transaction=True)
def test_document_write_queues_indexing_after_commit(staff_user, profile):
    services.create_document(
        title="Doc", description="", content=services.ContentInput(text="hello"), actor=staff_user
    )

    build = IndexBuild.objects.get()
    assert build.status == IndexBuild.Status.COMPLETED
    assert build.is_active


@pytest.mark.django_db(transaction=True)
def test_rolled_back_write_does_not_queue_indexing(staff_user, profile):
    with pytest.raises(RuntimeError), transaction.atomic():
        services.create_document(
            title="Doc",
            description="",
            content=services.ContentInput(text="hello"),
            actor=staff_user,
        )
        raise RuntimeError("boom")

    assert IndexBuild.objects.count() == 0


@pytest.mark.django_db(transaction=True)
def test_auto_index_disabled_queues_nothing(staff_user, profile, settings):
    settings.RAG = {**settings.RAG, "AUTO_INDEX": False}

    services.create_document(
        title="Doc", description="", content=services.ContentInput(text="hello"), actor=staff_user
    )

    assert IndexBuild.objects.count() == 0


def test_dispatch_is_a_noop_when_disabled(settings):
    settings.RAG = {**settings.RAG, "AUTO_INDEX": False}
    dispatch.schedule_indexing()  # must not raise or touch the database
