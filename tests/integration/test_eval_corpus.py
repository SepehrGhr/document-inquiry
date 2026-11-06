import pytest

from documents import selectors
from rag.evaluation.corpus import corpus_documents, load_corpus, remove_corpus
from rag.evaluation.dataset import load_corpus as read_corpus

CORPUS_ROOT = "tests/fixtures/corpus"


@pytest.fixture
def corpus():
    return read_corpus(CORPUS_ROOT)


@pytest.mark.django_db
def test_load_creates_every_document(staff_user, corpus):
    result = load_corpus(corpus, actor=staff_user)

    assert result.created == len(corpus.documents)
    assert result.updated == 0
    assert selectors.active_documents().count() == len(corpus.documents)
    assert len(corpus_documents(corpus)) == len(corpus.documents)


@pytest.mark.django_db
def test_loading_twice_is_idempotent(staff_user, corpus):
    load_corpus(corpus, actor=staff_user)

    result = load_corpus(corpus, actor=staff_user)

    assert result.created == 0
    assert result.updated == 0
    assert result.unchanged == len(corpus.documents)
    assert selectors.active_documents().count() == len(corpus.documents)


@pytest.mark.django_db
def test_reset_removes_only_corpus_documents(staff_user, corpus):
    from documents import services

    services.create_document(
        title="Unrelated",
        description="",
        content=services.ContentInput(text="keep me"),
        actor=staff_user,
    )
    load_corpus(corpus, actor=staff_user)

    result = load_corpus(corpus, actor=staff_user, reset=True)

    assert result.removed == len(corpus.documents)
    assert result.created == len(corpus.documents)
    assert selectors.active_documents().count() == len(corpus.documents) + 1
    assert selectors.active_documents().filter(title="Unrelated").exists()


@pytest.mark.django_db
def test_remove_corpus_leaves_other_documents(staff_user, corpus):
    from documents import services

    services.create_document(
        title="Unrelated",
        description="",
        content=services.ContentInput(text="keep me"),
        actor=staff_user,
    )
    load_corpus(corpus, actor=staff_user)

    removed = remove_corpus(corpus, actor=staff_user)

    assert removed == len(corpus.documents)
    assert selectors.active_documents().count() == 1


@pytest.mark.django_db
def test_loaded_text_matches_the_docx(staff_user, corpus):
    load_corpus(corpus, actor=staff_user)

    document = selectors.active_documents().get(title="About Sanasa Brick Works")
    revision = selectors.current_revision(document)

    assert "Sanasa" in revision.text
    assert revision.source == "text"
