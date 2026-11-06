import pytest

from documents import services
from documents.exceptions import DocumentNotFound, InvalidDocumentInput, RevisionConflict
from documents.models import Document, DocumentRevision


def _text(value):
    return services.ContentInput(text=value)


@pytest.mark.django_db
def test_create_stores_first_revision_with_exact_text(staff_user):
    text = "متن فارسی\nEnglish line\nمخلوط mixed"

    document = services.create_document(
        title="Sample", description="", content=_text(text), actor=staff_user
    )

    revision = document.revisions.get()
    assert revision.sequence == 1
    assert revision.text == text
    assert revision.source == DocumentRevision.Source.TEXT


@pytest.mark.django_db
def test_metadata_update_does_not_create_revision(staff_user):
    document = services.create_document(
        title="Old", description="", content=_text("body"), actor=staff_user
    )

    services.update_metadata(
        document_id=document.pk, title="New", description="desc", actor=staff_user
    )

    document.refresh_from_db()
    assert document.title == "New"
    assert document.revisions.count() == 1


@pytest.mark.django_db
def test_replace_content_appends_revision_and_keeps_history(staff_user):
    document = services.create_document(
        title="Doc", description="", content=_text("v1"), actor=staff_user
    )

    services.replace_content(
        document_id=document.pk, content=_text("v2"), expected_revision=1, actor=staff_user
    )

    assert list(document.revisions.order_by("sequence").values_list("text", flat=True)) == [
        "v1",
        "v2",
    ]


@pytest.mark.django_db
def test_replace_with_stale_revision_is_rejected(staff_user):
    document = services.create_document(
        title="Doc", description="", content=_text("v1"), actor=staff_user
    )
    services.replace_content(
        document_id=document.pk, content=_text("v2"), expected_revision=1, actor=staff_user
    )

    with pytest.raises(RevisionConflict):
        services.replace_content(
            document_id=document.pk, content=_text("v3"), expected_revision=1, actor=staff_user
        )


@pytest.mark.django_db
def test_deleted_document_cannot_be_edited(staff_user):
    document = services.create_document(
        title="Doc", description="", content=_text("v1"), actor=staff_user
    )
    services.delete_document(document_id=document.pk, actor=staff_user)

    with pytest.raises(DocumentNotFound):
        services.replace_content(
            document_id=document.pk, content=_text("v2"), expected_revision=1, actor=staff_user
        )
    assert Document.objects.get(pk=document.pk).deleted_at is not None


@pytest.mark.django_db
def test_docx_upload_is_stored_with_file_metadata(staff_user, make_docx):
    document = services.create_document(
        title="Docx",
        description="",
        content=services.ContentInput(docx_file=make_docx(paragraphs=("سلام",))),
        actor=staff_user,
    )

    revision = document.revisions.get()
    assert revision.source == DocumentRevision.Source.DOCX
    assert revision.text == "سلام"
    assert revision.original_filename == "sample.docx"
    assert revision.file_size > 0
    assert revision.file.storage.exists(revision.file.name)


@pytest.mark.django_db
def test_invalid_content_creates_no_document(staff_user):
    with pytest.raises(InvalidDocumentInput):
        services.create_document(
            title="Bad",
            description="",
            content=services.ContentInput(text="a", docx_file=object()),
            actor=staff_user,
        )

    assert not Document.objects.exists()
