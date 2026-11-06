import pytest
from django.urls import reverse

from documents.models import Document

pytestmark = pytest.mark.django_db

INLINE_FORM = {"revisions-TOTAL_FORMS": "0", "revisions-INITIAL_FORMS": "0"}


@pytest.fixture
def admin_client(client, staff_user):
    staff_user.is_superuser = True
    staff_user.save()
    client.force_login(staff_user)
    return client


def test_admin_creates_document_through_services(admin_client):
    response = admin_client.post(
        reverse("admin:documents_document_add"),
        {**INLINE_FORM, "title": "سند اداری", "description": "", "text": "متن از پنل مدیریت"},
    )

    assert response.status_code == 302
    document = Document.objects.get()
    assert document.revisions.get().text == "متن از پنل مدیریت"


def test_admin_edit_without_content_keeps_revision(admin_client):
    admin_client.post(
        reverse("admin:documents_document_add"),
        {**INLINE_FORM, "title": "Old", "description": "", "text": "body"},
    )
    document = Document.objects.get()

    response = admin_client.post(
        reverse("admin:documents_document_change", args=[document.pk]),
        {**INLINE_FORM, "title": "New", "description": "", "text": "", "file": ""},
    )

    assert response.status_code == 302
    document.refresh_from_db()
    assert document.title == "New"
    assert document.revisions.count() == 1


def test_admin_bulk_delete_is_logical(admin_client):
    admin_client.post(
        reverse("admin:documents_document_add"),
        {**INLINE_FORM, "title": "Doc", "description": "", "text": "body"},
    )
    document = Document.objects.get()

    admin_client.post(
        reverse("admin:documents_document_changelist"),
        {"action": "delete_selected", "_selected_action": [str(document.pk)]},
    )
    response = admin_client.post(
        reverse("admin:documents_document_changelist"),
        {"action": "delete_selected", "_selected_action": [str(document.pk)], "post": "yes"},
    )

    assert response.status_code == 302
    document.refresh_from_db()
    assert document.deleted_at is not None
    assert document.revisions.count() == 1
