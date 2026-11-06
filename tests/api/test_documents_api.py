import pytest
from django.contrib.auth import get_user_model
from rest_framework.test import APIClient

pytestmark = pytest.mark.django_db


def test_create_read_and_replace_document(api_client):
    created = api_client.post(
        "/api/v1/documents/",
        {"title": "راهنما", "text": "متن اولیه"},
        format="json",
    )
    assert created.status_code == 201
    document_id = created.json()["id"]
    assert created.json()["current_revision"] == 1

    detail = api_client.get(f"/api/v1/documents/{document_id}/")
    assert detail.json()["text"] == "متن اولیه"

    replaced = api_client.post(
        f"/api/v1/documents/{document_id}/revisions/",
        {"text": "متن دوم", "expected_revision": 1},
        format="json",
    )
    assert replaced.status_code == 201
    assert replaced.json()["sequence"] == 2

    stale = api_client.post(
        f"/api/v1/documents/{document_id}/revisions/",
        {"text": "متن سوم", "expected_revision": 1},
        format="json",
    )
    assert stale.status_code == 409

    history = api_client.get(f"/api/v1/documents/{document_id}/revisions/")
    assert [r["sequence"] for r in history.json()] == [2, 1]


def test_metadata_patch_keeps_revision(api_client):
    document_id = api_client.post(
        "/api/v1/documents/", {"title": "Old", "text": "body"}, format="json"
    ).json()["id"]

    response = api_client.patch(
        f"/api/v1/documents/{document_id}/", {"title": "New"}, format="json"
    )

    assert response.status_code == 200
    assert response.json()["title"] == "New"
    assert response.json()["current_revision"] == 1


def test_delete_hides_document(api_client):
    document_id = api_client.post(
        "/api/v1/documents/", {"title": "Doc", "text": "body"}, format="json"
    ).json()["id"]

    assert api_client.delete(f"/api/v1/documents/{document_id}/").status_code == 204
    assert api_client.get(f"/api/v1/documents/{document_id}/").status_code == 404
    assert api_client.get("/api/v1/documents/").json()["count"] == 0


def test_docx_upload_and_download(api_client, make_docx):
    created = api_client.post(
        "/api/v1/documents/",
        {"title": "Docx", "file": make_docx(paragraphs=("سلام دنیا",))},
        format="multipart",
    )
    assert created.status_code == 201
    document_id = created.json()["id"]

    download = api_client.get(f"/api/v1/documents/{document_id}/revisions/1/file/")
    assert download.status_code == 200
    assert download["Content-Disposition"].startswith("attachment")


def test_requires_authentication():
    response = APIClient().get("/api/v1/documents/")
    assert response.status_code in (401, 403)


def test_non_staff_user_is_denied():
    user = get_user_model().objects.create_user(username="plain", password="x")
    client = APIClient()
    client.force_authenticate(user=user)

    assert client.get("/api/v1/documents/").status_code == 403
