from django.urls import path

from documents.api import views

urlpatterns = [
    path("documents/", views.DocumentListCreateView.as_view(), name="document-list"),
    path(
        "documents/<uuid:document_id>/",
        views.DocumentDetailView.as_view(),
        name="document-detail",
    ),
    path(
        "documents/<uuid:document_id>/revisions/",
        views.RevisionListCreateView.as_view(),
        name="revision-list",
    ),
    path(
        "documents/<uuid:document_id>/revisions/<int:sequence>/",
        views.RevisionDetailView.as_view(),
        name="revision-detail",
    ),
    path(
        "documents/<uuid:document_id>/revisions/<int:sequence>/file/",
        views.RevisionFileView.as_view(),
        name="revision-file",
    ),
]
