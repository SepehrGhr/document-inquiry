from django.urls import path

from rag.api import views

urlpatterns = [
    path("questions/", views.QuestionListCreateView.as_view(), name="question-list"),
    path(
        "questions/<uuid:question_id>/",
        views.QuestionDetailView.as_view(),
        name="question-detail",
    ),
]
