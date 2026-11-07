import httpx
import pytest

from documents import services
from rag import indexing
from rag.models import Answer, IndexProfile, Question


@pytest.fixture
def indexed_corpus(db, staff_user):
    profile = IndexProfile.objects.create(
        name="default",
        provider="fake",
        model_id="hashing",
        dimension=64,
        chunker_version="char-1",
        chunk_size=200,
        chunk_overlap=20,
    )
    services.create_document(
        title="Refund policy",
        description="",
        content=services.ContentInput(
            text="Refunds are processed within 14 days. Contact support."
        ),
        actor=staff_user,
    )
    services.create_document(
        title="Shipping",
        description="",
        content=services.ContentInput(text="Orders ship within two business days."),
        actor=staff_user,
    )
    indexing.run_build(indexing.create_build(profile).id)
    return profile


@pytest.mark.django_db
@pytest.mark.parametrize("mode", ["dense", "hybrid_rrf"])
def test_question_returns_grounded_answer_with_sources(api_client, indexed_corpus, settings, mode):
    settings.RAG = {**settings.RAG, "RETRIEVAL_MODE": mode}
    response = api_client.post(
        "/api/v1/questions/", {"question": "How long do refunds take?"}, format="json"
    )

    assert response.status_code == 201
    body = response.json()
    assert body["answer"]["status"] == "answered"
    assert "[S1]" in body["answer"]["text"]
    assert body["answer"]["sources"][0]["document_title"] == "Refund policy"
    assert body["answer"]["sources"][0]["cited"] is True
    assert Answer.objects.count() == 1


@pytest.mark.django_db
def test_reranked_evidence_is_used_for_answers_and_citation_history(
    api_client, indexed_corpus, settings, monkeypatch
):
    settings.RAG = {
        **settings.RAG,
        "RERANKER": "jina",
        "JINA_API_KEY": "test-key",
        "TOP_K": 1,
    }

    def post(url, **kwargs):
        payload = kwargs["json"]
        assert len(payload["documents"]) == 2  # Not truncated to TOP_K before reranking.
        index = next(i for i, text in enumerate(payload["documents"]) if "Orders ship" in text)
        return httpx.Response(
            200,
            request=httpx.Request("POST", url),
            json={"results": [{"index": index, "relevance_score": 0.95}]},
        )

    monkeypatch.setattr("rag.adapters.jina.httpx.post", post)
    response = api_client.post(
        "/api/v1/questions/", {"question": "When do orders ship?"}, format="json"
    )

    assert response.status_code == 201
    answer = response.json()["answer"]
    assert answer["status"] == "answered"
    assert "[S1]" in answer["text"]
    assert len(answer["sources"]) == 1
    assert answer["sources"][0]["document_title"] == "Shipping"
    assert answer["sources"][0]["cited"] is True
    source = Answer.objects.get().sources.get()
    assert source.text == "Orders ship within two business days."
    assert source.score == pytest.approx(0.95)
    assert source.rank == 1


@pytest.mark.django_db
def test_reranker_failure_is_recorded_without_silent_fallback(
    api_client, indexed_corpus, settings, monkeypatch
):
    settings.RAG = {**settings.RAG, "RERANKER": "jina", "JINA_API_KEY": "test-key"}

    def post(url, **kwargs):
        return httpx.Response(429, request=httpx.Request("POST", url))

    monkeypatch.setattr("rag.adapters.jina.httpx.post", post)
    response = api_client.post(
        "/api/v1/questions/", {"question": "When do orders ship?"}, format="json"
    )

    assert response.status_code == 201
    assert response.json()["answer"]["status"] == "failed"
    assert "Reranking request failed" in Answer.objects.get().reason


@pytest.mark.django_db
def test_question_without_active_index_returns_503(api_client):
    response = api_client.post("/api/v1/questions/", {"question": "anything"}, format="json")

    assert response.status_code == 503
    assert Question.objects.count() == 0


@pytest.mark.django_db
def test_blank_question_is_rejected(api_client, indexed_corpus):
    response = api_client.post("/api/v1/questions/", {"question": "   "}, format="json")

    assert response.status_code == 400


@pytest.mark.django_db
def test_question_history_is_listed_and_retrievable(api_client, indexed_corpus):
    created = api_client.post(
        "/api/v1/questions/", {"question": "When do orders ship?"}, format="json"
    ).json()

    listing = api_client.get("/api/v1/questions/")
    detail = api_client.get(f"/api/v1/questions/{created['id']}/")

    assert listing.status_code == 200
    assert listing.json()["count"] == 1
    assert detail.json()["answer"]["status"] == "answered"


@pytest.mark.django_db
def test_questions_require_staff(client, indexed_corpus):
    response = client.post("/api/v1/questions/", {"question": "anything"}, format="json")

    assert response.status_code in (401, 403)
