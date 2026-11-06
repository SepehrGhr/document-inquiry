import json

import pytest

from rag.evaluation.dataset import CorpusError, load_corpus


def _write_corpus(root, documents, questions):
    (root / "docs").mkdir(parents=True)
    for document in documents:
        (root / document["file"]).write_text("x", encoding="utf-8")
    (root / "manifest.json").write_text(
        json.dumps({"company": "Sanasa", "documents": documents}), encoding="utf-8"
    )
    (root / "questions.jsonl").write_text(
        "\n".join(json.dumps(question) for question in questions), encoding="utf-8"
    )


def _document(file="docs/a.docx"):
    return {
        "file": file,
        "title": "A",
        "language": "en",
        "type": "policy",
        "status": "current",
    }


def _question(**overrides):
    base = {
        "id": "q1",
        "question": "Where?",
        "language": "en",
        "difficulty": "simple",
        "answerable": True,
        "expected_answer": "Here.",
        "expected_sources": ["docs/a.docx"],
    }
    return {**base, **overrides}


def test_loads_a_valid_corpus(tmp_path):
    _write_corpus(tmp_path, [_document()], [_question()])

    corpus = load_corpus(tmp_path)

    assert corpus.company == "Sanasa"
    assert corpus.documents[0].title == "A"
    assert corpus.questions[0].expected_sources == ("docs/a.docx",)


def test_missing_manifest_is_rejected(tmp_path):
    with pytest.raises(CorpusError, match="manifest.json"):
        load_corpus(tmp_path)


def test_manifest_file_that_does_not_exist_is_rejected(tmp_path):
    _write_corpus(tmp_path, [_document()], [_question()])
    (tmp_path / "docs/a.docx").unlink()

    with pytest.raises(CorpusError, match="do not exist"):
        load_corpus(tmp_path)


def test_unknown_difficulty_is_rejected(tmp_path):
    _write_corpus(tmp_path, [_document()], [_question(difficulty="impossible")])

    with pytest.raises(CorpusError, match="unknown difficulty"):
        load_corpus(tmp_path)


def test_question_citing_unknown_source_is_rejected(tmp_path):
    _write_corpus(tmp_path, [_document()], [_question(expected_sources=["docs/missing.docx"])])

    with pytest.raises(CorpusError, match="unknown sources"):
        load_corpus(tmp_path)


def test_answerable_question_without_sources_is_rejected(tmp_path):
    _write_corpus(tmp_path, [_document()], [_question(expected_sources=[])])

    with pytest.raises(CorpusError, match="no expected sources"):
        load_corpus(tmp_path)


def test_duplicate_question_ids_are_rejected(tmp_path):
    _write_corpus(tmp_path, [_document()], [_question(), _question()])

    with pytest.raises(CorpusError, match="Duplicate question IDs"):
        load_corpus(tmp_path)


def test_real_corpus_loads():
    corpus = load_corpus("tests/fixtures/corpus")

    assert len(corpus.documents) == 49
    assert len(corpus.questions) == 71
    assert {q.difficulty for q in corpus.questions} == {
        "simple",
        "exact_term",
        "multi_hop",
        "ambiguous",
        "unanswerable",
        "adversarial",
    }
