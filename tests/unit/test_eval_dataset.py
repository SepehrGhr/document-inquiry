import json

import pytest

from rag.evaluation.dataset import (
    Corpus,
    CorpusError,
    EvidenceAnnotation,
    EvidenceSpan,
    load_corpus,
)


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


def test_legacy_corpus_construction_keeps_optional_annotations_empty(tmp_path):
    corpus = Corpus(tmp_path, "Sanasa", (), ())
    assert corpus.evidence == {}
    assert corpus.splits == {}
    assert Corpus(tmp_path, "Sanasa", (), (), {"q1": EvidenceAnnotation((), ())}).evidence


@pytest.fixture
def annotated_corpus(tmp_path, make_docx):
    documents = [_document(), {**_document("docs/b.docx"), "title": "B"}]
    questions = [_question(), _question(id="q2", answerable=False, expected_sources=[])]
    _write_corpus(tmp_path, documents, questions)
    text = "First paragraph\n\nسلام می‌کنم — café"
    for document in documents:
        (tmp_path / document["file"]).write_bytes(make_docx(paragraphs=text.split("\n\n")).read())
    (tmp_path / "extracted").mkdir()
    (tmp_path / "extracted/a.txt").write_text(text, encoding="utf-8")
    quote = "سلام می‌کنم — café"
    span = {
        "id": "s1",
        "file": "docs/a.docx",
        "start": text.index(quote),
        "end": len(text),
        "quote": quote,
        "generator_notes": "ignored",
    }
    evidence = [
        {
            "question_id": "q1",
            "evidence_spans": [span, {**span, "id": "s2", "file": "docs/b.docx"}],
            "sufficient_evidence_sets": [["s1"], ["s2"]],
            "answer_behavior": "answer",
            "chunking_challenges": ["unicode"],
        },
        {"question_id": "q2", "evidence_spans": [], "sufficient_evidence_sets": []},
    ]
    _write_evidence(tmp_path, evidence)
    return tmp_path, evidence


def _write_evidence(root, entries):
    (root / "evidence.jsonl").write_text(
        "\n".join(json.dumps(entry, ensure_ascii=False) for entry in entries), encoding="utf-8"
    )


@pytest.mark.parametrize("wrapped", [False, True])
def test_loads_alternative_unicode_evidence_and_splits(annotated_corpus, wrapped, monkeypatch):
    root, _ = annotated_corpus
    splits = {"development": ["q1"], "test": ["q2"]}
    split_data = {"splits": splits} if wrapped else splits
    split_data["fact_families"] = {"q1": "policy", "q2": "missing"}
    (root / "splits.json").write_text(json.dumps(split_data), encoding="utf-8")
    # Both alternatives and repeated question references share cached canonical extraction.
    from rag.evaluation import dataset

    original = dataset.parse_docx
    parsed = []

    def parse_once(upload):
        parsed.append(upload.name)
        return original(upload)

    monkeypatch.setattr(dataset, "parse_docx", parse_once)
    entries = annotated_corpus[1]
    entries[0]["evidence_spans"].append({**entries[0]["evidence_spans"][0], "id": "s3"})
    entries[1]["evidence_spans"] = [entries[0]["evidence_spans"][0]]
    _write_evidence(root, entries)

    corpus = load_corpus(root)

    annotation = corpus.evidence["q1"]
    assert annotation.spans[0] == EvidenceSpan("s1", "docs/a.docx", 17, 35, "سلام می‌کنم — café")
    assert annotation.sufficient_sets == (("s1",), ("s2",))
    assert annotation.answer_behavior == "answer"
    assert annotation.challenges == ("unicode",)
    assert corpus.evidence["q2"].sufficient_sets == ()
    assert corpus.splits == {"development": ("q1",), "test": ("q2",)}
    assert len(parsed) == len(set(parsed)) == 2


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"start": True}, "integers"),
        ({"end": 35.0}, "integers"),
        ({"start": -1}, "bounds"),
        ({"end": 999}, "bounds"),
        ({"quote": "wrong"}, "quote"),
        ({"quote": ""}, "nonempty"),
        ({"file": "docs/missing.docx"}, "unknown source"),
        ({"id": ""}, "span ID"),
    ],
)
def test_invalid_evidence_spans_are_rejected(annotated_corpus, change, message):
    root, entries = annotated_corpus
    entries[0]["evidence_spans"][0].update(change)
    _write_evidence(root, entries)
    with pytest.raises(CorpusError, match=message):
        load_corpus(root)


def test_mismatched_extracted_text_is_rejected(annotated_corpus):
    root, _ = annotated_corpus
    (root / "extracted/a.txt").write_text("normalized or stale text", encoding="utf-8")
    with pytest.raises(CorpusError, match="differs from parser output"):
        load_corpus(root)


@pytest.mark.parametrize("sets", [[], [[]], [["unknown"]], [["s1", "s1"]], [["s2"]]])
def test_invalid_sufficient_sets_are_rejected(annotated_corpus, sets):
    root, entries = annotated_corpus
    entries[0]["sufficient_evidence_sets"] = sets
    _write_evidence(root, entries)
    with pytest.raises(CorpusError, match="sufficient"):
        load_corpus(root)


@pytest.mark.parametrize(
    "problem", ["unknown", "duplicate", "missing", "unsupported", "unanswerable", "span_duplicate"]
)
def test_invalid_annotation_membership_and_shape_are_rejected(annotated_corpus, problem):
    root, entries = annotated_corpus
    if problem == "unknown":
        entries[0]["question_id"] = "unknown"
    elif problem == "duplicate":
        entries.append(entries[0])
    elif problem == "missing":
        entries.pop()
    elif problem == "unsupported":
        entries[0] = {"question_id": "q1", "spans": []}
    elif problem == "unanswerable":
        entries[1].update(entries[0], question_id="q2")
    else:
        entries[0]["evidence_spans"].append(entries[0]["evidence_spans"][0])
    _write_evidence(root, entries)
    with pytest.raises(CorpusError):
        load_corpus(root)


@pytest.mark.parametrize(
    "splits",
    [
        {"development": ["q1"], "test": ["unknown"]},
        {"development": ["q1", "q1"], "test": ["q2"]},
        {"development": ["q1"], "test": ["q1", "q2"]},
        {"development": ["q1"], "test": []},
        {"development": ["q1"], "test": ["q2"], "fact_families": {"q1": "f"}},
        {"development": ["q1"], "test": ["q2"], "fact_families": {"q1": "f", "q2": "f"}},
        {"development": "q1", "test": ["q2"]},
        {"splits": []},
        [],
    ],
)
def test_invalid_splits_and_family_leakage_are_rejected(tmp_path, splits):
    _write_corpus(tmp_path, [_document()], [_question(), _question(id="q2")])
    (tmp_path / "splits.json").write_text(json.dumps(splits), encoding="utf-8")
    with pytest.raises(CorpusError):
        load_corpus(tmp_path)


def test_duplicate_titles_are_rejected(tmp_path):
    _write_corpus(tmp_path, [_document(), _document("docs/b.docx")], [_question()])
    with pytest.raises(CorpusError, match="Duplicate document titles"):
        load_corpus(tmp_path)


@pytest.mark.parametrize("file", ["../outside.docx", "/tmp/outside.docx"])
def test_manifest_paths_cannot_escape_corpus(tmp_path, file):
    _write_corpus(tmp_path, [_document()], [_question()])
    manifest = {"documents": [_document(file)]}
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(CorpusError, match="inside the corpus"):
        load_corpus(tmp_path)
