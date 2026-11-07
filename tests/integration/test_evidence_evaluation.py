import json
from types import SimpleNamespace

import pytest
from django.core.management import call_command
from django.core.management.base import CommandError

from documents import services
from documents.ingestion.docx import parse_docx
from rag import factory, indexing
from rag.adapters.chunk_queries import PASSAGE_FIELDS, chunks_for_build, to_passage
from rag.evaluation.corpus import load_corpus as ingest
from rag.evaluation.dataset import load_corpus
from rag.evaluation.runner import run_retrieval
from rag.exceptions import ProviderError
from rag.models import IndexBuild, IndexProfile


@pytest.fixture
def annotated_corpus(tmp_path, make_docx):
    (tmp_path / "docs").mkdir()
    texts = {}
    for name, paragraphs in {
        "a": (
            "Policy A",
            "Approval requires QC.",
            "Background material. " * 20,
            "Approval requires Finance.",
        ),
        "b": ("Policy B", "An alternative rule applies."),
    }.items():
        upload = make_docx(paragraphs=paragraphs)
        data = upload.getvalue()
        (tmp_path / f"docs/{name}.docx").write_bytes(data)
        texts[name] = parse_docx(make_docx(paragraphs=paragraphs)).text
    (tmp_path / "manifest.json").write_text(
        json.dumps(
            {
                "company": "Test",
                "documents": [
                    {
                        "file": f"docs/{name}.docx",
                        "title": f"Policy {name.upper()}",
                        "language": "en",
                        "type": "policy",
                        "status": "current",
                    }
                    for name in texts
                ],
            }
        )
    )
    questions = [
        {
            "id": "ld001",
            "question": "Which approvals?",
            "language": "en",
            "difficulty": "multi_hop",
            "answerable": True,
            "expected_answer": "QC and Finance",
            "expected_sources": ["docs/a.docx"],
        },
        {
            "id": "ld002",
            "question": "Which alternative?",
            "language": "en",
            "difficulty": "simple",
            "answerable": True,
            "expected_answer": "The alternative rule",
            "expected_sources": ["docs/b.docx"],
        },
    ]
    (tmp_path / "questions.jsonl").write_text("\n".join(map(json.dumps, questions)))
    annotations = []
    for q, name, quotes in [
        (questions[0], "a", ("Approval requires QC.", "Approval requires Finance.")),
        (questions[1], "b", ("An alternative rule applies.",)),
    ]:
        spans = [
            {
                "id": str(i),
                "file": f"docs/{name}.docx",
                "quote": quote,
                "start": texts[name].index(quote),
                "end": texts[name].index(quote) + len(quote),
            }
            for i, quote in enumerate(quotes)
        ]
        annotations.append(
            {
                "question_id": q["id"],
                "evidence_spans": spans,
                "sufficient_evidence_sets": [[s["id"] for s in spans]],
                "chunking_challenges": ["distant_sections"],
            }
        )
    (tmp_path / "evidence.jsonl").write_text("\n".join(map(json.dumps, annotations)))
    (tmp_path / "splits.json").write_text(
        json.dumps(
            {
                "development": ["ld001"],
                "test": ["ld002"],
                "fact_families": {"ld001": "a", "ld002": "b"},
            }
        )
    )
    return load_corpus(tmp_path)


def test_validate_command_needs_no_database_or_provider(annotated_corpus, capsys):
    call_command("validate_eval_corpus", corpus=str(annotated_corpus.root))
    assert "2 evidence annotations" in capsys.readouterr().out


def profile(name, size):
    return IndexProfile.objects.create(
        name=name,
        provider="fake",
        model_id="hashing",
        dimension=64,
        chunker_version="char-1",
        chunk_size=size,
        chunk_overlap=10,
    )


def all_chunks(build):
    return [
        to_passage(row, score=1.0, rank=i)
        for i, row in enumerate(
            chunks_for_build(build).order_by("document__title", "ordinal").values(*PASSAGE_FIELDS),
            1,
        )
    ]


@pytest.mark.django_db
def test_inactive_build_evaluation_selects_split_and_scores_actual_passages(
    annotated_corpus, staff_user, tmp_path, monkeypatch
):
    ingest(annotated_corpus, actor=staff_user)
    live = indexing.run_build(indexing.create_build(profile("live", 120)).id)
    experimental = profile("small", 80)
    call_command("rebuild_index", profile=experimental.name, sync=True, no_activate=True)
    inactive = IndexBuild.objects.get(profile=experimental)
    assert not inactive.is_active
    live.refresh_from_db()
    assert live.is_active
    passages = all_chunks(inactive)
    assert len(passages) > 5
    monkeypatch.setattr(
        factory,
        "build_retriever",
        lambda *a, **kw: SimpleNamespace(retrieve=lambda text, limit: passages[:limit]),
    )

    output = tmp_path / "result"
    call_command(
        "evaluate_rag",
        corpus=str(annotated_corpus.root),
        profile="small",
        build=str(inactive.id),
        split="development",
        top_k=100,
        output=str(output),
    )
    report = json.loads((output / "retrieval.json").read_text())
    assert report["metadata"]["question_ids"] == ["ld001"]
    assert report["metadata"]["status"] == "completed"
    assert report["metadata"]["index_chunks"] == len(passages)
    assert report["evidence"]["overall"]["sufficient_evidence@10"] == 1.0
    assert report["questions"][0]["passages"][0]["start"] == 0
    assert report["questions"][0]["evidence"]["sufficient_evidence"]
    assert "Evidence coverage" in (output / "retrieval.md").read_text()
    assert report["metadata"]["corpus_hashes"]["evidence.jsonl"]
    live.refresh_from_db()
    assert live.is_active


@pytest.mark.django_db
def test_character_budget_scores_clipped_context_not_the_full_retrieval(
    annotated_corpus, staff_user, monkeypatch
):
    ingest(annotated_corpus, actor=staff_user)
    build = indexing.run_build(indexing.create_build(profile("large", 2000)).id)
    passages = all_chunks(build)
    monkeypatch.setattr(
        factory,
        "build_retriever",
        lambda *a, **kw: SimpleNamespace(retrieve=lambda text, limit: passages[:limit]),
    )
    run = run_retrieval(annotated_corpus, build, top_k=5, split="development", context_chars=12)
    from rag.evaluation.report import retrieval_report

    report = retrieval_report(run, top_k=5)
    assert report["questions"][0]["context"]["characters"] == 12
    assert report["questions"][0]["passages"][0]["end"] == 12
    assert not report["questions"][0]["evidence"]["sufficient_evidence"]
    assert report["overall"]["recall@5"] == 1.0  # Right document, insufficient actual context.


@pytest.mark.django_db
def test_stale_corpus_text_and_missing_split_fail_before_provider_construction(
    annotated_corpus, staff_user, monkeypatch
):
    ingest(annotated_corpus, actor=staff_user)
    doc = services.Document.objects.get(title="Policy A")
    services.replace_content(
        document_id=doc.id,
        content=services.ContentInput(text="Changed policy"),
        expected_revision=1,
        actor=staff_user,
    )
    build = indexing.run_build(indexing.create_build(profile("stale", 120)).id)

    def forbidden(*a, **kw):
        pytest.fail("Provider constructed before corpus validation")

    monkeypatch.setattr(factory, "build_retriever", forbidden)
    from rag.evaluation.dataset import CorpusError

    with pytest.raises(CorpusError, match="differs from the corpus"):
        run_retrieval(annotated_corpus, build, top_k=5)
    with pytest.raises(CorpusError, match="not defined"):
        run_retrieval(annotated_corpus, build, top_k=5, split="unknown")


@pytest.mark.django_db
def test_failed_questions_remain_in_evidence_denominator_and_cli_saves_report(
    annotated_corpus, staff_user, monkeypatch, tmp_path
):
    ingest(annotated_corpus, actor=staff_user)
    build = indexing.run_build(indexing.create_build(profile("failure", 2000)).id)
    passages = all_chunks(build)

    def retrieve(text, *, limit):
        if "alternative" in text:
            raise ProviderError("simulated failure")
        return passages[:limit]

    monkeypatch.setattr(
        factory, "build_retriever", lambda *a, **kw: SimpleNamespace(retrieve=retrieve)
    )
    output = tmp_path / "failed"
    with pytest.raises(CommandError, match="partial reports"):
        call_command(
            "evaluate_rag",
            corpus=str(annotated_corpus.root),
            profile="failure",
            top_k=10,
            output=str(output),
        )
    report = json.loads((output / "retrieval.json").read_text())
    assert report["metadata"]["questions_failed"] == 1
    assert report["metadata"]["status"] == "failed"
    assert report["overall"]["recall@5"] == 1.0
    assert report["overall_all_requested"]["recall@5"] == 0.5
    assert report["evidence"]["overall"]["questions"] == 2
    assert report["evidence"]["overall"]["sufficient_evidence@5"] == 0.5
