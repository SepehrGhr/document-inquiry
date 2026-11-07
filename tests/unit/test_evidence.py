from dataclasses import replace

import pytest

from rag.contracts import RetrievedPassage
from rag.evaluation.dataset import EvidenceAnnotation, EvidenceSpan
from rag.evaluation.evidence import context_prefix, context_stats, evidence_at


def passage(start, end, *, title="A", revision="rev"):
    return RetrievedPassage(
        str(start), title, title, revision, 1, 0, "x" * (end - start), 1.0, 1, start, end
    )


def test_spans_can_be_covered_across_chunks_without_double_counting_overlap():
    annotation = EvidenceAnnotation((EvidenceSpan("rule", "a", 2, 12, "x" * 10),), (("rule",),))
    files = {"A": "a"}
    partial = evidence_at(annotation, [passage(0, 7), passage(3, 7)], files)
    assert partial["evidence_coverage"] == 0.5
    assert not partial["sufficient_evidence"]
    complete = evidence_at(annotation, [passage(0, 7), passage(5, 15)], files)
    assert complete["evidence_coverage"] == 1.0
    assert complete["sufficient_evidence"]


def test_alternative_sets_are_ored_but_complementary_spans_are_required():
    annotation = EvidenceAnnotation(
        (
            EvidenceSpan("rule", "a", 0, 5, "xxxxx"),
            EvidenceSpan("en", "b", 0, 5, "xxxxx"),
            EvidenceSpan("fa", "c", 0, 5, "xxxxx"),
        ),
        (("rule", "en"), ("rule", "fa")),
    )
    files = {"A": "a", "B": "b", "C": "c"}
    assert not evidence_at(annotation, [passage(0, 5)], files)["sufficient_evidence"]
    assert evidence_at(annotation, [passage(0, 5), passage(0, 5, title="C")], files)[
        "sufficient_evidence"
    ]
    assert not evidence_at(annotation, [passage(0, 5, title="Unrelated")], files)[
        "sufficient_evidence"
    ]


def test_character_budget_preserves_rank_and_clips_last_passage_offsets():
    passages = [passage(0, 10), passage(5, 15)]
    context = context_prefix(passages, 13)
    assert [len(p.text) for p in context] == [10, 3]
    assert context[1].start == 5 and context[1].end == 8
    assert context_stats(context) == {
        "characters": 13,
        "unique_characters": 10,
        "overlap_fraction": pytest.approx(3 / 13),
    }
    assert passages[1].end == 15
    assert context_prefix(passages, None) is passages


def test_context_overlap_is_revision_scoped_and_empty_annotations_are_not_sufficient():
    context = [passage(0, 5), replace(passage(0, 5), revision_id="other")]
    assert context_stats(context)["overlap_fraction"] == 0
    assert not evidence_at(EvidenceAnnotation((), ()), context, {"A": "a"})["sufficient_evidence"]
