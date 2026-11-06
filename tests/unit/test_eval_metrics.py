import pytest

from rag.evaluation.metrics import (
    AnswerScore,
    RetrievalScore,
    answer_summary,
    citation_precision,
    citation_recall,
    mean,
    percentile,
    retrieval_summary,
)


def test_recall_and_hit_at_k():
    score = RetrievalScore(retrieved=("a", "b", "c"), expected=("c", "d"))

    assert score.hit_at(1) is False
    assert score.hit_at(3) is True
    assert score.recall_at(3) == 0.5
    assert score.recall_at(10) == 0.5


def test_reciprocal_rank_uses_first_expected_document():
    assert RetrievalScore(("x", "y", "a"), ("a",)).reciprocal_rank() == pytest.approx(1 / 3)
    assert RetrievalScore(("x", "y"), ("a",)).reciprocal_rank() == 0.0


def test_retrieval_summary_skips_questions_without_expected_sources():
    scores = [
        RetrievalScore(("a",), ("a",)),
        RetrievalScore(("b",), ("a",)),
        RetrievalScore(("c",), ()),
    ]

    summary = retrieval_summary(scores)

    assert summary["questions"] == 2
    assert summary["recall@1"] == 0.5
    assert summary["mrr"] == pytest.approx(0.5)


def test_citation_precision_and_recall():
    score = AnswerScore(
        answerable=True,
        abstained=False,
        correct=True,
        cited_documents=("a", "b"),
        expected_documents=("a", "c"),
    )

    assert citation_precision(score) == 0.5
    assert citation_recall(score) == 0.5


def test_answer_summary_separates_answerable_and_unanswerable():
    scores = [
        AnswerScore(True, False, True, ("a",), ("a",), latency_ms=100),
        AnswerScore(True, True, False, (), ("a",), latency_ms=200),
        AnswerScore(False, True, True, (), (), latency_ms=300),
        AnswerScore(False, False, False, ("a",), (), latency_ms=400),
    ]

    summary = answer_summary(scores)

    assert summary["answer_accuracy"] == 0.5
    assert summary["false_abstention"] == 0.5
    assert summary["correct_abstention"] == 0.5
    assert summary["latency_p50_ms"] == 250.0


def test_mean_and_percentile_handle_empty_input():
    assert mean([]) == 0.0
    assert percentile([], 0.5) == 0.0
    assert percentile([10, 20, 30, 40], 0.5) == 25.0
    assert percentile([10, 20, 30, 40], 0.0) == 10.0
    assert percentile([10, 20, 30, 40], 1.0) == 40.0
