"""Pure retrieval and answer metrics.

Everything here takes plain values and returns numbers, so it can be unit
tested without a database, a model, or a network.
"""

from dataclasses import dataclass

RECALL_KS = (1, 3, 5, 10)


@dataclass(frozen=True)
class RetrievalScore:
    """How well one question's retrieved documents match its expected sources."""

    retrieved: tuple[str, ...]
    expected: tuple[str, ...]

    def hit_at(self, k: int) -> bool:
        return bool(set(self.retrieved[:k]) & set(self.expected))

    def recall_at(self, k: int) -> float:
        if not self.expected:
            return 0.0
        found = set(self.retrieved[:k]) & set(self.expected)
        return len(found) / len(set(self.expected))

    def reciprocal_rank(self) -> float:
        expected = set(self.expected)
        for rank, document in enumerate(self.retrieved, start=1):
            if document in expected:
                return 1.0 / rank
        return 0.0


def mean(values) -> float:
    values = list(values)
    return sum(values) / len(values) if values else 0.0


def percentile(values, fraction: float) -> float:
    """Linear-interpolated percentile; good enough for small evaluation sets."""
    values = sorted(values)
    if not values:
        return 0.0
    position = fraction * (len(values) - 1)
    lower = int(position)
    upper = min(lower + 1, len(values) - 1)
    weight = position - lower
    return float(values[lower] * (1 - weight) + values[upper] * weight)


def retrieval_summary(scores: list[RetrievalScore]) -> dict:
    """Aggregate retrieval scores. Questions with no expected sources are skipped."""
    scored = [score for score in scores if score.expected]
    return {
        "questions": len(scored),
        **{f"recall@{k}": mean(s.recall_at(k) for s in scored) for k in RECALL_KS},
        **{f"hit@{k}": mean(s.hit_at(k) for s in scored) for k in RECALL_KS},
        "mrr": mean(s.reciprocal_rank() for s in scored),
    }


@dataclass(frozen=True)
class AnswerScore:
    """The judged outcome of one answer."""

    answerable: bool
    abstained: bool
    correct: bool
    cited_documents: tuple[str, ...]
    expected_documents: tuple[str, ...]
    latency_ms: int | None = None


def citation_precision(score: AnswerScore) -> float:
    cited = set(score.cited_documents)
    if not cited:
        return 0.0
    return len(cited & set(score.expected_documents)) / len(cited)


def citation_recall(score: AnswerScore) -> float:
    expected = set(score.expected_documents)
    if not expected:
        return 0.0
    return len(set(score.cited_documents) & expected) / len(expected)


def answer_summary(scores: list[AnswerScore]) -> dict:
    answerable = [s for s in scores if s.answerable]
    unanswerable = [s for s in scores if not s.answerable]
    latencies = [s.latency_ms for s in scores if s.latency_ms is not None]
    return {
        "questions": len(scores),
        "answer_accuracy": mean(s.correct for s in answerable),
        "correct_abstention": mean(s.abstained for s in unanswerable),
        "false_abstention": mean(s.abstained for s in answerable),
        "citation_precision": mean(citation_precision(s) for s in answerable),
        "citation_recall": mean(citation_recall(s) for s in answerable),
        "latency_p50_ms": percentile(latencies, 0.5),
        "latency_p95_ms": percentile(latencies, 0.95),
    }


def group_by(items, key) -> dict:
    """Group items by a key function, preserving first-seen order."""
    groups: dict = {}
    for item in items:
        groups.setdefault(key(item), []).append(item)
    return groups
