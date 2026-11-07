"""Rank fusion over application-owned passages, independent of storage/providers."""

from dataclasses import replace

from rag.contracts import RetrievedPassage


def reciprocal_rank_fusion(
    *rankings: list[RetrievedPassage], limit: int, rank_constant: int = 60
) -> list[RetrievedPassage]:
    """Give each unique chunk one vote per list; use ranks, never raw scores."""
    if limit < 1 or rank_constant < 1:
        raise ValueError("limit and rank_constant must be positive.")
    scores: dict[str, float] = {}
    passages: dict[str, RetrievedPassage] = {}
    for ranking in rankings:
        seen = set()
        for rank, passage in enumerate(ranking, start=1):
            key = passage.chunk_id
            if key in seen:
                continue
            seen.add(key)
            passages.setdefault(key, passage)
            scores[key] = scores.get(key, 0.0) + 1.0 / (rank_constant + rank)
    ordered = sorted(scores, key=lambda key: (-scores[key], key))[:limit]
    return [
        replace(passages[key], score=scores[key], rank=rank)
        for rank, key in enumerate(ordered, start=1)
    ]
