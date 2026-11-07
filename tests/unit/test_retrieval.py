from dataclasses import replace

import pytest

from rag.contracts import RetrievedPassage
from rag.retrieval import reciprocal_rank_fusion
from rag.search_text import normalize_search_text


def passage(key, score=1.0):
    return RetrievedPassage(key, "doc", "Title", "rev", 1, 0, "exact text", score, 1)


def test_rrf_rewards_agreement_without_mixing_raw_scores():
    dense = [passage("a", 0.99), passage("c", 0.98), passage("b", 0.1)]
    lexical = [passage("b", 1000), passage("d", 999)]

    results = reciprocal_rank_fusion(dense, lexical, limit=3)

    assert [p.chunk_id for p in results] == ["b", "a", "c"]
    assert [p.rank for p in results] == [1, 2, 3]
    assert results[0].score == pytest.approx(1 / 63 + 1 / 61)
    assert results[0].text == "exact text"


def test_rrf_deduplicates_chunks_per_list_and_breaks_ties_stably():
    a, b = passage("a"), passage("b")
    results = reciprocal_rank_fusion([b, b], [a], limit=5)
    assert [p.chunk_id for p in results] == ["a", "b"]
    assert all(p.score == pytest.approx(1 / 61) for p in results)
    assert reciprocal_rank_fusion([], [], limit=5) == []


def test_rrf_keeps_different_chunks_from_the_same_document():
    a = passage("a")
    b = replace(a, chunk_id="b", ordinal=1)
    assert len(reciprocal_rank_fusion([a, b], [], limit=5)) == 2


def test_lexical_normalization_handles_persian_variants_and_preserves_codes():
    assert normalize_search_text("آجر كاوه می‌رود ۲۴۰ ٢٤٠ KAV-240P") == (
        "آجر کاوه می رود 240 240 kav-240p"
    )
    assert normalize_search_text("  كَــاوه\n\tساناسا  ") == "کاوه ساناسا"
