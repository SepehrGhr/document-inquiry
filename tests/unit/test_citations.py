from rag.citations import extract_citations, validate_citations


def test_extracts_distinct_citations_in_order():
    assert extract_citations("A [S2] and B [S1][S2].") == [2, 1]


def test_validate_splits_supported_and_unsupported():
    valid, invalid = validate_citations("See [S1] and [S9].", passage_count=3)

    assert valid == [1]
    assert invalid == [9]
