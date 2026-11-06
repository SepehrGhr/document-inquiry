from rag.chunking import CharacterChunker


def test_short_text_is_one_chunk_with_exact_offsets():
    text = "متن کوتاه\nEnglish line"
    chunks = CharacterChunker(chunk_size=100, overlap=10).split(text)

    assert len(chunks) == 1
    assert chunks[0].text == text
    assert (chunks[0].start, chunks[0].end) == (0, len(text))


def test_long_text_is_split_with_overlap_and_offsets_point_into_source():
    text = " ".join(f"sentence {i}." for i in range(200))
    chunks = CharacterChunker(chunk_size=200, overlap=40).split(text)

    assert len(chunks) > 1
    assert [c.ordinal for c in chunks] == list(range(len(chunks)))
    for chunk in chunks:
        assert text[chunk.start : chunk.end] == chunk.text
    for previous, current in zip(chunks, chunks[1:], strict=False):
        assert current.start < previous.end  # overlap


def test_blank_text_produces_no_chunks():
    assert CharacterChunker().split("   \n\n ") == []
