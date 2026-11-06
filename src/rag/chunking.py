"""Deterministic, character-based chunking.

Chunks are fixed-size windows with overlap, cut at paragraph or sentence
boundaries when one is close to the window edge. The source text is never
normalized: offsets always point into the exact stored revision text, which
matters for Persian half-spaces and zero-width characters.
"""

from rag.contracts import Chunk

CHUNKER_VERSION = "char-1"

DEFAULT_CHUNK_SIZE = 1200
DEFAULT_OVERLAP = 200

# How far back from the window edge we look for a nicer cut point.
_BOUNDARY_SEARCH = 200
_PARAGRAPH_BREAKS = ("\n\n", "\n")
_SENTENCE_BREAKS = (". ", "؟ ", "! ", "? ", "。", "। ")


class CharacterChunker:
    version = CHUNKER_VERSION

    def __init__(self, chunk_size: int = DEFAULT_CHUNK_SIZE, overlap: int = DEFAULT_OVERLAP):
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if not 0 <= overlap < chunk_size:
            raise ValueError("overlap must be in [0, chunk_size)")
        self.chunk_size = chunk_size
        self.overlap = overlap

    def split(self, text: str) -> list[Chunk]:
        text = text.strip()
        if not text:
            return []

        chunks: list[Chunk] = []
        start = 0
        length = len(text)
        while start < length:
            end = min(start + self.chunk_size, length)
            if end < length:
                end = self._cut_point(text, start, end)
            piece = text[start:end]
            if piece.strip():
                chunks.append(Chunk(ordinal=len(chunks), text=piece, start=start, end=end))
            if end >= length:
                break
            start = max(end - self.overlap, start + 1)
        return chunks

    def _cut_point(self, text: str, start: int, end: int) -> int:
        """Prefer a paragraph break, then a sentence break, near the window edge."""
        floor = max(start + self.chunk_size // 2, end - _BOUNDARY_SEARCH)
        window = text[floor:end]
        for breaks in (_PARAGRAPH_BREAKS, _SENTENCE_BREAKS):
            for marker in breaks:
                index = window.rfind(marker)
                if index != -1:
                    return floor + index + len(marker)
        return end
