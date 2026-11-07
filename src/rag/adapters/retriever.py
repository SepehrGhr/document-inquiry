"""Dense, lexical, and hybrid retrieval using one build and consistent limits."""

from rag.adapters import lexical, pgvector_index
from rag.retrieval import reciprocal_rank_fusion

RETRIEVAL_MODES = ("dense", "lexical", "hybrid_rrf")


class PostgresRetriever:
    def __init__(self, build, embedder, *, mode, candidate_k=20, rank_constant=60):
        if mode not in RETRIEVAL_MODES:
            raise ValueError(f"Unknown retrieval mode {mode!r}.")
        if candidate_k < 1 or rank_constant < 1:
            raise ValueError("candidate_k and rank_constant must be positive.")
        self.build = build
        self.embedder = embedder
        self.mode = mode
        self.candidate_k = candidate_k
        self.rank_constant = rank_constant

    def retrieve(self, text, *, limit):
        if limit < 1:
            raise ValueError("limit must be positive.")
        if self.mode == "lexical":
            return lexical.search(self.build, text, limit)
        vector = self.embedder.embed_query(text)
        pool = max(self.candidate_k, limit) if self.mode == "hybrid_rrf" else limit
        dense = pgvector_index.search(self.build, vector, pool)
        if self.mode == "dense":
            return dense
        return reciprocal_rank_fusion(
            dense,
            lexical.search(self.build, text, pool),
            limit=limit,
            rank_constant=self.rank_constant,
        )
