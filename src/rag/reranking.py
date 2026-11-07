"""Optional second-stage ranking over a larger retrieval shortlist."""

from rag.contracts import Reranker, RetrievedPassage, Retriever


class RerankingRetriever:
    def __init__(self, retriever: Retriever, reranker: Reranker, *, candidate_k: int = 20):
        if candidate_k < 1:
            raise ValueError("candidate_k must be positive.")
        self.retriever = retriever
        self.reranker = reranker
        self.candidate_k = candidate_k

    def retrieve(self, text: str, *, limit: int) -> list[RetrievedPassage]:
        if limit < 1:
            raise ValueError("limit must be positive.")
        candidates = self.retriever.retrieve(text, limit=max(self.candidate_k, limit))
        if not candidates:
            return []
        return self.reranker.rerank(text, candidates, limit=limit)
