class RagError(Exception):
    """Base class for RAG errors that map to client responses."""


class InvalidQuestion(RagError):
    """The question is empty or otherwise unusable."""


class IndexNotReady(RagError):
    """No active index build exists, so retrieval cannot run."""


class ProviderError(RagError):
    """The embedding or generation provider failed or returned unusable output."""
