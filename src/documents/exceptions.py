class DocumentError(Exception):
    """Base class for document lifecycle errors that map to client responses."""


class InvalidDocumentInput(DocumentError):
    """The request is malformed or the uploaded file cannot be accepted."""


class DocumentNotFound(DocumentError):
    """The document does not exist or has been deleted."""


class RevisionConflict(DocumentError):
    """The document changed since the caller last read it."""
