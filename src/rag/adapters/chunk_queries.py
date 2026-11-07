"""Shared revision/deletion scope and passage mapping for PostgreSQL retrievers."""

from rag.contracts import RetrievedPassage
from rag.models import IndexedChunk

PASSAGE_FIELDS = (
    "id",
    "document_id",
    "document__title",
    "revision_id",
    "revision__sequence",
    "ordinal",
    "text",
    "start",
    "end",
)


def chunks_for_build(build):
    return IndexedChunk.objects.filter(
        profile=build.profile,
        revision__in=build.revisions.values("revision_id"),
        document__deleted_at__isnull=True,
    )


def to_passage(row, *, score, rank):
    return RetrievedPassage(
        chunk_id=str(row["id"]),
        document_id=str(row["document_id"]),
        document_title=row["document__title"],
        revision_id=str(row["revision_id"]),
        revision_sequence=row["revision__sequence"],
        ordinal=row["ordinal"],
        text=row["text"],
        score=score,
        rank=rank,
        start=row["start"],
        end=row["end"],
    )
