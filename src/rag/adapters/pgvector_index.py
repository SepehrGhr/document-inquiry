"""Dense retrieval over PostgreSQL/pgvector.

Retrieval is always scoped to the active build's revision set, so superseded
revisions and logically deleted documents can never be returned.
"""

from pgvector.django import CosineDistance, L2Distance, MaxInnerProduct

from rag.contracts import RetrievedPassage
from rag.models import IndexBuild, IndexedChunk, IndexProfile

_DISTANCE = {
    IndexProfile.Metric.COSINE: CosineDistance,
    IndexProfile.Metric.L2: L2Distance,
    IndexProfile.Metric.INNER_PRODUCT: MaxInnerProduct,
}


def search(build: IndexBuild, query_vector: list[float], limit: int) -> list[RetrievedPassage]:
    distance = _DISTANCE[build.profile.metric]("embedding", query_vector)
    rows = (
        IndexedChunk.objects.filter(
            profile=build.profile,
            revision__in=build.revisions.values("revision_id"),
            document__deleted_at__isnull=True,
        )
        .annotate(distance=distance)
        .order_by("distance")
        .values(
            "id",
            "document_id",
            "document__title",
            "revision_id",
            "revision__sequence",
            "ordinal",
            "text",
            "distance",
        )[:limit]
    )
    return [
        RetrievedPassage(
            chunk_id=str(row["id"]),
            document_id=str(row["document_id"]),
            document_title=row["document__title"],
            revision_id=str(row["revision_id"]),
            revision_sequence=row["revision__sequence"],
            ordinal=row["ordinal"],
            text=row["text"],
            score=1.0 - row["distance"],
            rank=rank,
        )
        for rank, row in enumerate(rows, start=1)
    ]


def active_build() -> IndexBuild | None:
    return (
        IndexBuild.objects.filter(is_active=True, status=IndexBuild.Status.COMPLETED)
        .select_related("profile")
        .first()
    )
