"""Dense retrieval over PostgreSQL/pgvector.

Retrieval is always scoped to the active build's revision set, so superseded
revisions and logically deleted documents can never be returned.
"""

from pgvector.django import CosineDistance, L2Distance, MaxInnerProduct

from rag.adapters.chunk_queries import PASSAGE_FIELDS, chunks_for_build, to_passage
from rag.contracts import RetrievedPassage
from rag.models import IndexBuild, IndexProfile

_DISTANCE = {
    IndexProfile.Metric.COSINE: CosineDistance,
    IndexProfile.Metric.L2: L2Distance,
    IndexProfile.Metric.INNER_PRODUCT: MaxInnerProduct,
}


def search(build: IndexBuild, query_vector: list[float], limit: int) -> list[RetrievedPassage]:
    distance = _DISTANCE[build.profile.metric]("embedding", query_vector)
    rows = (
        chunks_for_build(build)
        .annotate(distance=distance)
        .order_by("distance", "id")
        .values(*PASSAGE_FIELDS, "distance")[:limit]
    )
    return [
        to_passage(row, score=1.0 - row["distance"], rank=rank)
        for rank, row in enumerate(rows, start=1)
    ]


def active_build() -> IndexBuild | None:
    return (
        IndexBuild.objects.filter(is_active=True, status=IndexBuild.Status.COMPLETED)
        .select_related("profile")
        .first()
    )
