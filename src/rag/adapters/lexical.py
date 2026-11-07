"""PostgreSQL full-text search with simple tokenization and cover-density ranking.

This uses ts_rank_cd, not BM25. OR across query terms keeps multi-document
questions from requiring every word in a single passage. Query terms are bound
parameters, not raw tsquery syntax.
"""

import re
from functools import reduce
from operator import or_

from django.contrib.postgres.search import SearchQuery, SearchRank
from django.db.models import F

from rag.adapters.chunk_queries import PASSAGE_FIELDS, chunks_for_build, to_passage
from rag.search_text import normalize_search_text

QUERY_VERSION = "or-stopwords-1"

# Common function/question words, independent of any evaluation corpus. Keep
# product names, identifiers, numbers, and negation available as search terms.
_QUERY_STOP_WORDS = set(
    """
a an the of for to on from how many much does do has have it its in at and or
be been being was were are is by as with that this these those which who whom
whose what where when why will would should can could may must
از در به با برای تا را که این آن و یا چه چند چگونه کدام آیا چیست است هست هستند
بود بودند شود شده می ها های
""".split()
)


def search(build, text: str, limit: int):
    # Keep hyphenated identifiers together so KAV-240P and KAV-240 remain distinct.
    terms = dict.fromkeys(
        term
        for term in re.findall(r"[^\W_]+(?:-[^\W_]+)*", normalize_search_text(text))
        if term not in _QUERY_STOP_WORDS
    )
    if not terms:
        return []
    query = reduce(or_, (SearchQuery(term, config="simple") for term in terms))
    rows = (
        chunks_for_build(build)
        .filter(search_vector=query)
        .annotate(lexical_score=SearchRank(F("search_vector"), query, cover_density=True))
        .order_by("-lexical_score", "id")
        .values(*PASSAGE_FIELDS, "lexical_score")[:limit]
    )
    return [
        to_passage(row, score=row["lexical_score"], rank=rank)
        for rank, row in enumerate(rows, start=1)
    ]
