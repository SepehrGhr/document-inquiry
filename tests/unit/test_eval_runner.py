from rag.contracts import RetrievedPassage
from rag.evaluation.dataset import Corpus, CorpusDocument, EvalQuestion
from rag.evaluation.runner import _document_files


def _passage(title, rank=1):
    return RetrievedPassage(
        chunk_id=f"c{rank}",
        document_id="d",
        document_title=title,
        revision_id="r",
        revision_sequence=1,
        ordinal=0,
        text="t",
        score=1.0,
        rank=rank,
    )


def _corpus():
    return Corpus(
        root=None,
        company="Sanasa",
        documents=(
            CorpusDocument("docs/a.docx", "A", "en", "policy", "current", None, None, ""),
            CorpusDocument("docs/b.docx", "B", "en", "policy", "current", None, None, ""),
        ),
        questions=(EvalQuestion("q1", "?", "en", "simple", True, "x", ("docs/a.docx",)),),
    )


def test_document_files_maps_titles_to_manifest_paths_in_rank_order():
    passages = [_passage("B", 1), _passage("A", 2), _passage("B", 3)]

    assert _document_files(passages, _corpus()) == ("docs/b.docx", "docs/a.docx")


def test_document_files_ignores_documents_outside_the_corpus():
    passages = [_passage("Unrelated", 1), _passage("A", 2)]

    assert _document_files(passages, _corpus()) == ("docs/a.docx",)
