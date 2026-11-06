"""Loading and validating the evaluation corpus.

The corpus is a folder with `manifest.json` (one entry per document) and
`questions.jsonl` (one question per line). Validation is strict: a malformed
corpus should fail loudly before any model calls are made.
"""

import json
from dataclasses import dataclass
from pathlib import Path

DIFFICULTIES = ("simple", "exact_term", "multi_hop", "ambiguous", "unanswerable", "adversarial")
LANGUAGES = ("en", "fa", "mixed", "de", "ar")


class CorpusError(Exception):
    """The corpus folder is missing files or contains invalid entries."""


@dataclass(frozen=True)
class CorpusDocument:
    file: str
    title: str
    language: str
    type: str
    status: str
    effective_date: str | None
    supersedes: str | None
    notes: str

    @property
    def path(self) -> str:
        return self.file


@dataclass(frozen=True)
class EvalQuestion:
    id: str
    question: str
    language: str
    difficulty: str
    answerable: bool
    expected_answer: str
    expected_sources: tuple[str, ...]


@dataclass(frozen=True)
class Corpus:
    root: Path
    company: str
    documents: tuple[CorpusDocument, ...]
    questions: tuple[EvalQuestion, ...]

    def document(self, file: str) -> CorpusDocument:
        for document in self.documents:
            if document.file == file:
                return document
        raise CorpusError(f"Unknown document {file!r}.")


def load_corpus(root) -> Corpus:
    root = Path(root)
    manifest_path = root / "manifest.json"
    questions_path = root / "questions.jsonl"
    if not manifest_path.is_file():
        raise CorpusError(f"Missing {manifest_path}.")
    if not questions_path.is_file():
        raise CorpusError(f"Missing {questions_path}.")

    manifest = _read_json(manifest_path)
    documents = tuple(_parse_document(entry) for entry in manifest.get("documents", []))
    if not documents:
        raise CorpusError("The manifest lists no documents.")

    files = [document.file for document in documents]
    duplicates = {file for file in files if files.count(file) > 1}
    if duplicates:
        raise CorpusError(f"Duplicate documents in manifest: {sorted(duplicates)}.")
    missing = [file for file in files if not (root / file).is_file()]
    if missing:
        raise CorpusError(f"Manifest lists files that do not exist: {missing}.")

    questions = tuple(_parse_question(line, number) for number, line in _read_jsonl(questions_path))
    if not questions:
        raise CorpusError("The questions file is empty.")
    _check_question_sources(questions, set(files))

    return Corpus(
        root=root,
        company=manifest.get("company", ""),
        documents=documents,
        questions=questions,
    )


def _read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise CorpusError(f"{path} is not valid JSON: {exc}") from exc


def _read_jsonl(path: Path):
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if line.strip():
            yield number, line


def _parse_document(entry: dict) -> CorpusDocument:
    for field in ("file", "title", "language", "type", "status"):
        if not entry.get(field):
            raise CorpusError(f"Document entry is missing {field!r}: {entry}.")
    return CorpusDocument(
        file=entry["file"],
        title=entry["title"],
        language=entry["language"],
        type=entry["type"],
        status=entry["status"],
        effective_date=entry.get("effective_date"),
        supersedes=entry.get("supersedes"),
        notes=entry.get("notes", ""),
    )


def _parse_question(line: str, number: int) -> EvalQuestion:
    try:
        entry = json.loads(line)
    except json.JSONDecodeError as exc:
        raise CorpusError(f"questions.jsonl line {number} is not valid JSON: {exc}") from exc

    for field in ("id", "question", "language", "difficulty", "expected_answer"):
        if not entry.get(field):
            raise CorpusError(f"Question on line {number} is missing {field!r}.")
    if entry["difficulty"] not in DIFFICULTIES:
        raise CorpusError(f"Question {entry['id']} has unknown difficulty {entry['difficulty']!r}.")
    if entry["language"] not in LANGUAGES:
        raise CorpusError(f"Question {entry['id']} has unknown language {entry['language']!r}.")
    if not isinstance(entry.get("answerable"), bool):
        raise CorpusError(f"Question {entry['id']} must set answerable to true or false.")

    return EvalQuestion(
        id=entry["id"],
        question=entry["question"],
        language=entry["language"],
        difficulty=entry["difficulty"],
        answerable=entry["answerable"],
        expected_answer=entry["expected_answer"],
        expected_sources=tuple(entry.get("expected_sources") or ()),
    )


def _check_question_sources(questions, known_files: set[str]) -> None:
    ids = [question.id for question in questions]
    duplicates = {qid for qid in ids if ids.count(qid) > 1}
    if duplicates:
        raise CorpusError(f"Duplicate question IDs: {sorted(duplicates)}.")
    for question in questions:
        unknown = [source for source in question.expected_sources if source not in known_files]
        if unknown:
            raise CorpusError(f"Question {question.id} cites unknown sources: {unknown}.")
        if question.answerable and not question.expected_sources:
            raise CorpusError(f"Answerable question {question.id} has no expected sources.")
