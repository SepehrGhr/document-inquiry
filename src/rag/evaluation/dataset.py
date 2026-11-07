"""Loading and validating the evaluation corpus.

The corpus is a folder with `manifest.json` (one entry per document) and
`questions.jsonl` (one question per line). Optional `evidence.jsonl` annotates
exact parser-text spans and alternative sufficient sets; `splits.json` defines
development/test membership. Validation is strict: a malformed corpus should
fail loudly before any model calls are made.
"""

import json
from dataclasses import dataclass
from dataclasses import field as dataclass_field
from pathlib import Path

from documents.exceptions import InvalidDocumentInput
from documents.ingestion.docx import parse_docx

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
class EvidenceSpan:
    id: str
    file: str
    start: int
    end: int
    quote: str


@dataclass(frozen=True)
class EvidenceAnnotation:
    spans: tuple[EvidenceSpan, ...]
    sufficient_sets: tuple[tuple[str, ...], ...]
    answer_behavior: str = ""
    challenges: tuple[str, ...] = ()


@dataclass(frozen=True)
class Corpus:
    root: Path
    company: str
    documents: tuple[CorpusDocument, ...]
    questions: tuple[EvalQuestion, ...]
    evidence: dict[str, EvidenceAnnotation] = dataclass_field(default_factory=dict)
    splits: dict[str, tuple[str, ...]] = dataclass_field(default_factory=dict)

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
    titles = [document.title for document in documents]
    if len(set(titles)) != len(titles):
        raise CorpusError("Duplicate document titles in manifest.")
    for file in files:
        _source_path(root, file)
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
        evidence=_load_evidence(root, questions, set(files)),
        splits=_load_splits(root, {question.id for question in questions}),
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


def _source_path(root: Path, file: str) -> Path:
    path = (root / file).resolve()
    if Path(file).is_absolute() or not path.is_relative_to(root.resolve()):
        raise CorpusError(f"Source path must stay inside the corpus: {file!r}.")
    return path


def _load_evidence(root, questions, known_files) -> dict[str, EvidenceAnnotation]:
    path = root / "evidence.jsonl"
    if not path.exists():
        return {}
    by_id = {question.id: question for question in questions}
    annotations = {}
    texts = {}
    for number, line in _read_jsonl(path):
        try:
            entry = json.loads(line)
        except json.JSONDecodeError as exc:
            raise CorpusError(f"evidence.jsonl line {number} is not valid JSON: {exc}") from exc
        if (
            not isinstance(entry, dict)
            or not {"question_id", "evidence_spans", "sufficient_evidence_sets"} <= entry.keys()
        ):
            raise CorpusError(
                f"Unsupported annotation shape on evidence.jsonl line {number}; expected "
                "question_id, evidence_spans (list of id/file/start/end/quote objects), "
                "and sufficient_evidence_sets (list of span-ID lists)."
            )
        qid = entry["question_id"]
        if not isinstance(qid, str) or qid not in by_id:
            raise CorpusError(f"Evidence references unknown question ID {qid!r}.")
        if qid in annotations:
            raise CorpusError(f"Duplicate evidence question ID {qid!r}.")
        spans = entry["evidence_spans"]
        sets = entry["sufficient_evidence_sets"]
        behavior = entry.get("answer_behavior", "")
        challenges = entry.get("chunking_challenges", [])
        if (
            not isinstance(spans, list)
            or not isinstance(sets, list)
            or not isinstance(behavior, str)
            or not isinstance(challenges, list)
            or any(not isinstance(value, str) for value in challenges)
        ):
            raise CorpusError(f"Invalid evidence annotation shape for {qid}.")
        parsed_spans = {}
        for span in spans:
            if (
                not isinstance(span, dict)
                or not {"id", "file", "start", "end", "quote"} <= span.keys()
            ):
                raise CorpusError(f"Evidence span for {qid} needs id/file/start/end/quote.")
            sid, file = span["id"], span["file"]
            start, end, quote = span["start"], span["end"], span["quote"]
            if not isinstance(sid, str) or not sid.strip() or sid in parsed_spans:
                raise CorpusError(f"Invalid or duplicate evidence span ID for {qid}: {sid!r}.")
            if not isinstance(file, str) or file not in known_files:
                raise CorpusError(f"Evidence {qid}/{sid} cites unknown source {file!r}.")
            if type(start) is not int or type(end) is not int:
                raise CorpusError(f"Evidence {qid}/{sid} offsets must be integers (not booleans).")
            if not isinstance(quote, str) or not quote:
                raise CorpusError(f"Evidence {qid}/{sid} quote must be nonempty text.")
            source = _source_path(root, file)
            if source not in texts:
                try:
                    with source.open("rb") as upload:
                        texts[source] = parse_docx(upload).text
                except InvalidDocumentInput as exc:
                    raise CorpusError(f"Cannot parse evidence source {file!r}: {exc}") from exc
                extracted = root / "extracted" / f"{source.stem}.txt"
                if extracted.exists():
                    with extracted.open(encoding="utf-8", newline="") as stream:
                        if stream.read() != texts[source]:
                            raise CorpusError(
                                f"Extracted text {extracted} differs from parser output."
                            )
            text = texts[source]
            if not 0 <= start < end <= len(text):
                raise CorpusError(f"Evidence {qid}/{sid} offsets are out of bounds.")
            if text[start:end] != quote:
                raise CorpusError(f"Evidence {qid}/{sid} quote does not match canonical text.")
            parsed_spans[sid] = EvidenceSpan(sid, file, start, end, quote)
        for sufficient in sets:
            if (
                not isinstance(sufficient, list)
                or not sufficient
                or any(not isinstance(sid, str) or sid not in parsed_spans for sid in sufficient)
                or len(set(sufficient)) != len(sufficient)
            ):
                raise CorpusError(f"Invalid sufficient evidence set for {qid}: {sufficient!r}.")
        question = by_id[qid]
        if question.answerable:
            if not sets:
                raise CorpusError(f"Answerable question {qid} needs a nonempty sufficient set.")
            if not any(
                {parsed_spans[sid].file for sid in sufficient} == set(question.expected_sources)
                for sufficient in sets
            ):
                raise CorpusError(
                    f"Question {qid} expected_sources match no sufficient evidence set."
                )
        elif sets:
            raise CorpusError(f"Unanswerable question {qid} must have no sufficient evidence sets.")
        annotations[qid] = EvidenceAnnotation(
            tuple(parsed_spans.values()),
            tuple(tuple(group) for group in sets),
            behavior,
            tuple(challenges),
        )
    missing = by_id.keys() - annotations.keys()
    if missing:
        raise CorpusError(f"Missing evidence annotations for questions: {sorted(missing)}.")
    return annotations


def _load_splits(root, question_ids) -> dict[str, tuple[str, ...]]:
    path = root / "splits.json"
    if not path.exists():
        return {}
    entry = _read_json(path)
    if not isinstance(entry, dict):
        raise CorpusError("splits.json must be an object with development and test lists.")
    groups = entry.get("splits", entry)
    allowed = {"development", "test"}
    if (
        not isinstance(groups, dict)
        or not allowed <= groups.keys()
        or groups.keys() - allowed - ({"fact_families"} if groups is entry else set())
        or ("splits" in entry and entry.keys() - {"splits", "fact_families"})
    ):
        raise CorpusError(
            "splits.json must contain development and test lists, optionally under splits."
        )
    membership = {}
    for name in ("development", "test"):
        if not isinstance(groups[name], list):
            raise CorpusError(f"Split {name} must be a list of question IDs.")
        for qid in groups[name]:
            if not isinstance(qid, str) or qid not in question_ids:
                raise CorpusError(f"Split {name} references unknown question ID {qid!r}.")
            if qid in membership:
                raise CorpusError(f"Duplicate question ID in splits: {qid!r}.")
            membership[qid] = name
    if membership.keys() != question_ids:
        raise CorpusError("Split membership must include every question exactly once.")
    if "fact_families" in entry:
        families = entry["fact_families"]
        if (
            not isinstance(families, dict)
            or families.keys() != question_ids
            or any(
                not isinstance(family, str) or not family.strip() for family in families.values()
            )
        ):
            raise CorpusError("fact_families must map every question ID to a nonempty family ID.")
        family_splits = {}
        for qid, family in families.items():
            if family in family_splits and family_splits[family] != membership[qid]:
                raise CorpusError(f"Fact family {family!r} crosses splits.")
            family_splits[family] = membership[qid]
    return {name: tuple(groups[name]) for name in ("development", "test")}
