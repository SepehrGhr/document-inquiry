"""Prepare an explicit, auditable evaluation copy; never execute corpus code.

Run trusted repository code with Python -I. Semantic annotation validation remains
 the responsibility of validate_eval_corpus after preparation.
"""

import argparse
import hashlib
import json
import shutil
from pathlib import Path


class PreparationError(ValueError):
    """The input cannot be prepared without guessing or overwriting data."""


def _read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def _rows(path, id_key):
    rows = [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]
    ids = []
    for row in rows:
        qid = row.get(id_key) if isinstance(row, dict) else None
        if not isinstance(qid, str) or not qid.strip() or qid in ids:
            raise PreparationError(f"Invalid or duplicate {id_key} in {path.name}: {qid!r}")
        ids.append(qid)
    if not ids:
        raise PreparationError(f"Empty {path.name}")
    return rows, set(ids)


def _splits(entry, question_ids):
    if not isinstance(entry, dict):
        raise PreparationError("splits.json must be an object")
    membership, families = {}, {}
    if "assignments" in entry:
        if not isinstance(entry["assignments"], list):
            raise PreparationError("assignments must be a list")
        assignments = entry["assignments"]
        for row in assignments:
            if not isinstance(row, dict):
                raise PreparationError("Each assignment must be an object")
            qid, name = row.get("question_id"), row.get("split")
            if not isinstance(qid, str) or qid not in question_ids or qid in membership:
                raise PreparationError(f"Unknown or duplicate split question ID: {qid!r}")
            if name not in ("development", "test"):
                raise PreparationError(f"Unknown split: {name!r}")
            membership[qid] = name
            families[qid] = row.get("fact_family_id")
    else:
        groups = entry.get("splits", entry)
        allowed = {"development", "test"}
        if (
            not isinstance(groups, dict)
            or not allowed <= groups.keys()
            or groups.keys() - allowed - ({"fact_families"} if groups is entry else set())
            or ("splits" in entry and entry.keys() - {"splits", "fact_families"})
        ):
            raise PreparationError("Expected development/test split lists")
        for name in ("development", "test"):
            if not isinstance(groups[name], list):
                raise PreparationError(f"Split {name} must be a list")
            for qid in groups[name]:
                if not isinstance(qid, str) or qid not in question_ids or qid in membership:
                    raise PreparationError(f"Unknown or duplicate split question ID: {qid!r}")
                membership[qid] = name
        families = entry.get("fact_families", {})
    if membership.keys() != question_ids:
        raise PreparationError("Split membership must include every question exactly once")
    if not isinstance(families, dict) or (
        families
        and (
            families.keys() != question_ids
            or any(not isinstance(value, str) or not value.strip() for value in families.values())
        )
    ):
        raise PreparationError("Fact families must map every question ID to a nonempty name")
    return membership, families


def prepare(source, output, exclusions=(), allow_family_leakage=False):
    source, output = Path(source).resolve(), Path(output).absolute()
    if output.exists() or output.is_symlink():
        raise PreparationError("Output must be a fresh directory (no overwrite)")
    if output.resolve().is_relative_to(source):
        raise PreparationError("Output must be outside the source corpus")

    def source_file(relative):
        path = source / relative
        if (
            Path(relative).is_absolute()
            or ".." in Path(relative).parts
            or not path.resolve().is_relative_to(source)
            or any(
                parent.is_symlink()
                for parent in (path, *path.parents)
                if parent != source and parent.is_relative_to(source)
            )
            or not path.is_file()
        ):
            raise PreparationError(f"Missing, unsafe, or symlinked source file: {relative!r}")
        return path

    manifest_path = source_file("manifest.json")
    questions_path = source_file("questions.jsonl")
    evidence_path = source_file("evidence.jsonl")
    splits_path = source_file("splits.json")
    questions, question_ids = _rows(questions_path, "id")
    evidence, evidence_ids = _rows(evidence_path, "question_id")
    if evidence_ids != question_ids:
        raise PreparationError("Evidence IDs must match question IDs exactly")
    excluded = set(exclusions)
    if excluded - question_ids:
        raise PreparationError(f"Unknown exclusions: {sorted(excluded - question_ids)}")
    retained = question_ids - excluded
    if not retained:
        raise PreparationError("Exclusions remove every question")
    membership, families = _splits(_read_json(splits_path), question_ids)
    family_members = {}
    for qid, family in families.items():
        if qid in retained:
            family_members.setdefault(family, {}).setdefault(membership[qid], []).append(qid)
    leakage = {
        family: {name: sorted(ids) for name, ids in groups.items()}
        for family, groups in family_members.items()
        if len(groups) > 1
    }
    if leakage and not allow_family_leakage:
        raise PreparationError(
            "Fact families cross splits; exploratory preparation requires --allow-family-leakage"
        )
    normalized = {
        name: [qid for qid, split in membership.items() if split == name and qid in retained]
        for name in ("development", "test")
    }
    if families and not leakage:
        normalized["fact_families"] = {qid: families[qid] for qid in membership if qid in retained}

    manifest = _read_json(manifest_path)
    documents = manifest.get("documents") if isinstance(manifest, dict) else None
    if not isinstance(documents, list) or not documents:
        raise PreparationError("Manifest must list documents")
    copies = {"manifest.json": manifest_path}
    for document in documents:
        relative = document.get("file") if isinstance(document, dict) else None
        if (
            not isinstance(relative, str)
            or len(Path(relative).parts) < 2
            or Path(relative).parts[0] != "docs"
            or Path(relative).suffix != ".docx"
            or relative in copies
        ):
            raise PreparationError(f"Invalid or duplicate manifest DOCX path: {relative!r}")
        copies[relative] = source_file(relative)
        extracted = f"extracted/{Path(relative).stem}.txt"
        if (source / extracted).exists() or (source / extracted).is_symlink():
            copies[extracted] = source_file(extracted)
    originals = {
        **copies,
        "questions.jsonl": questions_path,
        "evidence.jsonl": evidence_path,
        "splits.json": splits_path,
    }
    hashes = {
        name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in originals.items()
    }
    preparation = {
        "source": str(source),
        "excluded_question_ids": sorted(excluded),
        "retained_question_ids": sorted(retained),
        "source_sha256": hashes,
        "original_splits_file": "original-splits.json",
        "allow_family_leakage": allow_family_leakage,
        "family_leakage": leakage,
        "fact_families_omitted_from_normalized_splits": bool(leakage),
        "interpretation": (
            "Exploratory only: original family separation is invalid; no questions were moved."
            if leakage
            else "Prepared copy; semantic annotation validation is still required."
        ),
    }
    # All structural checks precede directory creation. mkdir is exclusive even
    # when another process creates the destination after the initial check.
    output.mkdir(parents=True, exist_ok=False)
    for relative, path in copies.items():
        target = output / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
    shutil.copyfile(splits_path, output / "original-splits.json")
    for name, rows, key in (
        ("questions.jsonl", questions, "id"),
        ("evidence.jsonl", evidence, "question_id"),
    ):
        (output / name).write_text(
            "".join(
                json.dumps(row, ensure_ascii=False) + "\n" for row in rows if row[key] in retained
            ),
            encoding="utf-8",
        )
    for name, data in (("splits.json", normalized), ("preparation.json", preparation)):
        (output / name).write_text(
            json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
    return preparation


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--exclude", action="append", default=[], metavar="QUESTION_ID")
    parser.add_argument("--allow-family-leakage", action="store_true")
    args = parser.parse_args()
    try:
        result = prepare(args.source, args.output, args.exclude, args.allow_family_leakage)
    except (PreparationError, OSError, ValueError) as exc:
        parser.exit(1, f"Preparation refused: {exc}\n")
    print(f"Prepared {len(result['retained_question_ids'])} questions in {args.output}")
    print(result["interpretation"])


if __name__ == "__main__":
    main()
