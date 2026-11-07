"""Exercise the trusted preparation CLI in isolation from corpus Python files."""

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "prepare_eval_corpus.py"


@pytest.fixture
def corpus(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "docs").mkdir()
    (source / "docs/a.docx").write_bytes(b"source document")
    (source / "extracted").mkdir()
    (source / "extracted/a.txt").write_text("source text")
    (source / "manifest.json").write_text(json.dumps({"documents": [{"file": "docs/a.docx"}]}))
    questions = [
        {"id": qid, "expected_sources": ["docs/a.docx"]} for qid in ("ld001", "ld016", "ld020")
    ]
    evidence = [{"question_id": row["id"], "original_metadata": "unchanged"} for row in questions]
    for name, rows in (("questions.jsonl", questions), ("evidence.jsonl", evidence)):
        (source / name).write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    (source / "splits.json").write_text(
        json.dumps(
            {
                "protocol": "original claim, not endorsed",
                "assignments": [
                    {"question_id": "ld001", "split": "development", "fact_family_id": "family-a"},
                    {"question_id": "ld016", "split": "development", "fact_family_id": "family-b"},
                    {"question_id": "ld020", "split": "test", "fact_family_id": "family-a"},
                ],
            }
        )
    )
    # Neither executed nor copied, even if the corpus is the working directory.
    (source / "json.py").write_text("raise RuntimeError('untrusted code executed')")
    (source / "generate.py").write_text("raise RuntimeError('untrusted code executed')")
    return source, tmp_path / "prepared"


def run_prepare(source, output, *args):
    return subprocess.run(
        [
            sys.executable,
            "-I",
            str(SCRIPT),
            "--source",
            str(source),
            "--output",
            str(output),
            *args,
        ],
        cwd=source,
        capture_output=True,
        text=True,
        check=False,
    )


def test_prepares_explicit_exploratory_copy_without_repairing_labels(corpus):
    source, output = corpus
    before = {
        str(path.relative_to(source)): path.read_bytes()
        for path in source.rglob("*")
        if path.is_file()
    }
    result = run_prepare(source, output, "--exclude", "ld016", "--allow-family-leakage")
    assert result.returncode == 0, result.stderr
    assert "Exploratory" in result.stdout
    assert {str(path.relative_to(output)) for path in output.rglob("*") if path.is_file()} == {
        "docs/a.docx",
        "extracted/a.txt",
        "manifest.json",
        "questions.jsonl",
        "evidence.jsonl",
        "splits.json",
        "original-splits.json",
        "preparation.json",
    }
    assert (output / "original-splits.json").read_bytes() == before["splits.json"]
    assert json.loads((output / "splits.json").read_text()) == {
        "development": ["ld001"],
        "test": ["ld020"],
    }
    for name, key in (("questions.jsonl", "id"), ("evidence.jsonl", "question_id")):
        assert [json.loads(line) for line in (output / name).read_text().splitlines()] == [
            row
            for line in before[name].decode().splitlines()
            if (row := json.loads(line))[key] != "ld016"
        ]
    notes = json.loads((output / "preparation.json").read_text())
    assert notes["excluded_question_ids"] == ["ld016"]
    assert notes["family_leakage"] == {
        "family-a": {"development": ["ld001"], "test": ["ld020"]},
    }
    for name, digest in notes["source_sha256"].items():
        assert digest == hashlib.sha256(before[name]).hexdigest()
    assert before == {
        str(path.relative_to(source)): path.read_bytes()
        for path in source.rglob("*")
        if path.is_file()
    }


@pytest.mark.parametrize("nested", [False, True])
def test_accepts_canonical_splits_with_valid_families_and_optional_extracted(corpus, nested):
    source, output = corpus
    groups = {"development": ["ld001", "ld016"], "test": ["ld020"]}
    families = {"ld001": "a", "ld016": "a", "ld020": "b"}
    entry = {"splits": groups} if nested else groups.copy()
    entry["fact_families"] = families
    (source / "splits.json").write_text(json.dumps(entry))
    (source / "extracted/a.txt").unlink()
    result = run_prepare(source, output)
    assert result.returncode == 0, result.stderr
    assert json.loads((output / "splits.json").read_text()) == {
        **groups,
        "fact_families": families,
    }
    assert not (output / "extracted").exists()


@pytest.mark.parametrize("failure", ["overwrite", "leakage", "unknown", "membership", "evidence"])
def test_refuses_invalid_or_unsafe_preparation(corpus, failure):
    source, output = corpus
    args = ["--exclude", "ld016", "--allow-family-leakage"]
    if failure == "overwrite":
        output.mkdir()
        (output / "keep.txt").write_text("do not change")
    elif failure == "leakage":
        args.remove("--allow-family-leakage")
    elif failure == "unknown":
        args.extend(["--exclude", "missing"])
    elif failure == "membership":
        splits = json.loads((source / "splits.json").read_text())
        splits["assignments"].append(splits["assignments"][0])
        (source / "splits.json").write_text(json.dumps(splits))
    else:
        (source / "evidence.jsonl").write_text('{"question_id": "unknown"}\n')
    result = run_prepare(source, output, *args)
    assert result.returncode != 0
    assert "Preparation refused" in result.stderr
    if failure == "overwrite":
        assert (output / "keep.txt").read_text() == "do not change"
        assert list(output.iterdir()) == [output / "keep.txt"]
    else:
        assert not output.exists()
