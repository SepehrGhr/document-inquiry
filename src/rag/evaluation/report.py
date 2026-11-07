"""Turning evaluation runs into JSON and markdown."""

import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path

from rag.adapters.lexical import QUERY_VERSION
from rag.evaluation.metrics import group_by, retrieval_summary
from rag.search_text import NORMALIZATION_VERSION


def run_metadata(build, *, top_k: int, retrieval_mode: str) -> dict:
    profile = build.profile
    return {
        "generated_at": datetime.now(UTC).isoformat(),
        "git_commit": _git_commit(),
        "profile": {
            "id": str(profile.id),
            "name": profile.name,
            "provider": profile.provider,
            "model_id": profile.model_id,
            "dimension": profile.dimension,
            "metric": profile.metric,
            "chunker_version": profile.chunker_version,
            "chunk_size": profile.chunk_size,
            "chunk_overlap": profile.chunk_overlap,
        },
        "build_id": str(build.id),
        "top_k": top_k,
        "retrieval_mode": retrieval_mode,
    }


def retrieval_report(run, *, top_k: int) -> dict:
    results = run.results
    metadata = run_metadata(run.build, top_k=top_k, retrieval_mode=run.retrieval_mode)
    metadata["reranking"] = {"provider": run.reranker}
    if run.reranker != "none":
        metadata["reranking"].update(model=run.rerank_model, candidate_k=run.rerank_candidate_k)
    if run.retrieval_mode in ("lexical", "hybrid_rrf"):
        metadata["lexical"] = {
            "configuration": "simple",
            "ranking": "ts_rank_cd",
            "normalization_version": NORMALIZATION_VERSION,
            "query_version": QUERY_VERSION,
        }
    if run.retrieval_mode == "hybrid_rrf":
        metadata["candidate_k"] = run.candidate_k
        metadata["rrf_k"] = run.rank_constant
    return {
        "metadata": metadata,
        "overall": retrieval_summary([r.score for r in results]),
        "by_difficulty": {
            difficulty: retrieval_summary([r.score for r in group])
            for difficulty, group in group_by(results, lambda r: r.question.difficulty).items()
        },
        "by_language": {
            language: retrieval_summary([r.score for r in group])
            for language, group in group_by(results, lambda r: r.question.language).items()
        },
        "errors": [{"question_id": qid, "error": message} for qid, message in run.errors],
        "questions": [
            {
                "id": r.question.id,
                "difficulty": r.question.difficulty,
                "language": r.question.language,
                "expected": list(r.question.expected_sources),
                "retrieved": list(r.retrieved_documents),
                "recall@5": r.score.recall_at(5),
                "reciprocal_rank": r.score.reciprocal_rank(),
                "latency_ms": r.latency_ms,
            }
            for r in results
        ],
    }


def write_json(report: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")


def retrieval_markdown(report: dict) -> str:
    meta = report["metadata"]
    profile = meta["profile"]
    lines = [
        "# Retrieval evaluation",
        "",
        f"- Generated: {meta['generated_at']}",
        f"- Commit: `{meta['git_commit']}`",
        f"- Profile: `{profile['name']}` ({profile['provider']} / {profile['model_id']}, "
        f"{profile['dimension']} dims, {profile['metric']})",
        f"- Chunking: {profile['chunker_version']}, {profile['chunk_size']} chars, "
        f"{profile['chunk_overlap']} overlap",
        f"- Build: `{meta['build_id']}`",
        f"- Top-k: {meta['top_k']}",
        f"- Mode: {meta['retrieval_mode']}",
        "",
        "## Overall",
        "",
        _table({"overall": report["overall"]}),
        "",
        "## By difficulty",
        "",
        _table(report["by_difficulty"]),
        "",
        "## By language",
        "",
        _table(report["by_language"]),
    ]
    if report["errors"]:
        lines += ["", "## Errors", ""]
        lines += [f"- `{error['question_id']}`: {error['error']}" for error in report["errors"]]
    if meta["retrieval_mode"] == "hybrid_rrf":
        lines += ["", f"Candidates per retriever: {meta['candidate_k']}; RRF k: {meta['rrf_k']}."]
    reranking = meta.get("reranking", {"provider": "none"})
    if reranking["provider"] == "none":
        lines += ["", "Reranking: disabled."]
    else:
        lines += [
            "",
            f"Reranking: {reranking['provider']} / `{reranking['model']}`; "
            f"candidate pool: {reranking['candidate_k']}.",
        ]
    return "\n".join(lines) + "\n"


_COLUMNS = ("questions", "recall@1", "recall@3", "recall@5", "recall@10", "hit@5", "mrr")


def _table(groups: dict) -> str:
    header = "| group | " + " | ".join(_COLUMNS) + " |"
    divider = "|---" * (len(_COLUMNS) + 1) + "|"
    rows = [header, divider]
    for name, summary in groups.items():
        cells = [name] + [_format(summary.get(column)) for column in _COLUMNS]
        rows.append("| " + " | ".join(cells) + " |")
    return "\n".join(rows)


def _format(value) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def _git_commit() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            timeout=5,
        ).stdout.strip()
    except Exception:
        return "unknown"
