"""Turning evaluation runs into JSON and markdown."""

import json
import subprocess
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

from rag.adapters.lexical import QUERY_VERSION
from rag.evaluation.evidence import context_stats, evidence_at, evidence_summary
from rag.evaluation.metrics import RetrievalScore, group_by, mean, percentile, retrieval_summary
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
    files = {d.title: d.file for d in run.corpus.documents} if run.corpus else {}
    annotations = run.corpus.evidence if run.corpus else {}
    by_id = {q.id: q for q in run.corpus.questions} if run.corpus else {}
    failed_scores = [
        RetrievalScore((), by_id[qid].expected_sources) for qid, _ in run.errors if qid in by_id
    ]
    metadata = run_metadata(run.build, top_k=top_k, retrieval_mode=run.retrieval_mode)
    available = sum(1 for q in by_id if run.split == "all" or q in run.corpus.splits[run.split])
    metadata.update(
        split=run.split,
        context_chars=run.context_chars,
        questions_requested=len(run.requested_ids),
        questions_completed=len(results),
        questions_failed=len(run.errors),
        question_ids=list(run.requested_ids),
        limit_applied=len(run.requested_ids) < available,
        status="failed"
        if run.errors
        else ("limited" if len(run.requested_ids) < available else "completed"),
        **run.provenance,
    )
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
    report = {
        "metadata": metadata,
        "overall": retrieval_summary([r.score for r in results]),
        "overall_all_requested": retrieval_summary([r.score for r in results] + failed_scores),
        "context": {
            "mean_characters": mean(context_stats(r.passages)["characters"] for r in results),
            "mean_overlap_fraction": mean(
                context_stats(r.passages)["overlap_fraction"]
                for r in results
                if context_stats(r.passages)["overlap_fraction"] is not None
            ),
            "latency_p50_ms": percentile([r.latency_ms for r in results], 0.5),
            "latency_p95_ms": percentile([r.latency_ms for r in results], 0.95),
        },
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
                "context": context_stats(r.passages),
                "retrieved_count": r.retrieved_count,
                "passages": [
                    {**asdict(p), "position": i, "file": files.get(p.document_title)}
                    for i, p in enumerate(r.passages, start=1)
                ],
                "evidence": evidence_at(annotations[r.question.id], r.passages, files)
                if r.question.id in annotations
                else None,
            }
            for r in results
        ],
    }
    if annotations:
        from rag.evaluation.runner import RetrievalResult

        attempts = results + [
            RetrievalResult(by_id[qid], (), (), 0) for qid, _ in run.errors if qid in by_id
        ]
        report["evidence"] = {
            "overall": evidence_summary(attempts, annotations, files),
            "by_difficulty": {
                key: evidence_summary(group, annotations, files)
                for key, group in group_by(attempts, lambda r: r.question.difficulty).items()
            },
            "by_language": {
                key: evidence_summary(group, annotations, files)
                for key, group in group_by(attempts, lambda r: r.question.language).items()
            },
            "by_challenge": {
                tag: evidence_summary(
                    [r for r in attempts if tag in annotations[r.question.id].challenges],
                    annotations,
                    files,
                )
                for tag in sorted(
                    {tag for r in attempts for tag in annotations[r.question.id].challenges}
                )
            },
        }
    return report


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
    if "questions_requested" in meta:
        lines += [
            "",
            f"Status: {meta['status']}; split: {meta['split']}; "
            f"requested/completed/failed: {meta['questions_requested']}/"
            f"{meta['questions_completed']}/{meta['questions_failed']}.",
            f"Source-text character budget: {meta['context_chars'] or 'unlimited'} (not tokens).",
            "Legacy document scores above use successful queries and deduplicated document ranks.",
            "",
            "## Documents (all requested; failures score zero)",
            "",
            _table({"overall": report["overall_all_requested"]}),
        ]
    if "evidence" in report:
        columns = (
            "questions",
            "evidence_coverage@1",
            "evidence_coverage@5",
            "sufficient_evidence@5",
            "sufficient_evidence@10",
        )
        evidence = report["evidence"]
        lines += [
            "",
            "## Evidence coverage (raw passage positions)",
            "",
            "Coverage unions overlapping offsets; any complete alternative set is sufficient. "
            "This is literal evidence-text coverage, not answer correctness. Failures score zero.",
            "",
            _table({"overall": evidence["overall"]}, columns),
            "",
            "### By difficulty",
            "",
            _table(evidence["by_difficulty"], columns),
            "",
            "### By language",
            "",
            _table(evidence["by_language"], columns),
            "",
            "### By challenge",
            "",
            _table(evidence["by_challenge"], columns),
        ]
    if "context" in report:
        context = report["context"]
        lines += [
            "",
            "## Context and latency",
            "",
            f"Mean source characters: {context['mean_characters']:.1f}; "
            f"mean overlap fraction: {context['mean_overlap_fraction']:.3f}.",
            f"Successful retrieval latency p50/p95: {context['latency_p50_ms']:.1f}/"
            f"{context['latency_p95_ms']:.1f} ms (no answer generation).",
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


def _table(groups: dict, columns=_COLUMNS) -> str:
    header = "| group | " + " | ".join(columns) + " |"
    divider = "|---" * (len(columns) + 1) + "|"
    rows = [header, divider]
    for name, summary in groups.items():
        cells = [name] + [_format(summary.get(column)) for column in columns]
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
