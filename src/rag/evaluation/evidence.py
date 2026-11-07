"""Span-based evidence coverage, independent of a particular chunking strategy.

Intervals use original revision character offsets. Overlap is counted once;
multiple retrieved chunks can jointly cover a span. Alternative sufficient
sets are ORed, while spans within one set are ANDed. This measures literal
text coverage, not semantic entailment or answer correctness.
"""

from dataclasses import replace

from rag.evaluation.metrics import RECALL_KS, mean


def union_length(intervals):
    end = total = 0
    for start, stop in sorted(intervals):
        total += max(0, stop - max(start, end))
        end = max(end, stop)
    return total


def span_coverage(span, passages, files):
    intervals = [
        (max(span.start, p.start), min(span.end, p.end))
        for p in passages
        if files.get(p.document_title) == span.file
        and p.start is not None
        and p.end is not None
        and p.start < span.end
        and p.end > span.start
    ]
    return union_length(intervals) / (span.end - span.start)


def evidence_at(annotation, passages, files):
    coverage = {s.id: span_coverage(s, passages, files) for s in annotation.spans}
    sets = annotation.sufficient_sets
    return {
        "span_coverage": coverage,
        "evidence_coverage": max((mean(coverage[s] for s in group) for group in sets), default=0.0),
        "sufficient_evidence": any(all(coverage[s] == 1.0 for s in group) for group in sets),
    }


def evidence_summary(results, annotations, files):
    scored = [
        r
        for r in results
        if r.question.id in annotations and annotations[r.question.id].sufficient_sets
    ]
    summary = {"questions": len(scored)}
    for k in RECALL_KS:
        scores = [evidence_at(annotations[r.question.id], r.passages[:k], files) for r in scored]
        summary[f"evidence_coverage@{k}"] = mean(s["evidence_coverage"] for s in scores)
        summary[f"sufficient_evidence@{k}"] = mean(s["sufficient_evidence"] for s in scores)
    return summary


def context_prefix(passages, max_chars):
    """A rank-preserving character budget, clipping only the last passage.

    The budget counts source text only, not prompt formatting. It is explicitly
    not a token estimate. Clipped offsets describe the actual evaluated text.
    """
    if max_chars is None:
        return passages
    if max_chars < 1:
        raise ValueError("context_chars must be positive.")
    selected = []
    remaining = max_chars
    for passage in passages:
        if remaining <= 0:
            break
        if len(passage.text) <= remaining:
            selected.append(passage)
        else:
            selected.append(
                replace(
                    passage,
                    text=passage.text[:remaining],
                    end=passage.start + remaining if passage.start is not None else None,
                )
            )
        remaining -= len(selected[-1].text)
    return selected


def context_stats(passages):
    chars = sum(len(p.text) for p in passages)
    if any(p.start is None or p.end is None for p in passages):
        return {"characters": chars, "unique_characters": None, "overlap_fraction": None}
    groups = {}
    for p in passages:
        groups.setdefault(p.revision_id, []).append((p.start, p.end))
    unique = sum(union_length(intervals) for intervals in groups.values())
    return {
        "characters": chars,
        "unique_characters": unique,
        "overlap_fraction": 1 - unique / chars if chars else 0.0,
    }
