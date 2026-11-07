# Long-document retrieval evaluation

The original corpus still works unchanged. A new corpus can add optional
`evidence.jsonl` and `splits.json` files for chunk-independent evaluation.
This is retrieval/context evaluation, not answer generation or an LLM judge.

## Workflow

Run with your usual Django/database environment and provider credentials.
Validation itself needs neither database access nor provider calls:

```bash
python manage.py validate_eval_corpus --corpus tests/fixtures/corpus_long_v1
```

This checks annotation structure, source paths, exact DOCX-parser text/offsets,
optional `extracted/<document-stem>.txt` equivalence, alternatives, canonical
source labels, and split/fact-family membership. It does not certify the
semantic truth or necessity of an annotated span. Review those separately.
Do not run scripts bundled with a downloaded corpus merely to validate it.

Load documents without automatically queueing new live index builds:

```bash
RAG_AUTO_INDEX=false python manage.py load_eval_corpus \
  --corpus tests/fixtures/corpus_long_v1 --actor YOUR_STAFF_USERNAME
```

Loading is still a document-store write. Use unique titles to avoid replacing
existing documents. The loader validates sidecars before ingestion. Setting
`RAG_AUTO_INDEX=false` is important: otherwise the existing document lifecycle
can queue live builds independently of the experimental commands below.

Create separate immutable profiles for each chunk size/overlap. For example:

```bash
python manage.py create_index_profile --name long-char-small --provider jina \
  --model jina-embeddings-v3 --dimension 1024 --chunk-size 600 --chunk-overlap 100
python manage.py create_index_profile --name long-char-baseline --provider jina \
  --model jina-embeddings-v3 --dimension 1024 --chunk-size 1200 --chunk-overlap 200
python manage.py create_index_profile --name long-char-large --provider jina \
  --model jina-embeddings-v3 --dimension 1024 --chunk-size 2400 --chunk-overlap 400

python manage.py rebuild_index --profile long-char-small --sync --no-activate
```

Repeat the last command for the other profiles and retain each printed build
UUID. Building may make embedding calls. `--no-activate` requires `--sync` and
leaves the current live build untouched. A completed inactive build can be
queried explicitly:

```bash
python manage.py evaluate_rag --retrieval-only \
  --corpus tests/fixtures/corpus_long_v1 \
  --profile long-char-small --build BUILD_UUID \
  --split development --mode dense --reranker none --top-k 10 \
  --output eval-results/long/small-development
```

For a context-size-controlled comparison, give every configuration the same
source-text character budget, for example:

```bash
python manage.py evaluate_rag --retrieval-only \
  --corpus tests/fixtures/corpus_long_v1 \
  --profile long-char-small --build BUILD_UUID \
  --split development --mode dense --reranker none --top-k 30 \
  --context-chars 6000 --output eval-results/long/small-budget-development
```

Use a sufficiently large, matched retrieval depth when comparing context
budgets. The budget preserves ranking order and clips only the final passage
that fits; it does not skip large passages. It counts source characters, not
prompt formatting, and is **not a token budget**. Multilingual token costs
can differ for the same character count. The reported offsets and coverage
refer to the clipped text actually evaluated. Reranking, if explicitly enabled,
runs before this evaluation-only context budget.

Use development questions for experiments. Run `--split test` only after choosing
the configuration. `--split all` remains the default for backward compatibility;
an undefined or empty requested split fails rather than silently using all
questions. `--limit` applies after split selection and marks a report limited.

Only the existing character chunker is implemented by this change. These commands
support its size/overlap experiments now; paragraph/table-aware strategies and
neighbor expansion are later pipeline changes, not silently enabled here.

## Evidence schema

Each line of `evidence.jsonl` is an object with:

- `question_id`: one known ID, exactly once; all questions must be annotated
  when the file exists.
- `evidence_spans`: list of objects with `id`, `file`, `start`, `end`, `quote`.
  Span IDs are unique within a question. Files are manifest-relative DOCX paths.
- `sufficient_evidence_sets`: list of nonempty span-ID lists. All spans inside
  one list are needed; any complete list is accepted as sufficient.
- Optional `answer_behavior`: descriptive string.
- Optional `chunking_challenges`: list of strings for grouped reporting.

Additional generation metadata, such as claims, reasoning, authority roles,
section references, and distractors, is allowed but is not semantically judged
by the evaluator.

Offsets are zero-based Python character positions with an exclusive end, into
the exact source text produced by the application's DOCX parser. Compute them,
do not estimate them:

```python
start = text.index(quote)
span = {
    "id": "rule",
    "file": "docs/policy.docx",
    "quote": quote,
    "start": start,
    "end": start + len(quote),
}
assert text[span["start"] : span["end"]] == span["quote"]
```

Example alternative logic:

```json
"sufficient_evidence_sets": [
  ["returns_rule", "english_product_spec"],
  ["returns_rule", "persian_catalogue"]
]
```

For answerable questions, at least one sufficient set must exist, and the legacy
`expected_sources` document set must match one of these alternatives. It must
not be the union of every alternative. Unanswerable questions have no sufficient
sets and are excluded from evidence retrieval aggregates. Their safe abstention
still requires future answer evaluation.

Evaluation refuses span scoring if a referenced document is missing/ambiguous
in the selected build, its indexed revision differs from the corpus, or stored
chunk offsets do not slice the revision text exactly. This prevents false
measurements against stale builds or differently parsed documents.

## Split schema

Supported direct form:

```json
{
  "development": ["ld001"],
  "test": ["ld002"],
  "fact_families": {"ld001": "family-a", "ld002": "family-b"}
}
```

The two split arrays may also be nested under `"splits"`, with `fact_families`
remaining at the top level. Every question belongs to exactly one split.
`fact_families` is optional; when provided it maps every question ID to a
nonempty family name, and no family may cross splits. Other shapes fail with
an error; adapt generator output explicitly rather than guessing membership.

## Reading the reports

JSON and Markdown include:

- **Legacy document metrics:** the original canonical-source recall/hit/MRR,
  using deduplicated mapped document ranks. `overall` remains successful-only
  for compatibility. `overall_all_requested` adds failed questions as zero.
- **Evidence coverage@k:** for each sufficient set, average fraction of each
  span's characters covered by the union of the first k actual passages; take
  the best alternative set. Partial matches earn partial literal coverage.
- **Sufficient evidence@k:** fraction of annotated answerable questions where
  at least one entire sufficient set is covered. Two adjacent chunks can jointly
  cover a span; overlapping characters never count twice.
- **Evidence breakdowns:** by difficulty, question language, and challenge tag.
  Evidence metrics count failed questions as zero, not as omitted samples.
- **Per-question context:** source characters, unique revision characters, and
  overlap fraction. Repeated text at distinct source offsets is not considered
  overlap; unknown offsets yield unavailable uniqueness statistics.
- **Raw selected-passage traces:** IDs, revision, ordinal, rank, score, original
  source offsets, actual evaluated text, and mapped file if known. Out-of-corpus
  passages still consume positions and appear in traces. They do not vanish
  before span-based scoring.
- **Run accounting:** selected question IDs, requested/completed/failed counts,
  limited/failed/completed status, split, context budget, corpus/sidecar hashes,
  RAG Python source hash, and index document/revision inventory, chunk count,
  and total indexed character count.
- **Latency:** successful retrieval/context-selection p50/p95. These are not
  generation latency or failed-attempt costs.

The source hash covers Python files under `src/rag/`; it does not capture every
environment variable, dependency, or external provider behavior. Keep provider
settings and comparison conditions consistent. The inventory includes all
searchable documents, not just the manifest: unrelated library documents can
serve as distractors and should be held constant across comparisons.

A failed evaluation writes partial reports and then exits nonzero. A wrong
document can never earn evidence credit just because it uses similar words;
span scoring uses the annotated file and validated source positions. Literal
coverage still does not establish answer correctness, authority handling, or
injection resistance.

## First long-corpus comparison (2026-11-07)

The received corpus had one inconsistent annotation (`ld016`), an unsupported
assignment-list split shape, and declared fact families crossing development/test.
At the user's request, a separate evaluation copy excluded `ld016`. Split lists
were converted to the supported shape without moving the remaining questions;
original assignments and leakage notes were retained. This is an **exploratory
development comparison**, not a clean held-out result. Original fixture files
were not edited. Ten documents were loaded with auto-indexing disabled.

All six runs completed 39 development questions without errors. Evidence metrics
score 33 questions; six unanswerable questions are excluded, and all 20 test
questions were left untouched. Dense retrieval used Jina v3 embeddings with no
reranking. All three builds cover the same source revisions and remain inactive;
the original live build `c7053813-fafe-4d46-bba6-d295ce81b25d` was preserved.

| Size / overlap | Index chunks (including existing distractors) | Top-10 complete evidence | Top-10 mean characters | Complete evidence at 6,000 characters |
|---|---:|---:|---:|---:|
| 600 / 100 | 736 | 27/33 | 4,686 | **27/33** |
| 1200 / 200 | 343 | 26/33 | 9,205 | 26/33 |
| 2400 / 400 | 193 | **31/33** | 17,386 | 23/33 |

At a matched 6,000-character budget (retrieval depth 30), literal evidence
coverage was 0.861 / 0.870 / 0.735 for small/baseline/large. Complete multi-hop
evidence was 7/10 / 6/10 / 5/10. At fixed top-10, large chunks recovered more
complete evidence, but consumed almost twice the baseline text and 3.7 times
the small-chunk text. This does not establish superior evidence efficiency.
Small chunks led complete-evidence success at the matched budget, while baseline
had slightly higher mean partial coverage. No default configuration was changed.

These are current character-chunker size/overlap comparisons, not evaluations
of section-aware chunking or answer quality. Overlap was varied proportionally
with size, not independently. Future structure-aware experiments should retain
the matched context budget and inspect per-question gains/losses.

The initial small/top-10 evaluation included live query embedding calls; its
report was retained separately and a provider-disabled warm-cache replay
reproduced the same retrieved passages before comparing latency. Full reports,
paired changes, copy-preparation notes, profiles/build IDs, and the original
split assignments are under gitignored
`eval-results/long-chunking-20261107T031907Z/`, including `comparison.md`.
