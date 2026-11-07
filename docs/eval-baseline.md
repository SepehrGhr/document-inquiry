# Retrieval baseline and reranking comparison

Measured on 2026-11-07 against the committed Sanasa corpus. Hybrid retrieval is
implemented, but **dense remains the default** because equal-weight RRF lowers
overall recall@5 on this dataset.

## Conditions

- 49 Sanasa DOCX documents plus three existing smoke-test documents in the dev
  database; the same 52 chunks were searched for every mode.
- All 71 questions executed, zero errors per mode. The 59 questions with
  expected sources are scored; 12 unanswerable questions are excluded from
  retrieval metrics.
- Profile: `jina-v3`, provider `jina`, model `jina-embeddings-v3`, 1024 dimensions,
  cosine distance. Profile ID: `519d702e-8e53-464a-ac45-f15eb2762b9e`.
- Build: `c7053813-fafe-4d46-bba6-d295ce81b25d`, chunker `char-1`, 1200 characters
  with 200 overlap. No re-embedding was needed for lexical search.
- Output depth 10; hybrid takes 20 candidates per retriever, uses equal RRF
  weights and constant 60, deduplicates by chunk ID, and returns 10 passages.
- Lexical: PostgreSQL `simple` tokenization and `ts_rank_cd`, OR terms,
  common English/Persian question words removed. Derived text uses `lexical-1`
  normalization; query policy is `or-stopwords-1`. This is not BM25.
- All query embeddings were read from cache. Calls to the underlying Jina
  query embedder were disabled to detect cache misses. No chat model or judge
  was called.
- Source: uncommitted hybrid changes based on `29030ac`. Raw JSON/markdown are
  gitignored under
  `eval-results/hybrid-comparison-20261107T011005Z/{dense,lexical,hybrid_rrf}/`.

## Results

| Mode | Recall@1 | Recall@3 | Recall@5 | Recall@10 | MRR |
|---|---:|---:|---:|---:|---:|
| dense | 0.551 | 0.749 | 0.814 | 0.924 | 0.808 |
| lexical | 0.393 | 0.579 | 0.653 | 0.720 | 0.627 |
| hybrid RRF | 0.418 | 0.667 | 0.791 | 0.924 | 0.698 |

Recall@5 by difficulty:

| Tier | Questions | Dense | Lexical | Hybrid RRF |
|---|---:|---:|---:|---:|
| simple | 18 | 0.944 | 0.833 | 0.889 |
| exact term | 12 | 1.000 | 0.667 | 1.000 |
| multi-hop | 18 | 0.648 | 0.556 | 0.685 |
| ambiguous | 7 | 0.619 | 0.286 | 0.333 |
| adversarial | 4 | 0.750 | 0.875 | 1.000 |

Recall@5 by question language:

| Language | Scored questions | Dense | Lexical | Hybrid RRF |
|---|---:|---:|---:|---:|
| English | 27 | 0.784 | 0.796 | 0.809 |
| Persian | 25 | 0.833 | 0.440 | 0.753 |
| mixed | 6 | 0.833 | 0.833 | 0.833 |
| German | 1 | 1.000 | 1.000 | 1.000 |

Hybrid improves multi-hop and adversarial-source recall@5 but lowers overall
recall by about 2.3 percentage points. Lexical search is particularly weak for Persian
questions, including those requiring evidence in another language. Normalizing
characters does not translate terms. Equal RRF voting can promote lexical
distractors and displace strong dense hits. Fusion provides a useful experiment
path; these results do not justify enabling it by default.

## Reproduce

With the corpus loaded and the desired Jina build active:

```bash
python manage.py migrate
python manage.py evaluate_rag --retrieval-only --profile jina-v3 --top-k 10 \
  --mode dense --output eval-results/dense
python manage.py evaluate_rag --retrieval-only --profile jina-v3 --top-k 10 \
  --mode lexical --output eval-results/lexical
python manage.py evaluate_rag --retrieval-only --profile jina-v3 --top-k 10 \
  --mode hybrid_rrf --candidate-k 20 --rrf-k 60 --output eval-results/hybrid
```

Lexical-only mode needs no provider credentials. Dense/hybrid use the configured
embedding adapter and cache; a cache miss normally calls the provider. The
offline guard described above was specific to this comparison run.

## Limits of this measurement

- Source labels are generated document-level expectations and have not all
  been independently audited. A hit does not establish that a particular chunk
  contains sufficient evidence.
- The current evaluator deduplicates documents and omits documents outside
  the manifest before scoring. Reported ranks are mapped document ranks within
  the returned passages, not raw chunk ranks; outside documents can consume
  retrieval slots without appearing in the metric list.
- Some expected-source lists include both current and obsolete evidence for
  ambiguity questions. Retrieving an obsolete source can count as relevant;
  this does not establish correct handling of authority or dates.
- Unanswerable questions and the adversarial tier require answer evaluation
  to assess abstention and prompt-injection resistance.
- This is a small synthetic corpus, with a single scored German question.
  Differences should not be generalized without a larger held-out evaluation.
- Preliminary experiments preceded a stored-vector ranking fix and should not
  be used for comparison. The final implementation ranks the stored vector
  directly using `F("search_vector")`, preserving the configured tokens and
  positions. The dataset has already informed development, so future tuning
  needs held-out examples.
- The earlier partial OpenRouter run (recall@5 0.907) omitted many hard
  questions due to rate limits and is not a valid comparison.

## Optional Jina reranking comparison (2026-11-07)

A second comparison reused the same active build, 52 chunks, all 71 questions,
output depth 10, and cached query embeddings. Dense, lexical, and hybrid
baselines reproduced the earlier per-question document rankings exactly.
Reranked modes used `jina-reranker-v2-base-multilingual` over 20 candidates;
hybrid used 20 candidates per channel and RRF constant 60 before reranking.
Embedding provider calls were disabled, no chat/judge calls were made, and
reranking used live Jina requests. Application source was still uncommitted.

| Mode | Recall@5 | Recall@10 | Hit@5 | MRR | All labeled sources@5 |
|---|---:|---:|---:|---:|---:|
| dense | 0.814 | 0.924 | 0.983 | 0.808 | 38/59 |
| lexical | 0.653 | 0.720 | 0.763 | 0.627 | 32/59 |
| hybrid RRF | 0.791 | 0.924 | 0.898 | 0.698 | 41/59 |
| dense + Jina | **0.853** | 0.918 | 0.915 | **0.836** | **46/59** |
| hybrid RRF + Jina | 0.845 | **0.935** | 0.915 | 0.836 | 45/59 |

All five configurations completed 71 questions, with 59 scored and zero
unresolved errors. Hybrid reranking encountered two SSL handshake timeouts
(q020 and q032). Successful results were checkpointed and retained; only
failed/pending questions were resumed. The last resume allowed one retry per
question for transport errors only; no further retry was needed. Original
partial reports and recovered-error metadata were retained.

Recall@5 by difficulty:

| Tier | n | Dense | Dense + Jina | Hybrid RRF | Hybrid RRF + Jina |
|---|---:|---:|---:|---:|---:|
| simple | 18 | 0.944 | 1.000 | 0.889 | 1.000 |
| exact term | 12 | 1.000 | 0.917 | 1.000 | 0.917 |
| multi-hop | 18 | 0.648 | 0.787 | 0.685 | 0.731 |
| ambiguous | 7 | 0.619 | 0.524 | 0.333 | 0.524 |
| adversarial | 4 | 0.750 | 0.875 | 1.000 | 1.000 |

Dense + Jina improved recall@5 on 10 questions, regressed on 5, and tied on 44
versus dense. Full labeled-source coverage for multi-hop questions rose from
5/18 to 11/18 (hybrid: 8/18; hybrid + Jina: 9/18). For q035 the product spec
moved from mapped rank 8 to 3 alongside the agreement; for q043 the spec and
sustainability report became ranks 1 and 2. However, q029's Arabic export-agent
source fell from dense rank 3 out of the reranked top 10. q053, q054, and q055
also lost all expected-source hits at five. The lower hit@5 means better mean
source coverage did not eliminate failures and actually introduced no-hit cases.

English recall@5 improved from 0.784 to 0.895 with dense + Jina; Persian changed
from 0.833 to 0.827, and mixed from 0.833 to 0.750. German remained 1.000 (n=1).
Candidate/source-language effects and alternative sufficient labels need review
before attributing these changes solely to language capability.

Successful retrieval latency p50/p95 was about 0.91s/1.15s for dense + Jina
and 0.93s/1.33s for hybrid + Jina, versus a few milliseconds for cached local
baselines. These exclude generation, live query embedding time, and abandoned
failed attempts; they do not describe full request latency or outage overhead.

**Decision:** keep reranking optional and dense as the unchanged default.
Dense + Jina is the strongest measured top-five candidate for further work,
but its exact-term/ambiguous regressions and lower hit rate need investigation.
This measures document-source retrieval, not answer quality, abstention,
authority handling, or injection resistance. Existing label and small-corpus
limitations still apply.

Full reports, paired changes, provenance hashes, and failure snapshots are under
`eval-results/rerank-comparison-20261107T020532Z/` (gitignored), including
`comparison.md` and `comparison.json`. Reproduce a reranked mode with:

```bash
python manage.py evaluate_rag --retrieval-only --profile jina-v3 --top-k 10 \
  --mode dense --reranker jina --rerank-candidate-k 20 \
  --output eval-results/dense-reranked
```

Use `--mode hybrid_rrf --candidate-k 20 --rrf-k 60` for the hybrid variant.
The checkpoint/resume and embedding-call guards were local evaluation scripts,
not additions to the management command or automatic production retries.
