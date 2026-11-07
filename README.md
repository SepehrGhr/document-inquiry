# Document Q&A

A Django service for managing text documents and DOCX uploads, and answering
questions grounded in them with a replaceable RAG pipeline (LangChain, pgvector,
Celery, and Redis).

## Reviewer quick start

This repository is designed to be reviewable without a frontend. Django Admin is
the UI, and the REST API is documented with OpenAPI/Swagger.

```bash
cp .env.example .env
# For a key-free local demo, use the values in the "Offline demo" section below.
docker compose up -d --build
docker compose exec web python manage.py migrate
docker compose exec web python manage.py createsuperuser
```

Open:

- Admin: http://localhost:8000/admin/
- API docs: http://localhost:8000/api/docs/
- Liveness: http://localhost:8000/health/live/
- Readiness: http://localhost:8000/health/ready/

For a live Jina/OpenRouter demo, set `JINA_KEY` and `OPENROUTER_KEY` in `.env`,
create a Jina profile, and build its index as shown below. The Compose file is a
local-review configuration: it uses Django's development server, plain HTTP, and
binds the web port to localhost. Do not use it as a production deployment.

## Offline demo

The default test settings use deterministic fake adapters. This is useful for
running the suite without credentials, but the normal application settings use
real providers. A reviewer can run the same flow locally without network calls:

```bash
export DJANGO_SETTINGS_MODULE=config.settings.test
export POSTGRES_HOST=127.0.0.1 POSTGRES_PORT=5434
.venv/bin/pytest -q
```

The test suite exercises document creation, indexing, retrieval, citations, API
history, hybrid retrieval, optional reranking, and the long-document evaluation
loader. It is the reproducible offline verification path, not a substitute for a
live answer-generation demo.

## What works today

- Create, read, update, and delete documents (deletion is logical).
- Plain-text content or `.docx` uploads, with the full extracted text stored.
- Immutable content revisions: every content replacement appends a revision;
  metadata edits do not.
- Optimistic concurrency: content replacement requires `expected_revision` and
  returns `409` if the document changed.
- Django Admin as the management UI, using the same services as the API.
- English, Persian, and mixed text are stored exactly as provided (no
  normalization).
- OpenAPI schema at `/api/schema/` and Swagger UI at `/api/docs/`.
- Dense retrieval over pgvector, grounded answers with validated citations,
  abstention on weak evidence, and question/answer history.
- Optional PostgreSQL lexical search and hybrid retrieval with reciprocal rank
  fusion (RRF), using the same retriever in the API and evaluation harness.
- Background indexing through Celery + Redis, triggered after a revision
  commits (from both the API and Admin).
- A retrieval evaluation harness with a committed 49-document synthetic corpus
  and 71 questions across six difficulty tiers.

## Run with Docker

```bash
cp .env.example .env          # then set DJANGO_SECRET_KEY and the RAG_* values
docker compose up -d --build
docker compose exec web python manage.py migrate
docker compose exec web python manage.py createsuperuser
```

The stack runs `web` (Django), `worker` (Celery), `redis`, and a
pgvector-enabled PostgreSQL. Before asking questions, create an index profile
and build the index:

```bash
docker compose exec web python manage.py create_index_profile \
  --name jina-v3 --provider jina --model jina-embeddings-v3 --dimension 1024
docker compose exec web python manage.py rebuild_index --profile jina-v3
```

Set `RAG_ACTIVE_PROFILE=jina-v3` in `.env` so new revisions are indexed under
that profile automatically. Embeddings are cached on disk (`.cache/embeddings/`),
so re-indexing unchanged text costs nothing.

- Admin: http://localhost:8000/admin/
- API docs: http://localhost:8000/api/docs/
- Health: `/health/live/` (process) and `/health/ready/` (database)

The Compose setup serves plain HTTP for local review. For a real deployment,
terminate TLS in front of the app; `config.settings.production` enables secure
cookies and HTTPS redirect.

## API

Authentication: Django session (Admin/browsable API) or DRF token. All
endpoints require an active staff user.

```bash
docker compose exec web python manage.py create_api_token <username>
```

```bash
TOKEN=...
BASE=http://localhost:8000/api/v1

# Create from text
curl -s -X POST $BASE/documents/ -H "Authorization: Token $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"title": "راهنمای فارسی", "text": "متن سند"}'

# Create from DOCX
curl -s -X POST $BASE/documents/ -H "Authorization: Token $TOKEN" \
  -F "title=Report" -F "file=@report.docx"

# Read, update metadata, replace content
curl -s $BASE/documents/<id>/ -H "Authorization: Token $TOKEN"
curl -s -X PATCH $BASE/documents/<id>/ -H "Authorization: Token $TOKEN" \
  -H "Content-Type: application/json" -d '{"title": "New title"}'
curl -s -X POST $BASE/documents/<id>/revisions/ -H "Authorization: Token $TOKEN" \
  -H "Content-Type: application/json" -d '{"text": "new body", "expected_revision": 1}'

# History and original file
curl -s $BASE/documents/<id>/revisions/ -H "Authorization: Token $TOKEN"
curl -s -OJ $BASE/documents/<id>/revisions/1/file/ -H "Authorization: Token $TOKEN"

# Delete (logical)
curl -s -X DELETE $BASE/documents/<id>/ -H "Authorization: Token $TOKEN"

# Ask a question (grounded answer with sources, or an abstention)
curl -s -X POST $BASE/questions/ -H "Authorization: Token $TOKEN" \
  -H "Content-Type: application/json" -d '{"question": "How long do refunds take?"}'

# Question history
curl -s $BASE/questions/ -H "Authorization: Token $TOKEN"
curl -s $BASE/questions/<id>/ -H "Authorization: Token $TOKEN"
```

`POST /questions/` returns `503` when no index is active yet.

## Retrieval modes

Run migrations after updating the app:

```bash
python manage.py migrate
```

The lexical-search migration backfills derived text for existing chunks and
creates a stored `tsvector` with a GIN index. It does not re-embed documents.
New chunks populate the derived text during indexing. Source text, citation
text, and embeddings are preserved.

Configure the question API with these environment variables, then restart web
processes:

```dotenv
RAG_RETRIEVAL_MODE=hybrid_rrf
RAG_RETRIEVAL_CANDIDATES=20
RAG_RRF_K=60
RAG_TOP_K=5
```

Modes are `dense` (default), `lexical`, and `hybrid_rrf`. Hybrid fetches at least
20 candidates from each search method, fuses them by chunk ID using
`sum(1 / (60 + rank))`, and returns the best `TOP_K` passages with fresh ranks.
Scores in hybrid answer sources are fusion scores when reranking is disabled,
not cosine similarities.
Both searches use the same build/profile/revision and deletion filters.

Lexical search uses PostgreSQL's `simple` configuration and `ts_rank_cd`
ranking. Query terms are combined with OR after filtering common English and
Persian question words. Derived text normalizes Arabic/Persian letter and digit
variants, vowel marks, and half-spaces; hyphenated product codes remain intact.
It does not perform stemming or translation and is not BM25. See
[Django's full-text search documentation](https://docs.djangoproject.com/en/5.2/ref/contrib/postgres/search/)
for the underlying database functions.

Dense remains the default because it currently outperforms equal-weight hybrid
RRF on this corpus. Hybrid is available for explicit evaluation and use.

### Optional Jina reranking

Reranking is disabled by default and works with all three retrieval modes.
To enable it for the question API, set these variables and restart web processes:

```dotenv
JINA_KEY=your-jina-key
RAG_RERANKER=jina
RAG_RERANK_MODEL=jina-reranker-v2-base-multilingual
RAG_RERANK_CANDIDATES=20
RAG_RERANK_TIMEOUT=60
RAG_TOP_K=5
```

The default model is Jina's multilingual cross-encoder; `RAG_RERANK_MODEL` can
select another text reranker supported by [Jina's API](https://jina.ai/reranker/).
The existing `RAG_JINA_BASE_URL` applies to both embedding and reranking requests.
No new index profile, migration, or re-embedding is needed for reranking.

The retriever fetches at least `RAG_RERANK_CANDIDATES` passages (never fewer than
`RAG_TOP_K`). In hybrid mode, RRF produces this larger shortlist **before**
reranking. Jina receives the question and candidate passage texts, returns their
ranked input indices, and the top `RAG_TOP_K` original passages go to generation.
Source IDs, text, and citation metadata are preserved. Persisted source scores
are then **reranker relevance scores**, not dense similarities or RRF scores;
they are not calibrated confidence probabilities.

Each nonempty shortlist makes a live Jina request, even when embeddings are
cached. Account for provider cost, quotas, extra latency, and sending document
text to Jina. HTTP/timeouts or malformed results fail the question through the
existing provider-error path; there is no silent fallback to unreranked results.
Set `RAG_RERANKER=none` to disable it. Relevance ranking does not establish factual
support, document authority, or prompt-injection safety. On the current corpus,
dense + Jina improves recall@5 (0.814 → 0.853), but lowers hit@5 and regresses
exact-term/ambiguous cases, so reranking remains opt-in. Full results and limits:
[docs/eval-baseline.md](docs/eval-baseline.md).

## Development

Requires Python 3.13 and a PostgreSQL instance.

```bash
python -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python manage.py migrate
.venv/bin/python manage.py runserver
```

Tests run against PostgreSQL (not SQLite) because the lifecycle relies on row
locks and database constraints, and the RAG app needs the `vector` extension:

```bash
docker run -d --name documents-test-db -p 5433:5432 \
  -e POSTGRES_DB=documents -e POSTGRES_USER=documents -e POSTGRES_PASSWORD=documents \
  pgvector/pgvector:pg17
.venv/bin/pytest
.venv/bin/ruff check .
```

The default suite uses deterministic fake adapters (`RAG_ANSWER_PROVIDER=fake`,
`CELERY_TASK_ALWAYS_EAGER=true`, `RAG_EMBEDDING_CACHE=false` in
`config.settings.test`), so it never needs Redis, an API key, or network access.

## Evaluation

`tests/fixtures/corpus/` holds a synthetic corpus about the fictional brick
factory **Sanasa (ساناسا)**: 49 `.docx` documents (English, Persian, mixed,
Arabic, German) and 71 questions across six difficulty tiers — simple,
exact-term, multi-hop, ambiguous, unanswerable, and adversarial. The corpus
includes outdated price lists, near-duplicate specs, and prompt-injection text
on purpose.

```bash
# Load the corpus (idempotent by title; --reset removes it first)
python manage.py load_eval_corpus --actor <username>

# Build the index for a profile
python manage.py rebuild_index --profile jina-v3 --sync

# Compare retrieval modes on the same active build (no chat calls)
python manage.py evaluate_rag --retrieval-only --profile jina-v3 --top-k 10 \
  --mode dense --output eval-results/dense
python manage.py evaluate_rag --retrieval-only --profile jina-v3 --top-k 10 \
  --mode lexical --output eval-results/lexical
python manage.py evaluate_rag --retrieval-only --profile jina-v3 --top-k 10 \
  --mode hybrid_rrf --candidate-k 20 --rrf-k 60 --output eval-results/hybrid

# Opt into Jina reranking (live rerank calls, but still no answer-model calls).
# Compare against the same mode/top-k without --reranker; use separate outputs.
python manage.py evaluate_rag --retrieval-only --profile jina-v3 --top-k 10 \
  --mode dense --reranker jina --rerank-candidate-k 20 \
  --output eval-results/dense-reranked
python manage.py evaluate_rag --retrieval-only --profile jina-v3 --top-k 10 \
  --mode hybrid_rrf --candidate-k 20 --rrf-k 60 \
  --reranker jina --rerank-candidate-k 20 --output eval-results/hybrid-reranked
```

Results are written to `eval-results/` (gitignored). The command defaults to
`--mode dense --reranker none`, independently of the API's settings. Reports
record the mode, candidate pool, RRF constant, lexical versions, and reranking
provider/model/shortlist depth. The reranker model comes from `RAG_RERANK_MODEL`.
Cached embeddings are reused when available; new question text still requires
an embedding call, and enabled reranking always calls Jina for nonempty results.

On the 71-question Jina v3 comparison (59 questions with expected sources, zero
errors), dense recall@5 is **0.814** and hybrid is **0.791**. The full summary,
conditions, and limitations are in [docs/eval-baseline.md](docs/eval-baseline.md).

### Long-document and chunking evaluation

Optional `evidence.jsonl` and `splits.json` sidecars add exact source-span coverage,
alternative sufficient evidence sets, and development/test selection. Validate
without provider calls or database access:

```bash
python manage.py validate_eval_corpus --corpus tests/fixtures/corpus_long_v1
```

Build experimental profiles with `rebuild_index --sync --no-activate`, then use
`evaluate_rag --build BUILD_UUID --split development` to evaluate a completed
inactive build without changing the live index. `--context-chars` adds an
explicit source-character budget (not a token estimate). Reports retain raw
selected passages/offsets, context overlap, index inventory, input hashes, and
requested/completed/failed counts. Failed runs save reports and exit nonzero.

For the supported annotation schema, metrics, and the full size/overlap comparison
workflow, see [docs/long-document-evaluation.md](docs/long-document-evaluation.md).
The original corpus remains supported without sidecars. This adds evaluation
infrastructure, not new paragraph/table chunkers or answer-quality judging.

## Design notes

- **Services own writes.** `documents/services.py` is the only write path; API
  views and Admin call it, so validation and revision rules cannot diverge.
- **Revisions are append-only.** The current content is the highest sequence.
  This gives the future indexing step a stable target and keeps history.
- **Logical deletion.** Deleted documents disappear from reads, downloads, and
  Admin, but revisions and files are retained. This is not privacy erasure; a
  purge workflow is deferred.
- **Files are private.** Uploads are served only through an authenticated
  endpoint, never from a public media directory.
- **DOCX parsing is bounded.** Upload size, archive member count, expanded
  size, and extracted text length are limited; macros, encryption, and DTDs are
  rejected. Only body paragraphs and tables are extracted — no headers,
  footers, text boxes, or page numbers.
- **RAG is a separate app with explicit contracts.** `rag/contracts.py` defines
  `Chunk`, `RetrievedPassage`, `AnswerResult`, and the `Chunker`/`Embedder`/
  `Retriever`/`Reranker`/`AnswerGenerator` interfaces. LangChain and the provider SDKs live only in
  `rag/adapters/`; `rag/factory.py` is the single composition point.
- **Embedders are swappable.** OpenRouter and Jina adapters both implement the
  same contract. The Jina adapter uses its `retrieval.passage` /
  `retrieval.query` task parameter, which is why it is preferred for retrieval.
- **Embeddings are cached on disk** by `(model, text)`, so evaluation re-runs
  and re-indexing unchanged text make no provider calls.
- **Embedding changes are data migrations.** `IndexProfile` records the model,
  dimension, metric, and chunker versions. Changing any of them means a new
  profile and a new build, not a settings flip.
- **Index builds are revision snapshots.** Chunks are keyed by profile and
  immutable revision, so a new build reuses unchanged vectors. A build is
  activated only after it completes and only if it still covers every current
  revision; retrieval always filters through the active build.
- **Indexing is queued after commit.** `documents/services.py` schedules a
  build with `transaction.on_commit`, so API and Admin writes share one path
  and a rolled-back revision never gets indexed.
- **Answers are grounded or abstained.** Retrieved text is treated as
  untrusted data; citations are validated against the supplied passages, and an
  answer with no supported citation is recorded as an abstention.

## Known limitations

- DOCX extraction is synchronous; very large documents are rejected rather
  than queued.
- A process crash between writing an upload and committing its revision can
  leave an orphaned file. `python manage.py audit_orphan_files` reports them;
  add `--delete` to remove files older than 24 hours.
- No per-user document ownership; all staff share one library.
- Equal-weight hybrid retrieval currently regresses overall recall on the
  Sanasa corpus. Dense is the default. Optional Jina reranking improves mean
  recall@5 but introduces some no-hit cases and exact-term/ambiguous regressions;
  see the evaluation report. BM25 is not implemented.
- Answer generation is synchronous inside the request. Slow free models can
  make `POST /questions/` slow; indexing is the only asynchronous step.
- Free models are rate-limited and can change or disappear. OpenRouter's free
  tier allows 50 requests/day; Jina has been reliable. Every answer records the
  provider and model used, but historical answers are not reproducible if the
  model changes.
- No vector index (HNSW/IVFFlat) is created; search is exact and fine for small
  corpora but will need an index at scale.
- Deleting a document excludes it from retrieval immediately, but its chunks
  remain stored until a new build or a purge.
- Answer-quality evaluation and an LLM judge are not implemented; only retrieval
  and evidence coverage are measured. Citation validation checks that cited
  references exist in the supplied passages; it does not semantically prove
  every claim in an answer.
- Reranking and live provider calls add latency and cost and require
  credentials. Each nonempty shortlist makes a live Jina request.
- The long-document pilot corpus has a known annotation issue (`ld016`) and
  fact families that cross development/test splits, so its results are
  exploratory development evidence, not a clean held-out benchmark. The
  original fixture was not edited; see
  [docs/long-document-evaluation.md](docs/long-document-evaluation.md).
