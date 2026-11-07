# Document Q&A

A Django service for managing text documents and DOCX uploads, and answering
questions grounded in them with a RAG pipeline (LangChain, pgvector, Celery).

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
docker run -d --name roshan-test-db -p 5433:5432 \
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

# Retrieval-only evaluation (no chat calls)
python manage.py evaluate_rag --retrieval-only --profile jina-v3 --top-k 10
```

Results are written to `eval-results/` (gitignored). The current dense baseline
with Jina v3 is recall@5 0.814 and MRR 0.808 over 59 scored questions; the
per-tier breakdown is in `HANDOFF.md`.

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
  `AnswerGenerator` interfaces. LangChain and the provider SDKs live only in
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
- Retrieval is dense-only (vanilla RAG). No lexical/BM25 retrieval, rank
  fusion, or reranking yet; those are the next, evaluation-driven steps.
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
- The evaluation answer pass is not built yet; only retrieval is measured.
  Recall is document-level, not chunk-level.
