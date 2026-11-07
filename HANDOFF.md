# Handoff — Document Q&A project

Last updated: 2026-11-07. This file is the entry point for a new session.
Read it first, then `README.md`, then the source files it points to.

## What this project is

A take-home assignment for an AI Engineering role. The brief is in
`پروژه LLM.md` (Persian). Summary of the hard requirements:

- Django is mandatory.
- Must run with Docker.
- LangChain is mandatory for the processing flow and LLM communication.
- A complete, usable API.
- UI is Django Admin only — no separate frontend.
- Document CRUD, DOCX support, full text stored, ask questions, find relevant
  content, advanced retrieval, generate answers grounded in documents, store
  question/answer history.
- Simplicity and readability matter more than unnecessary complexity.

The user's stated goal: demonstrate the ability to design systems, choose the
right tools, and show awareness of common pitfalls and limitations of RAG
systems. Maintainability and clean layer boundaries are a top concern — the
RAG pipeline should be replaceable without touching the API or database layer,
and the embedding model should be swappable with minimal code changes.

## Current state

Three milestones are complete and committed:

1. **Foundation** — database, document lifecycle, DOCX ingestion, REST API,
   Django Admin.
2. **Vanilla RAG** — dense retrieval over pgvector, grounded answers with
   validated citations, abstention, question/answer history, Celery + Redis
   indexing.
3. **Evaluation harness** — a 49-document synthetic corpus, 71 questions, a
   retrieval evaluation command, and a measured dense baseline.

The **answer pass of the evaluation is not built yet**, and the answer model is
currently blocked (see "Blockers" below).

### Decisions taken with the user

- **Generation**: free models through OpenRouter (`stealth/crouton-alpha`).
- **Embeddings**: Jina `jina-embeddings-v3` (1024 dims), via a dedicated
  adapter that uses Jina's `retrieval.passage` / `retrieval.query` task
  parameter. OpenRouter's free embedding models were tried first but hit a
  50-requests/day cap.
- **Indexing**: Celery + Redis, triggered after commit from
  `documents/services.py`.
- **Scope**: working vanilla RAG first, then evaluation-driven improvements
  (lexical + RRF, reranker) — not speculative advanced techniques.
- **Evaluation results**: full JSON/markdown in gitignored `eval-results/`;
  headline numbers go into committed docs.

## Blockers

- **OpenRouter free tier is exhausted.** The daily cap is 50 free-model
  requests; the answer pass needs ~71 chat calls. The reset is at 00:00 UTC.
  Options: wait and batch across days, add $10 of credits (1,000/day), or add
  a different `AnswerGenerator` adapter (the contract is provider-neutral).
- **Docker end-to-end run** of web + worker + redis + pgvector has not been
  done since the RAG milestone.

## RAG architecture as built

```
src/rag/
  contracts.py        # Chunk, RetrievedPassage, AnswerResult, protocols
  chunking.py         # CharacterChunker (char-1), offsets into exact text
  citations.py        # [S1] extraction and validation
  models.py           # IndexProfile, IndexBuild, IndexBuildRevision,
                      # IndexedChunk (pgvector), Question, Answer, AnswerSource
  indexing.py         # create_build / run_build / activation rules
  dispatch.py         # transaction.on_commit -> Celery task
  services.py         # ask(): retrieve -> generate -> validate -> persist
  factory.py          # the only place adapters are chosen
  tasks.py            # run_index_build
  adapters/
    openrouter.py     # LangChain OpenAI-compatible chat + embeddings
    jina.py           # Jina embeddings with retrieval.passage/query tasks
    cache.py          # disk cache for embeddings (CachedEmbedder)
    pgvector_index.py # dense search scoped to the active build
    fakes.py          # HashingEmbedder, EchoAnswerGenerator (tests)
  evaluation/
    dataset.py        # load + validate manifest.json / questions.jsonl
    corpus.py         # load .docx into Documents, idempotent by title
    metrics.py        # pure retrieval/answer metrics
    runner.py         # retrieval pass
    report.py         # JSON + markdown
  api/                # POST/GET /api/v1/questions/, GET /questions/{id}/
  management/commands/
    create_index_profile.py
    rebuild_index.py
    load_eval_corpus.py
    evaluate_rag.py
```

### Key invariants

- Chunks are keyed by `(profile, revision, ordinal)`; builds reference
  revisions, so unchanged vectors are reused across builds.
- A build activates only if completed **and** it still covers every current
  revision; at most one build is active (partial unique constraint).
- Retrieval filters by active build and excludes logically deleted documents.
- An answer with no valid citation is stored as an abstention.
- Embeddings are cached on disk by `(model, text)`, so re-runs and re-indexing
  unchanged text cost nothing.

### Index profiles in the dev database

| Name | Provider | Model | Dims | Status |
|---|---|---|---|---|
| `default` | openrouter | `nvidia/nemotron-3-embed-1b:free` | 2048 | built, superseded |
| `jina-v3` | jina | `jina-embeddings-v3` | 1024 | **active** |

`RAG_ACTIVE_PROFILE` defaults to `default`; pass `--profile jina-v3` to
`rebuild_index` and `evaluate_rag` until the setting is changed.

## Evaluation

### Corpus

`tests/fixtures/corpus/` (committed, 1.9 MB):

- `docs/` — 49 `.docx` files about the fictional brick factory **Sanasa
  (ساناسا)**: company docs, spec sheets, price lists (current and outdated),
  emails, contracts, meeting notes, support tickets, policies, and deliberately
  messy/contradictory documents.
- `manifest.json` — every document with language, type, status
  (`current`/`outdated`/`draft`/`archived`), effective date, and notes on
  deliberate contradictions and prompt-injection documents.
- `questions.jsonl` — 71 questions across six difficulty tiers.
- `README.md` — corpus description.

Language mix: 31 English, 10 Persian, 6 mixed, 1 Arabic, 1 German.

Question distribution: 18 simple, 18 multi-hop, 12 exact-term, 12 unanswerable,
7 ambiguous, 4 adversarial.

The corpus was generated by another agent from a prompt kept in the gitignored
`model-guide/corpus-prompt.md`.

### Commands

```bash
# Load the corpus into the document store (idempotent by title)
manage.py load_eval_corpus --actor <username> [--reset]

# Build the index for a profile
manage.py rebuild_index --profile jina-v3 --sync

# Retrieval-only evaluation (no chat calls)
manage.py evaluate_rag --retrieval-only --profile jina-v3 --top-k 10
```

Results go to `eval-results/retrieval.json` and `retrieval.md` (gitignored).

### Dense baseline (Jina v3, top-k 10, 59 scored questions, 0 errors)

| Metric | Value |
|---|---|
| recall@1 | 0.551 |
| recall@3 | 0.749 |
| recall@5 | 0.814 |
| recall@10 | 0.924 |
| hit@5 | 0.983 |
| MRR | 0.808 |

By difficulty:

| Tier | n | r@1 | r@5 | r@10 | MRR |
|---|---|---|---|---|---|
| simple | 18 | 0.778 | 0.944 | 1.000 | 0.861 |
| exact_term | 12 | 0.750 | 1.000 | 1.000 | 0.847 |
| multi_hop | 18 | 0.361 | 0.648 | 0.870 | 0.863 |
| ambiguous | 7 | 0.214 | 0.619 | 0.762 | 0.498 |
| adversarial | 4 | 0.375 | 0.750 | 0.875 | 0.750 |

By language: English r@5 0.784, Persian 0.833, mixed 0.833, German 1.000 (n=1).

**Reading of the baseline:** simple and exact-term questions are nearly solved
at r@5; multi-hop (0.648) and ambiguous (0.619) are the weak tiers. The gap
between r@1 (0.551) and r@10 (0.924) says the right document is usually in the
top 10 but often not first — the classic case for hybrid retrieval + RRF.
Persian is not a weak spot.

An earlier partial OpenRouter run showed r@5 0.907, but it covered only 43 of
59 questions with the hard multi-hop ones missing. **Treat that number as
invalid.**

## Verification status

- `pytest` — 77 passed (needs a pgvector-enabled PostgreSQL; see below).
- `ruff check .` — clean; `ruff format` applied.
- `makemigrations --check --dry-run` — no pending changes.
- OpenAPI generation — 0 errors, 0 warnings.
- Celery app discovers `rag.tasks.run_index_build`.
- Live Jina embedding calls verified (English, Persian, mixed, batch).
- Live OpenRouter chat verified earlier (English, Persian, abstention).
- **Not verified:** the evaluation answer pass, and a full `docker compose`
  run of web + worker + redis.

### Git history (branch `master`, no remote)

```
e9d2a0d test: add sanasa evaluation corpus and questions
ccf5372 feat(rag): add retrieval evaluation harness and corpus loader
9501014 feat(rag): add jina embedder and disk embedding cache
9058e77 docs: document rag setup, api and design decisions
df5339b feat(rag): add vanilla rag pipeline with pgvector and celery
172dd54 docs: document test database recreation steps
cb9c1c0 docs: add handoff notes for the rag milestone
0a87c14 docs: document setup, api usage and design decisions
6f5e17a chore: add project skills and assignment brief
1d6c220 test: cover document lifecycle, docx parsing and api
96b6887 feat(api): add document rest api and admin management
5f45887 feat(documents): add revisioned document lifecycle and docx ingestion
9ba817f chore: bootstrap django project with docker and postgres
```

Working tree is clean except an untracked `.agents/` directory (a mirror of
`.claude/skills/` that appeared during a session; not created by us, left
alone).

### Gitignored local-only paths

- `.env` — secrets (`DJANGO_SECRET_KEY`, `OPENROUTER_KEY`, `JINA_KEY`).
- `model-guide/` — provider notes and the corpus-generation prompt.
- `eval-results/` — evaluation output.
- `.cache/` — the embedding cache.

## Architecture as built

```
src/
  config/
    settings/{base,development,test,production}.py
    celery.py          # Celery app, autodiscovers tasks
    urls.py            # admin, /api/v1/, /api/schema/, /api/docs/, /health/*
    views.py           # live + ready health checks
    wsgi.py
  documents/
    models.py          # Document, DocumentRevision
    services.py        # ALL writes go through here; schedules indexing
    selectors.py       # read-side queries
    exceptions.py      # DocumentError, InvalidDocumentInput, DocumentNotFound, RevisionConflict
    forms.py           # Admin form (text or DOCX, optional on edit)
    admin.py           # DocumentAdmin, DocumentRevisionAdmin (read-only)
    ingestion/
      types.py         # ParsedDocument(text, parser_version)
      docx.py          # bounded DOCX extraction, ORM-independent
    api/
      serializers.py
      views.py
      urls.py
    management/commands/
      create_api_token.py
      audit_orphan_files.py
  rag/                 # see "RAG architecture as built" above
tests/
  unit/test_docx.py            # parser, no DB
  unit/test_chunking.py        # chunker offsets and overlap
  unit/test_citations.py       # [S1] extraction and validation
  unit/test_openrouter_adapter.py
  unit/test_jina_adapter.py
  unit/test_embedding_cache.py
  unit/test_eval_dataset.py    # corpus validation
  unit/test_eval_metrics.py    # pure metrics
  unit/test_eval_runner.py
  unit/test_eval_report.py
  integration/test_services.py # lifecycle, locking, revisions
  integration/test_rag_indexing.py
  integration/test_rag_dispatch.py
  integration/test_eval_corpus.py
  api/test_documents_api.py    # REST endpoints
  api/test_questions_api.py    # question endpoint, history, auth
  api/test_admin.py            # Admin writes go through services
  fixtures/corpus/             # the Sanasa evaluation corpus
```

### Key design decisions (and why)

1. **Services own all writes.** `documents/services.py` is the only write path.
   API views and Admin both call it. This is the boundary that keeps the two
   interfaces from diverging. Do not add a second write path.

2. **Immutable, append-only revisions.** `DocumentRevision` rows are never
   updated or deleted by the application. Current content = highest `sequence`.
   There is deliberately **no** `Document.current_revision` FK — that would be a
   second source of truth. Index builds reference specific revision IDs.

3. **Optimistic concurrency.** Content replacement requires `expected_revision`
   and returns `409` on mismatch. Edits and deletion take a `select_for_update`
   row lock on the document.

4. **Logical deletion.** `deleted_at` / `deleted_by`. Deleted documents vanish
   from reads, downloads, Admin, and retrieval. Revisions and files are
   retained. This is **not** privacy erasure — documented as a known limitation.

5. **Private file storage.** Uploads are served only through an authenticated
   endpoint (`/api/v1/documents/{id}/revisions/{n}/file/`), never from a public
   media directory. Storage keys are server-generated.

6. **Bounded DOCX parsing.** Validates the actual OOXML/ZIP package, not the
   filename. Limits: 10 MiB upload, 50 MiB expanded, 2000 members, 5M chars
   extracted. Rejects macros, encryption, DTDs. Extracts body paragraphs and
   tables in document order. No headers/footers/text boxes/page numbers.

7. **No normalization of stored text.** Persian, English, mixed text, half-spaces
   and zero-width characters are stored exactly as provided. Any search-specific
   normalization belongs in a *derived* representation later, not in the source.

8. **Auth.** Staff-only throughout. Session auth (Admin/browsable API) + DRF
   token auth. Tokens via `manage.py create_api_token <username>`. No signup.

9. **Contracts, not frameworks.** `rag/contracts.py` defines `Chunk`,
   `RetrievedPassage`, `AnswerResult`, and the `Chunker`/`Embedder`/
   `AnswerGenerator` protocols. LangChain and provider SDKs live only in
   `rag/adapters/`; `rag/factory.py` is the single composition point.

10. **Embedding changes are data migrations.** `IndexProfile` records provider,
    model, dimension, metric, and chunker versions. Changing any of them means
    a new profile and a new build, not a settings flip.

11. **Indexing is queued after commit.** `documents/services.py` schedules a
    build with `transaction.on_commit`, so API and Admin writes share one path
    and a rolled-back revision never gets indexed.

12. **Answers are grounded or abstained.** Retrieved text is treated as
    untrusted data; citations are validated against the supplied passages, and
    an answer with no supported citation is recorded as an abstention.

### Data model

`Document`: UUID pk, title, description, created_by/at, updated_at,
deleted_at/deleted_by.

`DocumentRevision`: UUID pk, document FK, `sequence` (unique per document,
>= 1), `text` (full text), `source` (`text` | `docx`), `file` (FileField,
blank for text), `original_filename`, `file_size`, `content_sha256`,
`parser_version`, created_by/at. Check constraint ties source to file presence.

`IndexProfile`: immutable embedding/index identity — name, provider, model id,
model revision, dimension, metric, normalize, parser/chunker versions, chunk
size and overlap.

`IndexBuild`: profile FK, status (`pending`/`running`/`completed`/`failed`),
`is_active` (at most one true), error, chunk count, timestamps.

`IndexBuildRevision`: which immutable revisions a build covers.

`IndexedChunk`: profile FK, document FK, revision FK, ordinal, text, start/end
offsets, content hash, pgvector `embedding`. Unique on
`(profile, revision, ordinal)`.

`Question`: text, asked_by, created_at.

`Answer`: question FK, status (`answered`/`abstained`/`failed`), text, reason,
build FK, provider, model, token counts, latency.

`AnswerSource`: answer FK, chunk/document/revision FKs, title, revision
sequence, ordinal, rank, score, `cited`, text snapshot.

### API surface (`/api/v1/`)

| Method | Path | Purpose |
|---|---|---|
| GET/POST | `/documents/` | list (paginated, `?search=` on title) / create from text or DOCX |
| GET/PATCH/DELETE | `/documents/{id}/` | detail with current text / metadata update / logical delete |
| GET/POST | `/documents/{id}/revisions/` | history / replace content (`expected_revision` required) |
| GET | `/documents/{id}/revisions/{n}/` | one revision with text |
| GET | `/documents/{id}/revisions/{n}/file/` | authorized original download |
| GET/POST | `/questions/` | question history / ask a question |
| GET | `/questions/{id}/` | one question with its answer and sources |

`POST /questions/` returns `503` when no index is active. OpenAPI at
`/api/schema/`, Swagger at `/api/docs/`. Health at `/health/live/` and
`/health/ready/`.

## Environment gotchas (important)

- **Tests need a pgvector-enabled PostgreSQL.** `config/settings/test.py` reads
  `TEST_POSTGRES_PORT` (default 5433). Plain `postgres:17` cannot run the `rag`
  migration. Recreate the container with:

  ```bash
  docker run -d --name roshan-test-db -p 5433:5432 \
    -e POSTGRES_DB=documents -e POSTGRES_USER=documents -e POSTGRES_PASSWORD=documents \
    pgvector/pgvector:pg17
  ```

  During the RAG work a second container `roshan-test-db-pgvector` was run on
  port 5434 and tests were run with `TEST_POSTGRES_PORT=5434`. Either works.
- **This machine cannot resolve `files.pythonhosted.org` from inside Docker.**
  `docker compose build` fails at `pip install`. The image was verified by
  temporarily vendoring wheels (`pip download` into `wheelhouse/`, then
  `pip install --no-index --find-links=wheelhouse`), and the Dockerfile was
  reverted to the standard version afterwards. On a normal network the standard
  Dockerfile should build. **Do not commit a wheelhouse workaround.**
- The host pip index is a private Nexus mirror (`nexus.arusha.dev`) configured
  in `~/.config/pip/pip.conf`. It is slow but works. Do not print or commit
  credentials from it.
- Local run: `export DJANGO_SETTINGS_MODULE=config.settings.development
  POSTGRES_HOST=127.0.0.1 POSTGRES_PORT=5433`, then `manage.py migrate` /
  `runserver`. Celery needs Redis (`CELERY_BROKER_URL`, default
  `redis://localhost:6379/0`).
- **Free provider limits.** OpenRouter free models: 50 requests/day, and 429s
  under rapid successive calls. Jina has been reliable. The embedding cache
  (`.cache/embeddings/`) means repeated text is never re-embedded.

## What is NOT done (next steps)

- **Evaluation answer pass** — `evaluate_rag` only implements
  `--retrieval-only`. Still needed: `judge.py` (deterministic checks plus an
  LLM judge for multi-hop/ambiguous/adversarial), answer metrics in the report,
  and the full baseline run. Blocked on the answer model.
- **Lexical retrieval + RRF**, then an optional reranker, only if evaluation
  justifies it. This does not need the answer model — it can be measured with
  the retrieval pass alone.
- **Docker end-to-end run** of web + worker + redis + pgvector db.
- **Vector index (HNSW/IVFFlat)** once the corpus is large enough to need it.
- **Async answer generation** if free-model latency makes the request path
  unacceptable.
- **Chunk-level ground truth** for a subset of questions. `expected_sources` is
  document-level, so recall counts a chunk from the right document even if it
  is the wrong chunk. If the baseline looks suspiciously good, this is the next
  measurement to tighten.

### The agreed plan for the RAG milestone

- **Contracts, not frameworks.** Small explicit interfaces; a reranker contract
  only when a reranker is actually added.
- **Application-owned types** crossing the boundary — not ORM objects or
  LangChain `Document` objects everywhere.
- **`rag/adapters/`** holds provider-, database-, and framework-specific code.
  `rag/factory.py` is the single composition point.
- **Embedding changes are data migrations, not config flips.**
- **Index builds reference immutable revision IDs.** Keep "latest content
  revision" separate from "active indexed revision".
- **Retrieval path:** dense baseline → add lexical → fuse with reciprocal rank
  fusion → add a reranker only if evaluation justifies it. Do not add raw
  lexical and vector scores together.
- **Abstain on weak evidence** rather than forcing an answer.
- **Validate citations** against supplied evidence; treat retrieved text as
  untrusted data, not instructions (prompt injection).
- **RAG changes must be additive.** Document-management endpoint contracts
  should not change.

## Working conventions in this repo

- **Commits:** single-line Conventional Commits, no AI attribution, no body.
  See `.claude/skills/git-commit/SKILL.md`. Commit in logical milestones.
- **Skills available:** `karpathy-guidelines` (simplicity, surgical changes,
  surface assumptions) and `pragmatic-testing` (test behavior, not
  implementation; few high-value tests). Both are in `.claude/skills/`.
- **Tests:** PostgreSQL-backed for anything touching locks or constraints.
  Fixtures are built programmatically (`tests/conftest.py` has `make_docx`).
  The evaluation corpus is the only committed document set, and it is entirely
  fictional.
- **Style:** Ruff, line length 100, migrations exempt from E501.
- **Do not** initialize a remote, push, or amend pushed commits without being
  asked.

## Known limitations to keep in mind (and mention in the write-up)

- DOCX extraction is synchronous; large documents are rejected, not queued.
- A crash between writing an upload and committing its revision can leave an
  orphaned file. `manage.py audit_orphan_files` reports them; `--delete` removes
  files older than 24h.
- Deletion is logical, not erasure. No restore or purge UI. Deleted documents
  are excluded from retrieval immediately, but their chunks remain stored until
  a new build or a purge.
- No per-user ownership; all staff share one library.
- Admin chrome is LTR; Persian text renders correctly but the UI is not RTL.
- No full-text search yet — `?search=` is a title substring filter only.
- Retrieval is dense-only (vanilla RAG). No lexical/BM25 retrieval, rank
  fusion, or reranking yet.
- Answer generation is synchronous inside the request. Slow free models can
  make `POST /questions/` slow; indexing is the only asynchronous step.
- Free models are rate-limited and can change or disappear. Every answer
  records the provider and model used, but historical answers are not
  reproducible if the model changes.
- No vector index (HNSW/IVFFlat) is created; search is exact and fine for small
  corpora but will need an index at scale.
- Evaluation recall is document-level, not chunk-level.
