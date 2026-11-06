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

## Current state: foundation + vanilla RAG

The Django foundation (database, document lifecycle, DOCX ingestion, REST API,
Django Admin) is complete. The RAG milestone is implemented as a working
**vanilla** pipeline: dense retrieval over pgvector, grounded answers with
validated citations, abstention, question/answer history, and Celery + Redis
indexing. It is not yet committed as of this update.

Decisions taken with the user for this milestone:

- Provider: free models through **OpenRouter** (exact model names/endpoints to
  be supplied by the user). The embedding model is a `create_index_profile`
  argument, not a hard-coded setting.
- Indexing: **Celery + Redis**, triggered after commit from
  `documents/services.py`.
- Scope: working vanilla RAG first, then evaluation-driven improvements
  (lexical + RRF, reranker) — not speculative advanced techniques.

### RAG architecture as built

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
    pgvector_index.py # dense search scoped to the active build
    fakes.py          # HashingEmbedder, EchoAnswerGenerator (tests)
  api/                # POST/GET /api/v1/questions/, GET /questions/{id}/
  management/commands/create_index_profile.py, rebuild_index.py
```

Key invariants:

- Chunks are keyed by `(profile, revision, ordinal)`; builds reference
  revisions, so unchanged vectors are reused across builds.
- A build activates only if completed **and** it still covers every current
  revision; at most one build is active (partial unique constraint).
- Retrieval filters by active build and excludes logically deleted documents.
- An answer with no valid citation is stored as an abstention.

### Verification status (RAG milestone)

- `pytest` — 44 passed (needs a pgvector-enabled PostgreSQL; see below).
- `ruff check .` — clean; `ruff format` applied to new files.
- `makemigrations --check --dry-run` — no pending changes.
- OpenAPI generation — 0 errors, 0 warnings.
- Celery app discovers `rag.tasks.run_index_build`.
- **Not yet verified:** a live OpenRouter call (no model/key supplied yet) and
  a full `docker compose` run of web + worker + redis.

### Test database now needs pgvector

The old `roshan-test-db` (plain `postgres:17`) cannot create the `vector`
extension. Recreate it with the pgvector image:

```bash
docker rm -f roshan-test-db
docker run -d --name roshan-test-db -p 5433:5432 \
  -e POSTGRES_DB=documents -e POSTGRES_USER=documents -e POSTGRES_PASSWORD=documents \
  pgvector/pgvector:pg17
```

The `rag` migration runs `CREATE EXTENSION IF NOT EXISTS vector` itself.

### Git history (branch `master`, no remote)

```
0a87c14 docs: document setup, api usage and design decisions
6f5e17a chore: add project skills and assignment brief
1d6c220 test: cover document lifecycle, docx parsing and api
96b6887 feat(api): add document rest api and admin management
5f45887 feat(documents): add revisioned document lifecycle and docx ingestion
9ba817f chore: bootstrap django project with docker and postgres
```

Working tree is clean except an untracked `.agents/` directory (a mirror of
`.claude/skills/` that appeared during the session; not created by us, left
alone). `.env` exists locally and is gitignored.

## Architecture as built

```
src/
  config/
    settings/{base,development,test,production}.py
    urls.py            # admin, /api/v1/, /api/schema/, /api/docs/, /health/*
    views.py           # live + ready health checks
    wsgi.py
  documents/
    models.py          # Document, DocumentRevision
    services.py        # ALL writes go through here
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
tests/
  unit/test_docx.py            # parser, no DB
  integration/test_services.py # lifecycle, locking, revisions
  api/test_documents_api.py    # REST endpoints
  api/test_admin.py            # Admin writes go through services
```

### Key design decisions (and why)

1. **Services own all writes.** `documents/services.py` is the only write path.
   API views and Admin both call it. This is the boundary that keeps the two
   interfaces from diverging. Do not add a second write path.

2. **Immutable, append-only revisions.** `DocumentRevision` rows are never
   updated or deleted by the application. Current content = highest `sequence`.
   There is deliberately **no** `Document.current_revision` FK — that would be a
   second source of truth. This gives the future indexing step a stable target:
   an index build can reference a specific revision ID.

3. **Optimistic concurrency.** Content replacement requires `expected_revision`
   and returns `409` on mismatch. Edits and deletion take a `select_for_update`
   row lock on the document.

4. **Logical deletion.** `deleted_at` / `deleted_by`. Deleted documents vanish
   from reads, downloads, and Admin. Revisions and files are retained. This is
   **not** privacy erasure — documented as a known limitation.

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

### Data model

`Document`: UUID pk, title, description, created_by/at, updated_at,
deleted_at/deleted_by.

`DocumentRevision`: UUID pk, document FK, `sequence` (unique per document,
>= 1), `text` (full text), `source` (`text` | `docx`), `file` (FileField,
blank for text), `original_filename`, `file_size`, `content_sha256`,
`parser_version`, created_by/at. Check constraint ties source to file presence.

### API surface (`/api/v1/`)

| Method | Path | Purpose |
|---|---|---|
| GET/POST | `/documents/` | list (paginated, `?search=` on title) / create from text or DOCX |
| GET/PATCH/DELETE | `/documents/{id}/` | detail with current text / metadata update / logical delete |
| GET/POST | `/documents/{id}/revisions/` | history / replace content (`expected_revision` required) |
| GET | `/documents/{id}/revisions/{n}/` | one revision with text |
| GET | `/documents/{id}/revisions/{n}/file/` | authorized original download |

OpenAPI at `/api/schema/`, Swagger at `/api/docs/`. Health at `/health/live/`
and `/health/ready/`.

## Verification status

- `pytest` — 20 passed (requires PostgreSQL; see below).
- `ruff check .` — clean.
- `makemigrations --check --dry-run` — no pending changes.
- OpenAPI generation — 0 errors.
- Session auth + token auth smoke-tested end to end.
- Docker image builds and runs; migrations, health, Admin, and docs verified
  inside the container.

### Environment gotchas (important)

- **Tests need PostgreSQL on port 5433.** A container named `roshan-test-db`
  was created for this. It can disappear when Docker Desktop restarts — if
  `docker start roshan-test-db` says "No such container", recreate it:

  ```bash
  docker run -d --name roshan-test-db -p 5433:5432 \
    -e POSTGRES_DB=documents -e POSTGRES_USER=documents -e POSTGRES_PASSWORD=documents \
    pgvector/pgvector:pg17
  ```

  `config/settings/test.py` reads `TEST_POSTGRES_PORT` (default 5433). The
  image must be pgvector-enabled; plain `postgres:17` cannot run the `rag`
  migration.
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
  `runserver`.

## What is NOT done (next steps)

- **Live OpenRouter verification — DONE (2026-11-07).** Models are in the
  gitignored `model-guide/` folder: chat `stealth/crouton-alpha`, embeddings
  `nvidia/nemotron-3-embed-1b:free` (2048 dims). Key is `OPENROUTER_KEY` in
  `.env`. Profile `default` was created and a build activated. Live smoke test
  passed: English, Persian, and an unanswerable question (correctly abstained).
  Note: free models return `429` under rapid successive calls; the adapter
  retries twice, but the evaluation harness should pace requests.
- **Docker end-to-end run** of web + worker + redis + pgvector db.
- **Evaluation harness** (dataset + metrics) — the next milestone, and the
  gate for any retrieval upgrade.
- **Lexical retrieval + RRF**, then an optional reranker, only if evaluation
  justifies it.
- **Persian/mixed-language retrieval quality testing.**
- **Vector index (HNSW/IVFFlat)** once the corpus is large enough to need it.
- **Async answer generation** if free-model latency makes the request path
  unacceptable.

### The agreed plan for the RAG milestone

The user approved this direction in the planning session. The vanilla part is
now built; the rest is the intended shape for the evaluation-driven phase:

- **Contracts, not frameworks.** Small explicit interfaces: `DocumentParser`,
  `Chunker`, `Embedder`, `ChunkIndex`, `Retriever`, `AnswerGenerator`. A
  reranker contract only when a reranker is actually added.
- **Application-owned types** crossing the boundary (`Chunk`, `RetrievedPassage`,
  `AnswerResult`) — not ORM objects or LangChain `Document` objects everywhere.
- **`rag/adapters/`** holds provider-, database-, and framework-specific code.
  `rag/factory.py` is the single composition point.
- **Embedding changes are data migrations, not config flips.** Store an
  `IndexProfile` (provider, model id, revision, dimension, metric, normalization,
  parser/chunking versions). Each index build belongs to a profile. Query
  embeddings must use the same profile. Changing models means re-embedding into
  a new index, evaluating, then switching the active profile.
- **Index builds reference immutable revision IDs.** Keep "latest content
  revision" separate from "active indexed revision". Activate a new build only
  after it completes; filter retrieval to the active build.
- **Retrieval path:** dense baseline → add lexical → fuse with reciprocal rank
  fusion → add a reranker only if evaluation justifies it. Do not add raw
  lexical and vector scores together.
- **Abstain on weak evidence** rather than forcing an answer.
- **Validate citations** against supplied evidence; treat retrieved text as
  untrusted data, not instructions (prompt injection).
- **Evaluation set** with answerable, unanswerable, exact-term, multi-document,
  edited/deleted-content, and Persian/mixed cases. Measure retrieval recall,
  answer correctness, citation support, abstention, latency, cost.
- **RAG changes must be additive.** Document-management endpoint contracts
  should not change.

### Open questions to settle with the user

- Exact OpenRouter chat model and embedding model/endpoint (user will supply).
- Whether OpenRouter serves embeddings at all; otherwise which embedding
  provider or local model to use.
- Whether the evaluation harness should be a separate milestone (current plan)
  or part of the next commit.

## Working conventions in this repo

- **Commits:** single-line Conventional Commits, no AI attribution, no body.
  See `.claude/skills/git-commit/SKILL.md`. Commit in logical milestones.
- **Skills available:** `karpathy-guidelines` (simplicity, surgical changes,
  surface assumptions) and `pragmatic-testing` (test behavior, not
  implementation; few high-value tests). Both are in `.claude/skills/`.
- **Tests:** PostgreSQL-backed for anything touching locks or constraints.
  Fixtures are built programmatically (`tests/conftest.py` has `make_docx`).
  No private or company documents in the repo.
- **Style:** Ruff, line length 100, migrations exempt from E501.
- **Do not** initialize a remote, push, or amend pushed commits without being
  asked.

## Known limitations to keep in mind (and mention in the write-up)

- DOCX extraction is synchronous; large documents are rejected, not queued.
- A crash between writing an upload and committing its revision can leave an
  orphaned file. `manage.py audit_orphan_files` reports them; `--delete` removes
  files older than 24h.
- Deletion is logical, not erasure. No restore or purge UI.
- No per-user ownership; all staff share one library.
- Admin chrome is LTR; Persian text renders correctly but the UI is not RTL.
- No full-text search yet — `?search=` is a title substring filter only.
