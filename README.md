# Document Q&A — Foundation

A Django service for managing text documents and DOCX uploads. This milestone
covers the database, document lifecycle, REST API, and Django Admin. Retrieval
and answer generation (LangChain) are the next milestone and are not part of
this code yet.

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

## Run with Docker

```bash
cp .env.example .env          # then set DJANGO_SECRET_KEY
docker compose up -d --build
docker compose exec web python manage.py migrate
docker compose exec web python manage.py createsuperuser
```

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
```

## Development

Requires Python 3.13 and a PostgreSQL instance.

```bash
python -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python manage.py migrate
.venv/bin/python manage.py runserver
```

Tests run against PostgreSQL (not SQLite) because the lifecycle relies on row
locks and database constraints:

```bash
docker run -d --name roshan-test-db -p 5433:5432 \
  -e POSTGRES_DB=documents -e POSTGRES_USER=documents -e POSTGRES_PASSWORD=documents \
  postgres:17
.venv/bin/pytest
.venv/bin/ruff check .
```

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
- **No AI dependencies yet.** No embedder, vector store, or model provider is
  required to run or test this milestone.

## Known limitations

- DOCX extraction is synchronous; very large documents are rejected rather
  than queued.
- A process crash between writing an upload and committing its revision can
  leave an orphaned file. `python manage.py audit_orphan_files` reports them;
  add `--delete` to remove files older than 24 hours.
- No per-user document ownership; all staff share one library.
