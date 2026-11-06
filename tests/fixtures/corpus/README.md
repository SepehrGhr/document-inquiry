# Sanasa synthetic RAG evaluation corpus

Sanasa (ساناسا) is an entirely fictional brick manufacturer. This fixture contains 49 `.docx` documents plus 71 evaluation questions for retrieval and grounded-answer testing. It includes English, Persian, mixed-language, Arabic, and German material; current and obsolete versions; deliberate distractors; messy records; and prompt-injection-like text that must be treated only as document data.

Use `manifest.json` to enumerate the documents and their authority/status. In particular, use the current 2025 price list and approved technical sheets over drafts, archives, and OCR imports. Each line of `questions.jsonl` is an independent evaluation item. `expected_sources` lists the documents supporting the expected answer; unanswerable items intentionally have no supporting evidence.

The corpus is fictional and intended only for tests.
