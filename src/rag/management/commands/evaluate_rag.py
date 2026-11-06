from pathlib import Path

from django.conf import settings
from django.core.management.base import BaseCommand, CommandError

from rag.adapters import pgvector_index
from rag.evaluation.dataset import CorpusError
from rag.evaluation.dataset import load_corpus as read_corpus
from rag.evaluation.report import retrieval_markdown, retrieval_report, write_json
from rag.evaluation.runner import run_retrieval
from rag.models import IndexProfile


class Command(BaseCommand):
    help = "Evaluate retrieval (and later answers) against the corpus questions."

    def add_arguments(self, parser):
        parser.add_argument("--corpus", default="tests/fixtures/corpus")
        parser.add_argument("--profile", default=settings.RAG["ACTIVE_PROFILE"])
        parser.add_argument("--top-k", type=int, default=settings.RAG["TOP_K"])
        parser.add_argument("--limit", type=int, default=0, help="Only the first N questions.")
        parser.add_argument("--delay", type=float, default=0.0, help="Seconds between questions.")
        parser.add_argument("--output", default="eval-results")
        parser.add_argument(
            "--retrieval-only",
            action="store_true",
            help="Skip answer generation (the only mode implemented so far).",
        )

    def handle(self, *args, **options):
        try:
            corpus = read_corpus(options["corpus"])
        except CorpusError as exc:
            raise CommandError(str(exc)) from exc

        profile = IndexProfile.objects.filter(name=options["profile"]).first()
        if profile is None:
            raise CommandError(f"No index profile named {options['profile']!r}.")
        build = pgvector_index.active_build()
        if build is None or build.profile_id != profile.id:
            raise CommandError(
                f"No active build for profile {profile.name!r}. Run rebuild_index first."
            )

        run = run_retrieval(
            corpus,
            build,
            top_k=options["top_k"],
            limit=options["limit"],
            delay=options["delay"],
            progress=self._progress,
        )
        report = retrieval_report(run, top_k=options["top_k"])

        output = Path(options["output"])
        write_json(report, output / "retrieval.json")
        (output / "retrieval.md").write_text(retrieval_markdown(report), encoding="utf-8")

        overall = report["overall"]
        self.stdout.write(
            f"{overall['questions']} questions: recall@5={overall['recall@5']:.3f} "
            f"mrr={overall['mrr']:.3f} errors={len(report['errors'])}"
        )
        self.stdout.write(f"Wrote {output / 'retrieval.json'} and {output / 'retrieval.md'}")

    def _progress(self, index, total, question, result):
        self.stdout.write(
            f"[{index}/{total}] {question.id} {question.difficulty:<11} "
            f"rr={result.score.reciprocal_rank():.2f} {result.latency_ms}ms"
        )
