from django.core.management.base import BaseCommand, CommandError

from rag.evaluation.dataset import CorpusError, load_corpus


class Command(BaseCommand):
    help = (
        "Validate corpus files, evidence offsets, alternatives, and splits without provider calls."
    )
    requires_system_checks = []

    def add_arguments(self, parser):
        parser.add_argument("--corpus", required=True)

    def handle(self, *args, **options):
        try:
            corpus = load_corpus(options["corpus"])
        except CorpusError as exc:
            raise CommandError(str(exc)) from exc
        self.stdout.write(
            f"Valid: {len(corpus.documents)} documents, {len(corpus.questions)} questions, "
            f"{len(corpus.evidence)} evidence annotations."
        )
        if corpus.splits:
            for name, ids in corpus.splits.items():
                self.stdout.write(f"{name}: {len(ids)} questions")
        else:
            self.stdout.write("No splits.json: only --split all is available.")
        if not corpus.evidence:
            self.stdout.write("No evidence.jsonl: document-level metrics only.")
        self.stdout.write("This validates annotation structure/text, not semantic correctness.")
