from django.contrib.auth import get_user_model
from django.core.management.base import BaseCommand, CommandError

from rag.evaluation.corpus import load_corpus
from rag.evaluation.dataset import CorpusError
from rag.evaluation.dataset import load_corpus as read_corpus


class Command(BaseCommand):
    help = "Load the evaluation corpus into the document store (idempotent by title)."

    def add_arguments(self, parser):
        parser.add_argument("--corpus", default="tests/fixtures/corpus")
        parser.add_argument("--actor", default=None, help="Username to attribute writes to.")
        parser.add_argument(
            "--reset",
            action="store_true",
            help="Logically delete existing corpus documents before loading.",
        )

    def handle(self, *args, **options):
        try:
            corpus = read_corpus(options["corpus"])
        except CorpusError as exc:
            raise CommandError(str(exc)) from exc

        actor = self._actor(options["actor"])
        result = load_corpus(corpus, actor=actor, reset=options["reset"])
        self.stdout.write(f"{len(corpus.documents)} documents: {result}")

    def _actor(self, username):
        users = get_user_model().objects
        if username:
            user = users.filter(username=username).first()
            if user is None:
                raise CommandError(f"User {username!r} does not exist.")
            return user
        user = users.filter(is_staff=True, is_active=True).order_by("id").first()
        if user is None:
            raise CommandError("No staff user exists; pass --actor or create one first.")
        return user
