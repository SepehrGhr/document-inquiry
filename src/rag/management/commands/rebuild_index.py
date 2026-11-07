from django.conf import settings
from django.core.management.base import BaseCommand, CommandError

from rag import indexing
from rag.models import IndexProfile


class Command(BaseCommand):
    help = "Build and activate an index for the active profile."

    def add_arguments(self, parser):
        parser.add_argument("--profile", default=settings.RAG["ACTIVE_PROFILE"])
        parser.add_argument(
            "--no-activate",
            action="store_true",
            help="Build for evaluation without changing the live index (requires --sync).",
        )
        parser.add_argument(
            "--sync", action="store_true", help="Run inline instead of queueing a Celery task."
        )

    def handle(self, *args, **options):
        if options["no_activate"] and not options["sync"]:
            raise CommandError("--no-activate requires --sync.")
        profile = IndexProfile.objects.filter(name=options["profile"]).first()
        if profile is None:
            raise CommandError(f"No index profile named {options['profile']!r}.")
        build = indexing.create_build(profile)
        if options["sync"]:
            build = indexing.run_build(build.id, activate=not options["no_activate"])
            self.stdout.write(f"Build {build.id}: {build.status}, {build.chunk_count} chunks.")
            return
        from rag.tasks import run_index_build

        run_index_build.delay(str(build.id))
        self.stdout.write(f"Queued build {build.id}.")
