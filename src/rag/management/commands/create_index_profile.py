from django.conf import settings
from django.core.management.base import BaseCommand, CommandError

from rag.chunking import CHUNKER_VERSION, DEFAULT_CHUNK_SIZE, DEFAULT_OVERLAP
from rag.models import IndexProfile


class Command(BaseCommand):
    help = "Create an immutable index profile (or return the existing identical one)."

    def add_arguments(self, parser):
        parser.add_argument("--name", default=settings.RAG["ACTIVE_PROFILE"])
        parser.add_argument("--provider", default="openrouter")
        parser.add_argument("--model", required=True, help="Embedding model ID.")
        parser.add_argument("--model-revision", default="")
        parser.add_argument("--dimension", type=int, required=True)
        parser.add_argument("--metric", default=IndexProfile.Metric.COSINE)
        parser.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE)
        parser.add_argument("--chunk-overlap", type=int, default=DEFAULT_OVERLAP)

    def handle(self, *args, **options):
        if options["metric"] not in IndexProfile.Metric.values:
            raise CommandError(f"Unknown metric {options['metric']!r}.")
        profile, created = IndexProfile.objects.get_or_create(
            name=options["name"],
            defaults={
                "provider": options["provider"],
                "model_id": options["model"],
                "model_revision": options["model_revision"],
                "dimension": options["dimension"],
                "metric": options["metric"],
                "chunker_version": CHUNKER_VERSION,
                "chunk_size": options["chunk_size"],
                "chunk_overlap": options["chunk_overlap"],
            },
        )
        if not created:
            self.stdout.write(f"Profile {profile.name!r} already exists ({profile.id}).")
            return
        self.stdout.write(f"Created profile {profile.name!r} ({profile.id}).")
