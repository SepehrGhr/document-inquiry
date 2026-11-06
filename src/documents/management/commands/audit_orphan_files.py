import time
from pathlib import Path

from django.conf import settings
from django.core.management.base import BaseCommand

from documents.models import DocumentRevision


class Command(BaseCommand):
    help = "Report (and optionally delete) stored files not referenced by any revision."

    def add_arguments(self, parser):
        parser.add_argument("--delete", action="store_true", help="Delete orphaned files.")
        parser.add_argument(
            "--min-age-hours",
            type=int,
            default=24,
            help="Only consider files older than this (default: 24).",
        )

    def handle(self, *args, **options):
        root = Path(settings.MEDIA_ROOT)
        referenced = set(DocumentRevision.objects.exclude(file="").values_list("file", flat=True))
        cutoff = time.time() - options["min_age_hours"] * 3600
        orphans = [
            path
            for path in root.rglob("*")
            if path.is_file()
            and path.stat().st_mtime < cutoff
            and path.relative_to(root).as_posix() not in referenced
        ]
        for path in orphans:
            self.stdout.write(str(path.relative_to(root)))
            if options["delete"]:
                path.unlink()
        self.stdout.write(f"{len(orphans)} orphaned file(s) found.")
