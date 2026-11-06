from django.contrib.auth import get_user_model
from django.core.management.base import BaseCommand, CommandError
from rest_framework.authtoken.models import Token


class Command(BaseCommand):
    help = "Create or return an API token for an existing staff user."

    def add_arguments(self, parser):
        parser.add_argument("username")

    def handle(self, *args, **options):
        user = get_user_model().objects.filter(username=options["username"]).first()
        if user is None:
            raise CommandError(f"User {options['username']!r} does not exist.")
        if not user.is_active or not user.is_staff:
            raise CommandError("The user must be an active staff member.")
        token, _ = Token.objects.get_or_create(user=user)
        self.stdout.write(token.key)
