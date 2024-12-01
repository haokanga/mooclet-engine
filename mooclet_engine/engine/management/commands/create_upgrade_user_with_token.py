# app/management/commands/createsuperuser_with_token.py

from django.core.management.base import BaseCommand
from django.contrib.auth.models import User
from rest_framework.authtoken.models import Token

class Command(BaseCommand):
    def handle(self, *args, **options):
        self.stdout.write('Starting upgrade user creation...')
        if not User.objects.filter(username="upgrade_root").exists():
            user = User.objects.create_superuser("upgrade_root", "doswalt@carnegielearning.com", "upgrade")
            token = Token.objects.create(user=user)

            env_path = '/shared/.env.docker.local'
            with open(env_path, 'r') as f:
                lines = f.readlines()

            with open(env_path, 'w') as f:
                for line in lines:
                    if line.startswith('MOOCLET_API_KEY'):
                        f.write(f'MOOCLET_API_KEY={token.key}\n')
                    else:
                        f.write(line)
            self.stdout.write(self.style.SUCCESS('Successfully created superuser and token for upgrade'))