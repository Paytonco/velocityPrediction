from django.core.management.base import BaseCommand, CommandError

# from rna_vel_pred import training
from rna_vel_pred.models import Experiment


class Command(BaseCommand):
    help = "Train a single experiment"
    requires_migrations_checks = True

    def add_arguments(self, parser):
        parser.add_argument("alt_id", help="Alternative ID of the experiment to train")

    def handle(self, *args, **options):
        alt_id = options["alt_id"]
        try:
            experiment = Experiment.objects.get(alt_id=alt_id)
        except Experiment.DoesNotExist as exc:
            raise CommandError(f"Experiment {alt_id!r} does not exist") from exc

        try:
            pass
            # training.train(experiment)
        except NotImplementedError as exc:
            raise CommandError(str(exc)) from exc
