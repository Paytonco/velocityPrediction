from django.core.management.base import BaseCommand

from django_experiment_tracker.experiment_generation import create_parameterized_model_from_parameters
from notebooks.generate_generalization_experiments import build_experiment_parameters, get_latest_commit, get_tags
from rna_vel_pred.models import Experiment, ExperimentParameter


class Command(BaseCommand):
    help = "Record git commit metadata into django_experiment_tracker.GitCommit"
    output_transaction = True
    requires_migrations_checks = True

    def add_arguments(self, parser):
        parser.add_argument(
            "--backfill",
            type=int,
            default=0,
            help="Record last N commits reachable from HEAD",
        )
        parser.add_argument(
            "--stdin-rewrite-map",
            action="store_true",
            help="Read post-rewrite stdin map (old_sha new_sha) and record new SHAs",
        )

    def handle(self, *args, **options):
        create_parameterized_model_from_parameters(
            experiment_model=Experiment,
            experiment_parameter_model=ExperimentParameter,
            experiment_parameters=build_experiment_parameters(),
            experiment_model_kwargs=dict(git_commit=get_latest_commit()),
            tags=get_tags(),
        )
        self.stdout.write(self.style.SUCCESS('Done!'))
