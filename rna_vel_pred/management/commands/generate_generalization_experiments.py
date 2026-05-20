from django.core.management.base import BaseCommand, CommandError
from django.utils import timezone

import rna_vel_pred.utils
from notebooks.generate_generalization_experiments import build_experiment_parameters, get_latest_commit, get_tags


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
        # latest_git_commit = GitCommit.objects.order_by('-commit_time').first()
        # try:
        #     no_group_param_group = ParameterGroup.objects.get(parameter_group_name='-NoGroup-')
        # except ParameterGroup.DoesNotExist as e:
        #     raise CommandError(f"{e} Query: parameter_group_name={'-NoGroup-'!r}")
        rna_vel_pred.utils.create_experiments_from_parameters(
            build_experiment_parameters(),
            dict(git_commit=get_latest_commit()),
            get_tags(),
        )
        self.stdout.write(self.style.SUCCESS('Done!'))

