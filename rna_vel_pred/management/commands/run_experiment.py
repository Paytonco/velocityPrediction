import subprocess
import sys

from django.conf import settings
from django.core.management.base import BaseCommand, CommandError
from django.utils import timezone

from django_experiment_tracker.models import GitCommit
from rna_vel_pred.models import Experiment


class Command(BaseCommand):
    help = "Run an experiment in a Django subprocess and record its completion"
    requires_migrations_checks = True

    def add_arguments(self, parser):
        parser.add_argument("alt_id", help="Alternative ID of the experiment to run")
        parser.add_argument(
            "--force",
            action="store_true",
            help="Run the experiment even if it already has completion data",
        )

    def handle(self, *args, **options):
        alt_id = options["alt_id"]
        try:
            experiment = Experiment.objects.get(alt_id=alt_id)
        except Experiment.DoesNotExist as exc:
            raise CommandError(f"Experiment {alt_id!r} does not exist") from exc

        if not options["force"] and experiment.exit_code == 0:
            self.stdout.write(f"Experiment {alt_id} has already completed successfully; skipping.")
            return

        current_commit = self._get_current_commit()
        experiment.git_commit_valid_for = current_commit
        experiment.save(update_fields=["git_commit_valid_for"])

        command = [
            sys.executable,
            str(settings.BASE_DIR / "manage.py"),
            "train_experiment",
            alt_id,
        ]
        run_dir = experiment.run_dir()
        try:
            run_dir.mkdir(parents=True, exist_ok=True)
            with (
                (run_dir / "stdout.log").open("w") as stdout,
                (run_dir / "stderr.log").open("w") as stderr,
            ):
                process = subprocess.run(
                    command,
                    cwd=settings.BASE_DIR,
                    check=False,
                    stdout=stdout,
                    stderr=stderr,
                )
        except OSError as exc:
            raise CommandError(f"Could not launch experiment {alt_id}: {exc}") from exc

        experiment.time_completed = timezone.now()
        experiment.exit_code = process.returncode
        experiment.save(update_fields=["time_completed", "exit_code"])

        self.stdout.write(
            f"Experiment {alt_id} completed with exit code {process.returncode}."
        )

    def _get_current_commit(self):
        if self._git("status", "--porcelain"):
            raise CommandError("Cannot run an experiment with uncommitted changes")

        commit_sha = self._git("rev-parse", "HEAD")
        try:
            return GitCommit.objects.get(commit_sha=commit_sha)
        except GitCommit.DoesNotExist as exc:
            raise CommandError(
                f"Current commit {commit_sha} has not been recorded; run "
                "'python manage.py record_git_commit' first"
            ) from exc

    def _git(self, *args):
        try:
            process = subprocess.run(
                ["git", *args],
                cwd=settings.BASE_DIR,
                check=True,
                capture_output=True,
                text=True,
            )
        except FileNotFoundError as exc:
            raise CommandError("git is not available in PATH") from exc
        except subprocess.CalledProcessError as exc:
            stderr = (exc.stderr or "").strip()
            raise CommandError(f"git {' '.join(args)} failed: {stderr}") from exc
        return process.stdout.strip()
