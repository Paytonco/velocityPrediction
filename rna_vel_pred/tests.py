import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from django.conf import settings
from django.core.management import call_command
from django.core.management.base import CommandError
from django.test import TestCase
from django.utils import timezone

from django_experiment_tracker.models import GitCommit
from rna_vel_pred.management.commands.run_experiment import Command as RunExperimentCommand
from rna_vel_pred.models import Experiment

get_current_commit = RunExperimentCommand._get_current_commit


class ExperimentCommandTestCase(TestCase):
    @classmethod
    def setUpTestData(cls):
        commit = GitCommit.objects.create(
            commit_time=timezone.now(),
            branch="test",
            commit_sha="a" * 40,
        )
        cls.experiment = Experiment.objects.create(
            alt_id="exp_test0001",
            git_commit_created=commit,
            git_commit_valid_for=commit,
        )

    def setUp(self):
        self.temporary_directory = TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        temporary_directory = Path(self.temporary_directory.name)
        runs_patch = patch.object(
            Experiment,
            "run_dir",
            lambda experiment: temporary_directory / experiment.alt_id,
        )
        runs_patch.start()
        self.addCleanup(runs_patch.stop)
        commit_patch = patch(
            "rna_vel_pred.management.commands.run_experiment.Command._get_current_commit",
            return_value=self.experiment.git_commit_valid_for,
        )
        self.get_current_commit = commit_patch.start()
        self.addCleanup(commit_patch.stop)

    @patch("rna_vel_pred.management.commands.run_experiment.subprocess.run")
    def test_run_experiment_records_completion_and_redirects_output(self, run):
        def write_output(*args, **kwargs):
            kwargs["stdout"].write("training output\n")
            kwargs["stderr"].write("training warning\n")
            return subprocess.CompletedProcess([], 0)

        run.side_effect = write_output
        call_command("run_experiment", self.experiment.alt_id)

        stdout = run.call_args.kwargs["stdout"]
        stderr = run.call_args.kwargs["stderr"]
        self.assertEqual(Path(stdout.name), self.experiment.run_dir() / "stdout.log")
        self.assertEqual(Path(stderr.name), self.experiment.run_dir() / "stderr.log")
        self.assertTrue(stdout.closed)
        self.assertTrue(stderr.closed)
        self.assertEqual((self.experiment.run_dir() / "stdout.log").read_text(), "training output\n")
        self.assertEqual((self.experiment.run_dir() / "stderr.log").read_text(), "training warning\n")

        self.experiment.refresh_from_db()
        self.assertIsNotNone(self.experiment.time_completed)
        self.assertEqual(self.experiment.exit_code, 0)
        self.assertEqual(
            self.experiment.git_commit_valid_for,
            self.get_current_commit.return_value,
        )
        run.assert_called_once_with(
            [
                sys.executable,
                str(settings.BASE_DIR / "manage.py"),
                "train_experiment",
                self.experiment.alt_id,
            ],
            cwd=settings.BASE_DIR,
            check=False,
            stdout=stdout,
            stderr=stderr,
        )

    @patch("rna_vel_pred.management.commands.run_experiment.subprocess.run")
    def test_run_experiment_records_nonzero_exit_without_raising(self, run):
        run.return_value = subprocess.CompletedProcess([], 23)

        call_command("run_experiment", self.experiment.alt_id)

        self.experiment.refresh_from_db()
        self.assertEqual(self.experiment.exit_code, 23)
        self.assertIsNotNone(self.experiment.time_completed)

    @patch("rna_vel_pred.management.commands.run_experiment.subprocess.run")
    def test_run_experiment_skips_completed_experiment(self, run):
        self.experiment.time_completed = timezone.now()
        self.experiment.exit_code = 0
        self.experiment.save(update_fields=["time_completed", "exit_code"])

        call_command("run_experiment", self.experiment.alt_id)

        run.assert_not_called()

    @patch("rna_vel_pred.management.commands.run_experiment.subprocess.run")
    def test_run_experiment_force_reruns_completed_experiment(self, run):
        self.experiment.time_completed = timezone.now()
        self.experiment.exit_code = 0
        self.experiment.save(update_fields=["time_completed", "exit_code"])
        run.return_value = subprocess.CompletedProcess([], 4)

        call_command("run_experiment", self.experiment.alt_id, force=True)

        self.experiment.refresh_from_db()
        self.assertEqual(self.experiment.exit_code, 4)
        run.assert_called_once()

    def test_run_experiment_rejects_unknown_experiment(self):
        with self.assertRaisesMessage(CommandError, "does not exist"):
            call_command("run_experiment", "exp_missing")

    @patch("rna_vel_pred.management.commands.run_experiment.subprocess.run")
    def test_run_experiment_rejects_uncommitted_changes(self, run):
        self.get_current_commit.side_effect = CommandError(
            "Cannot run an experiment with uncommitted changes"
        )

        with self.assertRaisesMessage(CommandError, "uncommitted changes"):
            call_command("run_experiment", self.experiment.alt_id)

        run.assert_not_called()

    @patch("rna_vel_pred.management.commands.run_experiment.subprocess.run")
    def test_run_experiment_updates_commit_before_launch(self, run):
        current_commit = GitCommit.objects.create(
            commit_time=timezone.now(),
            branch="current",
            commit_sha="b" * 40,
        )
        self.get_current_commit.return_value = current_commit

        def check_commit(*args, **kwargs):
            self.experiment.refresh_from_db()
            self.assertEqual(self.experiment.git_commit_valid_for, current_commit)
            return subprocess.CompletedProcess([], 0)

        run.side_effect = check_commit
        call_command("run_experiment", self.experiment.alt_id)

        run.assert_called_once()

    def test_git_preflight_excludes_sqlite_database(self):
        command = RunExperimentCommand()
        with patch.object(
            command,
            "_git",
            side_effect=["", self.experiment.git_commit_valid_for.commit_sha],
        ) as git:
            current_commit = get_current_commit(command)

        self.assertEqual(current_commit, self.experiment.git_commit_valid_for)
        self.assertEqual(
            git.call_args_list[0].args,
            (
                "status",
                "--porcelain",
                "--",
                ".",
                ":(top,exclude)db.sqlite3",
            ),
        )

    def test_git_preflight_rejects_other_changes(self):
        command = RunExperimentCommand()
        with patch.object(command, "_git", return_value=" M rna_vel_pred/models.py"):
            with self.assertRaisesMessage(CommandError, "uncommitted changes"):
                get_current_commit(command)

    def test_train_experiment_rejects_unknown_experiment(self):
        with self.assertRaisesMessage(CommandError, "does not exist"):
            call_command("train_experiment", "exp_missing")
