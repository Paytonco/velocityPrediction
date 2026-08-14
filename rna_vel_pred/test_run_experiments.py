import signal
import subprocess
from io import StringIO
from unittest.mock import patch

from django.core.management import call_command
from django.core.management.base import CommandError
from django.test import TestCase, override_settings
from django.utils import timezone

from django_experiment_tracker.models import GitCommit
from django_experiment_tracker.management.commands.run_experiments import (
    Command as RunExperimentsCommand,
    RunningExperiment,
)
from rna_vel_pred.models import Experiment


class CompletedProcess:
    _next_pid = 1000

    def __init__(self, returncode=0):
        self.returncode = returncode
        self.pid = self._next_pid
        type(self)._next_pid += 1

    def poll(self):
        return self.returncode


class RunExperimentsCommandTestCase(TestCase):
    @classmethod
    def setUpTestData(cls):
        commit = GitCommit.objects.create(
            commit_time=timezone.now(),
            branch="test",
            commit_sha="c" * 40,
        )
        cls.experiments = [
            Experiment.objects.create(
                alt_id=f"exp_test{i:04d}",
                git_commit_created=commit,
                git_commit_valid_for=commit,
                exit_code=0,
            )
            for i in range(7)
        ]

    def run_command(self, *alt_ids, **options):
        return call_command(
            "run_experiments",
            *alt_ids,
            runner="run_experiment",
            resource="CUDA_VISIBLE_DEVICES=0,1",
            stdout=StringIO(),
            **options,
        )

    @patch(
        "django_experiment_tracker.management.commands.run_experiments.subprocess.Popen"
    )
    def test_assigns_each_resource_up_to_its_slot_limit(self, popen):
        popen.side_effect = [CompletedProcess() for _ in self.experiments]

        self.run_command(
            *(experiment.alt_id for experiment in self.experiments),
            slots_per_resource=2,
        )

        assigned_resources = [
            call.kwargs["env"]["CUDA_VISIBLE_DEVICES"]
            for call in popen.call_args_list
        ]
        self.assertEqual(assigned_resources[:4], ["0", "0", "1", "1"])
        self.assertEqual(assigned_resources[4:], ["0", "0", "1"])
        for call, experiment in zip(popen.call_args_list, self.experiments):
            self.assertEqual(
                call.args[0][-2:], ["run_experiment", experiment.alt_id]
            )
            self.assertTrue(call.kwargs["start_new_session"])

    @patch(
        "django_experiment_tracker.management.commands.run_experiments.subprocess.Popen"
    )
    def test_continues_after_runner_failure_and_reports_batch_failure(self, popen):
        popen.side_effect = [
            CompletedProcess(4),
            CompletedProcess(),
            CompletedProcess(),
        ]
        alt_ids = [experiment.alt_id for experiment in self.experiments[:3]]

        with self.assertRaisesMessage(CommandError, alt_ids[0]):
            self.run_command(*alt_ids, slots_per_resource=1)

        self.assertEqual(popen.call_count, 3)

    @patch(
        "django_experiment_tracker.management.commands.run_experiments.subprocess.Popen",
        return_value=CompletedProcess(),
    )
    def test_treats_missing_experiment_exit_code_as_failure(self, popen):
        experiment = self.experiments[0]
        experiment.exit_code = None
        experiment.save(update_fields=["exit_code"])

        with self.assertRaisesMessage(CommandError, experiment.alt_id):
            self.run_command(experiment.alt_id)

    @patch(
        "django_experiment_tracker.management.commands.run_experiments.subprocess.Popen"
    )
    def test_validates_all_experiments_before_launching(self, popen):
        with self.assertRaisesMessage(CommandError, "exp_missing"):
            self.run_command(self.experiments[0].alt_id, "exp_missing")

        popen.assert_not_called()

    def test_rejects_duplicate_experiments(self):
        alt_id = self.experiments[0].alt_id
        with self.assertRaisesMessage(CommandError, "duplicates"):
            self.run_command(alt_id, alt_id)

    @override_settings(EXPERIMENT_TRACKER_EXPERIMENT_MODEL=None)
    def test_requires_experiment_model_setting(self):
        with self.assertRaisesMessage(
            CommandError, "EXPERIMENT_TRACKER_EXPERIMENT_MODEL"
        ):
            self.run_command(self.experiments[0].alt_id)

    @override_settings(EXPERIMENT_TRACKER_EXPERIMENT_MODEL="auth.User")
    def test_rejects_model_that_is_not_an_experiment(self):
        with self.assertRaisesMessage(CommandError, "concrete subclass"):
            self.run_command(self.experiments[0].alt_id)

    def test_rejects_unknown_runner(self):
        with self.assertRaisesMessage(CommandError, "Unknown management command"):
            call_command(
                "run_experiments",
                self.experiments[0].alt_id,
                runner="missing_runner",
                resource="GPU=0",
            )

    def test_rejects_itself_as_runner(self):
        with self.assertRaisesMessage(CommandError, "cannot use itself"):
            call_command(
                "run_experiments",
                self.experiments[0].alt_id,
                runner="run_experiments",
                resource="GPU=0",
            )

    def test_validates_resource_configuration(self):
        invalid_resources = ["GPU", "1GPU=0", "GPU=", "GPU=0,,1", "GPU=0,0"]
        for resource in invalid_resources:
            with self.subTest(resource=resource):
                with self.assertRaises(CommandError):
                    call_command(
                        "run_experiments",
                        self.experiments[0].alt_id,
                        runner="run_experiment",
                        resource=resource,
                    )

        with self.assertRaisesMessage(CommandError, "at least 1"):
            self.run_command(self.experiments[0].alt_id, slots_per_resource=0)

    @patch(
        "django_experiment_tracker.management.commands.run_experiments.os.killpg"
    )
    def test_terminates_active_process_groups(self, killpg):
        process = CompletedProcess()
        process.returncode = None

        def poll_after_termination():
            if killpg.called:
                process.returncode = -signal.SIGTERM
            return process.returncode

        process.poll = poll_after_termination
        running = [RunningExperiment("exp_test0000", "0", process)]

        RunExperimentsCommand()._terminate_all(running)

        killpg.assert_called_once_with(process.pid, signal.SIGTERM)
