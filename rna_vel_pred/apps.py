from django.apps import AppConfig
from django.db.models.signals import post_migrate
from django.core.management import call_command


class RnaVelPredConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'rna_vel_pred'

    def ready(self):
        post_migrate.connect(self.add_git_commits, sender=self)
        post_migrate.connect(self.add_parameters_and_groups, sender=self)

    @staticmethod
    def add_git_commits(**kwargs):
        call_command('record_git_commit')

    @staticmethod
    def add_parameters_and_groups(**kwargs):
        from django_experiment_tracker import models as tracker_models

        ungrouped_parameters = (
            tracker_models.Parameter.objects.get_or_create(
                parameter_name='batch_size', parameter_default_value=str(1024),
            )[0],
        )
        pg, pg_created = tracker_models.ParameterGroup.objects.get_or_create(parameter_group_name='-ungrouped-')
        if pg_created:
            pg.parameters.set(ungrouped_parameters)

        # scvelo dataset file parameters
        scvelo_dataset_file_parameters = (
            tracker_models.Parameter.objects.get_or_create(
                parameter_name='umap_dimension',
                parameter_type=tracker_models.ParameterType.INT,
                parameter_default_value=str(2),
            )[0],
        )
        scvelo_datasets = (
            'pancreas',
            'dentate_gyrus',
        )
        for ds in scvelo_datasets:
            pg, pg_created = tracker_models.ParameterGroup.objects.get_or_create(parameter_group_name=f'dataset_file_{ds}')
            if pg_created:
                pg.parameters.set(scvelo_dataset_file_parameters)

        graph_dataset_parameters = (
            tracker_models.Parameter.objects.get_or_create(
                parameter_name='sparsity_step',
                parameter_type=tracker_models.ParameterType.INT,
                parameter_default_value=str(10),
            )[0],
            tracker_models.Parameter.objects.get_or_create(
                parameter_name='neighbor_count',
                parameter_type=tracker_models.ParameterType.INT,
                parameter_default_value=str(10),
            )[0],
        )

        scvelo_dataset_parameters = graph_dataset_parameters + (
            tracker_models.Parameter.objects.get_or_create(
                parameter_name='dataset_file_alt_id',
                parameter_type=tracker_models.ParameterType.STRING,
                parameter_default_value='???',
            )[0],
        )
        for ds in scvelo_datasets:
            pg, pg_created = tracker_models.ParameterGroup.objects.get_or_create(parameter_group_name=f'dataset_{ds}')
            if pg_created:
                pg.parameters.set(scvelo_dataset_parameters)

        generated_dataset_parameters =  graph_dataset_parameters + (
            tracker_models.Parameter.objects.get_or_create(
                parameter_name='measurement_count',
                parameter_type=tracker_models.ParameterType.INT,
                parameter_default_value=str(4000),
            )[0],
            tracker_models.Parameter.objects.get_or_create(
                parameter_name='initial_noise_scale',
                parameter_type=tracker_models.ParameterType.FLOAT,
                parameter_default_value=str(0.5),
            )[0],
        )
        generated_datasets = (
            'simple',
            'oscillation',
            'bifurcation',
            'detransition',
        )
        for ds in generated_datasets:
            pg, pg_created = tracker_models.ParameterGroup.objects.get_or_create(parameter_group_name=f'dataset_{ds}')
            if pg_created:
                pg.parameters.set(generated_dataset_parameters)

        # pg, pg_created = tracker_models.ParameterGroup.objects.get_or_create(parameter_group_name='graph_dataset')
        # if pg_created:
        #     pg.parameters.set(graph_dataset_parameters)
