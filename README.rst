Inferring RNA Velocity Landscapes
=================================

Recent advancements in single-cell RNA sequencing have enabled the measurement of a cell’s velocity through transcriptional space, termed as RNA velocity.
This information is invaluable to contemporary methods in cell fate trajectory reconstruction.
However, these modern approaches do not accommodate legacy single-cell RNA sequencing datasets that lack RNA velocity information.
To address this limitation, we introduce a machine learning-based method for inferring RNA velocities.

Our approach is grounded in an extrinsic noise model of cell differentiation and leverages the group symmetries of the RNA velocity field.
We found that our approach accurately and precisely predicts RNA velocities from legacy scRNA-seq datasets in a variety of developmental biology contexts.
This model can be applied to any legacy single-cell RNA sequencing dataset to approximate the RNA velocity field.

In summary, this paper presents a numerical method of inferring RNA velocity landscapes from scRNA-seq datasets that were made without incorporating velocity information.
This advancement will enable experimentalists to modernize existing RNA seq datasets for use in velocity-based analysis without expending the time and resources to produce an entirely new dataset.
Additionally, this approach may be used to gain some insight into the regulatory dynamics which govern cell differentiation.

Installation
============

#. Install ``uv``:

   .. code:: bash

      curl -LsSf https://astral.sh/uv/install.sh | sh

#. Install Python dependencies using ``uv``:

   .. code:: bash

      uv sync

#. Change ``Conf.out_dir`` in ``conf/conf.py`` to the directory where the outputs of all experiments should be stored.

#. Change ``Conf.data_subdir`` in ``conf/conf.py`` to the directory where the processed data should be stored.

#. Create a ``runs`` directory inside ``Conf.out_dir``.

#. Test your installation by running pytest:

   .. code-block:: bash

      uv run pytest tests

Supplementary Documentation
---------------------------

* `Hydra <https://hydra.cc/docs/1.3/intro/>`_: The command-line inferface configuration library used to configure the experiments in this project.
* `Hydra ORM <https://github.com/reepoi/hydra-orm>`_: Library for saving experiment configurations to an `SQLite <https://sqlite.org/>`_ database.
* `PyTorch Geometric <https://pytorch-geometric.readthedocs.io/en/2.4.0/index.html>`_: The graph neural networks library utlized to implement the flocking models.
* `PyTorch Lightning <https://lightning.ai/docs/pytorch/2.2.2/>`_: The library that handles the model training and logging.

Training the models
===================

Example commands:

.. code:: bash

   # Motif
   python src/rna_vel_pred/main.py model=Second "datasets=[{_target_:conf.datasets.BifurcationMotif,time_step_count_sparsify:10,neighbor_count:10}]" use_directionless_loss=true
   # Dentate Gyrus
   python src/rna_vel_pred/main.py model=Second "datasets=[{_target_:conf.datasets.H5adUMap,dataset:DENTATE_GYRUS,time_step_count_sparsify:10,neighbor_count:120}]" use_directionless_loss=true

Evaluating the models
=====================

Each model training run is assigned an *alt_id*, a random string of eight characters that uniquely identifies the run.
Use this to indicate what model to evaluate below.

Example commands:

.. code:: bash

   # Motif
   python src/rna_vel_pred/main.py fit=false predict=true model=Trained model.conf=<alt_id> "datasets=[{_target_:conf.datasets.BifurcationMotif,time_step_count_sparsify:10,neighbor_count:10}]" use_directionless_loss=true
   # Dentate Gyrus
   python src/rna_vel_pred/main.py fit=false predict=true model=Trained model.conf=<alt_id> "datasets=[{_target_:conf.datasets.H5adUMap,dataset:DENTATE_GYRUS,time_step_count_sparsify:10,neighbor_count:120}]" use_directionless_loss=true
