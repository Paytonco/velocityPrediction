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

Get Started with Development
----------------------------

#. Install Python 3.8.10 or later.
#. Clone this repository with ``git clone``.
#. Create a Python virtual environment and activate it:

   .. code-block:: bash

      python3 -m venv .venv
      source .venv/bin/activate

#. Install the ``hatchling`` Python build tool:

   .. code-block:: bash

      pip install hatchling==1.25.0

#. Install `PyTorch 1.13.1 <pytorch.org/get-started/previous-versions/#v1131>`_.

#. Install the project Python package:

   .. code-block:: bash

      pip install -e .

#. Test your installation by running pytest:

   .. code-block:: bash

      pytest tests

Supplementary Documentation
---------------------------

* `Hydra <https://hydra.cc/docs/1.3/intro/>`_: The command-line inferface configuration library used to configure the experiments in this project.
* `PyTorch Geometric <https://pytorch-geometric.readthedocs.io/en/2.4.0/index.html>`_: The graph neural networks library utlized to implement the flocking models.
* `PyTorch Lightning <https://lightning.ai/docs/pytorch/2.2.2/>`_: The library that handles the model training and logging to WandB.
* `WandB API <https://docs.wandb.ai/ref/python/public-api/api>`_: The WandB API library used to retrieve the configurations of experiments.

Running Experiments
-------------------

Evaluating our trained model
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

We provide a pretrained model.
You can evaluate the model on your dataset by following these steps:

#. Create a new directory in the ``data`` directory of this repository with the name of your dataset ``<dataset-name>``.
#. Put your data into a CSV file with the following columns:

   - ``t``: the pseudotime of the measurement
   - ``x<integer>``: the components of the measurement's state in (projected) transcriptional space.
     For example, a three-dimensional transcriptional space would have the columns ``x1``, ``x2``, and ``x3``.
   - ``v<integer>``: (optional) the components of the measurement's RNA velocity.
     For example, a three-dimensional RNA velocity would have the columns ``v1``, ``v2``, and ``v3``.

#. Name your CSV file ``<dataset-name>__umap_n_components_<integer>`` where ``<integer>`` is the transcriptional space dimension of your data, and put it into the directory created in the first step.
#. Evaluate the model on your dataset by running the following command:

   .. code-block:: bash

      python3 src/main.py +dataset@dataset.<dataset-name>=SCVeloSaved dataset.<dataset-name>.data_subdir=<dataset-name> dataset.<dataset-name>.umap.n_components=<data-dimension> dataset.<dataset-name>.num_neighbors=<num-neighbors> dataset.<dataset-name>.sparsify_step_time=<sparsity-step> model=PretrainedModel trainer.fit=false trainer.pred=true

   where the angle braket values are replaced as follows:

   * ``<dataset-name>``: name of the dataset to evaluate the model on
   * ``<data-dimension>``: integer transcriptional space dimension of your data
   * ``<num-neighbors>``: integer size of the neighbor sets
   * ``<sparsity-step>``: the sparsification step to use

Example command that evaluates the pretrained model on the pancreas dataset projected to 17 dimensions with 20 neighbors and a sparsity step of 18:

.. code-block:: bash

   python3 src/main.py model=PretrainedModel trainer.fit=false trainer.pred=true +dataset@dataset.pancreas=SCVeloSaved dataset.pancreas.umap.n_components=17 dataset.pancreas.data_subdir=pancreas dataset.pancreas.num_neighbors=20 dataset.pancreas.sparsify_step_time=18

Training
^^^^^^^^

Train a model using this command:

.. code-block:: bash

   python3 src/main.py +dataset@dataset.<dataset-name>=SCVeloSaved dataset.<dataset-name>.data_subdir=<dataset-name> dataset.<dataset-name>.umap.n_components=<data-dimension> dataset.<dataset-name>.num_neighbors=<num-neighbors> dataset.<dataset-name>.sparsify_step_time=<sparsity-step> model.hidden.layers=<model-hidden-layers> model.hidden.dim=<model-hidden-dimension> model.bias=<model-bias> model.activation=<model-activation>

where the angle braket values are replaced as follows:

* ``<dataset-name>``: name of the dataset to train the model on
* ``<data-dimension>``: integer transcriptional space dimension of your data
* ``<num-neighbors>``: integer size of the neighbor sets
* ``<sparsity-step>``: the sparsification step to use
* ``<model-hidden-layers>``: integer number of hidden layers in the model
* ``<model-hidden-dimension>``: boolean indicating whether the model's linear layers use a bias term
* ``<model-bias>``: the model's activation function (e.g., ReLU)

Example of training a model on all the SCVelo datasets used in our paper:

.. code-block:: bash

   python3 src/main.py trainer.max_epochs=35 trainer.check_val_every_n_epoch=5 +dataset@dataset.pancreas=SCVeloSaved dataset.pancreas.umap.n_components=17 dataset.pancreas.data_subdir=pancreas dataset.pancreas.num_neighbors=20 dataset.pancreas.sparsify_step_time=18 +dataset@dataset.dentategyrus=SCVeloSaved dataset.dentategyrus.umap.n_components=17 dataset.dentategyrus.data_subdir=dentategyrus dataset.dentategyrus.num_neighbors=20 dataset.dentategyrus.sparsify_step_time=14 +dataset@dataset.bonemarrow=SCVeloSaved dataset.bonemarrow.umap.n_components=17 dataset.bonemarrow.data_subdir=bonemarrow dataset.bonemarrow.num_neighbors=20 dataset.bonemarrow.sparsify_step_time=28 +dataset@dataset.forebrain=SCVeloSaved dataset.forebrain.umap.n_components=17 dataset.forebrain.data_subdir=forebrain dataset.forebrain.num_neighbors=20 dataset.forebrain.sparsify_step_time=100 model.hidden.layers=9 model.hidden.dim=7 model.bias=false model.activation=ReLU dataset.forebrain.reverse_velocities=true
