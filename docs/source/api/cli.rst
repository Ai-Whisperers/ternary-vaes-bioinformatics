CLI Reference
=============

The Ternary VAE CLI provides commands for training, analysis, and data management.

Installation
------------

After installing the package, the CLI is available as ``ternary-vae`` or ``tvae``:

.. code-block:: bash

    ternary-vae --help
    tvae --help


Train Commands
--------------

.. automodule:: src.cli.train
   :members:
   :undoc-members:
   :show-inheritance:


Analyze Commands
----------------

.. automodule:: src.cli.analyze
   :members:
   :undoc-members:
   :show-inheritance:


Data Commands
-------------

.. automodule:: src.cli.data
   :members:
   :undoc-members:
   :show-inheritance:


Examples
--------

Training a Model
~~~~~~~~~~~~~~~~

.. code-block:: bash

    # Basic training
    ternary-vae train run --config config/train_ternary_vae.yaml

    # HIV-specific training
    ternary-vae train hiv --drug-class pi --epochs 100

    # Resume from checkpoint
    ternary-vae train resume --checkpoint outputs/checkpoints/epoch_50.pt


Running Analysis
~~~~~~~~~~~~~~~~

.. code-block:: bash

    # Analyze Stanford HIVDB data
    ternary-vae analyze stanford --drug-class all

    # Analyze CATNAP neutralization data
    ternary-vae analyze catnap --antibody VRC01

    # Run full analysis pipeline
    ternary-vae analyze all


Managing Data
~~~~~~~~~~~~~

.. code-block:: bash

    # Check data status
    ternary-vae data status

    # Download datasets
    ternary-vae data download huggingface

    # Validate data integrity
    ternary-vae data validate
