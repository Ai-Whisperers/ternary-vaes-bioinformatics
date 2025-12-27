CLI Reference Guide
===================

The Ternary VAE command-line interface provides commands for training, analysis, and data management.


Installation
------------

After installing the package, the CLI is available as ``ternary-vae`` or ``tvae``:

.. code-block:: bash

    pip install -e .
    ternary-vae --help


Command Structure
-----------------

.. code-block::

    ternary-vae <command-group> <command> [options]

    Command Groups:
      train     Training commands
      analyze   Analysis commands
      data      Data management commands


Train Commands
--------------

ternary-vae train run
~~~~~~~~~~~~~~~~~~~~~

Train a Ternary VAE model:

.. code-block:: bash

    ternary-vae train run [OPTIONS]

Options:

--config PATH          Path to config file
--epochs INT           Number of training epochs [default: 100]
--batch-size INT       Training batch size [default: 64]
--latent-dim INT       Latent space dimension [default: 16]
--device TEXT          Device to use (cuda/cpu) [default: cuda]
--output-dir PATH      Output directory [default: outputs/]


ternary-vae train hiv
~~~~~~~~~~~~~~~~~~~~~

Train for HIV analysis:

.. code-block:: bash

    ternary-vae train hiv [OPTIONS]

Options:

--drug-class TEXT      Drug class: pi, nrti, nnrti, ini, all [required]
--epochs INT           Number of epochs [default: 100]
--latent-dim INT       Latent dimension [default: 32]


ternary-vae train resume
~~~~~~~~~~~~~~~~~~~~~~~~

Resume training from checkpoint:

.. code-block:: bash

    ternary-vae train resume --checkpoint PATH


Analyze Commands
----------------

ternary-vae analyze stanford
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Analyze Stanford HIVDB data:

.. code-block:: bash

    ternary-vae analyze stanford [OPTIONS]

Options:

--drug-class TEXT      Drug class to analyze [default: all]
--output-dir PATH      Output directory [default: outputs/analysis/stanford]
--save-figures         Save visualization figures


ternary-vae analyze catnap
~~~~~~~~~~~~~~~~~~~~~~~~~~

Analyze CATNAP neutralization data:

.. code-block:: bash

    ternary-vae analyze catnap [OPTIONS]

Options:

--antibody TEXT        Specific antibody to analyze [default: all]
--min-records INT      Minimum records per antibody [default: 100]


ternary-vae analyze ctl
~~~~~~~~~~~~~~~~~~~~~~~

Analyze CTL epitope data:

.. code-block:: bash

    ternary-vae analyze ctl [OPTIONS]

Options:

--protein TEXT         Protein to analyze [default: all]
--hla-type TEXT        HLA restriction type to filter


ternary-vae analyze tropism
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Analyze coreceptor tropism:

.. code-block:: bash

    ternary-vae analyze tropism [OPTIONS]


ternary-vae analyze all
~~~~~~~~~~~~~~~~~~~~~~~

Run complete analysis pipeline:

.. code-block:: bash

    ternary-vae analyze all [OPTIONS]


Data Commands
-------------

ternary-vae data status
~~~~~~~~~~~~~~~~~~~~~~~

Show dataset status:

.. code-block:: bash

    ternary-vae data status

Example output:

.. code-block::

    Dataset Status
    ==============

    Stanford HIVDB:
      ✓ PI: 2,171 records
      ✓ NRTI: 1,867 records
      ✓ NNRTI: 2,270 records
      ✓ INI: 846 records

    CATNAP: ✓ 189,879 records
    CTL Summary: ✓ 2,116 epitopes


ternary-vae data download
~~~~~~~~~~~~~~~~~~~~~~~~~

Download datasets:

.. code-block:: bash

    ternary-vae data download [SOURCE]

Sources:

- ``huggingface`` - HuggingFace datasets
- ``all`` - All available sources


ternary-vae data validate
~~~~~~~~~~~~~~~~~~~~~~~~~

Validate data integrity:

.. code-block:: bash

    ternary-vae data validate


Global Options
--------------

These options apply to all commands:

--help                 Show help message
--version              Show version
--verbose / -v         Enable verbose output
--quiet / -q           Suppress output


Environment Variables
---------------------

TERNARY_VAE_DATA_DIR
    Base directory for datasets [default: data/]

TERNARY_VAE_OUTPUT_DIR
    Base directory for outputs [default: outputs/]

TERNARY_VAE_CONFIG_DIR
    Base directory for configs [default: config/]


Examples
--------

Complete workflow:

.. code-block:: bash

    # Check data status
    ternary-vae data status

    # Download missing data
    ternary-vae data download huggingface

    # Train for HIV analysis
    ternary-vae train hiv --drug-class pi --epochs 200

    # Run full analysis
    ternary-vae analyze all --save-figures
