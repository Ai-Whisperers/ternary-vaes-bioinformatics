Analysis Guide
==============

This guide covers running analyses on HIV datasets using trained Ternary VAE models.


Available Analyses
------------------

Stanford HIVDB (Drug Resistance)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Analyze drug resistance mutations from the Stanford HIV Drug Resistance Database:

.. code-block:: bash

    ternary-vae analyze stanford --drug-class all

**Key outputs:**

- Fold-change vs hyperbolic distance correlations
- Primary vs accessory mutation classification
- Cross-resistance pattern mapping
- Drug class-specific geometric signatures


CATNAP (Neutralization)
~~~~~~~~~~~~~~~~~~~~~~~

Analyze antibody neutralization data (189,879 records):

.. code-block:: bash

    ternary-vae analyze catnap --antibody VRC01

**Key outputs:**

- bnAb sensitivity geometric signatures
- Neutralization breadth vs epitope centrality
- Escape pathway prediction models


CTL Epitopes
~~~~~~~~~~~~

Analyze cytotoxic T-lymphocyte escape patterns (2,116 epitopes):

.. code-block:: bash

    ternary-vae analyze ctl --protein gag

**Key outputs:**

- HLA-stratified escape landscapes
- Epitope conservation vs radial position
- Protein-specific escape velocity


Tropism Analysis
~~~~~~~~~~~~~~~~

Analyze coreceptor tropism switching:

.. code-block:: bash

    ternary-vae analyze tropism

**Key outputs:**

- CCR5 vs CXCR4 hyperbolic separation
- Switching trajectory mapping
- Glycan shield correlation


Full Pipeline
~~~~~~~~~~~~~

Run all analyses:

.. code-block:: bash

    ternary-vae analyze all


Programmatic Analysis
---------------------

Run analyses directly in Python:

.. code-block:: python

    from src.data.hiv import load_stanford_hivdb
    from src.encoders import CodonEncoder
    from src.geometry import hyperbolic_distance

    # Load data
    df = load_stanford_hivdb(drug_class="pi")

    # Encode mutations
    encoder = CodonEncoder()
    embeddings = encoder.encode_mutations(df["CompMutList"])

    # Analyze geometric patterns
    distances = hyperbolic_distance(embeddings)


Cross-Dataset Integration
-------------------------

Combine multiple datasets for integrated analysis:

.. code-block:: python

    from scripts.hiv.analysis import cross_dataset_integration

    # Find positions under multiple selection pressures
    results = cross_dataset_integration.run(
        include_stanford=True,
        include_catnap=True,
        include_ctl=True
    )


Visualization
-------------

Generate publication-quality figures:

.. code-block:: bash

    ternary-vae analyze stanford --drug-class pi --save-figures

Figures are saved to ``outputs/figures/``.


Output Files
------------

Analysis outputs are saved to ``outputs/analysis/``:

.. code-block::

    outputs/analysis/
    ├── stanford/
    │   ├── resistance_distance_correlation.csv
    │   ├── mutation_classification.csv
    │   └── cross_resistance_matrix.csv
    ├── catnap/
    │   ├── bnab_sensitivity_signatures.csv
    │   └── escape_prediction_model.pkl
    ├── ctl/
    │   ├── hla_escape_landscapes.csv
    │   └── protein_escape_velocity.csv
    └── integrated/
        ├── vaccine_targets_ranked.csv
        └── constraint_map.csv
