Quickstart Guide
================

This guide will help you get started with Ternary VAE in minutes.


Installation
------------

Install from source:

.. code-block:: bash

    git clone https://github.com/Ai-Whisperers/ternary-vaes-bioinformatics.git
    cd ternary-vaes-bioinformatics
    pip install -e ".[all]"


Basic Training
--------------

Train a Ternary VAE model on the complete ternary operation space:

.. code-block:: python

    import torch
    from src import TernaryVAE, TrainingConfig
    from src.data import generate_all_ternary_operations
    from src.training import TernaryVAETrainer

    # Generate all 19,683 ternary operations
    x, indices = generate_all_ternary_operations()
    x_tensor = torch.tensor(x, dtype=torch.float32)

    # Create model
    model = TernaryVAE(
        latent_dim=16,
        hidden_dim=64,
        max_radius=0.95,
        curvature=1.0,
    )

    # Configure training
    config = TrainingConfig(
        epochs=100,
        batch_size=512,
        learning_rate=1e-3,
    )

    # Train
    trainer = TernaryVAETrainer(model, config)
    trainer.train(x_tensor)


Using the CLI
-------------

The CLI provides quick access to common operations:

.. code-block:: bash

    # Check package info
    ternary-vae info

    # Train with config file
    ternary-vae train run --config configs/ternary.yaml

    # Check available data
    ternary-vae data status


HIV Analysis
------------

Analyze HIV drug resistance data:

.. code-block:: python

    from src.data.hiv import load_stanford_hivdb, get_stanford_drug_columns

    # Load protease inhibitor data
    df = load_stanford_hivdb("pi")
    print(f"Loaded {len(df)} records")

    # Get drug columns
    drugs = get_stanford_drug_columns("pi")
    print(f"Drug columns: {drugs}")


Next Steps
----------

- Read the :doc:`guide/training` guide for detailed training options
- Explore the :doc:`api/models` for model architectures
- Check :doc:`guide/analysis` for HIV analysis workflows
