Training Guide
==============

This guide covers training Ternary VAE models for various bioinformatics applications.


Basic Training
--------------

The simplest way to train a model is using the CLI:

.. code-block:: bash

    ternary-vae train run --epochs 100


Configuration
~~~~~~~~~~~~~

Training is configured through YAML files:

.. code-block:: yaml

    # config/train_ternary_vae.yaml
    model:
      type: ternary_vae
      latent_dim: 16
      hidden_dim: 64
      num_layers: 3

    training:
      epochs: 100
      batch_size: 64
      learning_rate: 0.001
      optimizer: adam

    data:
      type: ternary_operations
      train_split: 0.8


HIV-Specific Training
---------------------

For HIV drug resistance analysis:

.. code-block:: bash

    ternary-vae train hiv \
        --drug-class pi \
        --epochs 100 \
        --latent-dim 32


Drug Classes
~~~~~~~~~~~~

- **PI** - Protease Inhibitors (2,171 records)
- **NRTI** - Nucleoside RT Inhibitors (1,867 records)
- **NNRTI** - Non-Nucleoside RT Inhibitors (2,270 records)
- **INI** - Integrase Inhibitors (846 records)


Programmatic Training
---------------------

Training directly in Python:

.. code-block:: python

    from src.models import TernaryVAE
    from src.training import TernaryVAETrainer
    from src.data import generate_all_ternary_operations

    # Prepare data
    x, indices = generate_all_ternary_operations()

    # Create model
    model = TernaryVAE(
        latent_dim=16,
        hidden_dim=64,
        num_layers=3,
        use_hyperbolic=True
    )

    # Create trainer
    trainer = TernaryVAETrainer(
        model=model,
        learning_rate=1e-3,
        use_curriculum=True
    )

    # Train
    history = trainer.train(
        train_data=x,
        epochs=100,
        batch_size=64
    )


Resuming Training
-----------------

Resume from a checkpoint:

.. code-block:: bash

    ternary-vae train resume --checkpoint outputs/checkpoints/epoch_50.pt


Monitoring Progress
-------------------

Training progress can be monitored via:

1. **Console output** - Loss and metrics printed each epoch
2. **TensorBoard** - ``tensorboard --logdir outputs/runs``
3. **Checkpoint files** - Saved in ``outputs/checkpoints/``


Best Practices
--------------

1. **Start small** - Begin with smaller latent dimensions and increase
2. **Use curriculum learning** - Gradually increase complexity
3. **Monitor KL divergence** - Avoid posterior collapse
4. **Save checkpoints** - Enable recovery from failures
5. **Validate regularly** - Track validation loss to detect overfitting
