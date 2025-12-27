Full Training Tutorial
======================

This tutorial covers complete training workflows for all Ternary VAE variants.


Training Modes
--------------

The unified training launcher supports multiple modes:

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - Mode
     - Description
   * - ``v5.11``
     - Standard Ternary VAE with hyperbolic geometry
   * - ``v5.11.11``
     - Enhanced VAE with homeostatic control
   * - ``epsilon``
     - Epsilon-VAE for checkpoint exploration
   * - ``hiv``
     - HIV-specific training pipeline
   * - ``swarm``
     - Multi-agent swarm training
   * - ``predictors``
     - Train resistance/escape predictors


V5.11 Training
--------------

The standard Ternary VAE with Poincare ball latent space:

.. code-block:: bash

    python src/train.py --mode v5.11 \
        --epochs 200 \
        --batch-size 512 \
        --learning-rate 1e-3 \
        --latent-dim 16 \
        --hidden-dim 64 \
        --save-dir outputs/v5.11

Key parameters:

- ``--latent-dim``: Hyperbolic space dimensionality (8-32)
- ``--hidden-dim``: Encoder/decoder hidden size (32-256)
- ``--curvature``: Poincare ball curvature (-2.0 to -0.5)


V5.11.11 Training (Homeostatic)
-------------------------------

Enhanced version with homeostatic control for stable training:

.. code-block:: bash

    python src/train.py --mode v5.11.11 \
        --epochs 200 \
        --batch-size 512 \
        --learning-rate 1e-3 \
        --target-kl 0.5 \
        --homeostasis-strength 0.1 \
        --save-dir outputs/v5.11.11

The homeostatic control maintains:

- **Target KL**: Prevents posterior collapse
- **Reconstruction balance**: Keeps ELBO components balanced
- **Gradient stability**: Smooths training dynamics


Programmatic Training
---------------------

For fine-grained control, use Python directly:

.. code-block:: python

    import torch
    from src.models.vae_v5_11 import TernaryVAEV5_11
    from src.training.trainers import TernaryVAETrainerV5_11
    from src.data import generate_all_ternary_operations

    # Create model
    model = TernaryVAEV5_11(
        input_dim=27,
        latent_dim=16,
        hidden_dim=64,
        curvature=-1.0,
        use_hyperbolic=True
    )

    # Generate data
    x, indices = generate_all_ternary_operations()

    # Create trainer
    trainer = TernaryVAETrainerV5_11(
        model=model,
        learning_rate=1e-3,
        beta=1.0,
        device="cuda"
    )

    # Train
    history = trainer.train(
        train_data=x,
        epochs=200,
        batch_size=512,
        validation_split=0.2
    )


Resuming Training
-----------------

Resume from a checkpoint:

.. code-block:: bash

    # Via CLI
    ternary-vae train resume checkpoints/model_epoch_100.pt \
        --epochs 100 \
        --device cuda

    # Via Python
    from src.models.vae_v5_11 import TernaryVAEV5_11

    model = TernaryVAEV5_11.load("checkpoints/model_epoch_100.pt")
    # Continue training...


Training with HIV Data
----------------------

For HIV-specific analysis:

.. code-block:: bash

    python src/train.py --mode hiv \
        --drug-class pi \
        --epochs 100 \
        --save-dir outputs/hiv

Or use the full pipeline:

.. code-block:: bash

    python scripts/hiv/run_full_hiv_pipeline.py \
        --drug-class pi \
        --epochs 100


Monitoring Training
-------------------

Training metrics are logged to:

- Console output (progress bars)
- ``outputs/<mode>/training_history.json``
- TensorBoard (if enabled)

Visualize with TensorBoard:

.. code-block:: bash

    tensorboard --logdir outputs/


Hyperparameter Tuning
---------------------

Key hyperparameters to tune:

.. code-block:: python

    # Learning rate schedule
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs
    )

    # Beta annealing for KL
    beta_schedule = torch.linspace(0.0, 1.0, warmup_epochs)

    # Gradient clipping
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)


Multi-GPU Training
------------------

For larger models:

.. code-block:: python

    import torch.nn as nn

    model = nn.DataParallel(model)
    model = model.cuda()


Checkpointing
-------------

Save checkpoints regularly:

.. code-block:: python

    # Every N epochs
    if epoch % save_every == 0:
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
        }, f'checkpoint_epoch_{epoch}.pt')


Next Steps
----------

- :doc:`epsilon_vae` - Explore checkpoint landscape
- :doc:`predictors` - Build downstream predictors
- :doc:`meta_learning` - Few-shot adaptation
