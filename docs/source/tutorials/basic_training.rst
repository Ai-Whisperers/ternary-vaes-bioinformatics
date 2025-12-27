Basic Training Tutorial
=======================

This tutorial covers the fundamentals of training a Ternary VAE model.


Overview
--------

The Ternary VAE learns to embed ternary operations (base-3 arithmetic) in a
hyperbolic latent space. This geometric structure captures the natural
hierarchy of codon relationships.


Setup
-----

.. code-block:: python

    import torch
    from src.models import TernaryVAE
    from src.training import TernaryVAETrainer
    from src.data import generate_all_ternary_operations


Understanding the Data
----------------------

Ternary operations are base-3 calculations. Each "digit" can be 0, 1, or 2:

.. code-block:: python

    # Generate all possible ternary operations
    x, indices = generate_all_ternary_operations()

    print(f"Number of operations: {len(x)}")  # 27 = 3^3
    print(f"Input shape: {x.shape}")          # (27, 27)


Building the Model
------------------

The Ternary VAE consists of:

1. **Encoder** - Maps inputs to hyperbolic latent space
2. **Decoder** - Reconstructs inputs from latent codes
3. **Hyperbolic operations** - Maintain geometric structure

.. code-block:: python

    model = TernaryVAE(
        input_dim=27,
        latent_dim=16,      # Latent space dimensionality
        hidden_dim=64,      # Hidden layer size
        num_layers=3,       # Encoder/decoder depth
        use_hyperbolic=True # Enable hyperbolic geometry
    )


Training Configuration
----------------------

.. code-block:: python

    trainer = TernaryVAETrainer(
        model=model,
        learning_rate=1e-3,
        beta=1.0,           # KL divergence weight
        use_curriculum=True # Gradual difficulty increase
    )


Running Training
----------------

.. code-block:: python

    history = trainer.train(
        train_data=x,
        epochs=100,
        batch_size=16,
        validation_split=0.2
    )


Monitoring Progress
-------------------

.. code-block:: python

    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Total loss
    axes[0].plot(history['loss'], label='Train')
    axes[0].plot(history['val_loss'], label='Val')
    axes[0].set_title('Total Loss')
    axes[0].legend()

    # Reconstruction loss
    axes[1].plot(history['recon_loss'])
    axes[1].set_title('Reconstruction Loss')

    # KL divergence
    axes[2].plot(history['kl_loss'])
    axes[2].set_title('KL Divergence')

    plt.tight_layout()
    plt.show()


Saving and Loading
------------------

.. code-block:: python

    # Save model
    torch.save(model.state_dict(), 'model.pt')

    # Load model
    model.load_state_dict(torch.load('model.pt'))


Analyzing the Latent Space
--------------------------

.. code-block:: python

    from src.visualization import plot_latent_space

    # Get latent embeddings
    with torch.no_grad():
        z, _, _ = model.encode(x)

    # Visualize
    plot_latent_space(z, indices, title="Ternary Operations in Hyperbolic Space")


Next Steps
----------

- Try different ``latent_dim`` values (8, 16, 32)
- Experiment with ``beta`` for KL weighting
- Apply to HIV codon data with :doc:`hiv_resistance`
