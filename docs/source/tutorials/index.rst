Tutorials
=========

Step-by-step tutorials for common tasks with Ternary VAE.

.. toctree::
   :maxdepth: 1

   basic_training
   hiv_resistance
   codon_analysis


Getting Started Tutorial
------------------------

This tutorial walks through training your first Ternary VAE model.


Prerequisites
~~~~~~~~~~~~~

1. Python 3.10+
2. PyTorch 2.0+
3. Ternary VAE installed (``pip install -e .``)


Step 1: Generate Data
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from src.data import generate_all_ternary_operations

    # Generate all 27 ternary operations
    x, indices = generate_all_ternary_operations()
    print(f"Generated {len(x)} samples")


Step 2: Create Model
~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from src.models import TernaryVAE

    model = TernaryVAE(
        input_dim=27,
        latent_dim=16,
        hidden_dim=64,
        use_hyperbolic=True
    )


Step 3: Train
~~~~~~~~~~~~~

.. code-block:: python

    from src.training import TernaryVAETrainer

    trainer = TernaryVAETrainer(model)
    history = trainer.train(x, epochs=100)


Step 4: Analyze Results
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    import matplotlib.pyplot as plt

    # Plot training loss
    plt.plot(history['loss'])
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Progress')
    plt.show()

    # Visualize latent space
    from src.visualization import plot_latent_space
    plot_latent_space(model, x)


Next Steps
~~~~~~~~~~

- :doc:`hiv_resistance` - Analyze HIV drug resistance
- :doc:`codon_analysis` - Deep dive into codon embeddings
- :doc:`/guide/analysis` - Full analysis guide
