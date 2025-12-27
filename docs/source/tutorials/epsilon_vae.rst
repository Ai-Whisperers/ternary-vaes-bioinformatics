Epsilon-VAE Tutorial
====================

Epsilon-VAE enables meta-learning over checkpoint landscapes, treating
trained model checkpoints as data points to explore.


What is Epsilon-VAE?
--------------------

Traditional VAEs learn representations of data. Epsilon-VAE instead learns
representations of **model checkpoints** - the states of a trained network
at different points in training.

This enables:

- **Checkpoint interpolation**: Blend models smoothly
- **Pareto frontier exploration**: Find optimal trade-offs
- **Meta-optimization**: Learn to navigate loss landscapes


Architecture
------------

.. code-block:: text

    Checkpoint --> WeightBlockEmbedder --> CheckpointEncoder --> Latent Space
                                                                      |
                                                                      v
    Reconstructed <-- CheckpointDecoder <-- MetricPredictor <-- Latent Code


Key Components
--------------

**WeightBlockEmbedder**
    Converts weight matrices into fixed-size embeddings

**CheckpointEncoder**
    Maps checkpoint embeddings to latent space

**MetricPredictor**
    Predicts training metrics (loss, accuracy) from latent codes

**CheckpointDecoder**
    Reconstructs checkpoint weights from latent representations


Basic Usage
-----------

.. code-block:: python

    from src.models.epsilon_vae import EpsilonVAE
    import torch

    # Create model
    epsilon_vae = EpsilonVAE(
        checkpoint_dim=1024,  # Flattened checkpoint size
        latent_dim=32,
        hidden_dim=256,
        n_metrics=3  # e.g., loss, recon, kl
    )

    # Prepare checkpoint data
    checkpoints = torch.randn(10, 1024)  # 10 checkpoints
    metrics = torch.randn(10, 3)  # Corresponding metrics

    # Forward pass
    output = epsilon_vae(checkpoints)
    z_mean, z_logvar = output['z_mean'], output['z_logvar']
    reconstructed = output['reconstructed']
    predicted_metrics = output['predicted_metrics']


Training Epsilon-VAE
--------------------

Collect checkpoints from a training run:

.. code-block:: python

    from src.models.vae_v5_11 import TernaryVAEV5_11

    # Train base model and save checkpoints
    checkpoints = []
    metrics_history = []

    for epoch in range(epochs):
        loss, recon, kl = train_epoch(model, data)

        if epoch % checkpoint_interval == 0:
            checkpoints.append(flatten_model(model))
            metrics_history.append([loss, recon, kl])

    # Stack into tensors
    checkpoint_tensor = torch.stack(checkpoints)
    metrics_tensor = torch.tensor(metrics_history)

Train Epsilon-VAE on collected checkpoints:

.. code-block:: python

    epsilon_vae = EpsilonVAE(
        checkpoint_dim=checkpoint_tensor.shape[1],
        latent_dim=32
    )

    optimizer = torch.optim.Adam(epsilon_vae.parameters(), lr=1e-3)

    for epoch in range(epsilon_epochs):
        output = epsilon_vae(checkpoint_tensor)
        loss = epsilon_vae.compute_loss(
            output,
            checkpoint_tensor,
            metrics_tensor
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


Checkpoint Interpolation
------------------------

Interpolate between two trained checkpoints:

.. code-block:: python

    # Encode two checkpoints
    z1 = epsilon_vae.encode(checkpoint1)
    z2 = epsilon_vae.encode(checkpoint2)

    # Interpolate in latent space
    alphas = torch.linspace(0, 1, 10)
    interpolated = []

    for alpha in alphas:
        z_interp = (1 - alpha) * z1 + alpha * z2
        checkpoint_interp = epsilon_vae.decode(z_interp)
        interpolated.append(checkpoint_interp)


Pareto Frontier Exploration
---------------------------

Find optimal trade-offs between metrics:

.. code-block:: python

    # Find Pareto-optimal checkpoints
    pareto_indices = epsilon_vae.find_pareto_frontier(
        metrics_tensor,
        objectives=['minimize', 'minimize', 'minimize']
    )

    pareto_checkpoints = checkpoint_tensor[pareto_indices]

    # Visualize Pareto frontier
    import matplotlib.pyplot as plt

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(
        metrics_tensor[:, 0],
        metrics_tensor[:, 1],
        metrics_tensor[:, 2],
        c='blue', alpha=0.3, label='All'
    )

    ax.scatter(
        metrics_tensor[pareto_indices, 0],
        metrics_tensor[pareto_indices, 1],
        metrics_tensor[pareto_indices, 2],
        c='red', s=100, label='Pareto'
    )

    ax.set_xlabel('Loss')
    ax.set_ylabel('Reconstruction')
    ax.set_zlabel('KL')
    ax.legend()
    plt.show()


Metric Prediction
-----------------

Predict metrics for novel checkpoint positions:

.. code-block:: python

    # Sample random points in latent space
    z_samples = torch.randn(100, latent_dim)

    # Predict metrics without decoding
    predicted = epsilon_vae.predict_metrics(z_samples)

    # Find point with best predicted loss
    best_idx = predicted[:, 0].argmin()
    best_z = z_samples[best_idx]

    # Decode to get actual checkpoint
    best_checkpoint = epsilon_vae.decode(best_z)


Command-Line Training
---------------------

.. code-block:: bash

    python src/train.py --mode epsilon \
        --checkpoint-dir outputs/v5.11/checkpoints \
        --epochs 100 \
        --latent-dim 32 \
        --save-dir outputs/epsilon


Applications
------------

1. **Model Selection**: Navigate checkpoint landscape to find optimal models
2. **Ensemble Creation**: Interpolate checkpoints for diverse ensembles
3. **Transfer Learning**: Find good initialization points for new tasks
4. **Architecture Search**: Explore learned representations of architectures


Next Steps
----------

- :doc:`meta_learning` - MAML for few-shot adaptation
- :doc:`predictors` - Build downstream predictors
- :doc:`full_training` - Training fundamentals
