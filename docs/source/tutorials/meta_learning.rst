Meta-Learning Tutorial
======================

This tutorial covers few-shot learning and rapid adaptation to new
biological domains using MAML and related algorithms.


Overview
--------

Meta-learning enables models to quickly adapt to new tasks with minimal data.
This is crucial for:

- New virus variants with limited sequences
- Emerging drug classes with few resistance examples
- Novel pathogens requiring rapid response


Available Algorithms
--------------------

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - Algorithm
     - Description
   * - ``MAML``
     - Model-Agnostic Meta-Learning - learns initialization
   * - ``Reptile``
     - Simplified MAML using averaged gradients
   * - ``FewShotAdapter``
     - Prototypical networks for metric learning
   * - ``PAdicTaskSampler``
     - P-adic task sampling based on biological hierarchy


MAML (Model-Agnostic Meta-Learning)
-----------------------------------

MAML learns an initialization that adapts quickly to new tasks:

.. code-block:: python

    from src.experimental import MAML, Task
    import torch
    import torch.nn as nn

    # Define base model
    class SimpleClassifier(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Linear(32, 5)  # 5-way classification
            )

        def forward(self, x):
            return self.net(x)

    # Create MAML wrapper
    model = SimpleClassifier()
    maml = MAML(
        model=model,
        inner_lr=0.01,      # Learning rate for adaptation
        n_inner_steps=5,    # Gradient steps per task
        first_order=True    # Use first-order approximation
    )


Task Definition
---------------

Tasks contain support (training) and query (test) sets:

.. code-block:: python

    from src.experimental import Task

    # Create a 5-way 5-shot task
    task = Task(
        support_x=torch.randn(25, 64),     # 5 samples x 5 classes
        support_y=torch.tensor([0]*5 + [1]*5 + [2]*5 + [3]*5 + [4]*5),
        query_x=torch.randn(50, 64),       # 10 samples x 5 classes
        query_y=torch.tensor([0]*10 + [1]*10 + [2]*10 + [3]*10 + [4]*10),
        task_id=0,
        metadata={"source": "synthetic"}
    )

    print(f"Support size: {task.n_support}")
    print(f"Query size: {task.n_query}")


Adaptation
----------

Adapt model to new task using support set:

.. code-block:: python

    # Adapt model to task
    adapted_model = maml.adapt(task.support_x, task.support_y)

    # Evaluate on query set
    with torch.no_grad():
        logits = adapted_model(task.query_x)
        predictions = logits.argmax(dim=1)
        accuracy = (predictions == task.query_y).float().mean()

    print(f"Adapted accuracy: {accuracy:.2%}")


Meta-Training
-------------

Train MAML across multiple tasks:

.. code-block:: python

    optimizer = torch.optim.Adam(maml.model.parameters(), lr=1e-3)

    for epoch in range(100):
        # Sample batch of tasks
        tasks = sample_task_batch(n_tasks=4)

        # Meta-training step
        metrics = maml.meta_train_step(tasks, optimizer)

        if epoch % 10 == 0:
            print(f"Epoch {epoch}: loss={metrics['meta_loss']:.4f}, "
                  f"acc={metrics['meta_accuracy']:.2%}")


Reptile Algorithm
-----------------

Simpler alternative to MAML:

.. code-block:: python

    from src._future.meta.meta_learning import Reptile

    reptile = Reptile(
        model=model,
        inner_lr=0.01,
        n_inner_steps=10,
        meta_step_size=0.1
    )

    # Train on task
    param_diffs = reptile.train_on_task(task)

    # Meta-step across batch
    metrics = reptile.meta_step(tasks)


P-adic Task Sampling
--------------------

Sample tasks based on biological hierarchy using p-adic valuations:

.. code-block:: python

    from src._future.meta.meta_learning import PAdicTaskSampler

    # Prepare data
    data_x = torch.randn(1000, 64)
    data_y = torch.randint(0, 5, (1000,))
    padic_indices = torch.arange(1000)  # Hierarchical indices

    sampler = PAdicTaskSampler(
        data_x=data_x,
        data_y=data_y,
        padic_indices=padic_indices,
        n_support=5,
        n_query=10,
        prime=3,              # p=3 for ternary structure
        valuation_threshold=2
    )

    # Sample single task
    task = sampler.sample_task()

    # Sample batch
    tasks = sampler.sample_batch(n_tasks=4)


Few-Shot Adaptation
-------------------

Prototypical networks for metric-based classification:

.. code-block:: python

    from src._future.meta.meta_learning import FewShotAdapter
    import torch.nn as nn

    # Create encoder
    encoder = nn.Sequential(
        nn.Linear(64, 32),
        nn.ReLU(),
        nn.Linear(32, 16)
    )

    adapter = FewShotAdapter(
        encoder=encoder,
        prototype_dim=16,
        n_adapt_steps=3,
        adapt_lr=0.1
    )

    # Compute class prototypes
    prototypes = adapter.compute_prototypes(
        task.support_x,
        task.support_y
    )

    # Classify query samples
    logits = adapter(task.query_x, prototypes)
    predictions = logits.argmax(dim=1)

    # One-step adaptation and prediction
    predictions = adapter.adapt_and_predict(task)


Biological Applications
-----------------------

**New Variant Adaptation**

When a new variant emerges:

.. code-block:: python

    # Collect few labeled samples from new variant
    new_variant_task = Task(
        support_x=new_variant_embeddings[:20],
        support_y=new_variant_labels[:20],
        query_x=new_variant_embeddings[20:],
        query_y=new_variant_labels[20:],
        metadata={"variant": "XBB.1.5"}
    )

    # Adapt pre-trained model
    adapted = maml.adapt(
        new_variant_task.support_x,
        new_variant_task.support_y
    )

    # Predict for new samples
    predictions = adapted(new_variant_task.query_x)


**Cross-Pathogen Transfer**

Transfer knowledge between pathogens:

.. code-block:: python

    # Pre-train on HIV data
    hiv_tasks = sample_hiv_tasks(n_tasks=100)
    for task in hiv_tasks:
        maml.meta_train_step([task], optimizer)

    # Adapt to SARS-CoV-2
    sars_task = create_sars_task()
    sars_adapted = maml.adapt(
        sars_task.support_x,
        sars_task.support_y
    )


End-to-End Example
------------------

Complete few-shot resistance prediction:

.. code-block:: python

    import torch
    import torch.nn as nn
    from src.experimental import MAML, Task
    from src.encoders import HyperbolicFeatureExtractor

    # 1. Create feature extractor (from trained VAE)
    extractor = HyperbolicFeatureExtractor.load("vae_encoder.pt")

    # 2. Create classifier head
    classifier = nn.Sequential(
        nn.Linear(64, 32),
        nn.ReLU(),
        nn.Linear(32, 2)  # Resistant vs Susceptible
    )

    # 3. Create MAML
    maml = MAML(classifier, inner_lr=0.01, n_inner_steps=5)

    # 4. Create tasks from drug-specific data
    tasks = []
    for drug in drugs:
        sequences, labels = load_drug_data(drug)
        features = extractor(sequences)

        task = Task(
            support_x=features[:10],
            support_y=labels[:10],
            query_x=features[10:],
            query_y=labels[10:],
            metadata={"drug": drug}
        )
        tasks.append(task)

    # 5. Meta-train
    optimizer = torch.optim.Adam(maml.model.parameters())
    for epoch in range(100):
        batch = random.sample(tasks, k=4)
        maml.meta_train_step(batch, optimizer)

    # 6. Adapt to new drug
    new_drug_task = create_task_for_new_drug()
    adapted = maml.adapt(
        new_drug_task.support_x,
        new_drug_task.support_y
    )


Next Steps
----------

- :doc:`predictors` - Build downstream predictors
- :doc:`hiv_resistance` - HIV-specific analysis
- :doc:`epsilon_vae` - Checkpoint exploration
