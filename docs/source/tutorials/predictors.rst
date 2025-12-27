Predictors Tutorial
===================

This tutorial covers building and using downstream predictors for
HIV drug resistance, immune escape, and viral tropism.


Available Predictors
--------------------

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Predictor
     - Description
   * - ``ResistancePredictor``
     - Predicts drug resistance fold-change
   * - ``EscapePredictor``
     - Predicts CTL epitope escape probability
   * - ``NeutralizationPredictor``
     - Predicts antibody neutralization (IC50)
   * - ``TropismClassifier``
     - Classifies viral tropism (CCR5/CXCR4)


Quick Start
-----------

.. code-block:: python

    from src.models.predictors import ResistancePredictor
    import numpy as np

    # Create predictor
    predictor = ResistancePredictor(
        n_estimators=100,
        max_depth=10
    )

    # Prepare training data (features + fold-change)
    X = np.random.randn(1000, 64)  # Hyperbolic embeddings
    y = np.random.exponential(5, 1000)  # Fold-change values

    # Train
    predictor.fit(X, y)

    # Predict
    predictions = predictor.predict(X[:10])
    print(f"Predicted fold-changes: {predictions}")


Training from Sequences
-----------------------

Train directly from amino acid sequences:

.. code-block:: python

    from src.models.predictors import ResistancePredictor

    predictor = ResistancePredictor()

    # Training sequences and resistance values
    sequences = [
        "PQITLWQRPLVTIKIGGQLKEALLDTGADDTVLEEMSLPGRWKPKMIGGIGGFIKVRQYD",
        "PQITLWQRPLVTVKVGGQLKEALLDTGADDTVIEEMSLPGRWKPKMIGGIGGFIKVRQYD",
        # ... more sequences
    ]
    fold_changes = [1.0, 3.5, 12.0, 0.8, 45.0]  # Resistance values

    # Fit from sequences (uses HyperbolicFeatureExtractor internally)
    predictor.fit_from_sequences(sequences, fold_changes)

    # Predict new sequences
    new_sequences = ["PQITLWQRPLVTIKIGGQLKEALLDTGADDTVLEEMSLPGR..."]
    predictions = predictor.predict_from_sequences(new_sequences)


Escape Prediction
-----------------

Predict probability of CTL epitope escape:

.. code-block:: python

    from src.models.predictors import EscapePredictor

    predictor = EscapePredictor()

    # Features derived from mutations within epitopes
    X = np.random.randn(500, 64)
    y = np.random.beta(2, 5, 500)  # Escape probabilities [0, 1]

    predictor.fit(X, y)
    escape_probs = predictor.predict(X[:10])

    print(f"Escape probabilities: {escape_probs}")


Neutralization Prediction
-------------------------

Predict antibody neutralization (IC50 values):

.. code-block:: python

    from src.models.predictors import NeutralizationPredictor

    predictor = NeutralizationPredictor()

    # Features from envelope sequence
    X = np.random.randn(800, 64)
    y = np.random.lognormal(2, 1, 800)  # IC50 values (log-normal)

    predictor.fit(X, y)
    ic50_predictions = predictor.predict(X[:10])


Tropism Classification
----------------------

Classify viral tropism (R5 vs X4):

.. code-block:: python

    from src.models.predictors import TropismClassifier

    classifier = TropismClassifier()

    # V3 loop features
    X = np.random.randn(600, 64)
    y = np.random.randint(0, 2, 600)  # 0=R5, 1=X4

    classifier.fit(X, y)

    # Get predictions and probabilities
    predictions = classifier.predict(X[:10])
    probabilities = classifier.predict_proba(X[:10])

    print(f"Predictions: {predictions}")
    print(f"X4 probability: {probabilities[:, 1]}")


Model Evaluation
----------------

Evaluate predictor performance:

.. code-block:: python

    from sklearn.model_selection import train_test_split

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Train
    predictor.fit(X_train, y_train)

    # Evaluate
    metrics = predictor.evaluate(X_test, y_test)

    print(f"MSE: {metrics['mse']:.4f}")
    print(f"RMSE: {metrics['rmse']:.4f}")
    print(f"R2: {metrics['r2']:.4f}")
    print(f"Spearman R: {metrics['spearman_r']:.4f}")


Feature Importance
------------------

Analyze which features drive predictions:

.. code-block:: python

    importance = predictor.feature_importance

    # Sort by importance
    sorted_features = sorted(
        importance.items(),
        key=lambda x: x[1],
        reverse=True
    )

    print("Top 10 important features:")
    for name, score in sorted_features[:10]:
        print(f"  {name}: {score:.4f}")


Save and Load
-------------

Persist trained predictors:

.. code-block:: python

    # Save
    predictor.save("resistance_predictor.pkl")

    # Load
    from src.models.predictors import ResistancePredictor
    loaded = ResistancePredictor.load("resistance_predictor.pkl")


Command-Line Training
---------------------

Train predictors via CLI:

.. code-block:: bash

    # Train all predictors
    python scripts/train_predictors.py \
        --predictor all \
        --data-dir data/hiv \
        --save-dir outputs/predictors

    # Train specific predictor
    python scripts/train_predictors.py \
        --predictor resistance \
        --data-dir data/hiv \
        --n-estimators 200

Or via unified launcher:

.. code-block:: bash

    python src/train.py --mode predictors \
        --drug-class pi \
        --save-dir outputs/predictors


Full HIV Pipeline
-----------------

Run complete analysis pipeline:

.. code-block:: bash

    python scripts/hiv/run_full_hiv_pipeline.py \
        --stage all \
        --drug-class pi \
        --epochs 100 \
        --output-dir outputs/hiv_analysis

This runs:

1. Data download/preparation
2. VAE training
3. Embedding extraction
4. Predictor training
5. Analysis and visualization


Integration with VAE
--------------------

Use VAE embeddings as predictor features:

.. code-block:: python

    import torch
    from src.models.vae_v5_11 import TernaryVAEV5_11
    from src.models.predictors import ResistancePredictor
    from src.encoders import HyperbolicFeatureExtractor

    # Load trained VAE
    vae = TernaryVAEV5_11.load("outputs/v5.11/best_model.pt")

    # Create feature extractor using VAE encoder
    extractor = HyperbolicFeatureExtractor(vae.encoder)

    # Extract features from sequences
    features = extractor.extract_features(sequences)

    # Train predictor on VAE features
    predictor = ResistancePredictor()
    predictor.fit(features, fold_changes)


Cross-Validation
----------------

Robust evaluation with cross-validation:

.. code-block:: python

    from sklearn.model_selection import cross_val_score

    predictor = ResistancePredictor()

    scores = cross_val_score(
        predictor.model,  # Access underlying sklearn model
        X, y,
        cv=5,
        scoring='neg_mean_squared_error'
    )

    print(f"CV MSE: {-scores.mean():.4f} +/- {scores.std():.4f}")


Next Steps
----------

- :doc:`hiv_resistance` - HIV-specific analysis
- :doc:`meta_learning` - Few-shot adaptation
- :doc:`full_training` - Training fundamentals
