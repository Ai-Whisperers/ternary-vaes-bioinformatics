# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.

"""Tests for predictor modules.

Tests cover:
- ResistancePredictor
- EscapePredictor
- NeutralizationPredictor
- TropismClassifier
"""

from __future__ import annotations

import numpy as np
import pytest


# Check if sklearn is available
try:
    import sklearn
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

pytestmark = pytest.mark.skipif(
    not HAS_SKLEARN,
    reason="scikit-learn not installed"
)


class TestResistancePredictor:
    """Tests for ResistancePredictor class."""

    @pytest.fixture
    def predictor(self):
        """Create a ResistancePredictor instance."""
        from src.models.predictors import ResistancePredictor
        return ResistancePredictor(n_estimators=10, max_depth=3)

    @pytest.fixture
    def sample_data(self):
        """Generate sample training data."""
        np.random.seed(42)
        n_samples = 100
        n_features = 4

        X = np.random.randn(n_samples, n_features)
        y = np.abs(np.random.randn(n_samples)) * 10 + 1  # Positive fold-change

        return X, y

    def test_init(self, predictor):
        """Test initialization."""
        assert predictor.n_estimators == 10
        assert predictor.max_depth == 3
        assert predictor.is_fitted is False

    def test_fit(self, predictor, sample_data):
        """Test fitting."""
        X, y = sample_data

        result = predictor.fit(X, y)

        assert result is predictor
        assert predictor.is_fitted is True
        assert predictor.model is not None

    def test_predict(self, predictor, sample_data):
        """Test prediction."""
        X, y = sample_data

        predictor.fit(X, y)
        predictions = predictor.predict(X)

        assert predictions.shape == (len(X),)
        assert (predictions > 0).all()  # Fold-change should be positive

    def test_predict_unfitted(self, predictor, sample_data):
        """Test prediction without fitting raises error."""
        X, _ = sample_data

        with pytest.raises(ValueError, match="not fitted"):
            predictor.predict(X)

    def test_evaluate(self, predictor, sample_data):
        """Test evaluation."""
        X, y = sample_data

        predictor.fit(X, y)
        metrics = predictor.evaluate(X, y)

        assert "mse" in metrics
        assert "rmse" in metrics
        assert "r2" in metrics
        assert "spearman_r" in metrics

    def test_predict_from_mutations(self, predictor, sample_data):
        """Test prediction from mutation pairs."""
        X, y = sample_data

        predictor.fit(X, y)
        mutations = [("A", "V"), ("G", "D"), ("L", "P")]

        predictions = predictor.predict_from_mutations(mutations)

        assert predictions.shape == (3,)

    def test_feature_importance(self, predictor, sample_data):
        """Test feature importance."""
        X, y = sample_data

        predictor.fit(X, y)
        importance = predictor.feature_importance

        assert isinstance(importance, dict)
        assert len(importance) > 0


class TestEscapePredictor:
    """Tests for EscapePredictor class."""

    @pytest.fixture
    def predictor(self):
        """Create an EscapePredictor instance."""
        from src.models.predictors import EscapePredictor
        return EscapePredictor(n_estimators=10, max_depth=3)

    @pytest.fixture
    def sample_data(self):
        """Generate sample training data."""
        np.random.seed(42)
        n_samples = 100
        n_features = 4

        X = np.random.randn(n_samples, n_features)
        y = np.random.randint(0, 2, n_samples)  # Binary labels (0=no escape, 1=escape)

        return X, y

    def test_init(self, predictor):
        """Test initialization."""
        assert predictor.is_fitted is False

    def test_fit(self, predictor, sample_data):
        """Test fitting."""
        X, y = sample_data

        result = predictor.fit(X, y)

        assert result is predictor
        assert predictor.is_fitted is True

    def test_predict(self, predictor, sample_data):
        """Test prediction."""
        X, y = sample_data

        predictor.fit(X, y)
        predictions = predictor.predict(X)

        assert predictions.shape == (len(X),)
        # Should be binary predictions
        assert set(predictions).issubset({0, 1})


class TestNeutralizationPredictor:
    """Tests for NeutralizationPredictor class."""

    @pytest.fixture
    def predictor(self):
        """Create a NeutralizationPredictor instance."""
        from src.models.predictors import NeutralizationPredictor
        return NeutralizationPredictor(n_estimators=10, max_depth=3)

    @pytest.fixture
    def sample_data(self):
        """Generate sample training data."""
        np.random.seed(42)
        n_samples = 100
        n_features = 4

        X = np.random.randn(n_samples, n_features)
        y = np.abs(np.random.randn(n_samples)) * 10 + 0.01  # IC50 values

        return X, y

    def test_init(self, predictor):
        """Test initialization."""
        assert predictor.is_fitted is False

    def test_fit(self, predictor, sample_data):
        """Test fitting."""
        X, y = sample_data

        result = predictor.fit(X, y)

        assert result is predictor
        assert predictor.is_fitted is True

    def test_predict(self, predictor, sample_data):
        """Test prediction."""
        X, y = sample_data

        predictor.fit(X, y)
        predictions = predictor.predict(X)

        assert predictions.shape == (len(X),)
        assert (predictions > 0).all()  # IC50 should be positive


class TestTropismClassifier:
    """Tests for TropismClassifier class."""

    @pytest.fixture
    def classifier(self):
        """Create a TropismClassifier instance."""
        from src.models.predictors import TropismClassifier
        return TropismClassifier(n_estimators=10, max_depth=3)

    @pytest.fixture
    def sample_data(self):
        """Generate sample training data."""
        np.random.seed(42)
        n_samples = 100
        n_features = 4

        X = np.random.randn(n_samples, n_features)
        y = np.random.randint(0, 2, n_samples)  # Binary classification

        return X, y

    def test_init(self, classifier):
        """Test initialization."""
        assert classifier.is_fitted is False

    def test_fit(self, classifier, sample_data):
        """Test fitting."""
        X, y = sample_data

        result = classifier.fit(X, y)

        assert result is classifier
        assert classifier.is_fitted is True

    def test_predict(self, classifier, sample_data):
        """Test prediction."""
        X, y = sample_data

        classifier.fit(X, y)
        predictions = classifier.predict(X)

        assert predictions.shape == (len(X),)
        # Should be binary
        assert set(predictions).issubset({0, 1})

    def test_predict_proba(self, classifier, sample_data):
        """Test probability prediction."""
        X, y = sample_data

        classifier.fit(X, y)
        probas = classifier.predict_proba(X)

        assert probas.shape == (len(X), 2)
        # Probabilities should sum to 1
        assert np.allclose(probas.sum(axis=1), 1.0)


class TestPredictorSerialization:
    """Tests for predictor save/load functionality."""

    @pytest.fixture
    def trained_predictor(self):
        """Create and train a predictor."""
        from src.models.predictors import ResistancePredictor

        np.random.seed(42)
        predictor = ResistancePredictor(n_estimators=10)
        X = np.random.randn(50, 4)
        y = np.abs(np.random.randn(50)) * 10 + 1

        predictor.fit(X, y)
        return predictor, X

    def test_save_and_load(self, trained_predictor, tmp_path):
        """Test save and load roundtrip."""
        from src.models.predictors import ResistancePredictor

        predictor, X = trained_predictor

        # Get original predictions
        original_preds = predictor.predict(X)

        # Save
        save_path = tmp_path / "predictor.pkl"
        predictor.save(save_path)

        assert save_path.exists()

        # Load
        loaded = ResistancePredictor.load(save_path)

        # Compare predictions
        loaded_preds = loaded.predict(X)

        assert np.allclose(original_preds, loaded_preds)


class TestPredictorIntegration:
    """Integration tests for predictors with feature extractor."""

    def test_end_to_end_from_sequences(self):
        """Test end-to-end prediction from sequences."""
        from src.models.predictors import ResistancePredictor

        # Create and train predictor
        np.random.seed(42)
        predictor = ResistancePredictor(n_estimators=10)

        # Generate training data
        sequences = [
            "MKWVTFISLLLLFSSAYS",
            "MVLSPADKTNVKAAWGKV",
            "MGLSDGEWQLVLNVWGKV",
            "MHSSIVLATVLFVAIASASKTRELCMKSLEHAKVG",
        ] * 10

        # Generate synthetic targets
        targets = np.abs(np.random.randn(len(sequences))) * 10 + 1

        # Fit from sequences
        predictor.fit_from_sequences(sequences, targets)

        assert predictor.is_fitted

        # Predict from sequences
        predictions = predictor.predict_from_sequences(sequences[:2])

        assert len(predictions) == 2
        assert (predictions > 0).all()
