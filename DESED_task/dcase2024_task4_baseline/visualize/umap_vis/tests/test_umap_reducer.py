"""Tests for UMAPReducer class."""

import logging
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from visualize.umap_vis.umap_reducer import UMAPReducer


class TestUMAPReducer:
    """Test suite for UMAPReducer class."""

    def test_initialization_default_params(self) -> None:
        """Test UMAPReducer initialization with default parameters."""
        reducer = UMAPReducer()

        assert reducer.n_neighbors == 15
        assert reducer.min_dist == 0.1
        assert reducer.metric == "euclidean"
        assert reducer.random_state == 42
        assert reducer.n_components == 2
        assert reducer.reducer is None

    def test_initialization_custom_params(self) -> None:
        """Test UMAPReducer initialization with custom parameters."""
        reducer = UMAPReducer(
            n_neighbors=30,
            min_dist=0.05,
            metric="cosine",
            random_state=123,
            n_components=3,
        )

        assert reducer.n_neighbors == 30
        assert reducer.min_dist == 0.05
        assert reducer.metric == "cosine"
        assert reducer.random_state == 123
        assert reducer.n_components == 3

    def test_fit_transform_basic(self) -> None:
        """Test basic fit_transform functionality."""
        reducer = UMAPReducer(random_state=42)

        # Create sample features (100 samples, 384 dimensions)
        features = np.random.randn(100, 384).astype(np.float32)

        embeddings = reducer.fit_transform(features)

        # Check output shape
        assert embeddings.shape == (100, 2)
        assert embeddings.dtype == np.float32

        # Check that reducer is now fitted
        assert reducer.reducer is not None

    def test_fit_transform_large_features(self) -> None:
        """Test fit_transform with larger dataset."""
        reducer = UMAPReducer(random_state=42)

        # Create larger sample features (500 samples, 384 dimensions)
        features = np.random.randn(500, 384).astype(np.float32)

        embeddings = reducer.fit_transform(features)

        # Check output shape
        assert embeddings.shape == (500, 2)
        assert embeddings.dtype == np.float32

    def test_fit_transform_logging(self, caplog: pytest.LogCaptureFixture) -> None:
        """Test that fit_transform logs execution time."""
        caplog.set_level(logging.INFO)

        reducer = UMAPReducer(random_state=42)
        features = np.random.randn(100, 384).astype(np.float32)

        reducer.fit_transform(features)

        # Check that INFO log contains execution time
        log_messages = [record.message for record in caplog.records if record.levelno == logging.INFO]
        assert any("UMAP fit_transform completed" in msg for msg in log_messages)
        assert any("seconds" in msg for msg in log_messages)

    def test_fit_transform_small_n_neighbors_warning(
        self,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Test warning when n_samples < n_neighbors."""
        caplog.set_level(logging.WARNING)

        # n_neighbors=15 but only 10 samples
        reducer = UMAPReducer(n_neighbors=15, random_state=42)
        features = np.random.randn(10, 384).astype(np.float32)

        embeddings = reducer.fit_transform(features)

        # Should still work but with warning
        assert embeddings.shape == (10, 2)

        # Check for warning log
        warning_messages = [
            record.message for record in caplog.records if record.levelno == logging.WARNING
        ]
        assert any("n_neighbors" in msg and "adjusted" in msg.lower() for msg in warning_messages)

    def test_fit_transform_large_dataset_warning(
        self,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Test warning when n_samples > 10000."""
        caplog.set_level(logging.WARNING)

        reducer = UMAPReducer(random_state=42)

        # Create large dataset (> 10,000 samples)
        features = np.random.randn(10001, 384).astype(np.float32)

        reducer.fit_transform(features)

        # Check for warning about large dataset
        warning_messages = [
            record.message for record in caplog.records if record.levelno == logging.WARNING
        ]
        assert any("10,000" in msg or "large" in msg.lower() for msg in warning_messages)

    @patch("umap.UMAP")
    def test_fit_transform_memory_error(self, mock_umap_class: MagicMock) -> None:
        """Test MemoryError handling during fit_transform."""
        # Configure mock to raise MemoryError
        mock_umap_instance = MagicMock()
        mock_umap_instance.fit_transform.side_effect = MemoryError("Insufficient memory")
        mock_umap_class.return_value = mock_umap_instance

        reducer = UMAPReducer(random_state=42)
        features = np.random.randn(100, 384).astype(np.float32)

        with pytest.raises(MemoryError, match="reduce.*sample.*downsamp"):
            reducer.fit_transform(features)

    def test_transform_success(self) -> None:
        """Test transform with pre-fitted reducer."""
        reducer = UMAPReducer(random_state=42)

        # Fit with training data
        train_features = np.random.randn(100, 384).astype(np.float32)
        reducer.fit_transform(train_features)

        # Transform new data
        test_features = np.random.randn(50, 384).astype(np.float32)
        embeddings = reducer.transform(test_features)

        # Check output shape
        assert embeddings.shape == (50, 2)
        assert embeddings.dtype == np.float32

    def test_transform_not_fitted_error(self) -> None:
        """Test ValueError when transform is called before fit_transform."""
        reducer = UMAPReducer(random_state=42)

        # Try to transform without fitting first
        features = np.random.randn(50, 384).astype(np.float32)

        with pytest.raises(ValueError, match="fit_transform.*must be called first"):
            reducer.transform(features)

    def test_get_params(self) -> None:
        """Test get_params returns correct configuration."""
        reducer = UMAPReducer(
            n_neighbors=30,
            min_dist=0.05,
            metric="cosine",
            random_state=123,
            n_components=3,
        )

        params = reducer.get_params()

        assert params["n_neighbors"] == 30
        assert params["min_dist"] == 0.05
        assert params["metric"] == "cosine"
        assert params["random_state"] == 123
        assert params["n_components"] == 3

    def test_fit_transform_deterministic(self) -> None:
        """Test that same random_state produces deterministic results."""
        features = np.random.randn(100, 384).astype(np.float32)

        # Create two reducers with same random_state
        reducer1 = UMAPReducer(random_state=42)
        reducer2 = UMAPReducer(random_state=42)

        embeddings1 = reducer1.fit_transform(features)
        embeddings2 = reducer2.fit_transform(features)

        # Results should be identical
        np.testing.assert_array_almost_equal(embeddings1, embeddings2, decimal=5)

    def test_fit_transform_different_random_state(self) -> None:
        """Test that different random_states produce different results."""
        features = np.random.randn(100, 384).astype(np.float32)

        # Create two reducers with different random_state
        reducer1 = UMAPReducer(random_state=42)
        reducer2 = UMAPReducer(random_state=123)

        embeddings1 = reducer1.fit_transform(features)
        embeddings2 = reducer2.fit_transform(features)

        # Results should be different
        assert not np.allclose(embeddings1, embeddings2, atol=0.1)

    def test_fit_transform_with_nan(self) -> None:
        """Test that features with NaN raise ValueError."""
        reducer = UMAPReducer(random_state=42)

        features = np.random.randn(100, 384).astype(np.float32)
        features[0, 0] = np.nan

        with pytest.raises(ValueError, match="NaN.*Inf"):
            reducer.fit_transform(features)

    def test_fit_transform_with_inf(self) -> None:
        """Test that features with Inf raise ValueError."""
        reducer = UMAPReducer(random_state=42)

        features = np.random.randn(100, 384).astype(np.float32)
        features[0, 0] = np.inf

        with pytest.raises(ValueError, match="NaN.*Inf"):
            reducer.fit_transform(features)

    def test_fit_transform_empty_features(self) -> None:
        """Test that empty features raise ValueError."""
        reducer = UMAPReducer(random_state=42)

        features = np.empty((0, 384), dtype=np.float32)

        with pytest.raises(ValueError, match="empty|at least"):
            reducer.fit_transform(features)

    def test_fit_transform_wrong_dimension(self) -> None:
        """Test that features with wrong dimension raise ValueError."""
        reducer = UMAPReducer(random_state=42)

        # 1D features (should be 2D)
        features = np.random.randn(100).astype(np.float32)

        with pytest.raises((ValueError, IndexError)):
            reducer.fit_transform(features)

    def test_fit_transform_n_components_3d(self) -> None:
        """Test fit_transform with n_components=3."""
        reducer = UMAPReducer(n_components=3, random_state=42)

        features = np.random.randn(100, 384).astype(np.float32)
        embeddings = reducer.fit_transform(features)

        # Check 3D output
        assert embeddings.shape == (100, 3)
        assert embeddings.dtype == np.float32

    def test_fit_transform_custom_metric_cosine(self) -> None:
        """Test fit_transform with cosine metric."""
        reducer = UMAPReducer(metric="cosine", random_state=42)

        features = np.random.randn(100, 384).astype(np.float32)
        embeddings = reducer.fit_transform(features)

        # Should work with cosine metric
        assert embeddings.shape == (100, 2)
        assert embeddings.dtype == np.float32

    def test_fit_transform_custom_n_neighbors(self) -> None:
        """Test fit_transform with custom n_neighbors."""
        reducer = UMAPReducer(n_neighbors=5, random_state=42)

        features = np.random.randn(100, 384).astype(np.float32)
        embeddings = reducer.fit_transform(features)

        # Should work with n_neighbors=5
        assert embeddings.shape == (100, 2)
        assert embeddings.dtype == np.float32

    def test_fit_transform_custom_min_dist(self) -> None:
        """Test fit_transform with custom min_dist."""
        reducer = UMAPReducer(min_dist=0.5, random_state=42)

        features = np.random.randn(100, 384).astype(np.float32)
        embeddings = reducer.fit_transform(features)

        # Should work with min_dist=0.5
        assert embeddings.shape == (100, 2)
        assert embeddings.dtype == np.float32
