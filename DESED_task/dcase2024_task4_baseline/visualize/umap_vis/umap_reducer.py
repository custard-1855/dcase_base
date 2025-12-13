"""UMAP dimensionality reduction wrapper for feature visualization."""

import logging
import time
from typing import Optional

import numpy as np
import umap
from numpy.typing import NDArray

# Configure logger
LOGGER = logging.getLogger(__name__)


class UMAPReducer:
    """UMAP dimensionality reduction wrapper.

    Provides a clean interface for UMAP-based dimensionality reduction
    with parameter management, validation, and logging.

    Attributes:
        n_neighbors: Number of nearest neighbors for UMAP (5-50 recommended).
        min_dist: Minimum distance in embedding space (0.001-0.5 recommended).
        metric: Distance metric for UMAP ('euclidean', 'cosine', etc.).
        random_state: Random seed for reproducibility.
        n_components: Output dimensionality (typically 2 for visualization).
        reducer: Fitted UMAP model instance (None until fit_transform is called).

    """

    def __init__(
        self,
        n_neighbors: int = 15,
        min_dist: float = 0.1,
        metric: str = "euclidean",
        random_state: int = 42,
        n_components: int = 2,
    ) -> None:
        """Initialize UMAPReducer.

        Args:
            n_neighbors: Number of nearest neighbors (default: 15).
            min_dist: Minimum distance in embedding space (default: 0.1).
            metric: Distance metric (default: 'euclidean').
            random_state: Random seed for reproducibility (default: 42).
            n_components: Output dimensionality (default: 2).

        """
        self.n_neighbors = n_neighbors
        self.min_dist = min_dist
        self.metric = metric
        self.random_state = random_state
        self.n_components = n_components
        self.reducer: umap.UMAP | None = None

    def fit_transform(self, features: NDArray[np.float32]) -> NDArray[np.float32]:
        """Reduce high-dimensional features to 2D embeddings.

        Args:
            features: Input features of shape (N, D) where N is number of samples
                     and D is feature dimensionality.

        Returns:
            embeddings: 2D embeddings of shape (N, n_components).

        Raises:
            ValueError: If features contain NaN/Inf, are empty, or have wrong shape.
            MemoryError: If insufficient memory for UMAP computation.

        """
        # Validate input features
        self._validate_features(features)

        # Adjust n_neighbors if necessary
        n_samples = features.shape[0]
        adjusted_n_neighbors = self.n_neighbors

        if n_samples <= self.n_neighbors:
            adjusted_n_neighbors = max(2, n_samples - 1)
            LOGGER.warning(
                f"n_neighbors ({self.n_neighbors}) >= n_samples ({n_samples}). "
                f"Adjusted to n_neighbors={adjusted_n_neighbors}.",
            )

        # Warn about large datasets
        if n_samples > 10000:
            LOGGER.warning(
                f"Large dataset detected ({n_samples} samples). "
                f"UMAP computation may take significant time and memory. "
                f"Consider downsampling if performance is an issue.",
            )

        # Create UMAP reducer
        self.reducer = umap.UMAP(
            n_neighbors=adjusted_n_neighbors,
            min_dist=self.min_dist,
            metric=self.metric,
            random_state=self.random_state,
            n_components=self.n_components,
        )

        # Execute UMAP fit_transform with timing
        start_time = time.time()
        LOGGER.info(
            f"Starting UMAP fit_transform with {n_samples} samples, "
            f"n_neighbors={adjusted_n_neighbors}, min_dist={self.min_dist}, "
            f"metric='{self.metric}'",
        )

        try:
            embeddings = self.reducer.fit_transform(features)
        except MemoryError as e:
            LOGGER.error(f"MemoryError during UMAP fit_transform: {e}")
            raise MemoryError(
                f"Insufficient memory for UMAP computation with {n_samples} samples. "
                "Please reduce the number of samples or use downsampling.",
            ) from e

        elapsed_time = time.time() - start_time
        LOGGER.info(f"UMAP fit_transform completed in {elapsed_time:.2f} seconds")

        # Ensure output dtype is float32
        return embeddings.astype(np.float32)

    def transform(self, features: NDArray[np.float32]) -> NDArray[np.float32]:
        """Transform new data using fitted UMAP model.

        Args:
            features: Input features of shape (N, D).

        Returns:
            embeddings: 2D embeddings of shape (N, n_components).

        Raises:
            ValueError: If fit_transform has not been called yet, or if features
                       contain NaN/Inf.

        """
        if self.reducer is None:
            raise ValueError(
                "fit_transform() must be called first before using transform(). "
                "The UMAP model has not been fitted yet.",
            )

        # Validate input features
        self._validate_features(features)

        # Transform using fitted reducer
        LOGGER.info(f"Transforming {features.shape[0]} samples using fitted UMAP model")
        embeddings = self.reducer.transform(features)

        return embeddings.astype(np.float32)

    def get_params(self) -> dict:
        """Get current UMAP parameters.

        Returns:
            params: Dictionary containing all UMAP parameters.

        """
        return {
            "n_neighbors": self.n_neighbors,
            "min_dist": self.min_dist,
            "metric": self.metric,
            "random_state": self.random_state,
            "n_components": self.n_components,
        }

    def _validate_features(self, features: NDArray[np.float32]) -> None:
        """Validate input features.

        Args:
            features: Input features to validate.

        Raises:
            ValueError: If features are invalid (wrong shape, contain NaN/Inf, or empty).

        """
        # Check if features are 2D
        if features.ndim != 2:
            raise ValueError(
                f"Features must be 2D array (N, D), got shape {features.shape}",
            )

        # Check if features are empty
        if features.shape[0] == 0:
            raise ValueError("Features array is empty (0 samples)")

        # Check for NaN or Inf
        if np.isnan(features).any() or np.isinf(features).any():
            raise ValueError(
                "Features contain NaN or Inf values. Please clean the data before UMAP reduction.",
            )
