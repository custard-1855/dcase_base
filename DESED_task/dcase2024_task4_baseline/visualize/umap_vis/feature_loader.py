"""Feature loading and preprocessing for UMAP visualization."""

import logging
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

LOGGER = logging.getLogger(__name__)


class FeatureLoader:
    """Load and preprocess features from .npz files for UMAP visualization."""

    def __init__(self, class_labels: dict[str, int]) -> None:
        """Initialize FeatureLoader with class label mapping.

        Args:
            class_labels: Dictionary mapping class names to indices

        """
        self.class_labels = class_labels
        self.class_names = list(class_labels.keys())

    def load_features(
        self,
        npz_path: str,
        model_type: str = "student",
    ) -> tuple[NDArray[np.float32], NDArray[np.int32], NDArray[np.float32], list[str]]:
        """Load features from .npz file.

        Args:
            npz_path: Path to .npz file
            model_type: "student" or "teacher"

        Returns:
            features: (N, 384) feature matrix
            primary_classes: (N,) primary class indices
            targets: (N, 27) multi-label ground truth
            filenames: (N,) list of filenames

        Raises:
            FileNotFoundError: File not found
            ValueError: Invalid shape
            KeyError: Missing required keys

        """
        # Check file exists
        path = Path(npz_path)
        if not path.exists():
            msg = f"File not found: {npz_path}"
            raise FileNotFoundError(msg)

        # Load npz file
        data = np.load(npz_path, allow_pickle=True)

        # Extract features based on model type
        feature_key = f"features_{model_type}"
        if feature_key not in data:
            msg = f"Missing key: {feature_key}"
            raise KeyError(msg)

        features = data[feature_key].astype(np.float32)

        # Extract targets
        if "targets" not in data:
            msg = "Missing key: targets"
            raise KeyError(msg)
        targets = data["targets"].astype(np.float32)

        # Extract filenames
        if "filenames" not in data:
            msg = "Missing key: filenames"
            raise KeyError(msg)
        filenames_array = data["filenames"]
        # Handle both string arrays and object arrays
        if filenames_array.dtype == object:
            filenames = filenames_array.tolist()
        else:
            filenames = [str(f) for f in filenames_array]

        # Validate shapes
        self.validate_features(features, expected_dim=384)

        n_samples = features.shape[0]
        if targets.shape != (n_samples, 27):
            msg = f"Expected targets shape ({n_samples}, 27), got {targets.shape}"
            raise ValueError(msg)

        # Extract primary classes
        primary_classes, _ = self.extract_primary_class(targets)

        LOGGER.info("Loaded %d samples from %s", n_samples, npz_path)
        LOGGER.info("Feature shape: %s", features.shape)

        return features, primary_classes, targets, filenames

    def extract_primary_class(
        self,
        targets: NDArray[np.float32],
    ) -> tuple[NDArray[np.int32], dict[int, list[int]]]:
        """Extract primary class from multi-label targets using argmax.

        Args:
            targets: (N, 27) multi-label array

        Returns:
            primary_classes: (N,) primary class indices
            multi_label_info: Sample index -> all class indices mapping

        """
        # Get primary class using argmax
        primary_classes = np.argmax(targets, axis=1).astype(np.int32)

        # Build multi-label info dictionary
        multi_label_info: dict[int, list[int]] = {}
        for i, target_row in enumerate(targets):
            # Get all classes with non-zero values
            active_classes = np.where(target_row > 0)[0].tolist()
            if active_classes:
                multi_label_info[i] = active_classes

        LOGGER.debug("Extracted primary classes for %d samples", len(targets))
        LOGGER.debug(
            "Multi-label samples: %d / %d",
            len(multi_label_info),
            len(targets),
        )

        return primary_classes, multi_label_info

    def extract_domain_labels(
        self,
        filenames: list[str],
    ) -> NDArray[np.int32]:
        """Extract domain labels from filenames.

        Identifies domain based on filename patterns:
        - desed_synthetic (0): contains "_synth" or "synthetic"
        - desed_real (1): contains "_real"
        - maestro_training (2): contains "maestro_train"
        - maestro_validation (3): contains "maestro_val"

        Args:
            filenames: List of filenames

        Returns:
            domain_labels: (N,) domain index array
                0: desed_synthetic
                1: desed_real
                2: maestro_training
                3: maestro_validation

        Raises:
            ValueError: Cannot identify domain from filename

        """
        domain_labels = np.zeros(len(filenames), dtype=np.int32)

        for i, filename in enumerate(filenames):
            # Convert to lowercase for case-insensitive matching
            filename_lower = filename.lower()

            # Check domain patterns
            if "_synth" in filename_lower or "synthetic" in filename_lower:
                domain_labels[i] = 0  # desed_synthetic
            elif "_real" in filename_lower:
                domain_labels[i] = 1  # desed_real
            elif "maestro_train" in filename_lower:
                domain_labels[i] = 2  # maestro_training
            elif "maestro_val" in filename_lower:
                domain_labels[i] = 3  # maestro_validation
            else:
                msg = (
                    f"Cannot identify domain from filename: {filename}. "
                    f"Valid patterns: '_synth'/'synthetic' (DESED synthetic), "
                    f"'_real' (DESED real), 'maestro_train' (MAESTRO training), "
                    f"'maestro_val' (MAESTRO validation)"
                )
                raise ValueError(msg)

        LOGGER.debug("Extracted domain labels for %d files", len(filenames))

        return domain_labels

    def load_multiple_datasets(
        self,
        npz_paths: list[str],
        model_type: str = "student",
    ) -> tuple[NDArray[np.float32], NDArray[np.int32], NDArray[np.float32], list[str]]:
        """Load and combine multiple datasets from .npz files.

        Args:
            npz_paths: List of .npz file paths
            model_type: "student" or "teacher"

        Returns:
            combined_features: (N_total, 384) concatenated features
            combined_classes: (N_total,) concatenated primary class indices
            combined_targets: (N_total, 27) concatenated targets
            combined_filenames: (N_total,) concatenated filenames

        Raises:
            ValueError: Empty list provided

        """
        if not npz_paths:
            msg = "Must provide at least one .npz file path"
            raise ValueError(msg)

        all_features = []
        all_classes = []
        all_targets = []
        all_filenames = []

        for npz_path in npz_paths:
            features, primary_classes, targets, filenames = self.load_features(
                npz_path,
                model_type=model_type,
            )
            all_features.append(features)
            all_classes.append(primary_classes)
            all_targets.append(targets)
            all_filenames.extend(filenames)

        # Concatenate all datasets
        combined_features = np.concatenate(all_features, axis=0)
        combined_classes = np.concatenate(all_classes, axis=0)
        combined_targets = np.concatenate(all_targets, axis=0)

        LOGGER.info(
            "Combined %d datasets into %d total samples",
            len(npz_paths),
            len(combined_features),
        )

        return combined_features, combined_classes, combined_targets, all_filenames

    def validate_features(
        self,
        features: NDArray[np.float32],
        expected_dim: int = 384,
    ) -> None:
        """Validate feature shape and statistics.

        Args:
            features: (N, D) feature array
            expected_dim: Expected feature dimension

        Raises:
            ValueError: Dimension mismatch or invalid values

        """
        # Check dimension
        if features.shape[1] != expected_dim:
            msg = f"Expected feature dimension {expected_dim}, got {features.shape[1]}"
            raise ValueError(msg)

        # Check for NaN or Inf
        if np.isnan(features).any() or np.isinf(features).any():
            msg = "Features contain NaN or Inf values"
            raise ValueError(msg)

        LOGGER.debug(
            "Feature validation passed: shape=%s, mean=%.4f, std=%.4f",
            features.shape,
            features.mean(),
            features.std(),
        )
