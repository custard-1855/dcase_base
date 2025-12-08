"""Tests for FeatureLoader class."""

from pathlib import Path
from typing import Any

import numpy as np
import pytest

from visualize.umap.feature_loader import FeatureLoader


class TestFeatureLoader:
    """Test suite for FeatureLoader class."""

    @pytest.fixture
    def sample_npz_data(self, tmp_path: Path) -> tuple[Path, dict[str, Any]]:
        """Create a sample .npz file for testing."""
        npz_path = tmp_path / "test_features.npz"

        # Create sample data with correct shapes
        n_samples = 100
        features_student = np.random.randn(n_samples, 384).astype(np.float32)
        features_teacher = np.random.randn(n_samples, 384).astype(np.float32)
        targets = np.random.rand(n_samples, 27).astype(np.float32)
        probs_student = np.random.rand(n_samples, 27).astype(np.float32)
        probs_teacher = np.random.rand(n_samples, 27).astype(np.float32)
        filenames = [f"audio_{i:03d}.wav" for i in range(n_samples)]

        # Save to npz
        np.savez(
            npz_path,
            features_student=features_student,
            features_teacher=features_teacher,
            targets=targets,
            probs_student=probs_student,
            probs_teacher=probs_teacher,
            filenames=np.array(filenames),
        )

        expected_data = {
            "features_student": features_student,
            "features_teacher": features_teacher,
            "targets": targets,
            "filenames": filenames,
        }

        return npz_path, expected_data

    @pytest.fixture
    def class_labels(self) -> dict[str, int]:
        """Sample class labels for testing."""
        from local.classes_dict import classes_labels_desed

        return dict(classes_labels_desed)

    def test_initialization(self, class_labels: dict[str, int]) -> None:
        """Test FeatureLoader initialization."""
        loader = FeatureLoader(class_labels)

        assert loader.class_labels == class_labels
        assert loader.class_names == list(class_labels.keys())

    def test_load_features_student_success(
        self,
        sample_npz_data: tuple[Path, dict[str, Any]],
        class_labels: dict[str, int],
    ) -> None:
        """Test successful loading of student features."""
        npz_path, expected = sample_npz_data
        loader = FeatureLoader(class_labels)

        features, primary_classes, targets, filenames = loader.load_features(
            str(npz_path),
            model_type="student",
        )

        # Check shapes
        assert features.shape == (100, 384)
        assert primary_classes.shape == (100,)
        assert targets.shape == (100, 27)
        assert len(filenames) == 100

        # Check data types
        assert features.dtype == np.float32
        assert primary_classes.dtype == np.int32
        assert targets.dtype == np.float32

        # Check values match
        np.testing.assert_array_equal(features, expected["features_student"])
        np.testing.assert_array_equal(targets, expected["targets"])
        assert filenames == expected["filenames"]

    def test_load_features_teacher_success(
        self,
        sample_npz_data: tuple[Path, dict[str, Any]],
        class_labels: dict[str, int],
    ) -> None:
        """Test successful loading of teacher features."""
        npz_path, expected = sample_npz_data
        loader = FeatureLoader(class_labels)

        features, primary_classes, targets, filenames = loader.load_features(
            str(npz_path),
            model_type="teacher",
        )

        # Check values match
        np.testing.assert_array_equal(features, expected["features_teacher"])

    def test_load_features_file_not_found(self, class_labels: dict[str, int]) -> None:
        """Test FileNotFoundError when file doesn't exist."""
        loader = FeatureLoader(class_labels)

        with pytest.raises(FileNotFoundError, match="not found"):
            loader.load_features("nonexistent_file.npz")

    def test_load_features_invalid_shape_features(
        self,
        tmp_path: Path,
        class_labels: dict[str, int],
    ) -> None:
        """Test ValueError when features have wrong shape."""
        npz_path = tmp_path / "invalid_features.npz"

        # Wrong feature dimension (should be 384)
        features_student = np.random.randn(100, 256).astype(np.float32)
        targets = np.random.rand(100, 27).astype(np.float32)
        filenames = np.array([f"audio_{i:03d}.wav" for i in range(100)])

        np.savez(
            npz_path,
            features_student=features_student,
            targets=targets,
            filenames=filenames,
        )

        loader = FeatureLoader(class_labels)

        with pytest.raises(ValueError, match=r"Expected.*384.*got.*256"):
            loader.load_features(str(npz_path))

    def test_load_features_invalid_shape_targets(
        self,
        tmp_path: Path,
        class_labels: dict[str, int],
    ) -> None:
        """Test ValueError when targets have wrong shape."""
        npz_path = tmp_path / "invalid_targets.npz"

        features_student = np.random.randn(100, 384).astype(np.float32)
        # Wrong target dimension (should be 27)
        targets = np.random.rand(100, 10).astype(np.float32)
        filenames = np.array([f"audio_{i:03d}.wav" for i in range(100)])

        np.savez(
            npz_path,
            features_student=features_student,
            targets=targets,
            filenames=filenames,
        )

        loader = FeatureLoader(class_labels)

        with pytest.raises(ValueError, match=r"Expected.*27.*got.*10"):
            loader.load_features(str(npz_path))

    def test_load_features_missing_keys(
        self,
        tmp_path: Path,
        class_labels: dict[str, int],
    ) -> None:
        """Test error when required keys are missing."""
        npz_path = tmp_path / "missing_keys.npz"

        # Only save some keys
        features_student = np.random.randn(100, 384).astype(np.float32)

        np.savez(npz_path, features_student=features_student)

        loader = FeatureLoader(class_labels)

        with pytest.raises(KeyError, match="targets"):
            loader.load_features(str(npz_path))

    def test_validate_features_success(self, class_labels: dict[str, int]) -> None:
        """Test successful feature validation."""
        loader = FeatureLoader(class_labels)
        features = np.random.randn(100, 384).astype(np.float32)

        # Should not raise any exception
        loader.validate_features(features, expected_dim=384)

    def test_validate_features_wrong_dimension(self, class_labels: dict[str, int]) -> None:
        """Test ValueError when feature dimension is wrong."""
        loader = FeatureLoader(class_labels)
        features = np.random.randn(100, 256).astype(np.float32)

        with pytest.raises(ValueError, match=r"Expected.*384.*got.*256"):
            loader.validate_features(features, expected_dim=384)

    def test_validate_features_contains_nan(
        self,
        class_labels: dict[str, int],
    ) -> None:
        """Test warning when features contain NaN."""
        loader = FeatureLoader(class_labels)
        features = np.random.randn(100, 384).astype(np.float32)
        features[0, 0] = np.nan

        with pytest.raises(ValueError, match=r"NaN.*Inf"):
            loader.validate_features(features)

    def test_validate_features_contains_inf(
        self,
        class_labels: dict[str, int],
    ) -> None:
        """Test warning when features contain Inf."""
        loader = FeatureLoader(class_labels)
        features = np.random.randn(100, 384).astype(np.float32)
        features[0, 0] = np.inf

        with pytest.raises(ValueError, match=r"NaN.*Inf"):
            loader.validate_features(features)

    def test_extract_primary_class_basic(
        self,
        class_labels: dict[str, int],
    ) -> None:
        """Test basic primary class extraction using argmax."""
        loader = FeatureLoader(class_labels)

        # Create multi-label targets where each sample has one dominant class
        targets = np.zeros((5, 27), dtype=np.float32)
        targets[0, 2] = 1.0  # Sample 0: class 2 is primary
        targets[1, 5] = 1.0  # Sample 1: class 5 is primary
        targets[2, 10] = 1.0  # Sample 2: class 10 is primary
        targets[3, 0] = 1.0  # Sample 3: class 0 is primary
        targets[4, 26] = 1.0  # Sample 4: class 26 is primary

        primary_classes, multi_label_info = loader.extract_primary_class(targets)

        # Check primary classes
        np.testing.assert_array_equal(primary_classes, [2, 5, 10, 0, 26])
        assert primary_classes.dtype == np.int32

        # Check multi-label info
        assert len(multi_label_info) == 5
        assert multi_label_info[0] == [2]
        assert multi_label_info[1] == [5]
        assert multi_label_info[2] == [10]
        assert multi_label_info[3] == [0]
        assert multi_label_info[4] == [26]

    def test_extract_primary_class_multiple_labels(
        self,
        class_labels: dict[str, int],
    ) -> None:
        """Test primary class extraction with multiple active labels."""
        loader = FeatureLoader(class_labels)

        # Create multi-label targets with multiple active classes
        targets = np.zeros((3, 27), dtype=np.float32)
        targets[0, [2, 5, 8]] = [0.8, 0.5, 0.3]  # Class 2 is primary
        targets[1, [1, 3, 7]] = [0.4, 0.9, 0.2]  # Class 3 is primary
        targets[2, [0, 10, 15]] = [0.3, 0.3, 0.7]  # Class 15 is primary

        primary_classes, multi_label_info = loader.extract_primary_class(targets)

        # Check primary classes (should be argmax)
        np.testing.assert_array_equal(primary_classes, [2, 3, 15])

        # Check multi-label info contains all active classes
        assert set(multi_label_info[0]) == {2, 5, 8}
        assert set(multi_label_info[1]) == {1, 3, 7}
        assert set(multi_label_info[2]) == {0, 10, 15}

    def test_extract_primary_class_tie_breaking(
        self,
        class_labels: dict[str, int],
    ) -> None:
        """Test tie-breaking behavior (should select smallest index)."""
        loader = FeatureLoader(class_labels)

        # Create targets with tied values
        targets = np.zeros((2, 27), dtype=np.float32)
        targets[0, [3, 7, 12]] = 1.0  # All tied, should select 3
        targets[1, [5, 10]] = 0.8  # All tied, should select 5

        primary_classes, _ = loader.extract_primary_class(targets)

        # argmax returns first occurrence in case of tie
        np.testing.assert_array_equal(primary_classes, [3, 5])

    def test_extract_primary_class_all_zeros(
        self,
        class_labels: dict[str, int],
    ) -> None:
        """Test behavior when all target values are zero."""
        loader = FeatureLoader(class_labels)

        # Create all-zero targets
        targets = np.zeros((3, 27), dtype=np.float32)

        primary_classes, multi_label_info = loader.extract_primary_class(targets)

        # argmax returns 0 when all values are equal
        np.testing.assert_array_equal(primary_classes, [0, 0, 0])

        # multi_label_info should be empty (no active classes)
        assert len(multi_label_info) == 0

    def test_extract_primary_class_logging(
        self,
        class_labels: dict[str, int],
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Test that primary class extraction logs correct information."""
        import logging

        caplog.set_level(logging.DEBUG)

        loader = FeatureLoader(class_labels)

        targets = np.zeros((10, 27), dtype=np.float32)
        # Set 7 samples with active labels
        for i in range(7):
            targets[i, i] = 1.0

        loader.extract_primary_class(targets)

        # Check that debug logs are present
        assert any("Extracted primary classes" in record.message for record in caplog.records)
        assert any("Multi-label samples: 7 / 10" in record.message for record in caplog.records)

    def test_extract_domain_labels_desed_synthetic(
        self,
        class_labels: dict[str, int],
    ) -> None:
        """Test domain extraction for DESED synthetic files."""
        loader = FeatureLoader(class_labels)

        filenames = [
            "Y123456_synth.wav",
            "Y789012_synth.wav",
            "desed_train/synthetic/Y345678_synth.wav",
        ]

        domain_labels = loader.extract_domain_labels(filenames)

        # All should be desed_synthetic (index 0)
        np.testing.assert_array_equal(domain_labels, [0, 0, 0])
        assert domain_labels.dtype == np.int32

    def test_extract_domain_labels_desed_real(
        self,
        class_labels: dict[str, int],
    ) -> None:
        """Test domain extraction for DESED real files."""
        loader = FeatureLoader(class_labels)

        filenames = [
            "Y123456_real.wav",
            "Y789012_real.wav",
            "desed_validation/Y345678_real.wav",
        ]

        domain_labels = loader.extract_domain_labels(filenames)

        # All should be desed_real (index 1)
        np.testing.assert_array_equal(domain_labels, [1, 1, 1])

    def test_extract_domain_labels_maestro_training(
        self,
        class_labels: dict[str, int],
    ) -> None:
        """Test domain extraction for MAESTRO training files."""
        loader = FeatureLoader(class_labels)

        filenames = [
            "maestro_train_123.wav",
            "maestro_training/audio_456.wav",
            "path/to/maestro_train_789.wav",
        ]

        domain_labels = loader.extract_domain_labels(filenames)

        # All should be maestro_training (index 2)
        np.testing.assert_array_equal(domain_labels, [2, 2, 2])

    def test_extract_domain_labels_maestro_validation(
        self,
        class_labels: dict[str, int],
    ) -> None:
        """Test domain extraction for MAESTRO validation files."""
        loader = FeatureLoader(class_labels)

        filenames = [
            "maestro_val_123.wav",
            "maestro_validation/audio_456.wav",
            "path/to/maestro_val_789.wav",
        ]

        domain_labels = loader.extract_domain_labels(filenames)

        # All should be maestro_validation (index 3)
        np.testing.assert_array_equal(domain_labels, [3, 3, 3])

    def test_extract_domain_labels_mixed_domains(
        self,
        class_labels: dict[str, int],
    ) -> None:
        """Test domain extraction with mixed domain files."""
        loader = FeatureLoader(class_labels)

        filenames = [
            "Y123456_synth.wav",  # desed_synthetic (0)
            "Y789012_real.wav",  # desed_real (1)
            "maestro_train_456.wav",  # maestro_training (2)
            "maestro_val_789.wav",  # maestro_validation (3)
        ]

        domain_labels = loader.extract_domain_labels(filenames)

        np.testing.assert_array_equal(domain_labels, [0, 1, 2, 3])

    def test_extract_domain_labels_invalid_filename(
        self,
        class_labels: dict[str, int],
    ) -> None:
        """Test ValueError for unidentifiable domain patterns."""
        loader = FeatureLoader(class_labels)

        filenames = [
            "Y123456_synth.wav",
            "unknown_file.wav",  # No domain pattern
        ]

        with pytest.raises(ValueError, match="Cannot identify domain"):
            loader.extract_domain_labels(filenames)

    def test_extract_domain_labels_error_message_format(
        self,
        class_labels: dict[str, int],
    ) -> None:
        """Test that error message contains valid pattern examples."""
        loader = FeatureLoader(class_labels)

        filenames = ["invalid_file.wav"]

        with pytest.raises(ValueError) as exc_info:
            loader.extract_domain_labels(filenames)

        # Error message should contain filename and valid patterns
        error_msg = str(exc_info.value)
        assert "invalid_file.wav" in error_msg
        assert "_synth" in error_msg or "synth" in error_msg.lower()
        assert "_real" in error_msg or "real" in error_msg.lower()
        assert "maestro" in error_msg.lower()

    def test_extract_domain_labels_case_insensitive(
        self,
        class_labels: dict[str, int],
    ) -> None:
        """Test that domain extraction is case-insensitive."""
        loader = FeatureLoader(class_labels)

        filenames = [
            "Y123456_SYNTH.wav",
            "Y789012_Real.wav",
            "MAESTRO_TRAIN_456.wav",
            "Maestro_Val_789.wav",
        ]

        domain_labels = loader.extract_domain_labels(filenames)

        np.testing.assert_array_equal(domain_labels, [0, 1, 2, 3])

    def test_load_multiple_datasets_basic(
        self,
        tmp_path: Path,
        class_labels: dict[str, int],
    ) -> None:
        """Test loading and combining multiple datasets."""
        loader = FeatureLoader(class_labels)

        # Create two sample .npz files
        npz_path1 = tmp_path / "dataset1.npz"
        npz_path2 = tmp_path / "dataset2.npz"

        # Dataset 1: 50 samples
        n1 = 50
        features1 = np.random.randn(n1, 384).astype(np.float32)
        targets1 = np.random.rand(n1, 27).astype(np.float32)
        filenames1 = [f"audio1_{i:03d}.wav" for i in range(n1)]

        np.savez(
            npz_path1,
            features_student=features1,
            features_teacher=np.random.randn(n1, 384).astype(np.float32),
            targets=targets1,
            probs_student=np.random.rand(n1, 27).astype(np.float32),
            probs_teacher=np.random.rand(n1, 27).astype(np.float32),
            filenames=np.array(filenames1),
        )

        # Dataset 2: 30 samples
        n2 = 30
        features2 = np.random.randn(n2, 384).astype(np.float32)
        targets2 = np.random.rand(n2, 27).astype(np.float32)
        filenames2 = [f"audio2_{i:03d}.wav" for i in range(n2)]

        np.savez(
            npz_path2,
            features_student=features2,
            features_teacher=np.random.randn(n2, 384).astype(np.float32),
            targets=targets2,
            probs_student=np.random.rand(n2, 27).astype(np.float32),
            probs_teacher=np.random.rand(n2, 27).astype(np.float32),
            filenames=np.array(filenames2),
        )

        # Load multiple datasets
        combined_features, combined_classes, combined_targets, combined_filenames = (
            loader.load_multiple_datasets([str(npz_path1), str(npz_path2)], model_type="student")
        )

        # Check combined shapes
        total_samples = n1 + n2
        assert combined_features.shape == (total_samples, 384)
        assert combined_classes.shape == (total_samples,)
        assert combined_targets.shape == (total_samples, 27)
        assert len(combined_filenames) == total_samples

        # Check data types
        assert combined_features.dtype == np.float32
        assert combined_classes.dtype == np.int32
        assert combined_targets.dtype == np.float32

        # Check that first n1 samples match dataset1
        np.testing.assert_array_equal(combined_features[:n1], features1)
        np.testing.assert_array_equal(combined_targets[:n1], targets1)
        assert combined_filenames[:n1] == filenames1

        # Check that next n2 samples match dataset2
        np.testing.assert_array_equal(combined_features[n1:], features2)
        np.testing.assert_array_equal(combined_targets[n1:], targets2)
        assert combined_filenames[n1:] == filenames2

    def test_load_multiple_datasets_single_file(
        self,
        tmp_path: Path,
        class_labels: dict[str, int],
    ) -> None:
        """Test loading single file via load_multiple_datasets (edge case)."""
        loader = FeatureLoader(class_labels)

        # Create one sample .npz file
        npz_path = tmp_path / "single_dataset.npz"

        n = 20
        features = np.random.randn(n, 384).astype(np.float32)
        targets = np.random.rand(n, 27).astype(np.float32)
        filenames = [f"audio_{i:03d}.wav" for i in range(n)]

        np.savez(
            npz_path,
            features_student=features,
            features_teacher=np.random.randn(n, 384).astype(np.float32),
            targets=targets,
            probs_student=np.random.rand(n, 27).astype(np.float32),
            probs_teacher=np.random.rand(n, 27).astype(np.float32),
            filenames=np.array(filenames),
        )

        # Load single file as a list
        combined_features, combined_classes, combined_targets, combined_filenames = (
            loader.load_multiple_datasets([str(npz_path)], model_type="student")
        )

        # Should match original data
        assert combined_features.shape == (n, 384)
        np.testing.assert_array_equal(combined_features, features)
        np.testing.assert_array_equal(combined_targets, targets)
        assert combined_filenames == filenames

    def test_load_multiple_datasets_three_files(
        self,
        tmp_path: Path,
        class_labels: dict[str, int],
    ) -> None:
        """Test loading three datasets."""
        loader = FeatureLoader(class_labels)

        npz_paths = []
        all_features = []
        all_targets = []
        all_filenames = []
        sample_counts = [10, 20, 15]

        for i, n in enumerate(sample_counts):
            npz_path = tmp_path / f"dataset{i}.npz"
            features = np.random.randn(n, 384).astype(np.float32)
            targets = np.random.rand(n, 27).astype(np.float32)
            filenames = [f"audio{i}_{j:03d}.wav" for j in range(n)]

            np.savez(
                npz_path,
                features_student=features,
                features_teacher=np.random.randn(n, 384).astype(np.float32),
                targets=targets,
                probs_student=np.random.rand(n, 27).astype(np.float32),
                probs_teacher=np.random.rand(n, 27).astype(np.float32),
                filenames=np.array(filenames),
            )

            npz_paths.append(str(npz_path))
            all_features.append(features)
            all_targets.append(targets)
            all_filenames.extend(filenames)

        # Load all three datasets
        combined_features, combined_classes, combined_targets, combined_filenames = (
            loader.load_multiple_datasets(npz_paths, model_type="student")
        )

        # Check total shape
        total_samples = sum(sample_counts)
        assert combined_features.shape == (total_samples, 384)
        assert combined_targets.shape == (total_samples, 27)
        assert len(combined_filenames) == total_samples

        # Check concatenation order
        expected_features = np.concatenate(all_features, axis=0)
        expected_targets = np.concatenate(all_targets, axis=0)
        np.testing.assert_array_equal(combined_features, expected_features)
        np.testing.assert_array_equal(combined_targets, expected_targets)
        assert combined_filenames == all_filenames

    def test_load_multiple_datasets_logging(
        self,
        tmp_path: Path,
        class_labels: dict[str, int],
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Test that load_multiple_datasets logs sample counts correctly."""
        import logging

        caplog.set_level(logging.INFO)

        loader = FeatureLoader(class_labels)

        # Create two datasets
        sample_counts = [25, 35]
        npz_paths = []

        for i, n in enumerate(sample_counts):
            npz_path = tmp_path / f"dataset{i}.npz"
            features = np.random.randn(n, 384).astype(np.float32)
            targets = np.random.rand(n, 27).astype(np.float32)
            filenames = [f"audio{i}_{j:03d}.wav" for j in range(n)]

            np.savez(
                npz_path,
                features_student=features,
                features_teacher=np.random.randn(n, 384).astype(np.float32),
                targets=targets,
                probs_student=np.random.rand(n, 27).astype(np.float32),
                probs_teacher=np.random.rand(n, 27).astype(np.float32),
                filenames=np.array(filenames),
            )
            npz_paths.append(str(npz_path))

        loader.load_multiple_datasets(npz_paths, model_type="student")

        # Check log contains individual dataset sample counts
        log_messages = [record.message for record in caplog.records]
        assert any("25 samples" in msg for msg in log_messages)
        assert any("35 samples" in msg for msg in log_messages)
        # Check log contains total sample count
        assert any("60" in msg for msg in log_messages)

    def test_load_multiple_datasets_empty_list(
        self,
        class_labels: dict[str, int],
    ) -> None:
        """Test that empty list raises appropriate error."""
        loader = FeatureLoader(class_labels)

        with pytest.raises(ValueError, match="at least one"):
            loader.load_multiple_datasets([], model_type="student")
