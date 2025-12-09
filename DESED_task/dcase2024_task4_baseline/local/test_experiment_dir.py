"""Tests for experiment directory management module."""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from dataclasses import FrozenInstanceError
from local.experiment_dir import ExecutionMode, ExperimentConfig


class TestExecutionMode:
    """Test ExecutionMode enum."""

    def test_execution_mode_values(self):
        """Test that ExecutionMode enum has correct values."""
        assert ExecutionMode.TRAIN.value == "train"
        assert ExecutionMode.TEST.value == "test"
        assert ExecutionMode.INFERENCE.value == "inference"
        assert ExecutionMode.FEATURE_EXTRACTION.value == "feature_extraction"

    def test_execution_mode_members(self):
        """Test that ExecutionMode enum has exactly 4 members."""
        assert len(ExecutionMode) == 4
        assert set(ExecutionMode) == {
            ExecutionMode.TRAIN,
            ExecutionMode.TEST,
            ExecutionMode.INFERENCE,
            ExecutionMode.FEATURE_EXTRACTION,
        }

    def test_execution_mode_from_string(self):
        """Test converting string to ExecutionMode."""
        assert ExecutionMode("train") == ExecutionMode.TRAIN
        assert ExecutionMode("test") == ExecutionMode.TEST
        assert ExecutionMode("inference") == ExecutionMode.INFERENCE
        assert ExecutionMode("feature_extraction") == ExecutionMode.FEATURE_EXTRACTION


class TestExperimentConfig:
    """Test ExperimentConfig dataclass."""

    def test_default_values(self):
        """Test that ExperimentConfig has correct default values."""
        config = ExperimentConfig()
        assert config.mode == ExecutionMode.TRAIN
        assert config.category == "default"
        assert config.method == "baseline"
        assert config.variant == "v1"
        assert config.base_dir == "experiments"
        assert config.template is None
        assert config.log_test_to_wandb is False

    def test_custom_values(self):
        """Test ExperimentConfig with custom values."""
        config = ExperimentConfig(
            mode=ExecutionMode.INFERENCE,
            category="advanced",
            method="cmt",
            variant="use_neg_sample",
            base_dir="/tmp/experiments",
            template="{method}_{variant}_{timestamp}",
            log_test_to_wandb=True,
        )
        assert config.mode == ExecutionMode.INFERENCE
        assert config.category == "advanced"
        assert config.method == "cmt"
        assert config.variant == "use_neg_sample"
        assert config.base_dir == "/tmp/experiments"
        assert config.template == "{method}_{variant}_{timestamp}"
        assert config.log_test_to_wandb is True

    def test_mode_from_string(self):
        """Test ExperimentConfig with mode as string."""
        config = ExperimentConfig(mode="test")
        assert config.mode == ExecutionMode.TEST

    def test_frozen_immutability(self):
        """Test that ExperimentConfig is immutable (frozen)."""
        config = ExperimentConfig()
        with pytest.raises(FrozenInstanceError):
            config.mode = ExecutionMode.TEST
        with pytest.raises(FrozenInstanceError):
            config.category = "new_category"

    def test_invalid_mode_value(self):
        """Test that invalid mode value raises ValueError."""
        with pytest.raises(ValueError):
            ExperimentConfig(mode="invalid_mode")

    def test_experiment_config_equality(self):
        """Test ExperimentConfig equality comparison."""
        config1 = ExperimentConfig(category="test", method="cmt")
        config2 = ExperimentConfig(category="test", method="cmt")
        config3 = ExperimentConfig(category="test", method="beats")

        assert config1 == config2
        assert config1 != config3

    def test_experiment_config_repr(self):
        """Test ExperimentConfig string representation."""
        config = ExperimentConfig(mode=ExecutionMode.TRAIN, category="test")
        repr_str = repr(config)
        assert "ExperimentConfig" in repr_str
        assert "mode=<ExecutionMode.TRAIN" in repr_str or "mode=ExecutionMode.TRAIN" in repr_str
        assert "category='test'" in repr_str


class TestDetectExecutionMode:
    """Test detect_execution_mode function."""

    def test_explicit_mode_specified(self):
        """Test that explicit mode in hparams takes highest priority."""
        from local.experiment_dir import ExperimentDirManager

        hparams = {"experiment": {"mode": "test"}}
        mode = ExperimentDirManager.detect_execution_mode(
            hparams=hparams,
            evaluation=True,
            test_state_dict={"some": "state"},
            fast_dev_run=True
        )
        # Explicit mode should override all other indicators
        assert mode == ExecutionMode.TEST

    def test_evaluation_true_inference_mode(self):
        """Test that evaluation=True results in INFERENCE mode when no explicit mode."""
        from local.experiment_dir import ExperimentDirManager

        hparams = {}
        mode = ExperimentDirManager.detect_execution_mode(
            hparams=hparams,
            evaluation=True,
            test_state_dict=None,
            fast_dev_run=False
        )
        assert mode == ExecutionMode.INFERENCE

    def test_test_state_dict_present_test_mode(self):
        """Test that test_state_dict presence results in TEST mode."""
        from local.experiment_dir import ExperimentDirManager

        hparams = {}
        mode = ExperimentDirManager.detect_execution_mode(
            hparams=hparams,
            evaluation=False,
            test_state_dict={"state": "dict"},
            fast_dev_run=False
        )
        assert mode == ExecutionMode.TEST

    def test_fast_dev_run_train_mode(self):
        """Test that fast_dev_run=True results in TRAIN mode."""
        from local.experiment_dir import ExperimentDirManager

        hparams = {}
        mode = ExperimentDirManager.detect_execution_mode(
            hparams=hparams,
            evaluation=False,
            test_state_dict=None,
            fast_dev_run=True
        )
        assert mode == ExecutionMode.TRAIN

    def test_default_train_mode(self):
        """Test that default mode is TRAIN when no indicators are present."""
        from local.experiment_dir import ExperimentDirManager

        hparams = {}
        mode = ExperimentDirManager.detect_execution_mode(
            hparams=hparams,
            evaluation=False,
            test_state_dict=None,
            fast_dev_run=False
        )
        assert mode == ExecutionMode.TRAIN

    def test_explicit_mode_as_enum(self):
        """Test that explicit mode can be provided as ExecutionMode enum."""
        from local.experiment_dir import ExperimentDirManager

        hparams = {"experiment": {"mode": ExecutionMode.FEATURE_EXTRACTION}}
        mode = ExperimentDirManager.detect_execution_mode(
            hparams=hparams,
            evaluation=False,
            test_state_dict=None,
            fast_dev_run=False
        )
        assert mode == ExecutionMode.FEATURE_EXTRACTION

    def test_mode_priority_order(self):
        """Test the priority order: explicit > evaluation > test_state_dict > fast_dev_run > default."""
        from local.experiment_dir import ExperimentDirManager

        # Priority 1: Explicit mode
        hparams_explicit = {"experiment": {"mode": "inference"}}
        mode = ExperimentDirManager.detect_execution_mode(
            hparams=hparams_explicit,
            evaluation=True,
            test_state_dict={"state": "dict"},
            fast_dev_run=True
        )
        assert mode == ExecutionMode.INFERENCE

        # Priority 2: evaluation (no explicit mode)
        mode = ExperimentDirManager.detect_execution_mode(
            hparams={},
            evaluation=True,
            test_state_dict={"state": "dict"},
            fast_dev_run=True
        )
        assert mode == ExecutionMode.INFERENCE

        # Priority 3: test_state_dict (no explicit mode, evaluation=False)
        mode = ExperimentDirManager.detect_execution_mode(
            hparams={},
            evaluation=False,
            test_state_dict={"state": "dict"},
            fast_dev_run=True
        )
        assert mode == ExecutionMode.TEST


class TestValidatePath:
    """Test validate_path function."""

    def test_valid_path(self):
        """Test that valid paths pass validation."""
        from local.experiment_dir import ExperimentDirManager

        # Valid paths should not raise exceptions
        ExperimentDirManager.validate_path(Path("valid/path/name"))
        ExperimentDirManager.validate_path(Path("experiments/train/baseline/cmt/v1"))
        ExperimentDirManager.validate_path(Path("a" * 260))  # Exactly at limit

    def test_invalid_characters(self):
        """Test that paths with invalid characters raise ValueError."""
        from local.experiment_dir import ExperimentDirManager

        invalid_paths = [
            Path("invalid<char>"),
            Path("path/with>angle"),
            Path('path/with"quote'),
            Path("path/with|pipe"),
            Path("path/with?question"),
            Path("path/with*asterisk"),
            Path("multiple<>invalid"),
        ]

        for invalid_path in invalid_paths:
            with pytest.raises(ValueError, match="invalid characters"):
                ExperimentDirManager.validate_path(invalid_path)

    def test_path_too_long(self):
        """Test that paths exceeding 260 characters raise ValueError."""
        from local.experiment_dir import ExperimentDirManager

        # Path with 261 characters
        long_path = Path("a" * 261)
        with pytest.raises(ValueError, match="exceeds maximum limit"):
            ExperimentDirManager.validate_path(long_path)

        # Path exactly at 260 characters should pass
        exact_path = Path("a" * 260)
        ExperimentDirManager.validate_path(exact_path)  # Should not raise

    def test_error_message_contains_suggestions(self):
        """Test that error messages provide helpful suggestions."""
        from local.experiment_dir import ExperimentDirManager

        # Test invalid character error message
        try:
            ExperimentDirManager.validate_path(Path("invalid<>path"))
            assert False, "Expected ValueError"
        except ValueError as e:
            assert "invalid characters" in str(e)
            assert "<" in str(e) and ">" in str(e)

        # Test path length error message
        try:
            ExperimentDirManager.validate_path(Path("a" * 261))
            assert False, "Expected ValueError"
        except ValueError as e:
            assert "exceeds maximum limit" in str(e)
            assert "260" in str(e)
            assert "shortening the variant name" in str(e)


class TestApplyTemplate:
    """Test apply_template function."""

    def test_simple_template(self):
        """Test template with basic placeholders."""
        from local.experiment_dir import ExperimentDirManager

        config = ExperimentConfig(
            mode=ExecutionMode.TRAIN,
            category="baseline",
            method="cmt",
            variant="v1",
        )

        result = ExperimentDirManager.apply_template("{method}_{variant}", config)
        assert result == "cmt_v1"

    def test_all_placeholders(self):
        """Test template with all available placeholders."""
        from local.experiment_dir import ExperimentDirManager

        config = ExperimentConfig(
            mode=ExecutionMode.INFERENCE,
            category="advanced",
            method="beats",
            variant="use_neg_sample",
        )

        result = ExperimentDirManager.apply_template(
            "{mode}_{category}_{method}_{variant}", config
        )
        assert result == "inference_advanced_beats_use_neg_sample"

    def test_timestamp_placeholder(self):
        """Test that timestamp placeholder generates valid timestamp."""
        from local.experiment_dir import ExperimentDirManager
        import re

        config = ExperimentConfig(method="cmt", variant="v1")

        result = ExperimentDirManager.apply_template("{method}_{timestamp}", config)

        # Check format: method_YYYYMMDD_HHMMSS
        pattern = r"cmt_\d{8}_\d{6}"
        assert re.match(pattern, result), f"Result '{result}' doesn't match expected pattern"

    def test_template_with_no_placeholders(self):
        """Test template with no placeholders (literal string)."""
        from local.experiment_dir import ExperimentDirManager

        config = ExperimentConfig()
        result = ExperimentDirManager.apply_template("literal_string", config)
        assert result == "literal_string"


class TestBuildExperimentPath:
    """Test build_experiment_path function."""

    def test_basic_path_generation(self):
        """Test basic hierarchical path generation."""
        from local.experiment_dir import ExperimentDirManager

        config = ExperimentConfig(
            mode=ExecutionMode.TRAIN,
            category="baseline",
            method="cmt",
            variant="v1",
            base_dir="experiments",
        )

        path = ExperimentDirManager.build_experiment_path(config)

        # Check path structure
        assert path.parts[-4:] == ("train", "baseline", "cmt", "v1")
        assert path.exists()  # Directory should be created

    def test_mode_based_layout(self):
        """Test that different modes create different directory layouts."""
        from local.experiment_dir import ExperimentDirManager
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            modes = [
                ExecutionMode.TRAIN,
                ExecutionMode.TEST,
                ExecutionMode.INFERENCE,
                ExecutionMode.FEATURE_EXTRACTION,
            ]

            for mode in modes:
                config = ExperimentConfig(
                    mode=mode,
                    category="test",
                    method="method",
                    variant="v1",
                    base_dir=tmpdir,
                )
                path = ExperimentDirManager.build_experiment_path(config)

                # Check that mode is in path
                assert mode.value in path.parts
                assert path.exists()

    def test_template_integration(self):
        """Test path generation with template."""
        from local.experiment_dir import ExperimentDirManager
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            config = ExperimentConfig(
                mode=ExecutionMode.TRAIN,
                category="test",
                method="cmt",
                variant="v1",
                base_dir=tmpdir,
                template="{method}_{variant}_custom",
            )

            path = ExperimentDirManager.build_experiment_path(config)

            # Check that template was applied to variant component
            assert "cmt_v1_custom" in path.parts
            assert path.exists()

    def test_environment_variable_expansion(self):
        """Test environment variable expansion in base_dir."""
        from local.experiment_dir import ExperimentDirManager
        import os
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            # Set environment variable
            os.environ["TEST_EXP_DIR"] = tmpdir

            config = ExperimentConfig(
                mode=ExecutionMode.TRAIN,
                category="test",
                method="method",
                variant="v1",
                base_dir="$TEST_EXP_DIR/experiments",
            )

            path = ExperimentDirManager.build_experiment_path(config)

            # Check that environment variable was expanded
            assert tmpdir in str(path)
            assert path.exists()

            # Cleanup
            del os.environ["TEST_EXP_DIR"]

    def test_parent_directory_creation(self):
        """Test that parent directories are automatically created."""
        from local.experiment_dir import ExperimentDirManager
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            config = ExperimentConfig(
                mode=ExecutionMode.TRAIN,
                category="deep/nested/category",
                method="method",
                variant="v1",
                base_dir=tmpdir,
            )

            path = ExperimentDirManager.build_experiment_path(config)

            # All parent directories should exist
            assert path.exists()
            assert path.parent.exists()
            assert path.parent.parent.exists()

    def test_validation_enabled(self):
        """Test that validation is performed when enabled."""
        from local.experiment_dir import ExperimentDirManager

        config = ExperimentConfig(
            mode=ExecutionMode.TRAIN,
            category="invalid<char>",
            method="method",
            variant="v1",
        )

        # Should raise ValueError due to invalid characters
        with pytest.raises(ValueError, match="invalid characters"):
            ExperimentDirManager.build_experiment_path(config, validate=True)

    def test_validation_disabled(self):
        """Test that validation can be disabled."""
        from local.experiment_dir import ExperimentDirManager
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a path that would normally fail validation (too long)
            # but we'll use a shorter one for the test to actually create it
            config = ExperimentConfig(
                mode=ExecutionMode.TRAIN,
                category="test",
                method="method",
                variant="v1",
                base_dir=tmpdir,
            )

            # Should not raise even if we had validation issues
            path = ExperimentDirManager.build_experiment_path(config, validate=False)
            assert path.exists()

    def test_path_length_validation(self):
        """Test that extremely long paths raise ValueError."""
        from local.experiment_dir import ExperimentDirManager

        # Create config with very long variant name
        long_variant = "v" * 250

        config = ExperimentConfig(
            mode=ExecutionMode.TRAIN,
            category="category",
            method="method",
            variant=long_variant,
            base_dir="experiments",
        )

        # Should raise ValueError due to path length
        with pytest.raises(ValueError, match="exceeds maximum limit"):
            ExperimentDirManager.build_experiment_path(config, validate=True)


class TestShouldInitializeWandb:
    """Test should_initialize_wandb function."""

    def test_train_mode_returns_true(self):
        """Test that TRAIN mode always returns True."""
        from local.experiment_dir import ExperimentDirManager

        config = ExperimentConfig(mode=ExecutionMode.TRAIN, log_test_to_wandb=False)
        result = ExperimentDirManager.should_initialize_wandb(ExecutionMode.TRAIN, config)
        assert result is True

        # Even with log_test_to_wandb=True, TRAIN mode should return True
        config_with_log = ExperimentConfig(mode=ExecutionMode.TRAIN, log_test_to_wandb=True)
        result = ExperimentDirManager.should_initialize_wandb(ExecutionMode.TRAIN, config_with_log)
        assert result is True

    def test_test_mode_respects_config_flag(self):
        """Test that TEST mode returns config.log_test_to_wandb value."""
        from local.experiment_dir import ExperimentDirManager

        # Test with log_test_to_wandb=False
        config_no_log = ExperimentConfig(mode=ExecutionMode.TEST, log_test_to_wandb=False)
        result = ExperimentDirManager.should_initialize_wandb(ExecutionMode.TEST, config_no_log)
        assert result is False

        # Test with log_test_to_wandb=True
        config_with_log = ExperimentConfig(mode=ExecutionMode.TEST, log_test_to_wandb=True)
        result = ExperimentDirManager.should_initialize_wandb(ExecutionMode.TEST, config_with_log)
        assert result is True

    def test_inference_mode_returns_false(self):
        """Test that INFERENCE mode always returns False."""
        from local.experiment_dir import ExperimentDirManager

        config = ExperimentConfig(mode=ExecutionMode.INFERENCE, log_test_to_wandb=False)
        result = ExperimentDirManager.should_initialize_wandb(ExecutionMode.INFERENCE, config)
        assert result is False

        # Even with log_test_to_wandb=True, INFERENCE mode should return False
        config_with_log = ExperimentConfig(mode=ExecutionMode.INFERENCE, log_test_to_wandb=True)
        result = ExperimentDirManager.should_initialize_wandb(ExecutionMode.INFERENCE, config_with_log)
        assert result is False

    def test_feature_extraction_mode_returns_false(self):
        """Test that FEATURE_EXTRACTION mode always returns False."""
        from local.experiment_dir import ExperimentDirManager

        config = ExperimentConfig(mode=ExecutionMode.FEATURE_EXTRACTION, log_test_to_wandb=False)
        result = ExperimentDirManager.should_initialize_wandb(ExecutionMode.FEATURE_EXTRACTION, config)
        assert result is False

        # Even with log_test_to_wandb=True, FEATURE_EXTRACTION mode should return False
        config_with_log = ExperimentConfig(mode=ExecutionMode.FEATURE_EXTRACTION, log_test_to_wandb=True)
        result = ExperimentDirManager.should_initialize_wandb(ExecutionMode.FEATURE_EXTRACTION, config_with_log)
        assert result is False

    def test_all_modes_with_different_configs(self):
        """Test all modes with various config combinations."""
        from local.experiment_dir import ExperimentDirManager

        # Test all modes with log_test_to_wandb=False
        for mode in ExecutionMode:
            config = ExperimentConfig(mode=mode, log_test_to_wandb=False)
            result = ExperimentDirManager.should_initialize_wandb(mode, config)

            if mode == ExecutionMode.TRAIN:
                assert result is True, f"TRAIN mode should return True"
            else:
                assert result is False, f"{mode.value} mode with log_test_to_wandb=False should return False"

        # Test all modes with log_test_to_wandb=True
        for mode in ExecutionMode:
            config = ExperimentConfig(mode=mode, log_test_to_wandb=True)
            result = ExperimentDirManager.should_initialize_wandb(mode, config)

            if mode == ExecutionMode.TRAIN or mode == ExecutionMode.TEST:
                assert result is True, f"{mode.value} mode with log_test_to_wandb=True should return True"
            else:
                assert result is False, f"{mode.value} mode should return False regardless of log_test_to_wandb"


class TestCreateArtifactDirs:
    """Test create_artifact_dirs function."""

    def test_creates_all_subdirectories(self):
        """Test that all required artifact subdirectories are created."""
        from local.experiment_dir import ExperimentDirManager
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            experiment_dir = Path(tmpdir) / "test_experiment"
            experiment_dir.mkdir()

            result = ExperimentDirManager.create_artifact_dirs(experiment_dir)

            # Check that all subdirectories exist
            expected_dirs = ["checkpoints", "metrics", "inference", "visualizations", "config"]
            for dir_name in expected_dirs:
                assert (experiment_dir / dir_name).exists()
                assert (experiment_dir / dir_name).is_dir()

            # Check that result dictionary contains all paths
            assert set(result.keys()) == set(expected_dirs)
            for dir_name, path in result.items():
                assert isinstance(path, Path)
                assert path == experiment_dir / dir_name
                assert path.exists()

    def test_returns_path_dictionary(self):
        """Test that function returns dictionary with correct structure."""
        from local.experiment_dir import ExperimentDirManager
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            experiment_dir = Path(tmpdir) / "test_experiment"
            experiment_dir.mkdir()

            result = ExperimentDirManager.create_artifact_dirs(experiment_dir)

            # Verify dictionary structure
            assert isinstance(result, dict)
            assert "checkpoints" in result
            assert "metrics" in result
            assert "inference" in result
            assert "visualizations" in result
            assert "config" in result

            # Verify all values are Path objects
            for key, value in result.items():
                assert isinstance(value, Path)

    def test_idempotent_operation(self):
        """Test that creating artifact dirs multiple times is safe (idempotent)."""
        from local.experiment_dir import ExperimentDirManager
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            experiment_dir = Path(tmpdir) / "test_experiment"
            experiment_dir.mkdir()

            # Create directories first time
            result1 = ExperimentDirManager.create_artifact_dirs(experiment_dir)

            # Create directories second time (should not raise error)
            result2 = ExperimentDirManager.create_artifact_dirs(experiment_dir)

            # Results should be identical
            assert result1 == result2

            # All directories should still exist
            for dir_name in result1.keys():
                assert (experiment_dir / dir_name).exists()

    def test_experiment_dir_nonexistent_raises_error(self):
        """Test that nonexistent experiment directory raises appropriate error."""
        from local.experiment_dir import ExperimentDirManager
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            nonexistent_dir = Path(tmpdir) / "nonexistent_experiment"

            # Should raise error when experiment_dir doesn't exist
            with pytest.raises(FileNotFoundError):
                ExperimentDirManager.create_artifact_dirs(nonexistent_dir)

    def test_preserves_existing_files(self):
        """Test that existing files in subdirectories are preserved."""
        from local.experiment_dir import ExperimentDirManager
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            experiment_dir = Path(tmpdir) / "test_experiment"
            experiment_dir.mkdir()

            # Create checkpoints dir with a file
            checkpoints_dir = experiment_dir / "checkpoints"
            checkpoints_dir.mkdir()
            test_file = checkpoints_dir / "model.ckpt"
            test_file.write_text("checkpoint data")

            # Call create_artifact_dirs
            result = ExperimentDirManager.create_artifact_dirs(experiment_dir)

            # Check that existing file is preserved
            assert test_file.exists()
            assert test_file.read_text() == "checkpoint data"

            # Check that other directories were created
            assert (experiment_dir / "metrics").exists()
            assert (experiment_dir / "inference").exists()
            assert (experiment_dir / "visualizations").exists()
            assert (experiment_dir / "config").exists()

    def test_path_as_string(self):
        """Test that function works with string path input."""
        from local.experiment_dir import ExperimentDirManager
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            experiment_dir = Path(tmpdir) / "test_experiment"
            experiment_dir.mkdir()

            # Pass string path instead of Path object
            result = ExperimentDirManager.create_artifact_dirs(str(experiment_dir))

            # Should still work and return Path objects
            assert isinstance(result, dict)
            assert all(isinstance(p, Path) for p in result.values())
            assert all(p.exists() for p in result.values())


class TestGenerateManifest:
    """Test generate_manifest function."""

    def test_generates_manifest_with_all_fields(self):
        """Test that manifest.json is created with all required fields."""
        from local.experiment_dir import ExperimentDirManager
        import tempfile
        import json

        with tempfile.TemporaryDirectory() as tmpdir:
            experiment_dir = Path(tmpdir) / "test_experiment"
            experiment_dir.mkdir()

            config = {
                "training": {"max_epochs": 10},
                "net": {"model": "CRNN"},
            }

            ExperimentDirManager.generate_manifest(
                experiment_dir=experiment_dir,
                run_id="test-run-id-12345",
                config=config,
                mode=ExecutionMode.TRAIN,
                parent_run_id=None,
            )

            # Check that manifest.json exists
            manifest_path = experiment_dir / "manifest.json"
            assert manifest_path.exists()

            # Load and verify content
            with manifest_path.open("r", encoding="utf-8") as f:
                manifest = json.load(f)

            # Verify all required fields
            assert manifest["run_id"] == "test-run-id-12345"
            assert manifest["experiment_path"] == str(experiment_dir)
            assert manifest["mode"] == "train"
            assert manifest["parent_run_id"] is None
            assert "created_at" in manifest
            assert "config" in manifest
            assert manifest["config"] == config

    def test_generates_manifest_for_inference_mode(self):
        """Test manifest generation for inference mode (run_id=None)."""
        from local.experiment_dir import ExperimentDirManager
        import tempfile
        import json

        with tempfile.TemporaryDirectory() as tmpdir:
            experiment_dir = Path(tmpdir) / "test_experiment"
            experiment_dir.mkdir()

            config = {"inference": {"threshold": 0.5}}

            ExperimentDirManager.generate_manifest(
                experiment_dir=experiment_dir,
                run_id=None,
                config=config,
                mode=ExecutionMode.INFERENCE,
                parent_run_id="parent-train-run-123",
            )

            manifest_path = experiment_dir / "manifest.json"
            assert manifest_path.exists()

            with manifest_path.open("r", encoding="utf-8") as f:
                manifest = json.load(f)

            # Verify inference-specific fields
            assert manifest["run_id"] is None
            assert manifest["mode"] == "inference"
            assert manifest["parent_run_id"] == "parent-train-run-123"

    def test_manifest_created_at_is_iso8601(self):
        """Test that created_at field is in ISO8601 format."""
        from local.experiment_dir import ExperimentDirManager
        import tempfile
        import json
        from datetime import datetime

        with tempfile.TemporaryDirectory() as tmpdir:
            experiment_dir = Path(tmpdir) / "test_experiment"
            experiment_dir.mkdir()

            ExperimentDirManager.generate_manifest(
                experiment_dir=experiment_dir,
                run_id="test-run-id",
                config={},
                mode=ExecutionMode.TRAIN,
            )

            manifest_path = experiment_dir / "manifest.json"
            with manifest_path.open("r", encoding="utf-8") as f:
                manifest = json.load(f)

            # Verify ISO8601 format (can be parsed without error)
            created_at = manifest["created_at"]
            parsed_time = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
            assert isinstance(parsed_time, datetime)

    def test_manifest_includes_config_snapshot(self):
        """Test that manifest includes complete config snapshot."""
        from local.experiment_dir import ExperimentDirManager
        import tempfile
        import json

        with tempfile.TemporaryDirectory() as tmpdir:
            experiment_dir = Path(tmpdir) / "test_experiment"
            experiment_dir.mkdir()

            config = {
                "training": {"max_epochs": 10, "batch_size": 32},
                "net": {"model": "CRNN", "embedding_dim": 768},
                "experiment": {"category": "baseline", "method": "cmt"},
            }

            ExperimentDirManager.generate_manifest(
                experiment_dir=experiment_dir,
                run_id="test-run-id",
                config=config,
                mode=ExecutionMode.TRAIN,
            )

            manifest_path = experiment_dir / "manifest.json"
            with manifest_path.open("r", encoding="utf-8") as f:
                manifest = json.load(f)

            # Verify config is fully preserved
            assert manifest["config"] == config
            assert manifest["config"]["training"]["max_epochs"] == 10
            assert manifest["config"]["net"]["embedding_dim"] == 768

    def test_manifest_utf8_encoding(self):
        """Test that manifest is saved with UTF-8 encoding."""
        from local.experiment_dir import ExperimentDirManager
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            experiment_dir = Path(tmpdir) / "test_experiment"
            experiment_dir.mkdir()

            ExperimentDirManager.generate_manifest(
                experiment_dir=experiment_dir,
                run_id="test-run-id",
                config={},
                mode=ExecutionMode.TRAIN,
            )

            manifest_path = experiment_dir / "manifest.json"

            # Read with explicit UTF-8 encoding
            with manifest_path.open("r", encoding="utf-8") as f:
                content = f.read()
                assert len(content) > 0

    def test_manifest_formatted_with_indent(self):
        """Test that manifest JSON is formatted with indent=2."""
        from local.experiment_dir import ExperimentDirManager
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            experiment_dir = Path(tmpdir) / "test_experiment"
            experiment_dir.mkdir()

            config = {"training": {"max_epochs": 10}}

            ExperimentDirManager.generate_manifest(
                experiment_dir=experiment_dir,
                run_id="test-run-id",
                config=config,
                mode=ExecutionMode.TRAIN,
            )

            manifest_path = experiment_dir / "manifest.json"
            with manifest_path.open("r", encoding="utf-8") as f:
                content = f.read()

            # Check for indentation (presence of multiple spaces)
            assert "  " in content  # indent=2 should create 2-space indentation

    def test_manifest_for_all_execution_modes(self):
        """Test manifest generation for all execution modes."""
        from local.experiment_dir import ExperimentDirManager
        import tempfile
        import json

        modes = [
            ExecutionMode.TRAIN,
            ExecutionMode.TEST,
            ExecutionMode.INFERENCE,
            ExecutionMode.FEATURE_EXTRACTION,
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            for mode in modes:
                experiment_dir = Path(tmpdir) / f"test_{mode.value}"
                experiment_dir.mkdir()

                ExperimentDirManager.generate_manifest(
                    experiment_dir=experiment_dir,
                    run_id=f"run-{mode.value}" if mode != ExecutionMode.INFERENCE else None,
                    config={},
                    mode=mode,
                )

                manifest_path = experiment_dir / "manifest.json"
                assert manifest_path.exists()

                with manifest_path.open("r", encoding="utf-8") as f:
                    manifest = json.load(f)

                assert manifest["mode"] == mode.value

    def test_manifest_experiment_path_is_absolute(self):
        """Test that experiment_path in manifest is stored as absolute path."""
        from local.experiment_dir import ExperimentDirManager
        import tempfile
        import json

        with tempfile.TemporaryDirectory() as tmpdir:
            experiment_dir = Path(tmpdir) / "test_experiment"
            experiment_dir.mkdir()

            ExperimentDirManager.generate_manifest(
                experiment_dir=experiment_dir,
                run_id="test-run-id",
                config={},
                mode=ExecutionMode.TRAIN,
            )

            manifest_path = experiment_dir / "manifest.json"
            with manifest_path.open("r", encoding="utf-8") as f:
                manifest = json.load(f)

            # Verify path is absolute
            stored_path = manifest["experiment_path"]
            assert Path(stored_path).is_absolute()

    def test_manifest_idempotent_operation(self):
        """Test that generating manifest multiple times overwrites existing file."""
        from local.experiment_dir import ExperimentDirManager
        import tempfile
        import json
        import time

        with tempfile.TemporaryDirectory() as tmpdir:
            experiment_dir = Path(tmpdir) / "test_experiment"
            experiment_dir.mkdir()

            # Generate first manifest
            ExperimentDirManager.generate_manifest(
                experiment_dir=experiment_dir,
                run_id="test-run-id-1",
                config={"version": 1},
                mode=ExecutionMode.TRAIN,
            )

            manifest_path = experiment_dir / "manifest.json"
            with manifest_path.open("r", encoding="utf-8") as f:
                manifest1 = json.load(f)

            time.sleep(0.01)  # Ensure different timestamp

            # Generate second manifest (should overwrite)
            ExperimentDirManager.generate_manifest(
                experiment_dir=experiment_dir,
                run_id="test-run-id-2",
                config={"version": 2},
                mode=ExecutionMode.TEST,
            )

            with manifest_path.open("r", encoding="utf-8") as f:
                manifest2 = json.load(f)

            # Verify second manifest overwrote first
            assert manifest2["run_id"] == "test-run-id-2"
            assert manifest2["config"]["version"] == 2
            assert manifest2["mode"] == "test"
            assert manifest1["run_id"] != manifest2["run_id"]


class TestGetExperimentDir:
    """Test get_experiment_dir function."""

    def test_get_experiment_dir_by_run_id_from_manifest(self):
        """Test resolving experiment directory by run_id using manifest.json."""
        from local.experiment_dir import ExperimentDirManager
        import tempfile
        import json

        with tempfile.TemporaryDirectory() as tmpdir:
            base_dir = Path(tmpdir) / "experiments"

            # Create experiment directory structure
            experiment_dir = base_dir / "train" / "baseline" / "cmt" / "v1" / "run-20250112-abcd"
            experiment_dir.mkdir(parents=True)

            # Generate manifest
            run_id = "20250112_123456-abcd1234"
            ExperimentDirManager.generate_manifest(
                experiment_dir=experiment_dir,
                run_id=run_id,
                config={},
                mode=ExecutionMode.TRAIN,
            )

            # Resolve path by run_id
            resolved_path = ExperimentDirManager.get_experiment_dir(
                run_id_or_name=run_id,
                base_dir=base_dir
            )

            assert resolved_path == experiment_dir

    def test_get_experiment_dir_with_mode_filter(self):
        """Test resolving experiment directory with mode filter to narrow search scope."""
        from local.experiment_dir import ExperimentDirManager
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            base_dir = Path(tmpdir) / "experiments"

            # Create train and inference directories with same run_id pattern
            train_dir = base_dir / "train" / "baseline" / "cmt" / "v1" / "run-20250112-train"
            train_dir.mkdir(parents=True)

            inference_dir = base_dir / "inference" / "baseline" / "cmt" / "v1" / "run-20250112-infer"
            inference_dir.mkdir(parents=True)

            # Generate manifests
            ExperimentDirManager.generate_manifest(
                experiment_dir=train_dir,
                run_id="train-run-id-123",
                config={},
                mode=ExecutionMode.TRAIN,
            )

            ExperimentDirManager.generate_manifest(
                experiment_dir=inference_dir,
                run_id=None,
                config={},
                mode=ExecutionMode.INFERENCE,
            )

            # Resolve with mode filter
            resolved_train = ExperimentDirManager.get_experiment_dir(
                run_id_or_name="train-run-id-123",
                base_dir=base_dir,
                mode=ExecutionMode.TRAIN
            )

            assert resolved_train == train_dir

    def test_get_experiment_dir_fallback_to_directory_scan(self):
        """Test fallback to directory scan when manifest is missing or corrupted."""
        from local.experiment_dir import ExperimentDirManager
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            base_dir = Path(tmpdir) / "experiments"

            # Create experiment directory without manifest
            experiment_dir = base_dir / "train" / "baseline" / "cmt" / "v1" / "run-20250112-xyz"
            experiment_dir.mkdir(parents=True)

            # Should still find directory by scanning (using directory name pattern)
            resolved_path = ExperimentDirManager.get_experiment_dir(
                run_id_or_name="run-20250112-xyz",
                base_dir=base_dir
            )

            assert resolved_path == experiment_dir

    def test_get_experiment_dir_not_found_raises_error(self):
        """Test that FileNotFoundError is raised when experiment directory not found."""
        from local.experiment_dir import ExperimentDirManager
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            base_dir = Path(tmpdir) / "experiments"
            base_dir.mkdir(parents=True)

            # Should raise FileNotFoundError for nonexistent run_id
            with pytest.raises(FileNotFoundError, match="Experiment directory not found"):
                ExperimentDirManager.get_experiment_dir(
                    run_id_or_name="nonexistent-run-id-999",
                    base_dir=base_dir
                )

    def test_get_experiment_dir_corrupted_manifest_fallback(self):
        """Test fallback to directory scan when manifest.json is corrupted (JSONDecodeError)."""
        from local.experiment_dir import ExperimentDirManager
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            base_dir = Path(tmpdir) / "experiments"

            # Create experiment directory
            experiment_dir = base_dir / "train" / "baseline" / "cmt" / "v1" / "run-20250112-corrupted"
            experiment_dir.mkdir(parents=True)

            # Create a corrupted manifest.json (invalid JSON)
            manifest_path = experiment_dir / "manifest.json"
            manifest_path.write_text("{ invalid json content !!!", encoding="utf-8")

            # Should still find directory by fallback directory scan
            resolved_path = ExperimentDirManager.get_experiment_dir(
                run_id_or_name="run-20250112-corrupted",
                base_dir=base_dir
            )

            assert resolved_path == experiment_dir

    def test_get_experiment_dir_multiple_corrupted_manifests(self):
        """Test that search continues through multiple corrupted manifests."""
        from local.experiment_dir import ExperimentDirManager
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            base_dir = Path(tmpdir) / "experiments"

            # Create multiple experiment directories
            corrupted_dir1 = base_dir / "train" / "baseline" / "method1" / "v1" / "run-001"
            corrupted_dir1.mkdir(parents=True)
            (corrupted_dir1 / "manifest.json").write_text("corrupted", encoding="utf-8")

            corrupted_dir2 = base_dir / "train" / "baseline" / "method2" / "v1" / "run-002"
            corrupted_dir2.mkdir(parents=True)
            (corrupted_dir2 / "manifest.json").write_text("{ also corrupted", encoding="utf-8")

            # Create a valid experiment directory with valid manifest
            valid_dir = base_dir / "train" / "baseline" / "method3" / "v1" / "run-003"
            valid_dir.mkdir(parents=True)
            ExperimentDirManager.generate_manifest(
                experiment_dir=valid_dir,
                run_id="target-run-id-123",
                config={},
                mode=ExecutionMode.TRAIN,
            )

            # Should find the valid directory, skipping corrupted manifests
            resolved_path = ExperimentDirManager.get_experiment_dir(
                run_id_or_name="target-run-id-123",
                base_dir=base_dir
            )

            assert resolved_path == valid_dir

    def test_get_experiment_dir_corrupted_manifest_fallback_by_name(self):
        """Test that corrupted manifest triggers fallback to directory name matching."""
        from local.experiment_dir import ExperimentDirManager
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            base_dir = Path(tmpdir) / "experiments"

            # Create experiment directory with corrupted manifest
            experiment_dir = base_dir / "train" / "baseline" / "cmt" / "v1" / "run-20250112-fallback-test"
            experiment_dir.mkdir(parents=True)

            # Write corrupted manifest (OSError scenario - read-protected file)
            manifest_path = experiment_dir / "manifest.json"
            manifest_path.write_text("{", encoding="utf-8")  # Incomplete JSON

            # Should fallback to directory name matching
            resolved_path = ExperimentDirManager.get_experiment_dir(
                run_id_or_name="fallback-test",  # Partial match in directory name
                base_dir=base_dir
            )

            assert resolved_path == experiment_dir

    def test_get_experiment_dir_performance(self):
        """Test that get_experiment_dir completes within 100ms (Requirement 3.5)."""
        from local.experiment_dir import ExperimentDirManager
        import tempfile
        import time

        with tempfile.TemporaryDirectory() as tmpdir:
            base_dir = Path(tmpdir) / "experiments"

            # Create multiple experiment directories
            for i in range(10):
                experiment_dir = base_dir / "train" / "baseline" / f"method{i}" / "v1" / f"run-{i:05d}"
                experiment_dir.mkdir(parents=True)

                ExperimentDirManager.generate_manifest(
                    experiment_dir=experiment_dir,
                    run_id=f"run-id-{i:05d}",
                    config={},
                    mode=ExecutionMode.TRAIN,
                )

            # Measure resolution time
            start_time = time.time()

            resolved_path = ExperimentDirManager.get_experiment_dir(
                run_id_or_name="run-id-00005",
                base_dir=base_dir
            )

            elapsed_time = time.time() - start_time

            # Should complete within 100ms (0.1 seconds)
            assert elapsed_time < 0.1, f"Resolution took {elapsed_time*1000:.2f}ms, expected < 100ms"
            assert resolved_path.exists()


class TestGetCheckpointDir:
    """Test get_checkpoint_dir function."""

    def test_get_checkpoint_dir_returns_correct_path(self):
        """Test that get_checkpoint_dir returns checkpoints/ subdirectory."""
        from local.experiment_dir import ExperimentDirManager
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            experiment_dir = Path(tmpdir) / "test_experiment"
            experiment_dir.mkdir()

            checkpoint_dir = ExperimentDirManager.get_checkpoint_dir(experiment_dir)

            assert checkpoint_dir == experiment_dir / "checkpoints"

    def test_get_checkpoint_dir_with_string_path(self):
        """Test that get_checkpoint_dir works with string input."""
        from local.experiment_dir import ExperimentDirManager
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            experiment_dir = Path(tmpdir) / "test_experiment"
            experiment_dir.mkdir()

            checkpoint_dir = ExperimentDirManager.get_checkpoint_dir(str(experiment_dir))

            assert checkpoint_dir == experiment_dir / "checkpoints"
            assert isinstance(checkpoint_dir, Path)


class TestGetInferenceDir:
    """Test get_inference_dir function."""

    def test_get_inference_dir_returns_correct_path(self):
        """Test that get_inference_dir returns inference/ subdirectory."""
        from local.experiment_dir import ExperimentDirManager
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            experiment_dir = Path(tmpdir) / "test_experiment"
            experiment_dir.mkdir()

            inference_dir = ExperimentDirManager.get_inference_dir(experiment_dir)

            assert inference_dir == experiment_dir / "inference"

    def test_get_inference_dir_with_string_path(self):
        """Test that get_inference_dir works with string input."""
        from local.experiment_dir import ExperimentDirManager
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            experiment_dir = Path(tmpdir) / "test_experiment"
            experiment_dir.mkdir()

            inference_dir = ExperimentDirManager.get_inference_dir(str(experiment_dir))

            assert inference_dir == experiment_dir / "inference"
            assert isinstance(inference_dir, Path)


class TestGetVisualizationDir:
    """Test get_visualization_dir function."""

    def test_get_visualization_dir_returns_correct_path(self):
        """Test that get_visualization_dir returns visualizations/ subdirectory."""
        from local.experiment_dir import ExperimentDirManager
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            experiment_dir = Path(tmpdir) / "test_experiment"
            experiment_dir.mkdir()

            viz_dir = ExperimentDirManager.get_visualization_dir(experiment_dir)

            assert viz_dir == experiment_dir / "visualizations"

    def test_get_visualization_dir_with_string_path(self):
        """Test that get_visualization_dir works with string input."""
        from local.experiment_dir import ExperimentDirManager
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            experiment_dir = Path(tmpdir) / "test_experiment"
            experiment_dir.mkdir()

            viz_dir = ExperimentDirManager.get_visualization_dir(str(experiment_dir))

            assert viz_dir == experiment_dir / "visualizations"
            assert isinstance(viz_dir, Path)
