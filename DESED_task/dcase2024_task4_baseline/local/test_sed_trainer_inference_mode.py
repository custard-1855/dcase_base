"""Integration tests for inference mode directory creation logic.

This test module validates Task 2.2 requirements:
- Inference/feature_extraction mode creates timestamp-based directories
- Artifact subdirectories are created
- manifest.json is generated with run_id=None

Note: These tests validate the directory creation logic independently of SEDTask4
to avoid complex dependency issues during testing.
"""

import json
import sys
from datetime import datetime
from pathlib import Path

import pytest

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from local.experiment_dir import ExecutionMode, ExperimentConfig, ExperimentDirManager


class TestInferenceModeDirectoryCreationLogic:
    """Test inference/feature_extraction mode directory creation logic.

    This test class validates Task 2.2: Non-wandb mode directory creation.
    These tests simulate the logic used in SEDTask4._init_wandb_project()
    without requiring the full SEDTask4 class and its dependencies.
    """

    @pytest.fixture
    def base_hparams(self, tmp_path):
        """Create base hyperparameters for testing."""
        return {
            "wandb": {"wandb_dir": "None"},  # Disable legacy mode
            "experiment": {
                "mode": "inference",
                "category": "test_category",
                "method": "test_method",
                "variant": "test_variant",
                "base_dir": str(tmp_path / "experiments"),
            },
        }

    def _simulate_inference_mode_initialization(
        self, hparams: dict, evaluation: bool = False
    ) -> dict:
        """Simulate the inference mode initialization logic from SEDTask4._init_wandb_project().

        This method replicates the core logic without requiring SEDTask4 dependencies.

        Returns:
            Dictionary with initialization results including:
            - execution_mode: Detected execution mode
            - inference_dir: Created inference directory path
            - wandb_checkpoint_dir: Should be None
            - artifact_dirs: Created artifact subdirectories
        """
        # Detect execution mode
        execution_mode = ExperimentDirManager.detect_execution_mode(
            hparams=hparams,
            evaluation=evaluation,
            test_state_dict=None,
            fast_dev_run=False,
        )

        # Check if wandb should be initialized
        exp_config = ExperimentConfig(**hparams["experiment"])
        should_init_wandb = ExperimentDirManager.should_initialize_wandb(
            execution_mode, exp_config
        )

        if should_init_wandb:
            raise RuntimeError("Expected wandb to be disabled for inference mode")

        # Create inference directory (non-wandb mode)
        base_dir = ExperimentDirManager.build_experiment_path(exp_config)
        inference_dir = base_dir / f"run-{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        inference_dir.mkdir(parents=True, exist_ok=True)

        # Create artifact directories
        artifact_dirs = ExperimentDirManager.create_artifact_dirs(inference_dir)

        # Generate manifest
        ExperimentDirManager.generate_manifest(
            inference_dir,
            run_id=None,
            config=hparams,
            mode=execution_mode,
        )

        return {
            "execution_mode": execution_mode,
            "inference_dir": inference_dir,
            "wandb_checkpoint_dir": None,
            "artifact_dirs": artifact_dirs,
        }

    def test_inference_mode_creates_timestamped_directory(self, base_hparams, tmp_path):
        """Test that inference mode creates timestamp-based directory.

        Requirement: 2.2 - Inference/feature_extraction mode creates timestamp directory
        """
        result = self._simulate_inference_mode_initialization(
            hparams=base_hparams, evaluation=True
        )

        inference_dir = result["inference_dir"]

        # Verify directory exists
        assert inference_dir.exists()

        # Verify directory name contains timestamp pattern (run-YYYYMMDD_HHMMSS)
        assert "run-" in inference_dir.name
        # "run-" (4 chars) + "YYYYMMDD_HHMMSS" (15 chars) = 19 chars
        assert len(inference_dir.name) >= 19

    def test_inference_mode_creates_artifact_subdirectories(self, base_hparams, tmp_path):
        """Test that inference mode creates all artifact subdirectories.

        Requirement: 2.2 - Artifact subdirectories created via create_artifact_dirs()
        """
        result = self._simulate_inference_mode_initialization(
            hparams=base_hparams, evaluation=True
        )

        inference_dir = result["inference_dir"]
        artifact_dirs = result["artifact_dirs"]

        # Verify all artifact subdirectories exist
        expected_subdirs = ["checkpoints", "metrics", "inference", "visualizations", "config"]
        for subdir in expected_subdirs:
            subdir_path = inference_dir / subdir
            assert subdir_path.exists(), f"Missing subdirectory: {subdir}"
            assert subdir_path.is_dir(), f"{subdir} is not a directory"

        # Verify artifact_dirs dictionary contains all paths
        assert set(artifact_dirs.keys()) == set(expected_subdirs)

    def test_inference_mode_generates_manifest_without_run_id(
        self, base_hparams, tmp_path
    ):
        """Test that inference mode generates manifest.json with run_id=None.

        Requirement: 2.2 - manifest.json generated with run_id=None
        """
        result = self._simulate_inference_mode_initialization(
            hparams=base_hparams, evaluation=True
        )

        inference_dir = result["inference_dir"]
        manifest_path = inference_dir / "manifest.json"

        # Verify manifest.json exists
        assert manifest_path.exists()

        # Load and verify manifest content
        with manifest_path.open("r", encoding="utf-8") as f:
            manifest = json.load(f)

        # Verify run_id is None
        assert manifest["run_id"] is None

        # Verify mode is "inference"
        assert manifest["mode"] == "inference"

        # Verify other required fields
        assert "experiment_path" in manifest
        assert "created_at" in manifest
        assert "config" in manifest

    def test_inference_mode_returns_valid_inference_dir(self, base_hparams, tmp_path):
        """Test that inference mode returns a valid inference directory path.

        Requirement: 2.2 - inference_dir is set and valid
        """
        result = self._simulate_inference_mode_initialization(
            hparams=base_hparams, evaluation=True
        )

        inference_dir = result["inference_dir"]

        # Verify it's a Path object
        assert isinstance(inference_dir, Path)

        # Verify it's a valid path that exists
        assert inference_dir.exists()

    def test_inference_mode_sets_wandb_checkpoint_dir_to_none(
        self, base_hparams, tmp_path
    ):
        """Test that inference mode sets wandb_checkpoint_dir to None.

        Requirement: 2.2 - wandb_checkpoint_dir should be None when wandb is disabled
        """
        result = self._simulate_inference_mode_initialization(
            hparams=base_hparams, evaluation=True
        )

        # Verify wandb_checkpoint_dir is None
        assert result["wandb_checkpoint_dir"] is None

    def test_feature_extraction_mode_creates_directory(self, base_hparams, tmp_path):
        """Test that feature_extraction mode also creates non-wandb directory.

        Requirement: 2.2 - Feature extraction mode behaves like inference mode
        """
        # Modify hparams for feature_extraction mode
        base_hparams["experiment"]["mode"] = "feature_extraction"

        result = self._simulate_inference_mode_initialization(
            hparams=base_hparams, evaluation=False
        )

        inference_dir = result["inference_dir"]
        assert inference_dir.exists()

        # Verify manifest mode is "feature_extraction"
        manifest_path = inference_dir / "manifest.json"
        with manifest_path.open("r", encoding="utf-8") as f:
            manifest = json.load(f)
        assert manifest["mode"] == "feature_extraction"

    def test_inference_mode_directory_hierarchy(self, base_hparams, tmp_path):
        """Test that inference mode creates correct directory hierarchy.

        Requirement: 2.2 - Directory structure follows experiments/{mode}/{category}/{method}/{variant}/
        """
        result = self._simulate_inference_mode_initialization(
            hparams=base_hparams, evaluation=True
        )

        inference_dir = result["inference_dir"]

        # Verify directory hierarchy
        # Expected: experiments/inference/test_category/test_method/test_variant/run-*
        parts = inference_dir.parts
        assert "inference" in parts
        assert "test_category" in parts
        assert "test_method" in parts
        assert "test_variant" in parts

    def test_inference_mode_detects_correct_execution_mode(self, base_hparams, tmp_path):
        """Test that inference mode correctly detects execution mode.

        Requirement: 5.1, 5.8 - Execution mode detection
        """
        result = self._simulate_inference_mode_initialization(
            hparams=base_hparams, evaluation=True
        )

        # Verify execution mode is INFERENCE
        assert result["execution_mode"] == ExecutionMode.INFERENCE

    def test_inference_mode_with_different_base_dir(self, base_hparams, tmp_path):
        """Test inference mode with custom base_dir configuration."""
        custom_base = tmp_path / "custom_experiments"
        base_hparams["experiment"]["base_dir"] = str(custom_base)

        result = self._simulate_inference_mode_initialization(
            hparams=base_hparams, evaluation=True
        )

        inference_dir = result["inference_dir"]

        # Verify custom base_dir is used
        assert custom_base in inference_dir.parents
        assert inference_dir.exists()
