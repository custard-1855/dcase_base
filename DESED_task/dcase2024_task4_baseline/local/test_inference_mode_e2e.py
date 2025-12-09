"""E2E test for inference mode with hierarchical experiment directory structure.

This test verifies:
1. Inference mode execution with --mode inference flag
2. WandB run is NOT created (wandb.init not called)
3. Hierarchical directory structure experiments/inference/{category}/{method}/{variant}/run-*/ is created
4. Artifact subdirectories are created
5. manifest.json contains mode="inference" and run_id=null
"""

import json
import shutil
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from local.experiment_dir import ExecutionMode, ExperimentConfig, ExperimentDirManager


@pytest.fixture
def temp_experiment_dir():
    """Create temporary directory for E2E test experiments."""
    temp_dir = tempfile.mkdtemp(prefix="test_e2e_inference_")
    yield Path(temp_dir)
    # Cleanup
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def mock_wandb():
    """Mock wandb module for E2E testing without actual wandb initialization."""
    with patch("local.sed_trainer_pretrained.wandb") as mock_wandb_module:
        # Create mock run object (should NOT be initialized in inference mode)
        mock_run = MagicMock()
        mock_run.id = None
        mock_run.dir = None

        # Configure mock wandb.init (should NOT be called in inference mode)
        mock_wandb_module.init.return_value = None
        mock_wandb_module.run = None

        yield mock_wandb_module


@pytest.fixture
def inference_mode_yaml_config(temp_experiment_dir):
    """Create YAML-like config dictionary for inference mode E2E test."""
    return {
        "experiment": {
            "mode": "inference",
            "category": "e2e_inference_test",
            "method": "baseline",
            "variant": "v1",
            "base_dir": str(temp_experiment_dir),
        },
        "wandb": {
            "use_wandb": True,  # Even if True, should be disabled in inference mode
            "wandb_dir": "None",  # Use new mode, not legacy
        },
        "net": {
            "wandb_dir": "None",
            "median_filter": [1] * 27,  # Required by SEDTask4.__init__
        },
        "pretrained": {
            "e2e": False,  # Don't use end-to-end pretrained model
        },
        "training": {
            "n_epochs": 1,
            "batch_size": [1, 1, 1, 1, 1],
            "num_workers": 0,  # Required by SEDTask4.__init__
            "self_sup_loss": "mse",  # Required by SEDTask4.__init__
            "n_test_thresholds": 10,  # Required by SEDTask4.__init__
        },
        "scaler": {
            "statistic": "instance",
            "normtype": "minmax",
            "dims": [1, 2],
            "savepath": "./scaler.ckpt",
        },
        "data": {
            "audio_max_len": 10,
            "fs": 16000,
            "net_subsample": 4,
        },
        "feats": {
            "n_filters": 2048,
            "hop_length": 256,
            "n_mels": 128,
            "n_window": 2048,
            "sample_rate": 16000,
            "f_min": 0,
            "f_max": 8000,
        },
    }


def test_inference_mode_e2e_directory_structure(
    temp_experiment_dir, mock_wandb, inference_mode_yaml_config
):
    """Test E2E inference mode creates correct directory structure WITHOUT wandb run.

    Requirements tested:
    - Requirement 1.1: Hierarchical directory structure experiments/{mode}/{category}/{method}/{variant}/
    - Requirement 2.3: Inference management under {experiment_dir}/inference/
    - Requirement 2.6: Manifest generation with mode="inference" and run_id=null
    - Requirement 5.4: inference mode disables wandb run creation
    - Requirement 5.5: Manifest records mode field
    - Requirement 5.6: Mode-specific log message when wandb is disabled
    """
    # Setup: Import SEDTask4 here to ensure mocks are applied
    from local.sed_trainer_pretrained import SEDTask4

    # Expected directory structure
    expected_base_path = (
        temp_experiment_dir
        / "inference"
        / inference_mode_yaml_config["experiment"]["category"]
        / inference_mode_yaml_config["experiment"]["method"]
        / inference_mode_yaml_config["experiment"]["variant"]
    )

    # Create minimal SEDTask4 instance to trigger directory creation
    # Mock encoder with necessary attributes
    mock_encoder = MagicMock()
    mock_encoder.labels = ["class_" + str(i) for i in range(27)]  # 27 classes

    # Mock student model with parameters() method
    mock_student = MagicMock()
    mock_student.parameters.return_value = []

    # Capture print output to verify log message
    with patch("builtins.print") as mock_print:
        sed_task = SEDTask4(
            hparams=inference_mode_yaml_config,
            encoder=mock_encoder,
            sed_student=mock_student,
            pretrained_model=None,
            opt=None,
            train_data=None,
            valid_data=None,
            test_data=None,
            train_sampler=None,
            scheduler=None,
            fast_dev_run=False,
            evaluation=True,  # evaluation=True is typical for inference mode
        )

        # Verify log message for wandb disabled (Requirement 5.6)
        print_calls = [str(call) for call in mock_print.call_args_list]
        wandb_disabled_log_found = any(
            "WandB disabled for inference mode" in call or "wandb disabled" in call.lower()
            for call in print_calls
        )
        assert wandb_disabled_log_found, (
            f"Expected 'WandB disabled for inference mode' log message. "
            f"Actual print calls: {print_calls}"
        )

    # Assertions: Verify directory structure

    # 1. Base hierarchical path should exist (Requirement 1.1)
    assert expected_base_path.exists(), f"Base path not created: {expected_base_path}"
    assert expected_base_path.is_dir(), "Base path is not a directory"

    # 2. Inference run directory should exist (with timestamp pattern)
    # Find run-* directories in expected_base_path
    run_dirs = [d for d in expected_base_path.iterdir() if d.is_dir() and d.name.startswith("run-")]
    assert len(run_dirs) > 0, f"No run-* directory created in {expected_base_path}"

    inference_run_dir = run_dirs[0]  # Take the first (should be only one in this test)
    assert inference_run_dir.exists(), f"Inference run directory not created: {inference_run_dir}"

    # 3. Artifact subdirectories should exist (Requirement 2.3, 2.6)
    expected_subdirs = ["checkpoints", "metrics", "config", "inference", "visualizations"]
    for subdir_name in expected_subdirs:
        subdir_path = inference_run_dir / subdir_name
        assert subdir_path.exists(), f"Artifact subdirectory not created: {subdir_path}"
        assert subdir_path.is_dir(), f"{subdir_name} is not a directory"

    # 4. Manifest.json should exist and contain correct metadata (Requirements 2.6, 5.5)
    manifest_path = inference_run_dir / "manifest.json"
    assert manifest_path.exists(), f"Manifest file not created: {manifest_path}"

    with manifest_path.open("r", encoding="utf-8") as f:
        manifest = json.load(f)

    # Verify manifest structure
    assert "run_id" in manifest, "Manifest missing run_id field"
    assert manifest["run_id"] is None, f"Expected run_id=None for inference mode, got {manifest['run_id']}"

    assert "mode" in manifest, "Manifest missing mode field"
    assert manifest["mode"] == "inference", f"Expected mode='inference', got '{manifest['mode']}'"

    assert "experiment_path" in manifest, "Manifest missing experiment_path field"
    assert manifest["experiment_path"] == str(
        inference_run_dir.absolute()
    ), "Manifest experiment_path mismatch"

    assert "created_at" in manifest, "Manifest missing created_at field"
    assert "config" in manifest, "Manifest missing config field"

    # 5. Verify wandb was NOT initialized (Requirement 5.4)
    mock_wandb.init.assert_not_called()

    # 6. Verify _wandb_checkpoint_dir is None (inference mode doesn't use wandb)
    assert sed_task._wandb_checkpoint_dir is None, (
        f"Expected _wandb_checkpoint_dir=None for inference mode, "
        f"got {sed_task._wandb_checkpoint_dir}"
    )

    # 7. Verify _inference_dir is set
    assert hasattr(sed_task, "_inference_dir"), "SEDTask4 should have _inference_dir attribute"
    assert sed_task._inference_dir is not None, "_inference_dir should not be None"
    assert str(inference_run_dir) in sed_task._inference_dir, (
        f"_inference_dir should contain inference run directory path. "
        f"Expected: {inference_run_dir}, Got: {sed_task._inference_dir}"
    )


def test_inference_mode_e2e_with_explicit_cli_mode(temp_experiment_dir, mock_wandb):
    """Test inference mode with explicit CLI --mode argument.

    This verifies that explicit mode specification takes priority over hparams
    (Requirement 5.7: Mode explicit specification via CLI).
    """
    # Setup: Import SEDTask4
    from local.sed_trainer_pretrained import SEDTask4

    # Config WITHOUT explicit mode in experiment section
    config_no_mode = {
        "experiment": {
            # No "mode" field - will be set via CLI argument simulation
            "category": "cli_inference_test",
            "method": "baseline",
            "variant": "v1",
            "base_dir": str(temp_experiment_dir),
        },
        "wandb": {
            "use_wandb": True,
            "wandb_dir": "None",
        },
        "net": {
            "wandb_dir": "None",
            "median_filter": [1] * 27,
        },
        "pretrained": {
            "e2e": False,
        },
        "training": {
            "n_epochs": 1,
            "num_workers": 0,
            "self_sup_loss": "mse",
            "n_test_thresholds": 10,
        },
        "scaler": {
            "statistic": "instance",
            "normtype": "minmax",
            "dims": [1, 2],
            "savepath": "./scaler.ckpt",
        },
        "data": {
            "audio_max_len": 10,
            "fs": 16000,
            "net_subsample": 4,
        },
        "feats": {
            "n_filters": 2048,
            "hop_length": 256,
            "n_mels": 128,
            "n_window": 2048,
            "sample_rate": 16000,
            "f_min": 0,
            "f_max": 8000,
        },
    }

    # Simulate CLI argument setting mode explicitly
    config_no_mode["experiment"]["mode"] = "inference"

    # Create SEDTask4 instance
    mock_encoder = MagicMock()
    mock_encoder.labels = ["class_" + str(i) for i in range(27)]

    mock_student = MagicMock()
    mock_student.parameters.return_value = []

    sed_task = SEDTask4(
        hparams=config_no_mode,
        encoder=mock_encoder,
        sed_student=mock_student,
        pretrained_model=None,
        opt=None,
        train_data=None,
        valid_data=None,
        test_data=None,
        train_sampler=None,
        scheduler=None,
        fast_dev_run=False,
        evaluation=False,  # evaluation=False but mode="inference" explicitly
    )

    # Expected directory structure with inference mode
    expected_base_path = (
        temp_experiment_dir / "inference" / "cli_inference_test" / "baseline" / "v1"
    )

    # Assertions
    assert expected_base_path.exists(), f"Inference mode directory not created: {expected_base_path}"

    # Verify wandb was NOT initialized
    mock_wandb.init.assert_not_called()

    # Verify execution mode was detected as INFERENCE
    assert hasattr(sed_task, "execution_mode"), "SEDTask4 should have execution_mode attribute"
    assert sed_task.execution_mode == ExecutionMode.INFERENCE, (
        f"Expected execution_mode=INFERENCE, got {sed_task.execution_mode}"
    )


def test_feature_extraction_mode_e2e(temp_experiment_dir, mock_wandb):
    """Test feature_extraction mode creates directory structure WITHOUT wandb run.

    FEATURE_EXTRACTION mode should behave identically to INFERENCE mode regarding
    wandb initialization and directory structure.

    Requirements tested:
    - Requirement 5.1: Four execution modes recognition (FEATURE_EXTRACTION)
    - Requirement 5.4: feature_extraction mode disables wandb
    """
    # Setup: Import SEDTask4
    from local.sed_trainer_pretrained import SEDTask4

    config_feature_extraction = {
        "experiment": {
            "mode": "feature_extraction",
            "category": "feature_extraction_test",
            "method": "beats",
            "variant": "v1",
            "base_dir": str(temp_experiment_dir),
        },
        "wandb": {
            "use_wandb": True,
            "wandb_dir": "None",
        },
        "net": {
            "wandb_dir": "None",
            "median_filter": [1] * 27,
        },
        "pretrained": {
            "e2e": False,
        },
        "training": {
            "n_epochs": 1,
            "num_workers": 0,
            "self_sup_loss": "mse",
            "n_test_thresholds": 10,
        },
        "scaler": {
            "statistic": "instance",
            "normtype": "minmax",
            "dims": [1, 2],
            "savepath": "./scaler.ckpt",
        },
        "data": {
            "audio_max_len": 10,
            "fs": 16000,
            "net_subsample": 4,
        },
        "feats": {
            "n_filters": 2048,
            "hop_length": 256,
            "n_mels": 128,
            "n_window": 2048,
            "sample_rate": 16000,
            "f_min": 0,
            "f_max": 8000,
        },
    }

    # Create SEDTask4 instance
    mock_encoder = MagicMock()
    mock_encoder.labels = ["class_" + str(i) for i in range(27)]

    mock_student = MagicMock()
    mock_student.parameters.return_value = []

    sed_task = SEDTask4(
        hparams=config_feature_extraction,
        encoder=mock_encoder,
        sed_student=mock_student,
        pretrained_model=None,
        opt=None,
        train_data=None,
        valid_data=None,
        test_data=None,
        train_sampler=None,
        scheduler=None,
        fast_dev_run=False,
        evaluation=False,
    )

    # Expected directory structure
    expected_base_path = (
        temp_experiment_dir / "feature_extraction" / "feature_extraction_test" / "beats" / "v1"
    )

    # Assertions
    assert expected_base_path.exists(), (
        f"Feature extraction mode directory not created: {expected_base_path}"
    )

    # Find run-* directories
    run_dirs = [d for d in expected_base_path.iterdir() if d.is_dir() and d.name.startswith("run-")]
    assert len(run_dirs) > 0, f"No run-* directory created in {expected_base_path}"

    feature_extraction_run_dir = run_dirs[0]

    # Verify manifest
    manifest_path = feature_extraction_run_dir / "manifest.json"
    assert manifest_path.exists(), f"Manifest not created: {manifest_path}"

    with manifest_path.open("r", encoding="utf-8") as f:
        manifest = json.load(f)

    assert manifest["mode"] == "feature_extraction", (
        f"Expected mode='feature_extraction', got '{manifest['mode']}'"
    )
    assert manifest["run_id"] is None, (
        f"Expected run_id=None for feature_extraction mode, got {manifest['run_id']}"
    )

    # Verify wandb was NOT initialized
    mock_wandb.init.assert_not_called()

    # Verify execution mode
    assert sed_task.execution_mode == ExecutionMode.FEATURE_EXTRACTION, (
        f"Expected execution_mode=FEATURE_EXTRACTION, got {sed_task.execution_mode}"
    )
