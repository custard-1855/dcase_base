"""E2E test for train mode with hierarchical experiment directory structure.

This test verifies:
1. Experiment starts with train mode configuration
2. Hierarchical directory structure experiments/train/{category}/{method}/{variant}/run-*/ is created
3. Artifact subdirectories (checkpoints/, metrics/, config/) are created
4. manifest.json contains correct mode and run_id
"""

import json
import shutil
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch

from local.experiment_dir import ExecutionMode, ExperimentConfig, ExperimentDirManager


@pytest.fixture
def temp_experiment_dir():
    """Create temporary directory for E2E test experiments."""
    temp_dir = tempfile.mkdtemp(prefix="test_e2e_train_")
    yield Path(temp_dir)
    # Cleanup
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def mock_wandb():
    """Mock wandb module for E2E testing without actual wandb initialization."""
    with patch("local.sed_trainer_pretrained.wandb") as mock_wandb_module:
        # Create mock run object
        mock_run = MagicMock()
        mock_run.id = "test_run_20250112_123456"
        mock_run.dir = None  # Will be set dynamically in test

        # Configure mock wandb.init to return mock run
        mock_wandb_module.init.return_value = mock_run
        mock_wandb_module.run = mock_run

        yield mock_wandb_module


@pytest.fixture
def train_mode_yaml_config(temp_experiment_dir):
    """Create YAML-like config dictionary for train mode E2E test."""
    return {
        "experiment": {
            "mode": "train",
            "category": "e2e_test",
            "method": "baseline",
            "variant": "v1",
            "base_dir": str(temp_experiment_dir),
        },
        "wandb": {
            "use_wandb": True,
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


def test_train_mode_e2e_directory_structure(
    temp_experiment_dir, mock_wandb, train_mode_yaml_config
):
    """Test E2E train mode creates correct hierarchical directory structure.

    Requirements tested:
    - Requirement 1.1: Hierarchical directory structure experiments/{mode}/{category}/{method}/{variant}/
    - Requirement 1.2: Automatic parent directory creation
    - Requirement 2.1: Checkpoints management under {experiment_dir}/checkpoints/
    - Requirement 2.2: Metrics management under {experiment_dir}/metrics/
    - Requirement 2.5: Config management under {experiment_dir}/config/
    - Requirement 2.6: Manifest generation with mode="train" and run_id
    - Requirement 5.2: Train mode creates new wandb run and directory
    - Requirement 5.5: Manifest records mode field
    """
    # Setup: Import SEDTask4 here to ensure mocks are applied
    from local.sed_trainer_pretrained import SEDTask4

    # Expected directory structure
    expected_base_path = (
        temp_experiment_dir
        / "train"
        / train_mode_yaml_config["experiment"]["category"]
        / train_mode_yaml_config["experiment"]["method"]
        / train_mode_yaml_config["experiment"]["variant"]
    )

    # Configure mock wandb run directory dynamically
    # Simulate wandb creating run-{timestamp}-{id} directory
    mock_run_dir = expected_base_path / f"run-{mock_wandb.run.id}"
    mock_run_dir.mkdir(parents=True, exist_ok=True)
    mock_wandb.run.dir = str(mock_run_dir)

    # Create minimal SEDTask4 instance to trigger wandb initialization
    # Mock encoder with necessary attributes
    mock_encoder = MagicMock()
    mock_encoder.labels = ["class_" + str(i) for i in range(27)]  # 27 classes

    # Mock student model with parameters() method
    mock_student = MagicMock()
    mock_student.parameters.return_value = []

    sed_task = SEDTask4(
        hparams=train_mode_yaml_config,
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

    # Assertions: Verify directory structure

    # 1. Base hierarchical path should exist (Requirement 1.1, 1.2)
    assert expected_base_path.exists(), f"Base path not created: {expected_base_path}"
    assert expected_base_path.is_dir(), "Base path is not a directory"

    # 2. Wandb run directory should exist (Requirement 5.2)
    assert mock_run_dir.exists(), f"Wandb run directory not created: {mock_run_dir}"

    # 3. Artifact subdirectories should exist (Requirements 2.1, 2.2, 2.5)
    expected_subdirs = ["checkpoints", "metrics", "config", "inference", "visualizations"]
    for subdir_name in expected_subdirs:
        subdir_path = mock_run_dir / subdir_name
        assert subdir_path.exists(), f"Artifact subdirectory not created: {subdir_path}"
        assert subdir_path.is_dir(), f"{subdir_name} is not a directory"

    # 4. Manifest.json should exist and contain correct metadata (Requirements 2.6, 5.5)
    manifest_path = mock_run_dir / "manifest.json"
    assert manifest_path.exists(), f"Manifest file not created: {manifest_path}"

    with manifest_path.open("r", encoding="utf-8") as f:
        manifest = json.load(f)

    # Verify manifest structure
    assert "run_id" in manifest, "Manifest missing run_id field"
    assert manifest["run_id"] == mock_wandb.run.id, "Manifest run_id mismatch"

    assert "mode" in manifest, "Manifest missing mode field"
    assert manifest["mode"] == "train", f"Expected mode='train', got '{manifest['mode']}'"

    assert "experiment_path" in manifest, "Manifest missing experiment_path field"
    assert manifest["experiment_path"] == str(
        mock_run_dir.absolute()
    ), "Manifest experiment_path mismatch"

    assert "created_at" in manifest, "Manifest missing created_at field"
    assert "config" in manifest, "Manifest missing config field"

    # parent_run_id should be None for train mode
    assert (
        manifest.get("parent_run_id") is None
    ), f"Train mode should have parent_run_id=None, got {manifest.get('parent_run_id')}"

    # 5. Verify wandb was initialized with correct parameters
    mock_wandb.init.assert_called_once()
    call_kwargs = mock_wandb.init.call_args.kwargs

    # Check wandb.init was called with custom dir parameter (Requirement 3.1, 3.2)
    assert "dir" in call_kwargs, "wandb.init should be called with 'dir' parameter"
    assert str(expected_base_path) in call_kwargs["dir"], "wandb.init dir parameter mismatch"

    # Check wandb.init name includes mode/category/method/variant
    expected_name_pattern = "train/e2e_test/baseline/v1"
    assert "name" in call_kwargs, "wandb.init should be called with 'name' parameter"
    assert (
        expected_name_pattern in call_kwargs["name"]
    ), f"wandb.init name should contain '{expected_name_pattern}'"


def test_train_mode_e2e_without_experiment_section(temp_experiment_dir, mock_wandb):
    """Test E2E train mode falls back to default when experiment section is missing.

    This verifies backward compatibility (Requirement 4.4: Default value fallback).
    """
    # Setup: Import SEDTask4 here
    from local.sed_trainer_pretrained import SEDTask4

    # Config without experiment section
    config_no_experiment = {
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

    # Configure mock wandb (default project mode)
    mock_run_dir = temp_experiment_dir / "default_run"
    mock_run_dir.mkdir(parents=True, exist_ok=True)
    mock_wandb.run.dir = str(mock_run_dir)

    # Create SEDTask4 instance with proper mocks
    mock_encoder = MagicMock()
    mock_encoder.labels = ["class_" + str(i) for i in range(27)]

    mock_student = MagicMock()
    mock_student.parameters.return_value = []

    sed_task = SEDTask4(
        hparams=config_no_experiment,
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

    # Assertions: Verify default fallback behavior
    # wandb.init should still be called (default mode)
    mock_wandb.init.assert_called_once()

    # _wandb_checkpoint_dir should be set (even in default mode)
    assert hasattr(
        sed_task, "_wandb_checkpoint_dir"
    ), "SEDTask4 should have _wandb_checkpoint_dir attribute"
