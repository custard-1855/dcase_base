"""Test for concurrent experiment separation.

This test verifies:
1. Multiple training experiments with the same category/method/variant are separated by run ID
2. Multiple inference executions are separated by different timestamp directories
3. No conflicts occur when running parallel experiments
"""

import json
import shutil
import tempfile
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from local.experiment_dir import ExecutionMode, ExperimentConfig, ExperimentDirManager


@pytest.fixture
def temp_experiment_dir():
    """Create temporary directory for concurrent experiment tests."""
    temp_dir = tempfile.mkdtemp(prefix="test_concurrent_")
    yield Path(temp_dir)
    # Cleanup
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def base_config(temp_experiment_dir):
    """Create base configuration for concurrent experiments."""
    return {
        "experiment": {
            "mode": "train",
            "category": "concurrent_test",
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
            "batch_size": [1, 1, 1, 1, 1],
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


def test_concurrent_training_experiments_separation(temp_experiment_dir, base_config):
    """Test that multiple training experiments with same config are separated by run ID.

    Requirements tested:
    - Requirement 1.5: When multiple experiments share the same hierarchy path,
                      unique identifiers (run ID) should prevent conflicts
    - Requirement 5.2: Train mode creates new wandb run and directory

    This test simulates multiple training experiments starting concurrently
    with the same category/method/variant configuration.
    """
    from local.sed_trainer_pretrained import SEDTask4

    # Expected base path for all concurrent experiments
    expected_base_path = (
        temp_experiment_dir
        / "train"
        / base_config["experiment"]["category"]
        / base_config["experiment"]["method"]
        / base_config["experiment"]["variant"]
    )

    # Create 3 concurrent training experiments
    num_experiments = 3
    experiment_dirs = []
    run_ids = []

    for i in range(num_experiments):
        with patch("local.sed_trainer_pretrained.wandb") as mock_wandb:
            # Create unique run ID for each experiment
            mock_run = MagicMock()
            mock_run.id = f"concurrent_run_{i}_{int(time.time() * 1000)}"
            run_ids.append(mock_run.id)

            # Create unique run directory (simulating wandb behavior)
            mock_run_dir = expected_base_path / f"run-{mock_run.id}"
            mock_run_dir.mkdir(parents=True, exist_ok=True)
            mock_run.dir = str(mock_run_dir)
            experiment_dirs.append(mock_run_dir)

            mock_wandb.init.return_value = mock_run
            mock_wandb.run = mock_run

            # Create minimal mocks
            mock_encoder = MagicMock()
            mock_encoder.labels = ["class_" + str(j) for j in range(27)]

            mock_student = MagicMock()
            mock_student.parameters.return_value = []

            # Initialize SEDTask4 to trigger wandb initialization
            sed_task = SEDTask4(
                hparams=base_config,
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

            # Small delay to ensure different timestamps if needed
            time.sleep(0.01)

    # Assertions: Verify all experiments are separated

    # 1. All run IDs should be unique
    assert len(set(run_ids)) == num_experiments, "Run IDs should be unique for each experiment"

    # 2. All experiment directories should exist and be separate
    assert (
        len(experiment_dirs) == num_experiments
    ), f"Expected {num_experiments} directories, got {len(experiment_dirs)}"

    for exp_dir in experiment_dirs:
        assert exp_dir.exists(), f"Experiment directory not created: {exp_dir}"
        assert exp_dir.is_dir(), f"Path is not a directory: {exp_dir}"

    # 3. Each directory should have its own manifest with unique run_id
    manifests = []
    for exp_dir in experiment_dirs:
        manifest_path = exp_dir / "manifest.json"
        assert manifest_path.exists(), f"Manifest not found: {manifest_path}"

        with manifest_path.open("r", encoding="utf-8") as f:
            manifest = json.load(f)
            manifests.append(manifest)

        # Verify manifest contains run_id and mode
        assert "run_id" in manifest, f"Manifest missing run_id: {manifest_path}"
        assert "mode" in manifest, f"Manifest missing mode: {manifest_path}"
        assert manifest["mode"] == "train", f"Expected mode='train', got '{manifest['mode']}'"

    # 4. All manifest run_ids should be unique
    manifest_run_ids = [m["run_id"] for m in manifests]
    assert len(set(manifest_run_ids)) == num_experiments, "Manifest run_ids should be unique"

    # 5. Verify all directories are under the same hierarchy path
    for exp_dir in experiment_dirs:
        assert exp_dir.parent == expected_base_path, (
            f"Experiment directory should be under {expected_base_path}, "
            f"but got {exp_dir.parent}"
        )


def test_concurrent_inference_executions_separation(temp_experiment_dir, base_config):
    """Test that multiple inference executions are separated by timestamp directories.

    Requirements tested:
    - Requirement 1.5: Unique identifiers (timestamp) prevent conflicts for inference mode
    - Requirement 5.4: Inference mode disables wandb and stores artifacts without synchronization

    This test verifies that inference executions without wandb are properly separated
    using timestamp-based directory names.
    """
    from local.sed_trainer_pretrained import SEDTask4

    # Modify config for inference mode
    inference_config = base_config.copy()
    inference_config["experiment"] = {
        "mode": "inference",
        "category": "concurrent_test",
        "method": "baseline",
        "variant": "v1",
        "base_dir": str(temp_experiment_dir),
    }

    expected_base_path = (
        temp_experiment_dir
        / "inference"
        / inference_config["experiment"]["category"]
        / inference_config["experiment"]["method"]
        / inference_config["experiment"]["variant"]
    )

    # Create 3 concurrent inference executions
    num_executions = 3
    inference_dirs = []

    for i in range(num_executions):
        with patch("local.sed_trainer_pretrained.wandb") as mock_wandb:
            # wandb should not be initialized in inference mode
            mock_wandb.init.return_value = None
            mock_wandb.run = None

            # Create minimal mocks
            mock_encoder = MagicMock()
            mock_encoder.labels = ["class_" + str(j) for j in range(27)]

            mock_student = MagicMock()
            mock_student.parameters.return_value = []

            # Initialize SEDTask4 with inference mode
            sed_task = SEDTask4(
                hparams=inference_config,
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
                evaluation=True,  # Inference mode indicator
            )

            # Verify wandb was not initialized (inference mode)
            assert not mock_wandb.init.called, "wandb.init should not be called in inference mode"

            # Verify _inference_dir was set
            assert hasattr(
                sed_task, "_inference_dir"
            ), "SEDTask4 should have _inference_dir attribute"
            assert (
                sed_task._inference_dir is not None
            ), "_inference_dir should be set in inference mode"

            inference_dirs.append(Path(sed_task._inference_dir))

            # Small delay to ensure different timestamps
            time.sleep(0.1)  # Longer delay for timestamp differentiation

    # Assertions: Verify all inference executions are separated

    # 1. All inference directories should exist and be separate
    assert (
        len(inference_dirs) == num_executions
    ), f"Expected {num_executions} directories, got {len(inference_dirs)}"

    for inf_dir in inference_dirs:
        assert inf_dir.exists(), f"Inference directory not created: {inf_dir}"
        assert inf_dir.is_dir(), f"Path is not a directory: {inf_dir}"

    # 2. All directories should have unique names (timestamp-based)
    dir_names = [d.name for d in inference_dirs]
    assert len(set(dir_names)) == num_executions, (
        f"Inference directories should have unique names. Got: {dir_names}"
    )

    # 3. Each directory should have its own manifest with mode="inference" and run_id=None
    for inf_dir in inference_dirs:
        manifest_path = inf_dir / "manifest.json"
        assert manifest_path.exists(), f"Manifest not found: {manifest_path}"

        with manifest_path.open("r", encoding="utf-8") as f:
            manifest = json.load(f)

        # Verify manifest structure for inference mode
        assert "mode" in manifest, f"Manifest missing mode: {manifest_path}"
        assert (
            manifest["mode"] == "inference"
        ), f"Expected mode='inference', got '{manifest['mode']}'"

        assert "run_id" in manifest, f"Manifest missing run_id: {manifest_path}"
        assert (
            manifest["run_id"] is None
        ), f"Inference mode should have run_id=None, got {manifest['run_id']}"

    # 4. Verify all directories are under the same hierarchy path
    for inf_dir in inference_dirs:
        assert inf_dir.parent == expected_base_path, (
            f"Inference directory should be under {expected_base_path}, " f"but got {inf_dir.parent}"
        )


def test_mixed_mode_concurrent_experiments_no_conflict(temp_experiment_dir, base_config):
    """Test that training and inference experiments with same config don't conflict.

    This test verifies that training (under experiments/train/) and inference
    (under experiments/inference/) can run concurrently without conflicts,
    even with the same category/method/variant.

    Requirements tested:
    - Requirement 1.1: Hierarchical structure with mode-based top-level separation
    - Requirement 5.2, 5.4: Mode-specific directory layouts prevent conflicts
    """
    from local.sed_trainer_pretrained import SEDTask4

    # 1. Create training experiment
    train_config = base_config.copy()
    train_config["experiment"]["mode"] = "train"

    expected_train_path = (
        temp_experiment_dir
        / "train"
        / train_config["experiment"]["category"]
        / train_config["experiment"]["method"]
        / train_config["experiment"]["variant"]
    )

    with patch("local.sed_trainer_pretrained.wandb") as mock_wandb_train:
        mock_run = MagicMock()
        mock_run.id = "train_run_concurrent"
        mock_run_dir = expected_train_path / f"run-{mock_run.id}"
        mock_run_dir.mkdir(parents=True, exist_ok=True)
        mock_run.dir = str(mock_run_dir)

        mock_wandb_train.init.return_value = mock_run
        mock_wandb_train.run = mock_run

        mock_encoder = MagicMock()
        mock_encoder.labels = ["class_" + str(i) for i in range(27)]
        mock_student = MagicMock()
        mock_student.parameters.return_value = []

        train_task = SEDTask4(
            hparams=train_config,
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

        train_experiment_dir = Path(mock_run.dir)

    # 2. Create inference execution with same category/method/variant
    inference_config = base_config.copy()
    inference_config["experiment"]["mode"] = "inference"

    expected_inference_path = (
        temp_experiment_dir
        / "inference"
        / inference_config["experiment"]["category"]
        / inference_config["experiment"]["method"]
        / inference_config["experiment"]["variant"]
    )

    with patch("local.sed_trainer_pretrained.wandb") as mock_wandb_inference:
        mock_wandb_inference.init.return_value = None
        mock_wandb_inference.run = None

        mock_encoder = MagicMock()
        mock_encoder.labels = ["class_" + str(i) for i in range(27)]
        mock_student = MagicMock()
        mock_student.parameters.return_value = []

        inference_task = SEDTask4(
            hparams=inference_config,
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
            evaluation=True,
        )

        inference_experiment_dir = Path(inference_task._inference_dir)

    # Assertions: Verify no conflicts and proper separation

    # 1. Both directories should exist
    assert train_experiment_dir.exists(), f"Training directory not found: {train_experiment_dir}"
    assert (
        inference_experiment_dir.exists()
    ), f"Inference directory not found: {inference_experiment_dir}"

    # 2. Directories should be in different top-level mode directories
    assert (
        "train" in train_experiment_dir.parts
    ), "Training experiment should be under 'train' directory"
    assert (
        "inference" in inference_experiment_dir.parts
    ), "Inference experiment should be under 'inference' directory"

    # 3. No overlap in paths
    assert (
        train_experiment_dir != inference_experiment_dir
    ), "Training and inference directories should be different"

    # 4. Both manifests should have correct mode
    train_manifest_path = train_experiment_dir / "manifest.json"
    inference_manifest_path = inference_experiment_dir / "manifest.json"

    assert train_manifest_path.exists(), f"Training manifest not found: {train_manifest_path}"
    assert (
        inference_manifest_path.exists()
    ), f"Inference manifest not found: {inference_manifest_path}"

    with train_manifest_path.open("r", encoding="utf-8") as f:
        train_manifest = json.load(f)
    with inference_manifest_path.open("r", encoding="utf-8") as f:
        inference_manifest = json.load(f)

    assert train_manifest["mode"] == "train", f"Expected train mode, got {train_manifest['mode']}"
    assert (
        inference_manifest["mode"] == "inference"
    ), f"Expected inference mode, got {inference_manifest['mode']}"

    # 5. Training has run_id, inference has run_id=None
    assert (
        train_manifest["run_id"] is not None
    ), "Training experiment should have non-null run_id"
    assert (
        inference_manifest["run_id"] is None
    ), "Inference execution should have run_id=None"
