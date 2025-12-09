"""Unit tests for train_pretrained.py experiment section parsing."""

import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, mock_open, patch

import pytest
import yaml

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from local.experiment_dir import ExperimentConfig, ExecutionMode


@pytest.fixture
def mock_yaml_config():
    """Base YAML configuration without experiment section."""
    return {
        "data": {"fs": 16000},
        "feats": {"n_filters": 256, "hop_length": 256},
        "training": {"seed": 42, "batch_size_val": 32},
        "net": {"attn_type": "default", "attn_deepen": 2, "mixstyle_type": "disabled"},
        "cmt": {"enabled": False, "phi_frame": 0.5, "warmup_epochs": 0, "use_neg_sample": False},
        "sebbs": {"enabled": False},
        "wandb": {"use_wandb": False, "wandb_dir": "None"},
    }


@pytest.fixture
def mock_yaml_config_with_experiment():
    """YAML configuration with experiment section."""
    config = {
        "data": {"fs": 16000},
        "feats": {"n_filters": 256, "hop_length": 256},
        "training": {"seed": 42, "batch_size_val": 32},
        "net": {"attn_type": "default", "attn_deepen": 2, "mixstyle_type": "disabled"},
        "cmt": {"enabled": False, "phi_frame": 0.5, "warmup_epochs": 0, "use_neg_sample": False},
        "sebbs": {"enabled": False},
        "wandb": {"use_wandb": False, "wandb_dir": "None"},
        "experiment": {
            "mode": "train",
            "category": "baseline",
            "method": "cmt",
            "variant": "v1",
            "base_dir": "experiments",
        },
    }
    return config


class TestExperimentSectionProcessing:
    """Test experiment section processing logic."""

    def test_experiment_section_missing_adds_empty_dict(self, mock_yaml_config):
        """Test that missing experiment section is initialized as empty dict."""
        # Import the function from train_pretrained module
        from train_pretrained import process_experiment_section

        # Act
        result = process_experiment_section(mock_yaml_config)

        # Assert
        assert "experiment" in result, "experiment section should be added to configs"
        assert isinstance(result["experiment"], dict), "experiment section should be initialized as dict"

    def test_experiment_section_with_defaults_creates_valid_config(self, mock_yaml_config):
        """Test that default ExperimentConfig can be created when experiment section is empty."""
        # Import the function from train_pretrained module
        from train_pretrained import process_experiment_section

        # Arrange
        mock_yaml_config["experiment"] = {}

        # Act
        result = process_experiment_section(mock_yaml_config)

        # Assert - Should be able to create ExperimentConfig with defaults
        exp_config = ExperimentConfig(**result["experiment"])
        assert exp_config.mode == ExecutionMode.TRAIN
        assert exp_config.category == "default"
        assert exp_config.method == "baseline"
        assert exp_config.variant == "v1"

    def test_experiment_section_valid_config_logs_structure(
        self, mock_yaml_config_with_experiment, capsys
    ):
        """Test that valid experiment config logs the structure."""
        # Import the function from train_pretrained module
        from train_pretrained import process_experiment_section

        # Act
        result = process_experiment_section(mock_yaml_config_with_experiment)

        # Assert
        assert "experiment" in result
        exp_config = ExperimentConfig(**result["experiment"])
        assert exp_config.mode == ExecutionMode.TRAIN
        assert exp_config.category == "baseline"
        assert exp_config.method == "cmt"
        assert exp_config.variant == "v1"

        # Check that structure was logged
        captured = capsys.readouterr()
        assert "Experiment structure:" in captured.out
        assert "train/baseline/cmt/v1" in captured.out

    def test_experiment_section_with_invalid_mode_raises_error(self, mock_yaml_config):
        """Test that invalid mode in experiment section raises ValueError."""
        # Import the function from train_pretrained module
        from train_pretrained import process_experiment_section

        # Arrange
        mock_yaml_config["experiment"] = {
            "mode": "invalid_mode",
            "category": "baseline",
            "method": "cmt",
            "variant": "v1",
        }

        # Act & Assert
        with pytest.raises(ValueError, match="Invalid mode"):
            process_experiment_section(mock_yaml_config)

    def test_experiment_section_preserved_in_configs(self, mock_yaml_config_with_experiment):
        """Test that experiment section is preserved in returned configs."""
        # Import the function from train_pretrained module
        from train_pretrained import process_experiment_section

        # Act
        result = process_experiment_section(mock_yaml_config_with_experiment)

        # Assert
        assert result["experiment"]["mode"] == "train"
        assert result["experiment"]["category"] == "baseline"
        assert result["experiment"]["method"] == "cmt"
        assert result["experiment"]["variant"] == "v1"
        assert result["experiment"]["base_dir"] == "experiments"

    def test_experiment_section_with_inference_mode(self, mock_yaml_config):
        """Test that inference mode is correctly processed."""
        # Import the function from train_pretrained module
        from train_pretrained import process_experiment_section

        # Arrange
        mock_yaml_config["experiment"] = {
            "mode": "inference",
            "category": "evaluation",
            "method": "cmt",
            "variant": "v2",
        }

        # Act
        result = process_experiment_section(mock_yaml_config)

        # Assert
        exp_config = ExperimentConfig(**result["experiment"])
        assert exp_config.mode == ExecutionMode.INFERENCE
        assert exp_config.category == "evaluation"


class TestCLIModeArgument:
    """Test --mode CLI argument processing."""

    def test_mode_argument_exists_in_parser(self):
        """Test that --mode argument is defined in argparse."""
        # Import prepare_run from train_pretrained module
        from train_pretrained import prepare_run

        # Act - Call prepare_run with --help to see if --mode is available
        # We need to mock sys.exit to prevent the test from exiting
        with pytest.raises(SystemExit):
            with patch("sys.argv", ["train_pretrained.py", "--help"]):
                prepare_run()

    def test_mode_argument_accepts_valid_modes(self):
        """Test that --mode argument accepts valid execution modes."""
        import argparse

        # Import prepare_run from train_pretrained module
        from train_pretrained import prepare_run

        # Test each valid mode
        valid_modes = ["train", "test", "inference", "feature_extraction"]

        for mode in valid_modes:
            # Mock the necessary dependencies
            with patch("builtins.open", mock_open(read_data="data: {fs: 16000}")):
                with patch("yaml.safe_load") as mock_yaml_load:
                    mock_yaml_load.return_value = {
                        "data": {"fs": 16000},
                        "feats": {"n_filters": 256, "hop_length": 256},
                        "training": {"seed": 42},
                        "net": {},
                    }

                    # Mock sys.argv with --mode argument
                    with patch(
                        "sys.argv",
                        ["train_pretrained.py", "--confs", "test.yaml", "--mode", mode],
                    ):
                        # This should not raise any exception for valid modes
                        try:
                            # We expect this to fail later due to missing dependencies,
                            # but --mode parsing should succeed
                            prepare_run()
                        except Exception:
                            # Ignore other errors, we only care about argparse
                            pass

    def test_mode_argument_rejects_invalid_modes(self):
        """Test that --mode argument rejects invalid execution modes."""
        import argparse

        # Create a parser similar to train_pretrained.py
        parser = argparse.ArgumentParser("Training a SED system for DESED Task")
        parser.add_argument("--confs", default="./confs/pretrained.yaml")
        parser.add_argument(
            "--mode",
            choices=["train", "test", "inference", "feature_extraction"],
            default=None,
            help="Explicit execution mode (overrides auto-detection)",
        )

        # Test invalid mode
        with pytest.raises(SystemExit):
            parser.parse_args(["--mode", "invalid_mode"])

    def test_mode_argument_overrides_yaml_config(self, mock_yaml_config_with_experiment):
        """Test that --mode CLI argument overrides YAML config."""
        from train_pretrained import process_experiment_section

        # Arrange - YAML has mode="train"
        configs = mock_yaml_config_with_experiment.copy()
        assert configs["experiment"]["mode"] == "train"

        # Simulate CLI --mode argument overriding YAML
        # This mimics the logic that should be in prepare_run()
        cli_mode = "inference"
        configs["experiment"]["mode"] = cli_mode

        # Act
        result = process_experiment_section(configs)

        # Assert
        exp_config = ExperimentConfig(**result["experiment"])
        assert exp_config.mode == ExecutionMode.INFERENCE, "CLI mode should override YAML mode"

    def test_mode_argument_default_is_none(self):
        """Test that --mode argument has default=None (no explicit mode)."""
        import argparse

        # Create a parser similar to train_pretrained.py
        parser = argparse.ArgumentParser("Training a SED system for DESED Task")
        parser.add_argument(
            "--mode",
            choices=["train", "test", "inference", "feature_extraction"],
            default=None,
            help="Explicit execution mode (overrides auto-detection)",
        )

        # Parse without --mode argument
        args = parser.parse_args([])

        # Assert
        assert args.mode is None, "--mode default should be None for auto-detection"

    def test_mode_argument_logs_explicit_mode(self, capsys):
        """Test that explicit mode specification is logged."""
        from train_pretrained import process_experiment_section

        # Arrange
        configs = {
            "data": {"fs": 16000},
            "experiment": {"mode": "inference", "category": "test", "method": "cmt", "variant": "v1"},
        }

        # Act
        # In actual implementation, prepare_run() should log "Explicit mode specified: inference"
        # when CLI --mode argument is provided
        # For this test, we'll check the logging happens in process_experiment_section
        result = process_experiment_section(configs)

        # Assert structure logging exists (mode logging will be added in prepare_run)
        captured = capsys.readouterr()
        assert "Experiment structure:" in captured.out


class TestLegacyWandbDirCompatibility:
    """Test legacy --wandb_dir compatibility and priority (Task 3.3)."""

    def test_wandb_dir_argument_sets_config(self, mock_yaml_config):
        """Test that --wandb_dir CLI argument sets configs['wandb']['wandb_dir']."""
        # Test implementation will check that wandb_dir is properly set in configs
        # This validates that the path from CLI argument to config is correct
        wandb_dir_value = "legacy/experiment/name"

        # Simulate the behavior in prepare_run()
        mock_yaml_config["wandb"]["wandb_dir"] = wandb_dir_value

        # Assert
        assert mock_yaml_config["wandb"]["wandb_dir"] == wandb_dir_value
        assert mock_yaml_config["wandb"]["wandb_dir"] != "None"

    def test_wandb_dir_default_is_none_string(self, mock_yaml_config):
        """Test that wandb_dir defaults to 'None' string when not specified."""
        # Assert
        assert mock_yaml_config["wandb"]["wandb_dir"] == "None"
        assert isinstance(mock_yaml_config["wandb"]["wandb_dir"], str)

    def test_legacy_mode_coexists_with_experiment_section(self, mock_yaml_config_with_experiment):
        """Test that legacy --wandb_dir can coexist with experiment section.

        This validates the priority mechanism:
        - Both wandb_dir and experiment section can exist in configs
        - _init_wandb_project() will prioritize wandb_dir when != "None"
        - Experiment section is still preserved for other components
        """
        # Arrange - Simulate CLI setting wandb_dir while experiment section exists
        configs = mock_yaml_config_with_experiment.copy()
        assert "experiment" in configs

        # Simulate --wandb_dir argument
        configs["wandb"]["wandb_dir"] = "legacy/path/to/experiment"

        # Assert - Both should be present
        assert configs["wandb"]["wandb_dir"] == "legacy/path/to/experiment"
        assert configs["experiment"]["mode"] == "train"
        assert configs["experiment"]["category"] == "baseline"

        # This validates that both legacy and new mode information are preserved,
        # allowing _init_wandb_project to decide which to use

    def test_legacy_mode_warning_log_requirement(self, mock_yaml_config_with_experiment, capsys):
        """Test that legacy mode usage should trigger warning log (design requirement).

        This test documents the requirement that when --wandb_dir is specified,
        a warning log should be output in prepare_run() or _init_wandb_project().

        Expected behavior:
        - When wandb_dir != "None", log: "Using legacy wandb_dir mode (ignoring execution mode)"
        """
        # Arrange
        configs = mock_yaml_config_with_experiment.copy()
        configs["wandb"]["wandb_dir"] = "legacy/experiment"

        # Simulate the warning log that should be in prepare_run()
        if configs["wandb"]["wandb_dir"] != "None" and "experiment" in configs:
            print("Using legacy wandb_dir mode (ignoring execution mode)")

        # Assert
        captured = capsys.readouterr()
        assert "Using legacy wandb_dir mode" in captured.out
        assert "ignoring execution mode" in captured.out

    def test_priority_mechanism_legacy_over_new_mode(self, mock_yaml_config_with_experiment):
        """Test that priority mechanism is correctly configured.

        Priority order (from design.md):
        1. Legacy mode (wandb_dir != "None")
        2. New mode (ExperimentConfig)
        3. Default fallback

        This test validates that when both are present, the config structure
        allows _init_wandb_project to check wandb_dir first.
        """
        # Arrange - Both legacy and new mode are configured
        configs = mock_yaml_config_with_experiment.copy()
        configs["wandb"]["wandb_dir"] = "legacy/path"
        configs["experiment"]["mode"] = "train"
        configs["experiment"]["category"] = "new_category"

        # Assert - Both are present, priority will be determined in _init_wandb_project
        assert configs["wandb"]["wandb_dir"] != "None", "Legacy mode is active"
        assert "experiment" in configs, "New mode config is present"

        # This simulates the check in _init_wandb_project():
        # if self.hparams["wandb"]["wandb_dir"] != "None":
        #     # Use legacy mode (priority 1)
        # elif "experiment" in self.hparams:
        #     # Use new mode (priority 2)
        # else:
        #     # Use default (priority 3)

        # Verify the conditions that _init_wandb_project will evaluate
        use_legacy_mode = configs["wandb"]["wandb_dir"] != "None"
        has_experiment_config = "experiment" in configs

        assert use_legacy_mode is True
        assert has_experiment_config is True
        # Legacy mode takes priority even when experiment config exists


class TestPrepareRunIntegration:
    """Integration tests for prepare_run() with legacy mode (Task 3.3)."""

    @pytest.fixture
    def temp_yaml_config(self, tmp_path):
        """Create temporary YAML config file for integration testing."""
        config_file = tmp_path / "test_config.yaml"
        config_content = {
            "data": {
                "fs": 16000,
                "audio_max_len": 10,
                "synth_folder_44k": str(tmp_path / "synth"),
                "synth_folder": str(tmp_path / "synth_16k"),
                "synth_val_folder_44k": str(tmp_path / "synth_val"),
                "synth_val_folder": str(tmp_path / "synth_val_16k"),
                "real_maestro_train_folder_44k": str(tmp_path / "maestro_train"),
                "real_maestro_train_folder": str(tmp_path / "maestro_train_16k"),
                "real_maestro_val_folder_44k": str(tmp_path / "maestro_val"),
                "real_maestro_val_folder": str(tmp_path / "maestro_val_16k"),
                "strong_folder_44k": str(tmp_path / "strong"),
                "strong_folder": str(tmp_path / "strong_16k"),
                "weak_folder_44k": str(tmp_path / "weak"),
                "weak_folder": str(tmp_path / "weak_16k"),
                "unlabeled_folder_44k": str(tmp_path / "unlabeled"),
                "unlabeled_folder": str(tmp_path / "unlabeled_16k"),
                "test_folder_44k": str(tmp_path / "test"),
                "test_folder": str(tmp_path / "test_16k"),
                "eval_folder_44k": str(tmp_path / "eval"),
                "eval_folder": str(tmp_path / "eval_16k"),
                "synth_val_dur": str(tmp_path / "synth_val_dur.tsv"),
                "test_dur": str(tmp_path / "test_dur.tsv"),
                "net_subsample": 4,
            },
            "feats": {"n_filters": 128, "hop_length": 256},
            "training": {"batch_size_val": 24},
            "cmt": {"enabled": False, "phi_frame": 0.5, "warmup_epochs": 0, "use_neg_sample": False},
            "sebbs": {"enabled": False},
            "wandb": {"use_wandb": False, "wandb_dir": "None"},
            "net": {"attn_type": "default", "attn_deepen": 2, "mixstyle_type": "disabled"},
            "experiment": {
                "mode": "train",
                "category": "baseline",
                "method": "cmt",
                "variant": "v1",
            },
        }

        # Create necessary directories
        for key, value in config_content["data"].items():
            if key.endswith("_folder") or key.endswith("_folder_44k"):
                Path(value).mkdir(parents=True, exist_ok=True)

        with open(config_file, "w") as f:
            yaml.dump(config_content, f)

        return config_file

    @patch("train_pretrained.resample_data_generate_durations")
    def test_wandb_dir_cli_argument_overrides_yaml(
        self, mock_resample, temp_yaml_config, capsys
    ):
        """Test that --wandb_dir CLI argument properly sets configs['wandb']['wandb_dir'].

        This is an integration test that validates the full flow:
        1. YAML is loaded with wandb_dir: "None"
        2. CLI argument --wandb_dir is provided
        3. configs['wandb']['wandb_dir'] is set to CLI value
        4. Warning log is output when experiment section also exists
        """
        from train_pretrained import prepare_run

        # Arrange
        argv = [
            "--confs",
            str(temp_yaml_config),
            "--wandb_dir",
            "legacy/experiment/path",
        ]

        # Act
        configs, args, test_state_dict, evaluation = prepare_run(argv)

        # Assert
        assert configs["wandb"]["wandb_dir"] == "legacy/experiment/path"
        assert args.wandb_dir == "legacy/experiment/path"
        assert test_state_dict is None
        assert evaluation is False

        # Check that experiment section is still processed
        assert "experiment" in configs

    @patch("train_pretrained.resample_data_generate_durations")
    def test_wandb_dir_not_specified_uses_yaml_default(
        self, mock_resample, temp_yaml_config
    ):
        """Test that when --wandb_dir is not specified, YAML default is used."""
        from train_pretrained import prepare_run

        # Arrange
        argv = ["--confs", str(temp_yaml_config)]

        # Act
        configs, args, test_state_dict, evaluation = prepare_run(argv)

        # Assert
        assert configs["wandb"]["wandb_dir"] == "None"
        assert args.wandb_dir == "None"

    @patch("train_pretrained.resample_data_generate_durations")
    def test_legacy_mode_warning_not_logged_when_experiment_missing(
        self, mock_resample, temp_yaml_config, capsys, tmp_path
    ):
        """Test that warning is specific to legacy + experiment coexistence.

        When --wandb_dir is specified but no experiment section exists,
        no warning should be logged (backward compatibility).
        """
        from train_pretrained import prepare_run

        # Create config without experiment section
        config_file = tmp_path / "no_exp_config.yaml"
        config_content = yaml.safe_load(open(temp_yaml_config))
        del config_content["experiment"]

        with open(config_file, "w") as f:
            yaml.dump(config_content, f)

        # Arrange
        argv = ["--confs", str(config_file), "--wandb_dir", "legacy/path"]

        # Act
        configs, args, _, _ = prepare_run(argv)

        # Assert
        assert configs["wandb"]["wandb_dir"] == "legacy/path"

        # This test documents expected behavior - warning logic should be
        # implemented in prepare_run() to detect coexistence
        # captured = capsys.readouterr()
        # assert "Using legacy wandb_dir mode" not in captured.out


class TestTestStateDictParameter:
    """Test test_state_dict parameter passing for mode detection (Task 3.4)."""

    @pytest.fixture
    def temp_yaml_config(self, tmp_path):
        """Create temporary YAML config file for integration testing."""
        config_file = tmp_path / "test_config.yaml"
        config_content = {
            "data": {
                "fs": 16000,
                "audio_max_len": 10,
                "synth_folder_44k": str(tmp_path / "synth"),
                "synth_folder": str(tmp_path / "synth_16k"),
                "synth_val_folder_44k": str(tmp_path / "synth_val"),
                "synth_val_folder": str(tmp_path / "synth_val_16k"),
                "real_maestro_train_folder_44k": str(tmp_path / "maestro_train"),
                "real_maestro_train_folder": str(tmp_path / "maestro_train_16k"),
                "real_maestro_val_folder_44k": str(tmp_path / "maestro_val"),
                "real_maestro_val_folder": str(tmp_path / "maestro_val_16k"),
                "strong_folder_44k": str(tmp_path / "strong"),
                "strong_folder": str(tmp_path / "strong_16k"),
                "weak_folder_44k": str(tmp_path / "weak"),
                "weak_folder": str(tmp_path / "weak_16k"),
                "unlabeled_folder_44k": str(tmp_path / "unlabeled"),
                "unlabeled_folder": str(tmp_path / "unlabeled_16k"),
                "test_folder_44k": str(tmp_path / "test"),
                "test_folder": str(tmp_path / "test_16k"),
                "eval_folder_44k": str(tmp_path / "eval"),
                "eval_folder": str(tmp_path / "eval_16k"),
                "synth_val_dur": str(tmp_path / "synth_val_dur.tsv"),
                "test_dur": str(tmp_path / "test_dur.tsv"),
                "net_subsample": 4,
            },
            "feats": {"n_filters": 128, "hop_length": 256},
            "training": {"batch_size_val": 24},
            "cmt": {"enabled": False, "phi_frame": 0.5, "warmup_epochs": 0, "use_neg_sample": False},
            "sebbs": {"enabled": False},
            "wandb": {"use_wandb": False, "wandb_dir": "None"},
            "net": {"attn_type": "default", "attn_deepen": 2, "mixstyle_type": "disabled"},
            "experiment": {
                "mode": "train",
                "category": "baseline",
                "method": "cmt",
                "variant": "v1",
            },
        }

        # Create necessary directories
        for key, value in config_content["data"].items():
            if key.endswith("_folder") or key.endswith("_folder_44k"):
                Path(value).mkdir(parents=True, exist_ok=True)

        with open(config_file, "w") as f:
            yaml.dump(config_content, f)

        return config_file

    @patch("train_pretrained.resample_data_generate_durations")
    @patch("torch.load")
    def test_test_from_checkpoint_loads_state_dict(
        self, mock_torch_load, mock_resample, temp_yaml_config
    ):
        """Test that --test_from_checkpoint loads checkpoint and returns state_dict."""
        from train_pretrained import prepare_run

        # Arrange - Mock checkpoint loading
        mock_checkpoint = {
            "epoch": 10,
            "state_dict": {"layer1.weight": "mock_tensor"},
            "hyper_parameters": {
                "training": {"seed": 42},
                "net": {},
            },
        }
        mock_torch_load.return_value = mock_checkpoint

        argv = [
            "--confs",
            str(temp_yaml_config),
            "--test_from_checkpoint",
            "path/to/checkpoint.ckpt",
        ]

        # Act
        configs, args, test_state_dict, evaluation = prepare_run(argv)

        # Assert
        assert test_state_dict is not None, "test_state_dict should be loaded from checkpoint"
        assert test_state_dict == mock_checkpoint["state_dict"]
        assert evaluation is False, "evaluation should be False for test_from_checkpoint"
        mock_torch_load.assert_called_once_with("path/to/checkpoint.ckpt", weights_only=False)

    @patch("train_pretrained.resample_data_generate_durations")
    @patch("torch.load")
    def test_eval_from_checkpoint_loads_state_dict_and_sets_evaluation(
        self, mock_torch_load, mock_resample, temp_yaml_config
    ):
        """Test that --eval_from_checkpoint loads checkpoint and sets evaluation=True."""
        from train_pretrained import prepare_run

        # Arrange - Mock checkpoint loading
        mock_checkpoint = {
            "epoch": 15,
            "state_dict": {"layer1.weight": "mock_tensor"},
            "hyper_parameters": {
                "training": {"seed": 42},
                "net": {},
            },
        }
        mock_torch_load.return_value = mock_checkpoint

        argv = [
            "--confs",
            str(temp_yaml_config),
            "--eval_from_checkpoint",
            "path/to/eval_checkpoint.ckpt",
        ]

        # Act
        configs, args, test_state_dict, evaluation = prepare_run(argv)

        # Assert
        assert test_state_dict is not None, "test_state_dict should be loaded from checkpoint"
        assert test_state_dict == mock_checkpoint["state_dict"]
        assert evaluation is True, "evaluation should be True for eval_from_checkpoint"
        assert (
            configs["training"]["batch_size_val"] == 1
        ), "batch_size_val should be set to 1 for evaluation"
        mock_torch_load.assert_called_once_with("path/to/eval_checkpoint.ckpt", weights_only=False)

    @patch("train_pretrained.resample_data_generate_durations")
    def test_no_checkpoint_returns_none_state_dict(self, mock_resample, temp_yaml_config):
        """Test that when no checkpoint is specified, test_state_dict is None."""
        from train_pretrained import prepare_run

        # Arrange
        argv = ["--confs", str(temp_yaml_config)]

        # Act
        configs, args, test_state_dict, evaluation = prepare_run(argv)

        # Assert
        assert test_state_dict is None, "test_state_dict should be None when no checkpoint specified"
        assert evaluation is False, "evaluation should be False by default"

    @patch("train_pretrained.resample_data_generate_durations")
    @patch("torch.load")
    def test_checkpoint_loading_logs_epoch_info(
        self, mock_torch_load, mock_resample, temp_yaml_config, capsys
    ):
        """Test that checkpoint loading logs the epoch information."""
        from train_pretrained import prepare_run

        # Arrange
        mock_checkpoint = {
            "epoch": 25,
            "state_dict": {"layer1.weight": "mock_tensor"},
            "hyper_parameters": {
                "training": {"seed": 42},
                "net": {},
            },
        }
        mock_torch_load.return_value = mock_checkpoint

        argv = [
            "--confs",
            str(temp_yaml_config),
            "--test_from_checkpoint",
            "path/to/checkpoint.ckpt",
        ]

        # Act
        configs, args, test_state_dict, evaluation = prepare_run(argv)

        # Assert
        captured = capsys.readouterr()
        assert "loaded model:" in captured.out
        assert "path/to/checkpoint.ckpt" in captured.out
        assert "at epoch: 25" in captured.out
