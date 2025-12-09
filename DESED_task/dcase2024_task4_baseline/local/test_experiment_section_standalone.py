"""Standalone tests for experiment section processing without heavy dependencies."""

import sys
from pathlib import Path

import pytest

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from local.experiment_dir import ExperimentConfig, ExecutionMode


def process_experiment_section(configs: dict) -> dict:
    """Process experiment section in configs - copied for standalone testing."""
    if "experiment" not in configs:
        configs["experiment"] = {}

    exp_config = ExperimentConfig(**configs["experiment"])

    structure = f"{exp_config.mode.value}/{exp_config.category}/{exp_config.method}/{exp_config.variant}"
    print(f"Experiment structure: {structure}")

    return configs


@pytest.fixture
def mock_yaml_config():
    """Base YAML configuration without experiment section."""
    return {
        "data": {"fs": 16000},
        "training": {"seed": 42},
        "wandb": {"use_wandb": False, "wandb_dir": "None"},
    }


@pytest.fixture
def mock_yaml_config_with_experiment():
    """YAML configuration with experiment section."""
    return {
        "data": {"fs": 16000},
        "training": {"seed": 42},
        "wandb": {"use_wandb": False, "wandb_dir": "None"},
        "experiment": {
            "mode": "train",
            "category": "baseline",
            "method": "cmt",
            "variant": "v1",
            "base_dir": "experiments",
        },
    }


class TestExperimentSectionProcessing:
    """Test experiment section processing logic."""

    def test_experiment_section_missing_adds_empty_dict(self, mock_yaml_config):
        """Test that missing experiment section is initialized as empty dict."""
        result = process_experiment_section(mock_yaml_config)

        assert "experiment" in result
        assert isinstance(result["experiment"], dict)

    def test_experiment_section_with_defaults_creates_valid_config(self, mock_yaml_config):
        """Test that default ExperimentConfig can be created when experiment section is empty."""
        mock_yaml_config["experiment"] = {}

        result = process_experiment_section(mock_yaml_config)

        exp_config = ExperimentConfig(**result["experiment"])
        assert exp_config.mode == ExecutionMode.TRAIN
        assert exp_config.category == "default"
        assert exp_config.method == "baseline"
        assert exp_config.variant == "v1"

    def test_experiment_section_valid_config_logs_structure(
        self, mock_yaml_config_with_experiment, capsys
    ):
        """Test that valid experiment config logs the structure."""
        result = process_experiment_section(mock_yaml_config_with_experiment)

        assert "experiment" in result
        exp_config = ExperimentConfig(**result["experiment"])
        assert exp_config.mode == ExecutionMode.TRAIN
        assert exp_config.category == "baseline"
        assert exp_config.method == "cmt"
        assert exp_config.variant == "v1"

        captured = capsys.readouterr()
        assert "Experiment structure:" in captured.out
        assert "train/baseline/cmt/v1" in captured.out

    def test_experiment_section_with_invalid_mode_raises_error(self, mock_yaml_config):
        """Test that invalid mode in experiment section raises ValueError."""
        mock_yaml_config["experiment"] = {
            "mode": "invalid_mode",
            "category": "baseline",
            "method": "cmt",
            "variant": "v1",
        }

        with pytest.raises(ValueError, match="Invalid mode"):
            process_experiment_section(mock_yaml_config)

    def test_experiment_section_preserved_in_configs(self, mock_yaml_config_with_experiment):
        """Test that experiment section is preserved in returned configs."""
        result = process_experiment_section(mock_yaml_config_with_experiment)

        assert result["experiment"]["mode"] == "train"
        assert result["experiment"]["category"] == "baseline"
        assert result["experiment"]["method"] == "cmt"
        assert result["experiment"]["variant"] == "v1"
        assert result["experiment"]["base_dir"] == "experiments"

    def test_experiment_section_with_inference_mode(self, mock_yaml_config):
        """Test that inference mode is correctly processed."""
        mock_yaml_config["experiment"] = {
            "mode": "inference",
            "category": "evaluation",
            "method": "cmt",
            "variant": "v2",
        }

        result = process_experiment_section(mock_yaml_config)

        exp_config = ExperimentConfig(**result["experiment"])
        assert exp_config.mode == ExecutionMode.INFERENCE
        assert exp_config.category == "evaluation"
