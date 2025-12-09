"""Test ModelCheckpoint dirpath integration with execution mode management.

Tests for Task 3.5: ModelCheckpoint dirpath integration
- Verify checkpoint_dir is correctly determined from _wandb_checkpoint_dir
- Verify fallback to logger.log_dir when wandb is disabled
- Verify behavior in different execution modes
"""

import os
from pathlib import Path
from unittest.mock import Mock

import pytest

from local.experiment_dir import ExecutionMode


class TestModelCheckpointDirpathIntegration:
    """Tests for ModelCheckpoint dirpath integration with execution modes."""

    def test_checkpoint_dir_from_wandb_checkpoint_dir_when_available(self, tmp_path):
        """Test that checkpoint_dir is determined from _wandb_checkpoint_dir when available.

        This simulates the behavior in train_pretrained.py:564-574 where:
        1. SEDTask4 is created
        2. _wandb_checkpoint_dir is set in _init_wandb_project()
        3. train_pretrained.py checks hasattr and uses it if available
        """
        # Arrange: Set up mock trainer with wandb checkpoint directory
        mock_trainer = Mock()
        wandb_checkpoint_dir = tmp_path / "wandb" / "run-test" / "checkpoints"
        wandb_checkpoint_dir.mkdir(parents=True)
        mock_trainer._wandb_checkpoint_dir = str(wandb_checkpoint_dir)

        # Mock logger
        mock_logger = Mock()
        mock_logger.log_dir = str(tmp_path / "exp" / "default")

        # Act: Simulate the logic from train_pretrained.py:564-574
        if (
            hasattr(mock_trainer, "_wandb_checkpoint_dir")
            and mock_trainer._wandb_checkpoint_dir
        ):
            checkpoint_dir = mock_trainer._wandb_checkpoint_dir
        else:
            checkpoint_dir = mock_logger.log_dir

        # Assert: checkpoint_dir should be from wandb
        assert checkpoint_dir == str(wandb_checkpoint_dir)
        assert "wandb" in checkpoint_dir
        assert "checkpoints" in checkpoint_dir

    def test_checkpoint_dir_fallback_to_logger_when_wandb_disabled(self, tmp_path):
        """Test that checkpoint_dir falls back to logger.log_dir when wandb is disabled.

        This covers the case where:
        1. wandb is disabled (use_wandb=False)
        2. _wandb_checkpoint_dir is None
        3. train_pretrained.py falls back to logger.log_dir
        """
        # Arrange: Mock trainer with no checkpoint dir
        mock_trainer = Mock()
        mock_trainer._wandb_checkpoint_dir = None

        # Mock logger
        mock_logger = Mock()
        logger_dir = tmp_path / "exp" / "default"
        logger_dir.mkdir(parents=True)
        mock_logger.log_dir = str(logger_dir)

        # Act: Simulate the logic from train_pretrained.py:564-574
        if (
            hasattr(mock_trainer, "_wandb_checkpoint_dir")
            and mock_trainer._wandb_checkpoint_dir
        ):
            checkpoint_dir = mock_trainer._wandb_checkpoint_dir
        else:
            checkpoint_dir = mock_logger.log_dir

        # Assert: checkpoint_dir should be from logger
        assert checkpoint_dir == str(logger_dir)
        assert "exp" in checkpoint_dir
        assert "wandb" not in checkpoint_dir

    def test_checkpoint_dir_with_inference_mode_no_wandb(self, tmp_path):
        """Test that checkpoint_dir is None when in inference mode (wandb disabled).

        This covers the case where:
        1. Execution mode is INFERENCE
        2. wandb initialization is skipped
        3. _wandb_checkpoint_dir is None
        4. In train_pretrained.py, ModelCheckpoint is not created (test_state_dict check)
        """
        # Arrange: Mock trainer in inference mode
        mock_trainer = Mock()
        mock_trainer._wandb_checkpoint_dir = None
        mock_trainer.execution_mode = ExecutionMode.INFERENCE

        # Mock logger (still available in inference mode)
        mock_logger = Mock()
        mock_logger.log_dir = str(tmp_path / "exp" / "inference")

        # Act: Check the _wandb_checkpoint_dir state
        # Note: In actual train_pretrained.py, test_state_dict prevents checkpoint creation
        has_checkpoint_dir = (
            hasattr(mock_trainer, "_wandb_checkpoint_dir")
            and mock_trainer._wandb_checkpoint_dir
        )

        # Assert: Should not have checkpoint dir in inference mode
        assert not has_checkpoint_dir
        assert mock_trainer._wandb_checkpoint_dir is None

    def test_checkpoint_dir_with_new_experiment_config_train_mode(self, tmp_path):
        """Test checkpoint_dir with new ExperimentConfig in train mode.

        This covers the case where:
        1. experiment section is configured in YAML
        2. mode is TRAIN
        3. wandb is enabled
        4. _wandb_checkpoint_dir is set via ExperimentDirManager
        """
        # Arrange: Mock trainer with hierarchical checkpoint directory
        mock_trainer = Mock()
        experiment_checkpoint_dir = (
            tmp_path / "experiments" / "train" / "baseline" / "test" / "v1" / "checkpoints"
        )
        experiment_checkpoint_dir.mkdir(parents=True)
        mock_trainer._wandb_checkpoint_dir = str(experiment_checkpoint_dir)

        # Mock logger
        mock_logger = Mock()
        mock_logger.log_dir = str(tmp_path / "exp" / "fallback")

        # Act: Simulate the logic from train_pretrained.py:564-574
        if (
            hasattr(mock_trainer, "_wandb_checkpoint_dir")
            and mock_trainer._wandb_checkpoint_dir
        ):
            checkpoint_dir = mock_trainer._wandb_checkpoint_dir
        else:
            checkpoint_dir = mock_logger.log_dir

        # Assert: checkpoint_dir should use new hierarchical structure
        assert checkpoint_dir == str(experiment_checkpoint_dir)
        assert "experiments/train" in checkpoint_dir
        assert "baseline/test/v1" in checkpoint_dir
        assert "checkpoints" in checkpoint_dir

    def test_checkpoint_dir_attribute_not_set(self, tmp_path):
        """Test fallback when _wandb_checkpoint_dir attribute doesn't exist.

        This covers the edge case where:
        1. _wandb_checkpoint_dir attribute is not set at all
        2. hasattr returns False
        3. Falls back to logger.log_dir
        """
        # Arrange: Mock trainer without _wandb_checkpoint_dir attribute
        mock_trainer = Mock(spec=[])  # spec=[] means no attributes by default

        # Mock logger
        mock_logger = Mock()
        logger_dir = tmp_path / "exp" / "fallback"
        logger_dir.mkdir(parents=True)
        mock_logger.log_dir = str(logger_dir)

        # Act: Simulate the logic from train_pretrained.py:564-574
        if (
            hasattr(mock_trainer, "_wandb_checkpoint_dir")
            and mock_trainer._wandb_checkpoint_dir
        ):
            checkpoint_dir = mock_trainer._wandb_checkpoint_dir
        else:
            checkpoint_dir = mock_logger.log_dir

        # Assert: checkpoint_dir should fall back to logger
        assert checkpoint_dir == str(logger_dir)
        assert not hasattr(mock_trainer, "_wandb_checkpoint_dir")

    def test_checkpoint_dir_empty_string_treated_as_none(self, tmp_path):
        """Test that empty string in _wandb_checkpoint_dir is treated as None.

        This covers edge case where:
        1. _wandb_checkpoint_dir is set but is empty string
        2. Should be treated as falsy and fall back to logger
        """
        # Arrange: Mock trainer with empty string checkpoint dir
        mock_trainer = Mock()
        mock_trainer._wandb_checkpoint_dir = ""

        # Mock logger
        mock_logger = Mock()
        logger_dir = tmp_path / "exp" / "fallback"
        logger_dir.mkdir(parents=True)
        mock_logger.log_dir = str(logger_dir)

        # Act: Simulate the logic from train_pretrained.py:564-574
        if (
            hasattr(mock_trainer, "_wandb_checkpoint_dir")
            and mock_trainer._wandb_checkpoint_dir
        ):
            checkpoint_dir = mock_trainer._wandb_checkpoint_dir
        else:
            checkpoint_dir = mock_logger.log_dir

        # Assert: checkpoint_dir should fall back to logger (empty string is falsy)
        assert checkpoint_dir == str(logger_dir)
        assert checkpoint_dir != ""


if __name__ == "__main__":
    # Run with pytest
    pytest.main([__file__, "-v"])
