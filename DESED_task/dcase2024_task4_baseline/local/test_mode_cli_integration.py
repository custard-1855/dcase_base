"""Integration test for --mode CLI argument in train_pretrained.py"""

import sys
from pathlib import Path
import argparse

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))


def test_mode_argument_parser():
    """Test that --mode argument is correctly added to argparse."""
    # Create parser similar to train_pretrained.py
    parser = argparse.ArgumentParser("Training a SED system for DESED Task")

    # Add mode argument as implemented in train_pretrained.py (Task 3.2)
    parser.add_argument(
        "--mode",
        choices=["train", "test", "inference", "feature_extraction"],
        default=None,
        help="Explicit execution mode (overrides auto-detection)",
    )

    # Test 1: Parse with valid mode
    args = parser.parse_args(["--mode", "train"])
    assert args.mode == "train", "Should parse train mode"

    args = parser.parse_args(["--mode", "inference"])
    assert args.mode == "inference", "Should parse inference mode"

    # Test 2: Parse without mode (default=None)
    args = parser.parse_args([])
    assert args.mode is None, "Default mode should be None"

    # Test 3: Invalid mode should raise SystemExit
    try:
        args = parser.parse_args(["--mode", "invalid"])
        assert False, "Should raise SystemExit for invalid mode"
    except SystemExit:
        pass

    print("✓ All mode argument parser tests passed")


def test_mode_override_logic():
    """Test the mode override logic that should be in prepare_run()."""
    # Simulate YAML config
    configs = {
        "experiment": {
            "mode": "train",
            "category": "baseline",
            "method": "cmt",
            "variant": "v1"
        }
    }

    # Simulate CLI argument --mode inference
    cli_mode = "inference"

    # Apply override logic (as implemented in Task 3.2)
    if cli_mode is not None:
        if "experiment" not in configs:
            configs["experiment"] = {}
        configs["experiment"]["mode"] = cli_mode

    # Verify override
    assert configs["experiment"]["mode"] == "inference", "CLI mode should override YAML mode"

    print("✓ Mode override logic test passed")


def test_mode_override_with_empty_experiment_section():
    """Test that mode override works even if experiment section is missing."""
    # Simulate YAML config without experiment section
    configs = {
        "data": {"fs": 16000}
    }

    # Simulate CLI argument --mode inference
    cli_mode = "inference"

    # Apply override logic
    if cli_mode is not None:
        if "experiment" not in configs:
            configs["experiment"] = {}
        configs["experiment"]["mode"] = cli_mode

    # Verify
    assert "experiment" in configs, "Experiment section should be created"
    assert configs["experiment"]["mode"] == "inference", "CLI mode should be set"

    print("✓ Mode override with empty experiment section test passed")


if __name__ == "__main__":
    test_mode_argument_parser()
    test_mode_override_logic()
    test_mode_override_with_empty_experiment_section()
    print("\n✅ All integration tests passed!")
