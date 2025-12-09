"""Pytest configuration for test_train_pretrained.py."""

import sys
from unittest.mock import MagicMock

import pytest


@pytest.fixture(scope="session", autouse=True)
def mock_sebbs_imports():
    """Mock sebbs submodule imports to avoid dependency issues in tests."""
    # Mock sebbs change_detection module
    sys.modules["sebbs.change_detection"] = MagicMock()
    # Mock sebbs csebbs module
    sys.modules["sebbs.sebbs.csebbs"] = MagicMock()

    yield

    # Cleanup is not strictly necessary as tests run in isolated processes
    # but we can remove the mocks if needed
    # del sys.modules["sebbs.change_detection"]
    # del sys.modules["sebbs.sebbs.csebbs"]
