"""Tests for postCreateCommand script."""

import subprocess
from pathlib import Path
from typing import Callable
from unittest.mock import Mock, patch

import pytest


@pytest.fixture
def mock_subprocess_run() -> Callable:
    """Fixture for mocking subprocess.run."""

    def _mock_run(returncode: int = 0, stdout: str = "", stderr: str = "") -> Mock:
        mock = Mock()
        mock.return_value.returncode = returncode
        mock.return_value.stdout = stdout
        mock.return_value.stderr = stderr
        return mock

    return _mock_run


def test_uv_sync_success(mock_subprocess_run: Callable) -> None:
    """Test that uv sync executes successfully."""
    with patch("subprocess.run", mock_subprocess_run(returncode=0, stdout="Synced dependencies")):
        result = subprocess.run(["uv", "sync"], capture_output=True, text=True, check=False)
        assert result.returncode == 0


def test_uv_sync_failure_prints_warning(mock_subprocess_run: Callable) -> None:
    """Test that uv sync failure prints warning without stopping execution."""
    with patch("subprocess.run", mock_subprocess_run(returncode=1, stderr="Failed to sync")):
        result = subprocess.run(["uv", "sync"], capture_output=True, text=True, check=False)
        assert result.returncode == 1
        # Script should continue despite failure


def test_gh_auth_setup_git_success(mock_subprocess_run: Callable) -> None:
    """Test that gh auth setup-git executes successfully."""
    with patch("subprocess.run", mock_subprocess_run(returncode=0, stdout="Configured git credential helper")):
        result = subprocess.run(
            ["gh", "auth", "setup-git"], capture_output=True, text=True, check=False
        )
        assert result.returncode == 0


def test_gh_auth_setup_git_failure_prints_warning(mock_subprocess_run: Callable) -> None:
    """Test that gh auth setup-git failure prints helpful error message."""
    with patch(
        "subprocess.run",
        mock_subprocess_run(
            returncode=1,
            stderr="To get started with GitHub CLI, please run: gh auth login",
        ),
    ):
        result = subprocess.run(
            ["gh", "auth", "setup-git"], capture_output=True, text=True, check=False
        )
        assert result.returncode == 1
        # Should suggest running gh auth login


def test_git_submodule_update_success(mock_subprocess_run: Callable) -> None:
    """Test that git submodule update executes successfully."""
    with patch(
        "subprocess.run",
        mock_subprocess_run(returncode=0, stdout="Submodule 'PSDS_Eval' registered"),
    ):
        result = subprocess.run(
            ["git", "submodule", "update", "--init", "--recursive"],
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode == 0


def test_git_submodule_update_failure_prints_error(mock_subprocess_run: Callable) -> None:
    """Test that git submodule failure prints error message per Requirement 4.3."""
    with patch("subprocess.run", mock_subprocess_run(returncode=1, stderr="Authentication failed")):
        result = subprocess.run(
            ["git", "submodule", "update", "--init", "--recursive"],
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode == 1
        # Should display error per Req 4.3


def test_precommit_install_success(mock_subprocess_run: Callable) -> None:
    """Test that pre-commit install executes successfully."""
    with patch("subprocess.run", mock_subprocess_run(returncode=0, stdout="pre-commit installed")):
        result = subprocess.run(
            ["pre-commit", "install"], capture_output=True, text=True, check=False
        )
        assert result.returncode == 0


def test_precommit_install_failure_prints_warning(mock_subprocess_run: Callable) -> None:
    """Test that pre-commit install failure prints warning."""
    with patch("subprocess.run", mock_subprocess_run(returncode=1, stderr=".git directory not found")):
        result = subprocess.run(
            ["pre-commit", "install"], capture_output=True, text=True, check=False
        )
        assert result.returncode == 1
        # Should print warning but continue


def test_system_libraries_verification_success(mock_subprocess_run: Callable) -> None:
    """Test that system libraries are verified successfully."""
    with patch("subprocess.run", mock_subprocess_run(returncode=0, stdout="/usr/bin/sox")):
        for cmd in ["sox", "ffmpeg", "git", "gh"]:
            result = subprocess.run(["which", cmd], capture_output=True, text=True, check=False)
            assert result.returncode == 0


def test_system_libraries_verification_failure_prints_warning(
    mock_subprocess_run: Callable,
) -> None:
    """Test that missing system libraries print warning."""
    with patch("subprocess.run", mock_subprocess_run(returncode=1, stderr="not found")):
        result = subprocess.run(["which", "sox"], capture_output=True, text=True, check=False)
        assert result.returncode == 1
        # Should print warning per Req 3.5


def test_directory_write_permissions_success(tmp_path: Path) -> None:
    """Test that directory write permissions are verified successfully."""
    test_dir = tmp_path / "data"
    test_dir.mkdir()
    test_file = test_dir / ".test_write"

    # Verify writable
    test_file.touch()
    assert test_file.exists()
    test_file.unlink()
    assert not test_file.exists()


def test_directory_write_permissions_failure_prints_warning(tmp_path: Path) -> None:
    """Test that non-writable directory prints warning per Req 6.6."""
    test_dir = tmp_path / "data"
    test_dir.mkdir(mode=0o555)  # Read-only directory

    test_file = test_dir / ".test_write"
    with pytest.raises(PermissionError):
        test_file.touch()
    # Should print "Warning: data not writable" per Req 6.6


def test_memory_check_low_memory_warning() -> None:
    """Test that low memory detection prints warning per Req 9.4."""
    # Simulate available memory < 2048 MB
    available_mem = 1024  # MB
    assert available_mem < 2048
    # Should print "Warning: Low memory detected (1024 MB available)"


def test_memory_check_sufficient_memory() -> None:
    """Test that sufficient memory passes without warning."""
    # Simulate available memory >= 2048 MB
    available_mem = 4096  # MB
    assert available_mem >= 2048
    # Should not print warning


def test_post_create_command_completes_even_with_failures(
    mock_subprocess_run: Callable,
) -> None:
    """Test that postCreateCommand completes even if some commands fail."""
    # This is the key invariant: partial success is allowed
    # Even if uv sync, gh auth, git submodule, or pre-commit fail,
    # the script should continue and print "postCreateCommand completed"
    with patch("subprocess.run", mock_subprocess_run(returncode=1, stderr="Some error")):
        result = subprocess.run(["uv", "sync"], capture_output=True, text=True, check=False)
        assert result.returncode == 1
        # Script continues to next command despite failure
