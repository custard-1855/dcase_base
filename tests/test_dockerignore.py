"""Tests for .dockerignore file configuration.

Tests verify that the .dockerignore file:
- Exists in the .devcontainer directory
- Excludes sensitive files (.env, credentials.json, .ssh/, .aws/)
- Excludes large data directories (data/, embeddings/, exp/, wandb/)
- Excludes unnecessary build artifacts (.git/, __pycache__, *.pyc, .DS_Store)
"""

import pytest
from pathlib import Path


class TestDockerignore:
    """Test suite for .dockerignore file."""

    @pytest.fixture
    def dockerignore_path(self) -> Path:
        """Return the path to .dockerignore file."""
        return Path(__file__).parent.parent / ".devcontainer" / ".dockerignore"

    def test_dockerignore_exists(self, dockerignore_path: Path) -> None:
        """Test that .dockerignore file exists."""
        assert dockerignore_path.exists(), f".dockerignore not found at {dockerignore_path}"

    def test_dockerignore_excludes_sensitive_files(self, dockerignore_path: Path) -> None:
        """Test that sensitive files are excluded from Docker build context."""
        content = dockerignore_path.read_text()

        # Sensitive file patterns
        sensitive_patterns = [
            ".env",
            "credentials.json",
            ".ssh/",
            ".aws/",
        ]

        for pattern in sensitive_patterns:
            assert pattern in content, f"Sensitive pattern '{pattern}' not found in .dockerignore"

    def test_dockerignore_excludes_large_data_directories(self, dockerignore_path: Path) -> None:
        """Test that large data directories are excluded (managed by volume mounts)."""
        content = dockerignore_path.read_text()

        # Large data directory patterns
        data_patterns = [
            "data/",
            "embeddings/",
            "exp/",
            "wandb/",
        ]

        for pattern in data_patterns:
            assert pattern in content, f"Data directory pattern '{pattern}' not found in .dockerignore"

    def test_dockerignore_excludes_build_artifacts(self, dockerignore_path: Path) -> None:
        """Test that unnecessary build artifacts are excluded."""
        content = dockerignore_path.read_text()

        # Build artifact patterns
        artifact_patterns = [
            ".git/",
            "__pycache__/",
            "*.pyc",
            ".DS_Store",
        ]

        for pattern in artifact_patterns:
            assert pattern in content, f"Artifact pattern '{pattern}' not found in .dockerignore"

    def test_dockerignore_includes_comments(self, dockerignore_path: Path) -> None:
        """Test that .dockerignore includes explanatory comments."""
        content = dockerignore_path.read_text()

        # At least one comment line should exist
        assert "#" in content, ".dockerignore should include explanatory comments"
