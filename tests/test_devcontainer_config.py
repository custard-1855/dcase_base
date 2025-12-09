"""
Test suite for devcontainer configuration validation.

Tests verify that devcontainer.json meets all requirements from spec.
"""

import json
import re
from pathlib import Path

import pytest


class TestDevcontainerConfig:
    """Test cases for .devcontainer/devcontainer.json validation."""

    @staticmethod
    def _remove_json_comments(json_str: str) -> str:
        """Remove single-line and multi-line comments from JSON string (JSONC support)."""
        # Remove single-line comments (// ...)
        json_str = re.sub(r"//.*?$", "", json_str, flags=re.MULTILINE)
        # Remove multi-line comments (/* ... */)
        json_str = re.sub(r"/\*.*?\*/", "", json_str, flags=re.DOTALL)
        return json_str

    @pytest.fixture
    def devcontainer_config(self) -> dict:
        """Load devcontainer.json configuration (supports JSONC - JSON with Comments)."""
        config_path = Path(".devcontainer/devcontainer.json")
        if not config_path.exists():
            pytest.fail(f"devcontainer.json not found at {config_path}")

        with open(config_path) as f:
            json_content = f.read()
            # Remove comments before parsing
            json_content = self._remove_json_comments(json_content)
            return json.loads(json_content)

    def test_devcontainer_file_exists(self):
        """Requirement 1.1: .devcontainer/devcontainer.json must exist."""
        config_path = Path(".devcontainer/devcontainer.json")
        assert config_path.exists(), "devcontainer.json file must exist"

    def test_basic_structure(self, devcontainer_config: dict):
        """Verify basic structure of devcontainer.json."""
        assert "name" in devcontainer_config
        assert "build" in devcontainer_config
        assert devcontainer_config["name"] == "DCASE 2024 Task 4 Dev Container (CPU)"

    def test_dockerfile_reference(self, devcontainer_config: dict):
        """Requirement 1.2: devcontainer.json must reference Dockerfile."""
        assert "build" in devcontainer_config
        assert "dockerfile" in devcontainer_config["build"]
        assert devcontainer_config["build"]["dockerfile"] == "Dockerfile"

    def test_claude_code_feature(self, devcontainer_config: dict):
        """Requirement 1.6: Claude Code feature must be integrated."""
        assert "features" in devcontainer_config
        features = devcontainer_config["features"]
        assert "ghcr.io/anthropics/devcontainer-features/claude-code:1" in features

    def test_vscode_extensions(self, devcontainer_config: dict):
        """Requirement 1.4: VS Code extensions (Python, Ruff, MyPy) must be configured."""
        assert "customizations" in devcontainer_config
        assert "vscode" in devcontainer_config["customizations"]
        vscode_config = devcontainer_config["customizations"]["vscode"]

        assert "extensions" in vscode_config
        extensions = vscode_config["extensions"]

        required_extensions = [
            "ms-python.python",
            "charliermarsh.ruff",
            "ms-python.mypy-type-checker",
            "ms-python.vscode-pylance"
        ]

        for ext in required_extensions:
            assert ext in extensions, f"Extension {ext} must be configured"

    def test_port_forwarding(self, devcontainer_config: dict):
        """Requirement 1.5: Port forwarding for TensorBoard and Optuna Dashboard must be configured."""
        assert "forwardPorts" in devcontainer_config
        ports = devcontainer_config["forwardPorts"]

        assert 6006 in ports, "TensorBoard port 6006 must be forwarded"
        assert 8080 in ports, "Optuna Dashboard port 8080 must be forwarded"

    def test_remote_user(self, devcontainer_config: dict):
        """Requirement 1.7: Container user must be set to 'vscode'."""
        assert "remoteUser" in devcontainer_config
        assert devcontainer_config["remoteUser"] == "vscode"

    def test_claude_config_env(self, devcontainer_config: dict):
        """Requirement 1.8: Environment variable CLAUDE_CONFIG_DIR must be set."""
        assert "remoteEnv" in devcontainer_config
        remote_env = devcontainer_config["remoteEnv"]

        assert "CLAUDE_CONFIG_DIR" in remote_env
        assert remote_env["CLAUDE_CONFIG_DIR"] == "/home/vscode/.claude"

    def test_named_volumes(self, devcontainer_config: dict):
        """Requirements 6.1-6.5: Named volumes for data, embeddings, exp, wandb, claude-config must be configured."""
        assert "mounts" in devcontainer_config
        mounts = devcontainer_config["mounts"]

        # Extract mount configurations as strings or dicts
        mount_targets = []
        for mount in mounts:
            if isinstance(mount, str):
                # Parse string format: "source=...,target=...,type=..."
                parts = dict(part.split("=", 1) for part in mount.split(","))
                mount_targets.append(parts.get("target", ""))
            elif isinstance(mount, dict):
                mount_targets.append(mount.get("target", ""))

        required_mounts = {
            "/workspace/data": "data volume",
            "/workspace/embeddings": "embeddings volume",
            "/workspace/exp": "exp volume",
            "/workspace/wandb": "wandb volume",
            "/home/vscode/.claude": "claude-code-config volume"
        }

        for target, description in required_mounts.items():
            assert target in mount_targets, f"{description} must be mounted at {target}"

    def test_postcreate_command(self, devcontainer_config: dict):
        """Verify postCreateCommand includes all required initialization steps."""
        assert "postCreateCommand" in devcontainer_config
        post_create = devcontainer_config["postCreateCommand"]

        # Check for required commands in postCreateCommand
        required_commands = [
            "uv sync",
            "gh auth setup-git",
            "git submodule update --init --recursive",
            "pre-commit install"
        ]

        for cmd in required_commands:
            assert cmd in post_create, f"postCreateCommand must include: {cmd}"

    def test_vscode_settings(self, devcontainer_config: dict):
        """Requirements 5.4-5.5: VS Code editor settings for auto-format and import organization."""
        vscode_config = devcontainer_config["customizations"]["vscode"]
        assert "settings" in vscode_config
        settings = vscode_config["settings"]

        # Check format on save
        assert settings.get("editor.formatOnSave") is True

        # Check code actions on save (import organization)
        assert "editor.codeActionsOnSave" in settings
        code_actions = settings["editor.codeActionsOnSave"]
        assert code_actions.get("source.organizeImports") is True

        # Check Python interpreter path
        assert "python.defaultInterpreterPath" in settings
        assert settings["python.defaultInterpreterPath"] == "/workspace/.venv/bin/python"

        # Check Python formatter
        assert "[python]" in settings
        assert settings["[python]"].get("editor.defaultFormatter") == "charliermarsh.ruff"
