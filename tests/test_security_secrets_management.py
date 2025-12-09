"""
Security Test: Secrets Management Verification

Task 5.2: 機密情報管理を検証する

Test Coverage:
- Requirement 7.1: Non-root user (vscode) execution
- Requirement 7.2: Minimal privilege process execution
- Requirement 7.3: Secrets as environment variables (not hardcoded)
- Requirement 7.4: Sensitive files excluded from build context

This test suite verifies that the devcontainer configuration follows
security best practices for secrets management.
"""

import json
import re
from pathlib import Path

import pytest

# Project root directory
PROJECT_ROOT = Path(__file__).parent.parent
DEVCONTAINER_DIR = PROJECT_ROOT / ".devcontainer"


class TestDockerignore:
    """Test .dockerignore file for sensitive file exclusions (Req 7.4)"""

    def test_dockerignore_exists(self):
        """Test that .dockerignore file exists"""
        dockerignore_path = DEVCONTAINER_DIR / ".dockerignore"
        assert dockerignore_path.exists(), ".dockerignore file must exist"

    def test_sensitive_files_excluded(self):
        """Test that sensitive files are listed in .dockerignore"""
        dockerignore_path = DEVCONTAINER_DIR / ".dockerignore"
        dockerignore_content = dockerignore_path.read_text()

        # List of sensitive files/patterns that MUST be excluded (Req 7.4)
        required_exclusions = [
            ".env",
            "credentials.json",
            ".ssh/",
            ".aws/",
        ]

        for exclusion in required_exclusions:
            assert (
                exclusion in dockerignore_content
            ), f"Sensitive file pattern '{exclusion}' must be in .dockerignore (Req 7.4)"

    def test_private_key_patterns_excluded(self):
        """Test that private key patterns are excluded"""
        dockerignore_path = DEVCONTAINER_DIR / ".dockerignore"
        dockerignore_content = dockerignore_path.read_text()

        # Private key extensions
        key_patterns = ["*.key", "*.pem"]

        for pattern in key_patterns:
            assert (
                pattern in dockerignore_content
            ), f"Private key pattern '{pattern}' must be in .dockerignore (Req 7.4)"


class TestDockerfile:
    """Test Dockerfile for security violations (Req 7.3)"""

    def test_dockerfile_exists(self):
        """Test that Dockerfile exists"""
        dockerfile_path = DEVCONTAINER_DIR / "Dockerfile"
        assert dockerfile_path.exists(), "Dockerfile must exist"

    def test_no_hardcoded_secrets_in_env(self):
        """Test that Dockerfile does not contain hardcoded secrets via ENV"""
        dockerfile_path = DEVCONTAINER_DIR / "Dockerfile"
        dockerfile_content = dockerfile_path.read_text()

        # Patterns that indicate potential hardcoded secrets (Req 7.3)
        # These patterns should NOT appear in the Dockerfile
        forbidden_patterns = [
            r"ENV\s+WANDB_API_KEY\s*=\s*['\"]?\w+",  # ENV WANDB_API_KEY=xxx
            r"ENV\s+.*API_KEY\s*=\s*['\"]?\w+",  # ENV *_API_KEY=xxx
            r"ENV\s+.*SECRET\s*=\s*['\"]?\w+",  # ENV *_SECRET=xxx
            r"ENV\s+.*PASSWORD\s*=\s*['\"]?\w+",  # ENV *_PASSWORD=xxx
            r"ENV\s+.*TOKEN\s*=\s*['\"]?\w+",  # ENV *_TOKEN=xxx
        ]

        for pattern in forbidden_patterns:
            matches = re.findall(pattern, dockerfile_content, re.IGNORECASE)
            assert (
                not matches
            ), f"Dockerfile must not contain hardcoded secrets matching pattern '{pattern}' (Req 7.3). Found: {matches}"

    def test_non_root_user_configured(self):
        """Test that Dockerfile configures non-root user (Req 7.1, 7.2)"""
        dockerfile_path = DEVCONTAINER_DIR / "Dockerfile"
        dockerfile_content = dockerfile_path.read_text()

        # Must create 'vscode' user (Req 7.1)
        assert (
            "useradd" in dockerfile_content and "vscode" in dockerfile_content
        ), "Dockerfile must create non-root user 'vscode' (Req 7.1)"

        # Must switch to non-root user (Req 7.2)
        assert (
            "USER vscode" in dockerfile_content
        ), "Dockerfile must switch to 'vscode' user for minimal privileges (Req 7.2)"


class TestDevcontainerJson:
    """Test devcontainer.json for security configuration (Req 7.1, 7.3)"""

    def test_devcontainer_json_exists(self):
        """Test that devcontainer.json exists"""
        devcontainer_path = DEVCONTAINER_DIR / "devcontainer.json"
        assert devcontainer_path.exists(), "devcontainer.json must exist"

    def test_remote_user_is_non_root(self):
        """Test that remoteUser is set to non-root 'vscode' (Req 7.1)"""
        devcontainer_path = DEVCONTAINER_DIR / "devcontainer.json"
        devcontainer_content = devcontainer_path.read_text()

        # Parse JSON (may have comments, so strip them first)
        json_content = re.sub(r"//.*", "", devcontainer_content)
        config = json.loads(json_content)

        assert (
            "remoteUser" in config
        ), "devcontainer.json must specify 'remoteUser' (Req 7.1)"
        assert (
            config["remoteUser"] == "vscode"
        ), "remoteUser must be 'vscode' (non-root) (Req 7.1)"

    def test_wandb_api_key_is_commented_optional(self):
        """Test that WANDB_API_KEY is commented out (optional feature) (Req 7.3)"""
        devcontainer_path = DEVCONTAINER_DIR / "devcontainer.json"
        devcontainer_content = devcontainer_path.read_text()

        # Check that WANDB_API_KEY line is commented (starts with //)
        wandb_lines = [
            line for line in devcontainer_content.split("\n") if "WANDB_API_KEY" in line
        ]

        if wandb_lines:
            # If WANDB_API_KEY exists, it should be commented or a placeholder
            for line in wandb_lines:
                stripped = line.strip()
                if '"WANDB_API_KEY"' in stripped:
                    # If it's not a comment, it must be in remoteEnv and use ${localEnv:...}
                    assert (
                        stripped.startswith("//") or "${localEnv:" in line
                    ), f"WANDB_API_KEY must be commented or use environment variable reference (Req 7.3). Found: {line.strip()}"

    def test_no_hardcoded_secrets_in_remote_env(self):
        """Test that remoteEnv does not contain hardcoded secrets (Req 7.3)"""
        devcontainer_path = DEVCONTAINER_DIR / "devcontainer.json"
        devcontainer_content = devcontainer_path.read_text()

        # Parse JSON (strip comments first)
        json_content = re.sub(r"//.*", "", devcontainer_content)
        config = json.loads(json_content)

        if "remoteEnv" in config:
            remote_env = config["remoteEnv"]
            # Check that no environment variables contain hardcoded secrets
            # Secrets should use ${localEnv:...} pattern
            for key, value in remote_env.items():
                if any(
                    keyword in key.upper()
                    for keyword in ["API_KEY", "SECRET", "PASSWORD", "TOKEN"]
                ):
                    assert isinstance(
                        value, str
                    ), f"Environment variable {key} value must be a string"
                    assert (
                        "${localEnv:" in value or value == ""
                    ), f"Secret environment variable '{key}' must use ${{localEnv:...}} pattern, not hardcoded value (Req 7.3)"


class TestSecurityDocumentation:
    """Test that security best practices are documented"""

    def test_devcontainer_guide_exists(self):
        """Test that DEVCONTAINER_GUIDE.md exists with security documentation"""
        guide_path = (
            PROJECT_ROOT / "DESED_task" / "dcase2024_task4_baseline" / "DEVCONTAINER_GUIDE.md"
        )
        assert guide_path.exists(), "DEVCONTAINER_GUIDE.md must exist"

        guide_content = guide_path.read_text()

        # Should document security practices
        security_keywords = [
            "secret",  # Secrets management section
            "環境変数",  # Environment variables (Japanese)
            "API",  # API key management
        ]

        # At least one security-related section should exist
        found_keywords = [kw for kw in security_keywords if kw in guide_content.lower()]
        assert (
            found_keywords
        ), f"DEVCONTAINER_GUIDE.md should document security practices. Expected keywords: {security_keywords}"


# Pytest configuration
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
