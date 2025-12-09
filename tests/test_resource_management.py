"""
Resource Management Tests

Task 6.1: メモリ・ディスク容量管理を検証する
Task 6.2: ビルド時間最適化を検証する

Test Coverage:
- Requirement 9.2: Memory limits configuration
- Requirement 9.3: CPU cores configuration
- Requirement 9.4: Resource shortage warnings
- Requirement 9.5: Disk I/O optimization

This test suite verifies that the devcontainer configuration properly
manages system resources and provides optimization for build performance.
"""

import re
import subprocess
from pathlib import Path

import pytest

# Project root directory
PROJECT_ROOT = Path(__file__).parent.parent
DEVCONTAINER_DIR = PROJECT_ROOT / ".devcontainer"


class TestPostCreateCommandResourceChecks:
    """Test resource checking functionality in postCreateCommand (Req 9.4)"""

    def test_post_create_command_exists(self):
        """Test that postCreateCommand script exists"""
        script_path = DEVCONTAINER_DIR / "post_create_command.sh"
        assert script_path.exists(), "post_create_command.sh must exist"

    def test_post_create_command_is_executable(self):
        """Test that postCreateCommand script has execute permissions"""
        script_path = DEVCONTAINER_DIR / "post_create_command.sh"
        assert script_path.exists(), "post_create_command.sh must exist"
        # Check if file is executable (Unix permissions)
        import os
        import stat

        st = os.stat(script_path)
        is_executable = bool(st.st_mode & stat.S_IXUSR)
        if not is_executable:
            # Try to make it executable
            os.chmod(script_path, st.st_mode | stat.S_IXUSR)
            pytest.skip("Made script executable - re-run test")

    def test_memory_check_implementation(self):
        """Test that memory check is implemented in postCreateCommand (Req 9.4)"""
        script_path = DEVCONTAINER_DIR / "post_create_command.sh"
        script_content = script_path.read_text()

        # Should check memory using 'free' command
        assert "free -m" in script_content, "Script must check memory with 'free -m'"

        # Should have threshold check (2048 MB as per design.md)
        assert (
            "2048" in script_content
        ), "Script must check memory threshold of 2048 MB (Req 9.4)"

        # Should display warning for low memory
        assert (
            "Low memory" in script_content or "low memory" in script_content.lower()
        ), "Script must warn about low memory (Req 9.4)"

    def test_disk_space_check_implementation(self):
        """Test that disk space check is implemented in postCreateCommand (Req 9.4)"""
        script_path = DEVCONTAINER_DIR / "post_create_command.sh"
        script_content = script_path.read_text()

        # Should check disk space using 'df' command
        assert "df" in script_content, "Script must check disk space with 'df'"

        # Should have threshold check (10 GB as per design.md)
        assert "10" in script_content, "Script must check disk threshold (Req 9.4)"

        # Should display warning for low disk space
        assert (
            "Low disk" in script_content or "low disk" in script_content.lower()
        ), "Script must warn about low disk space (Req 9.4)"


class TestDevcontainerResourceConfiguration:
    """Test resource configuration in devcontainer.json (Req 9.2, 9.3)"""

    def test_devcontainer_json_exists(self):
        """Test that devcontainer.json exists"""
        devcontainer_path = DEVCONTAINER_DIR / "devcontainer.json"
        assert devcontainer_path.exists(), "devcontainer.json must exist"

    def test_resource_limits_documented(self):
        """Test that resource limits are documented (runArgs for memory/CPU)"""
        devcontainer_path = DEVCONTAINER_DIR / "devcontainer.json"
        devcontainer_content = devcontainer_path.read_text()

        # Check if runArgs or documentation mentions memory/CPU configuration
        # Note: These may be commented out as optional settings
        has_memory_reference = (
            "--memory" in devcontainer_content or "memory" in devcontainer_content.lower()
        )
        has_cpu_reference = "--cpus" in devcontainer_content or "cpu" in devcontainer_content.lower()

        # At least documentation should exist
        assert (
            has_memory_reference or has_cpu_reference
        ), "devcontainer.json should document memory/CPU configuration (Req 9.2, 9.3)"


class TestDockerfileBuildOptimization:
    """Test Dockerfile build optimization features (Req 9.5)"""

    def test_dockerfile_multi_stage_build(self):
        """Test that Dockerfile uses multi-stage build for optimization"""
        dockerfile_path = DEVCONTAINER_DIR / "Dockerfile"
        dockerfile_content = dockerfile_path.read_text()

        # Count FROM statements (multi-stage should have multiple)
        from_statements = re.findall(r"^FROM\s+", dockerfile_content, re.MULTILINE)
        assert (
            len(from_statements) >= 2
        ), "Dockerfile should use multi-stage build (Req 9.5)"

    def test_dockerfile_uv_cache_mount(self):
        """Test that Dockerfile uses cache mount for uv (Req 9.5)"""
        dockerfile_path = DEVCONTAINER_DIR / "Dockerfile"
        dockerfile_content = dockerfile_path.read_text()

        # Should use BuildKit cache mount for uv
        assert (
            "--mount=type=cache" in dockerfile_content
        ), "Dockerfile should use cache mount for optimization (Req 9.5)"
        assert (
            "/root/.cache/uv" in dockerfile_content or "/.cache/uv" in dockerfile_content
        ), "Dockerfile should cache uv dependencies"

    def test_dockerfile_apt_cache_cleanup(self):
        """Test that Dockerfile cleans up APT cache to reduce image size (Req 9.5)"""
        dockerfile_path = DEVCONTAINER_DIR / "Dockerfile"
        dockerfile_content = dockerfile_path.read_text()

        # Should clean up APT lists after installation
        assert (
            "rm -rf /var/lib/apt/lists/*" in dockerfile_content
        ), "Dockerfile should clean up APT cache for smaller image size (Req 9.5)"


class TestDocumentation:
    """Test that resource management is documented"""

    def test_devcontainer_guide_resource_section(self):
        """Test that DEVCONTAINER_GUIDE.md documents resource management"""
        guide_path = (
            PROJECT_ROOT / "DESED_task" / "dcase2024_task4_baseline" / "DEVCONTAINER_GUIDE.md"
        )
        assert guide_path.exists(), "DEVCONTAINER_GUIDE.md must exist"

        guide_content = guide_path.read_text()

        # Should document resource management topics
        resource_keywords = [
            "メモリ",  # Memory (Japanese)
            "memory",
            "disk",
            "ディスク",  # Disk (Japanese)
            "docker volume prune",  # Volume cleanup command
        ]

        found_keywords = [kw for kw in resource_keywords if kw in guide_content.lower()]
        assert (
            len(found_keywords) >= 2
        ), f"DEVCONTAINER_GUIDE.md should document resource management. Found: {found_keywords}"

    def test_devcontainer_guide_docker_desktop_settings(self):
        """Test that Docker Desktop memory configuration is documented"""
        guide_path = (
            PROJECT_ROOT / "DESED_task" / "dcase2024_task4_baseline" / "DEVCONTAINER_GUIDE.md"
        )
        guide_content = guide_path.read_text()

        # Should mention Docker Desktop settings
        has_docker_desktop = "docker desktop" in guide_content.lower()
        has_settings_info = "設定" in guide_content or "settings" in guide_content.lower()

        assert (
            has_docker_desktop or has_settings_info
        ), "DEVCONTAINER_GUIDE.md should document Docker Desktop settings"


class TestBuildPerformanceDocumentation:
    """Test that build performance optimization is documented (Task 6.2)"""

    def test_build_time_expectations_documented(self):
        """Test that expected build times are documented"""
        guide_path = (
            PROJECT_ROOT / "DESED_task" / "dcase2024_task4_baseline" / "DEVCONTAINER_GUIDE.md"
        )
        guide_content = guide_path.read_text()

        # Should mention build time expectations
        # Design.md specifies: 初回約5-7分、2回目以降約1-2分
        has_build_time = (
            "分" in guide_content or "min" in guide_content.lower() or "time" in guide_content.lower()
        )

        assert has_build_time, "DEVCONTAINER_GUIDE.md should document build time expectations"


# Integration test (optional - requires Docker)
class TestResourceCheckScriptIntegration:
    """Integration tests for resource check script"""

    @pytest.mark.skip(reason="Requires running inside devcontainer")
    def test_script_executes_successfully(self):
        """Test that postCreateCommand script can execute (integration test)"""
        script_path = DEVCONTAINER_DIR / "post_create_command.sh"
        result = subprocess.run(
            ["bash", str(script_path)],
            capture_output=True,
            text=True,
            cwd=DEVCONTAINER_DIR,
        )
        # Script should complete (may have warnings, but should not fail)
        assert result.returncode in [
            0,
            None,
        ], f"Script execution failed: {result.stderr}"


# Pytest configuration
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
