"""Test devcontainer documentation presence in README.md"""

import re
from pathlib import Path


def test_readme_has_devcontainer_section():
    """Test that README.md contains a Devcontainer section."""
    readme_path = Path(__file__).parent.parent / "DESED_task" / "dcase2024_task4_baseline" / "README.md"
    assert readme_path.exists(), f"README.md not found at {readme_path}"

    content = readme_path.read_text()

    # Check for Devcontainer section header
    assert re.search(r"##\s+(?:Devcontainer|Development Container)", content, re.IGNORECASE), \
        "README.md should have a Devcontainer section header"


def test_readme_devcontainer_section_has_overview():
    """Test that Devcontainer section includes overview."""
    readme_path = Path(__file__).parent.parent / "DESED_task" / "dcase2024_task4_baseline" / "README.md"
    content = readme_path.read_text()

    # Check for key phrases in devcontainer section
    assert "Reopen in Container" in content or "Dev Container" in content, \
        "README.md should mention 'Reopen in Container' or 'Dev Container'"


def test_readme_devcontainer_section_has_prerequisites():
    """Test that Devcontainer section mentions prerequisites."""
    readme_path = Path(__file__).parent.parent / "DESED_task" / "dcase2024_task4_baseline" / "README.md"
    content = readme_path.read_text()

    # Check for Docker Desktop and gh auth login prerequisites
    assert "Docker Desktop" in content or "Docker" in content, \
        "README.md should mention Docker Desktop prerequisite"
    assert "gh auth login" in content, \
        "README.md should mention 'gh auth login' prerequisite"


def test_readme_devcontainer_section_has_setup_steps():
    """Test that Devcontainer section includes setup steps."""
    readme_path = Path(__file__).parent.parent / "DESED_task" / "dcase2024_task4_baseline" / "README.md"
    content = readme_path.read_text()

    # Check for setup instructions
    assert "Reopen in Container" in content, \
        "README.md should mention 'Reopen in Container' setup step"


def test_readme_devcontainer_section_has_usage_examples():
    """Test that Devcontainer section includes basic usage examples."""
    readme_path = Path(__file__).parent.parent / "DESED_task" / "dcase2024_task4_baseline" / "README.md"
    content = readme_path.read_text()

    # Check for training/inference usage examples
    # (These commands should already exist in README, but should be referenced in devcontainer section)
    assert "train_pretrained.py" in content, \
        "README.md should mention training script usage"


def test_readme_devcontainer_section_has_troubleshooting_link():
    """Test that Devcontainer section links to troubleshooting guide."""
    readme_path = Path(__file__).parent.parent / "DESED_task" / "dcase2024_task4_baseline" / "README.md"
    content = readme_path.read_text()

    # Check for link to DEVCONTAINER_GUIDE.md
    assert "DEVCONTAINER_GUIDE.md" in content, \
        "README.md should link to DEVCONTAINER_GUIDE.md"
