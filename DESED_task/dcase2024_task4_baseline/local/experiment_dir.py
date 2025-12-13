"""Experiment directory management and execution mode detection utilities.

This module provides hierarchical experiment directory structure management
and execution mode detection for the DCASE 2024 Task 4 baseline system.
"""

import json
import os
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Optional


class ExecutionMode(Enum):
    """Execution mode enumeration for experiment management.

    Attributes:
        TRAIN: Training mode - creates new wandb run and training directory
        TEST: Test-only mode - reuses training directory or creates minimal test directory
        INFERENCE: Inference mode - disables wandb, stores artifacts without synchronization
        FEATURE_EXTRACTION: Feature extraction mode - similar to inference, no wandb run

    """

    TRAIN = "train"
    TEST = "test"
    INFERENCE = "inference"
    FEATURE_EXTRACTION = "feature_extraction"


@dataclass(frozen=True)
class ExperimentConfig:
    """Experiment configuration data class with immutability enforcement.

    This dataclass holds experiment naming parameters and execution mode configuration.
    All fields are immutable (frozen=True) to ensure configuration consistency.

    Attributes:
        mode: Execution mode (train/test/inference/feature_extraction)
        category: Experiment category (e.g., "baseline", "advanced")
        method: Experiment method (e.g., "cmt", "beats")
        variant: Experiment variant (e.g., "v1", "use_neg_sample")
        base_dir: Base directory for experiments (default: "experiments")
        template: Optional template string for experiment naming
        log_test_to_wandb: Whether to log test-only runs to wandb (default: False)

    """

    mode: ExecutionMode = ExecutionMode.TRAIN
    category: str = "default"
    method: str = "baseline"
    variant: str = "v1"
    base_dir: str = "experiments"
    template: str | None = None
    log_test_to_wandb: bool = False

    def __post_init__(self) -> None:
        """Post-initialization validation and type conversion.

        Converts string mode to ExecutionMode enum if necessary.
        Raises ValueError for invalid mode values.
        """
        # Convert string mode to ExecutionMode enum
        if isinstance(self.mode, str):
            try:
                # Use object.__setattr__ because dataclass is frozen
                object.__setattr__(self, "mode", ExecutionMode(self.mode))
            except ValueError as e:
                valid_modes = [m.value for m in ExecutionMode]
                msg = f"Invalid mode '{self.mode}'. Valid modes: {valid_modes}"
                raise ValueError(msg) from e


class ExperimentDirManager:
    """Experiment directory management and execution mode detection utility class.

    This class provides static methods for:
    - Detecting execution mode from hparams and execution parameters
    - Building hierarchical experiment directory paths
    - Managing artifact directories and manifests
    - Resolving paths from run IDs or experiment names
    """

    @staticmethod
    def detect_execution_mode(
        hparams: dict,
        evaluation: bool = False,
        test_state_dict: dict | None = None,
        fast_dev_run: bool = False,
    ) -> ExecutionMode:
        """Detect execution mode from hparams and execution parameters.

        Priority order:
        1. Explicit mode specification in hparams["experiment"]["mode"]
        2. evaluation=True → INFERENCE
        3. test_state_dict is not None → TEST
        4. fast_dev_run=True → TRAIN
        5. Default → TRAIN

        Args:
            hparams: Hyperparameter dictionary (may contain "experiment.mode")
            evaluation: Evaluation parameter (indicator for inference mode)
            test_state_dict: Test state dictionary (indicator for test-only mode)
            fast_dev_run: Fast dev run flag

        Returns:
            Detected ExecutionMode

        Examples:
            >>> hparams = {"experiment": {"mode": "inference"}}
            >>> ExperimentDirManager.detect_execution_mode(hparams)
            ExecutionMode.INFERENCE

            >>> hparams = {}
            >>> ExperimentDirManager.detect_execution_mode(hparams, evaluation=True)
            ExecutionMode.INFERENCE

            >>> ExperimentDirManager.detect_execution_mode({}, test_state_dict={"state": "dict"})
            ExecutionMode.TEST
        """
        # Priority 1: Explicit mode in hparams
        if "experiment" in hparams and "mode" in hparams["experiment"]:
            explicit_mode = hparams["experiment"]["mode"]
            # Handle both string and ExecutionMode enum
            if isinstance(explicit_mode, ExecutionMode):
                return explicit_mode
            if isinstance(explicit_mode, str):
                return ExecutionMode(explicit_mode)

        # Priority 2: evaluation=True → INFERENCE
        if evaluation:
            return ExecutionMode.INFERENCE

        # Priority 3: test_state_dict present → TEST
        if test_state_dict is not None:
            return ExecutionMode.TEST

        # Priority 4 & 5: fast_dev_run=True or default → TRAIN
        return ExecutionMode.TRAIN

    @staticmethod
    def validate_path(path: Path) -> None:
        """Validate path for filesystem compatibility.

        Checks for:
        - Invalid characters (Windows: <>:"|?*)
        - Path length limit (260 characters for Windows compatibility)

        Args:
            path: Path to validate

        Raises:
            ValueError: If path contains invalid characters or exceeds length limit

        Examples:
            >>> ExperimentDirManager.validate_path(Path("valid/path/name"))
            # No exception raised

            >>> ExperimentDirManager.validate_path(Path("invalid<char>"))
            ValueError: Path contains invalid characters: <, >
        """
        path_str = str(path)

        # Check for invalid characters (Windows compatibility)
        invalid_chars = r'[<>:"|?*]'
        matches = re.findall(invalid_chars, path_str)
        if matches:
            unique_matches = sorted(set(matches))
            msg = f"Path contains invalid characters: {', '.join(unique_matches)}"
            raise ValueError(msg)

        # Check path length (Windows MAX_PATH limit)
        max_length = 260
        if len(path_str) > max_length:
            msg = (
                f"Path length ({len(path_str)} characters) exceeds maximum limit "
                f"({max_length} characters). Consider shortening the variant name."
            )
            raise ValueError(msg)

    @staticmethod
    def apply_template(template: str, config: ExperimentConfig) -> str:
        """Apply template placeholders to generate experiment name component.

        Supported placeholders:
        - {mode}: Execution mode value
        - {category}: Experiment category
        - {method}: Experiment method
        - {variant}: Experiment variant
        - {timestamp}: Current timestamp in YYYYMMDD_HHMMSS format

        Args:
            template: Template string with placeholders
            config: Experiment configuration

        Returns:
            Template string with placeholders replaced

        Examples:
            >>> config = ExperimentConfig(method="cmt", variant="v1")
            >>> ExperimentDirManager.apply_template("{method}_{variant}", config)
            'cmt_v1'

            >>> ExperimentDirManager.apply_template("{method}_{timestamp}", config)
            'cmt_20250112_153045'  # Actual timestamp will vary
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return template.format(
            mode=config.mode.value,
            category=config.category,
            method=config.method,
            variant=config.variant,
            timestamp=timestamp,
        )

    @staticmethod
    def build_experiment_path(
        config: ExperimentConfig,
        validate: bool = True,
    ) -> Path:
        """Build hierarchical experiment directory path with mode-based layout.

        Creates path structure: {base_dir}/{mode}/{category}/{method}/{variant}/
        Automatically creates parent directories if they don't exist.
        Supports environment variable expansion in base_dir.

        Args:
            config: Experiment configuration with naming parameters
            validate: Whether to validate path for filesystem compatibility

        Returns:
            Experiment directory parent path (without wandb run-* subdirectory)

        Raises:
            ValueError: If path validation fails (invalid characters, length exceeded)

        Examples:
            >>> config = ExperimentConfig(
            ...     mode=ExecutionMode.TRAIN,
            ...     category="baseline",
            ...     method="cmt",
            ...     variant="v1"
            ... )
            >>> path = ExperimentDirManager.build_experiment_path(config)
            >>> str(path)
            'experiments/train/baseline/cmt/v1'
        """
        # Expand environment variables in base_dir
        base_dir = Path(os.path.expandvars(config.base_dir))

        # Apply template if provided, otherwise use variant directly
        if config.template:
            variant_component = ExperimentDirManager.apply_template(config.template, config)
        else:
            variant_component = config.variant

        # Build hierarchical path: {base_dir}/{mode}/{category}/{method}/{variant}/
        experiment_path = (
            base_dir / config.mode.value / config.category / config.method / variant_component
        )

        # Validate path if requested
        if validate:
            ExperimentDirManager.validate_path(experiment_path)

        # Create parent directories
        experiment_path.mkdir(parents=True, exist_ok=True)

        return experiment_path

    @staticmethod
    def should_initialize_wandb(mode: ExecutionMode, config: ExperimentConfig) -> bool:
        """Determine whether to initialize wandb based on execution mode.

        Logic:
        - TRAIN mode → Always True
        - TEST mode → Depends on config.log_test_to_wandb
        - INFERENCE mode → Always False
        - FEATURE_EXTRACTION mode → Always False

        Args:
            mode: Execution mode
            config: Experiment configuration containing log_test_to_wandb flag

        Returns:
            True if wandb should be initialized, False otherwise

        Examples:
            >>> config = ExperimentConfig(log_test_to_wandb=False)
            >>> ExperimentDirManager.should_initialize_wandb(ExecutionMode.TRAIN, config)
            True

            >>> ExperimentDirManager.should_initialize_wandb(ExecutionMode.TEST, config)
            False

            >>> config_with_log = ExperimentConfig(log_test_to_wandb=True)
            >>> ExperimentDirManager.should_initialize_wandb(ExecutionMode.TEST, config_with_log)
            True

            >>> ExperimentDirManager.should_initialize_wandb(ExecutionMode.INFERENCE, config)
            False
        """
        if mode == ExecutionMode.TRAIN:
            return True
        if mode == ExecutionMode.TEST:
            return config.log_test_to_wandb
        # INFERENCE and FEATURE_EXTRACTION modes always return False
        return False

    @staticmethod
    def create_artifact_dirs(experiment_dir: Path | str) -> dict[str, Path]:
        """Create artifact subdirectories within experiment directory.

        Creates the following subdirectories:
        - checkpoints/: Model checkpoints and weights
        - metrics/: Training metrics and logs
        - inference/: Inference results
        - visualizations/: UMAP, reliability diagrams, etc.
        - config/: Experiment configuration snapshots

        This function is idempotent - calling it multiple times is safe
        and will not overwrite existing directories or files.

        Args:
            experiment_dir: Experiment directory path (Path object or string)

        Returns:
            Dictionary mapping subdirectory names to their Path objects
            {"checkpoints": Path, "metrics": Path, "inference": Path,
             "visualizations": Path, "config": Path}

        Raises:
            FileNotFoundError: If experiment_dir does not exist

        Examples:
            >>> experiment_dir = Path("experiments/train/baseline/cmt/v1/run-20250112-abcd")
            >>> artifact_dirs = ExperimentDirManager.create_artifact_dirs(experiment_dir)
            >>> artifact_dirs["checkpoints"]
            PosixPath('experiments/train/baseline/cmt/v1/run-20250112-abcd/checkpoints')
        """
        # Convert string to Path if necessary
        experiment_path = Path(experiment_dir) if isinstance(experiment_dir, str) else experiment_dir

        # Validate that experiment directory exists
        if not experiment_path.exists():
            msg = f"Experiment directory does not exist: {experiment_path}"
            raise FileNotFoundError(msg)

        # Define artifact subdirectories
        subdirs = {
            "checkpoints": experiment_path / "checkpoints",
            "metrics": experiment_path / "metrics",
            "inference": experiment_path / "inference",
            "visualizations": experiment_path / "visualizations",
            "config": experiment_path / "config",
        }

        # Create all subdirectories (exist_ok=True for idempotency)
        for subdir_path in subdirs.values():
            subdir_path.mkdir(parents=True, exist_ok=True)

        return subdirs

    @staticmethod
    def generate_manifest(
        experiment_dir: Path | str,
        run_id: Optional[str],
        config: dict,
        mode: ExecutionMode,
        parent_run_id: Optional[str] = None,
    ) -> None:
        """Generate manifest.json file with experiment metadata and mode information.

        Creates a manifest file in the experiment directory containing:
        - run_id: wandb run ID (or None for inference/feature_extraction modes)
        - experiment_path: Absolute path to experiment directory
        - mode: Execution mode (train/test/inference/feature_extraction)
        - created_at: ISO8601 timestamp
        - config: Snapshot of experiment configuration
        - parent_run_id: Reference to parent training run (for test/inference modes)

        The manifest file enables path resolution and experiment tracking
        without relying solely on wandb metadata.

        This function overwrites existing manifest files (not idempotent by design).

        Args:
            experiment_dir: Experiment directory path (Path object or string)
            run_id: wandb run ID (None for inference/feature_extraction modes)
            config: Complete experiment configuration dictionary
            mode: Execution mode
            parent_run_id: Optional reference to parent training run ID

        Examples:
            >>> ExperimentDirManager.generate_manifest(
            ...     experiment_dir=Path("experiments/train/baseline/cmt/v1/run-20250112-abcd"),
            ...     run_id="20250112_123456-abcd1234",
            ...     config={"training": {"max_epochs": 10}},
            ...     mode=ExecutionMode.TRAIN,
            ... )

            >>> ExperimentDirManager.generate_manifest(
            ...     experiment_dir=Path("experiments/inference/baseline/cmt/v1/run-20250112-efgh"),
            ...     run_id=None,
            ...     config={"inference": {"threshold": 0.5}},
            ...     mode=ExecutionMode.INFERENCE,
            ...     parent_run_id="parent-train-run-123",
            ... )
        """
        # Convert string to Path if necessary
        experiment_path = Path(experiment_dir) if isinstance(experiment_dir, str) else experiment_dir

        # Convert ExecutionMode to string value
        mode_str = mode.value if isinstance(mode, ExecutionMode) else mode

        # Create manifest data structure
        manifest = {
            "run_id": run_id,
            "experiment_path": str(experiment_path.absolute()),
            "mode": mode_str,
            "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "parent_run_id": parent_run_id,
            "config": config,
        }

        # Write manifest to JSON file with UTF-8 encoding and indent=2
        manifest_path = experiment_path / "manifest.json"
        with manifest_path.open("w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2, ensure_ascii=False)

    @staticmethod
    def _get_search_paths(
        base_path: Path, mode: ExecutionMode | None
    ) -> list[Path]:
        """Determine search paths based on mode parameter.

        Args:
            base_path: Base directory for experiments
            mode: Optional mode to narrow search scope

        Returns:
            List of paths to search
        """
        if mode is not None:
            # Search only in specific mode directory
            return [base_path / mode.value]

        # Search across all mode directories
        return [
            base_path / ExecutionMode.TRAIN.value,
            base_path / ExecutionMode.TEST.value,
            base_path / ExecutionMode.INFERENCE.value,
            base_path / ExecutionMode.FEATURE_EXTRACTION.value,
        ]

    @staticmethod
    def _search_by_manifest(search_paths: list[Path], run_id_or_name: str) -> Path | None:
        """Search for experiment directory by scanning manifest.json files.

        Args:
            search_paths: List of directories to search
            run_id_or_name: run_id to match in manifests

        Returns:
            Path to matching experiment directory, or None if not found
        """
        for search_path in search_paths:
            if not search_path.exists():
                continue

            # Walk through directory tree looking for manifest.json files
            for root, _dirs, files in os.walk(search_path):
                if "manifest.json" not in files:
                    continue

                manifest_path = Path(root) / "manifest.json"
                try:
                    with manifest_path.open("r", encoding="utf-8") as f:
                        manifest = json.load(f)

                    # Check if run_id matches
                    if manifest.get("run_id") == run_id_or_name:
                        return Path(root)

                except (json.JSONDecodeError, OSError):
                    # Manifest corrupted or unreadable, continue searching
                    continue

        return None

    @staticmethod
    def _search_by_directory_name(
        search_paths: list[Path], run_id_or_name: str
    ) -> Path | None:
        """Search for experiment directory by matching directory names.

        Args:
            search_paths: List of directories to search
            run_id_or_name: Pattern to match in directory names

        Returns:
            Path to matching experiment directory, or None if not found
        """
        for search_path in search_paths:
            if not search_path.exists():
                continue

            # Walk through directory tree looking for matching directory names
            for root, dirs, _files in os.walk(search_path):
                for dir_name in dirs:
                    # Check if directory name contains the run_id_or_name
                    if run_id_or_name in dir_name:
                        return Path(root) / dir_name

        return None

    @staticmethod
    def get_experiment_dir(
        run_id_or_name: str,
        base_dir: Path | str = Path("experiments"),
        mode: ExecutionMode | None = None,
    ) -> Path:
        """Resolve experiment directory from run ID or experiment name.

        This function searches for experiment directories using multiple strategies:
        1. Manifest search: Scan manifest.json files for matching run_id
        2. Directory scan fallback: Search for directories matching run_id_or_name pattern

        The mode parameter can narrow search scope to specific execution modes
        (train/test/inference/feature_extraction), improving performance.

        Performance requirement: Must complete within 100ms (Requirement 3.5)

        Args:
            run_id_or_name: wandb run ID or experiment name to search for
            base_dir: Base directory for experiments (default: "experiments")
            mode: Optional mode to narrow search scope (e.g., ExecutionMode.TRAIN)

        Returns:
            Path to experiment directory

        Raises:
            FileNotFoundError: If no matching experiment directory is found

        Examples:
            >>> # Search for experiment by run_id across all modes
            >>> exp_dir = ExperimentDirManager.get_experiment_dir("run-id-12345")

            >>> # Search only in train mode for better performance
            >>> exp_dir = ExperimentDirManager.get_experiment_dir(
            ...     "run-id-12345",
            ...     mode=ExecutionMode.TRAIN
            ... )

            >>> # Custom base directory
            >>> exp_dir = ExperimentDirManager.get_experiment_dir(
            ...     "run-id-12345",
            ...     base_dir=Path("/custom/experiments")
            ... )
        """
        # Convert string to Path if necessary
        base_path = Path(base_dir) if isinstance(base_dir, str) else base_dir

        # Determine search paths based on mode parameter
        search_paths = ExperimentDirManager._get_search_paths(base_path, mode)

        # Strategy 1: Search manifest.json files for matching run_id
        result = ExperimentDirManager._search_by_manifest(search_paths, run_id_or_name)
        if result is not None:
            return result

        # Strategy 2: Fallback to directory name pattern matching
        result = ExperimentDirManager._search_by_directory_name(search_paths, run_id_or_name)
        if result is not None:
            return result

        # If no match found, raise FileNotFoundError
        msg = (
            f"Experiment directory not found for run_id_or_name='{run_id_or_name}' "
            f"in base_dir='{base_path}'"
        )
        if mode is not None:
            msg += f" (mode filter: {mode.value})"
        raise FileNotFoundError(msg)

    @staticmethod
    def get_checkpoint_dir(experiment_dir: Path | str) -> Path:
        """Return checkpoints/ subdirectory path within experiment directory.

        Args:
            experiment_dir: Experiment directory path (Path object or string)

        Returns:
            Path to checkpoints subdirectory

        Examples:
            >>> exp_dir = Path("experiments/train/baseline/cmt/v1/run-123")
            >>> checkpoint_dir = ExperimentDirManager.get_checkpoint_dir(exp_dir)
            >>> str(checkpoint_dir)
            'experiments/train/baseline/cmt/v1/run-123/checkpoints'
        """
        # Convert string to Path if necessary
        experiment_path = Path(experiment_dir) if isinstance(experiment_dir, str) else experiment_dir
        return experiment_path / "checkpoints"

    @staticmethod
    def get_inference_dir(experiment_dir: Path | str) -> Path:
        """Return inference/ subdirectory path within experiment directory.

        Args:
            experiment_dir: Experiment directory path (Path object or string)

        Returns:
            Path to inference subdirectory

        Examples:
            >>> exp_dir = Path("experiments/train/baseline/cmt/v1/run-123")
            >>> inference_dir = ExperimentDirManager.get_inference_dir(exp_dir)
            >>> str(inference_dir)
            'experiments/train/baseline/cmt/v1/run-123/inference'
        """
        # Convert string to Path if necessary
        experiment_path = Path(experiment_dir) if isinstance(experiment_dir, str) else experiment_dir
        return experiment_path / "inference"

    @staticmethod
    def get_visualization_dir(experiment_dir: Path | str) -> Path:
        """Return visualizations/ subdirectory path within experiment directory.

        Args:
            experiment_dir: Experiment directory path (Path object or string)

        Returns:
            Path to visualizations subdirectory

        Examples:
            >>> exp_dir = Path("experiments/train/baseline/cmt/v1/run-123")
            >>> viz_dir = ExperimentDirManager.get_visualization_dir(exp_dir)
            >>> str(viz_dir)
            'experiments/train/baseline/cmt/v1/run-123/visualizations'
        """
        # Convert string to Path if necessary
        experiment_path = Path(experiment_dir) if isinstance(experiment_dir, str) else experiment_dir
        return experiment_path / "visualizations"
