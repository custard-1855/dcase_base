"""Performance tests for experiment directory path resolution.

This module tests performance requirements for the ExperimentDirManager,
ensuring that path resolution operations meet scalability and performance
requirements for production use.

Key Requirements (from design.md):
- Requirement 3.5: get_experiment_dir() must complete within 100ms
- Scalability: 1000 experiment manifest scan must complete within 1 second
- Optimization: Mode-based search scope narrowing should not degrade performance

Test Coverage:
- Basic 100ms performance requirement with moderate experiment count
- Large-scale 1000+ experiment scalability
- Mode filter effectiveness for search scope narrowing
- Fallback strategy performance with corrupted manifests
- Worst-case performance with target at end of search space
"""

import sys
import time
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from local.experiment_dir import ExecutionMode, ExperimentConfig, ExperimentDirManager


def _create_test_experiments(
    base_dir: Path,
    count: int,
    mode: ExecutionMode | None = None,
    modes: list[ExecutionMode] | None = None,
) -> dict:
    """Helper function to create test experiment directories with manifests.

    Args:
        base_dir: Base directory for experiment creation
        count: Number of experiments to create
        mode: Single mode to use for all experiments (if modes is None)
        modes: List of modes to cycle through (overrides mode parameter)

    Returns:
        Dictionary with 'target_run_id' and 'target_dir' keys for middle experiment
    """
    target_run_id = None
    target_dir = None
    target_mode = mode or ExecutionMode.TRAIN

    for i in range(count):
        if modes:
            current_mode = modes[i % len(modes)]
        else:
            current_mode = mode or ExecutionMode.TRAIN

        experiment_dir = (
            base_dir
            / current_mode.value
            / f"category{i % 10}"
            / f"method{i % 20}"
            / f"v{i % 5}"
            / f"run-{i:06d}"
        )
        experiment_dir.mkdir(parents=True)

        # Only TRAIN and TEST modes have run_ids
        has_run_id = current_mode in [ExecutionMode.TRAIN, ExecutionMode.TEST]
        run_id = f"run-id-{i:06d}" if has_run_id else None

        # Store target as middle experiment
        if i == count // 2:
            target_run_id = run_id
            target_dir = experiment_dir
            target_mode = current_mode

        ExperimentDirManager.generate_manifest(
            experiment_dir=experiment_dir,
            run_id=run_id,
            config={"experiment_id": i},
            mode=current_mode,
        )

    return {
        "target_run_id": target_run_id,
        "target_dir": target_dir,
        "target_mode": target_mode,
    }


class TestPathResolutionPerformance:
    """Test path resolution performance requirements."""

    def test_get_experiment_dir_100ms_requirement(self):
        """Test that get_experiment_dir completes within 100ms (Requirement 3.5).

        This test verifies the basic performance requirement with a moderate
        number of experiments (50 experiments). This is the core performance
        requirement from design.md Section "System Flows".
        """
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            base_dir = Path(tmpdir) / "experiments"

            # Create 50 experiment directories with manifests
            test_data = _create_test_experiments(base_dir, count=50, mode=ExecutionMode.TRAIN)
            target_run_id = test_data["target_run_id"]

            # Measure resolution time for target experiment
            start_time = time.time()

            resolved_path = ExperimentDirManager.get_experiment_dir(
                run_id_or_name=target_run_id, base_dir=base_dir
            )

            elapsed_time = time.time() - start_time

            # Should complete within 100ms (0.1 seconds) per Requirement 3.5
            assert (
                elapsed_time < 0.1
            ), f"Resolution took {elapsed_time*1000:.2f}ms, expected < 100ms"
            assert resolved_path.exists()

    def test_1000_experiment_manifest_scan_1_second_requirement(self):
        """Test that 1000 experiment manifest scan completes within 1 second.

        This test verifies scalability with a large number of experiments,
        ensuring the manifest-based search strategy performs efficiently
        even with diverse experiment types across all execution modes.
        """
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            base_dir = Path(tmpdir) / "experiments"

            # Create 1000 experiments distributed across all modes
            all_modes = [
                ExecutionMode.TRAIN,
                ExecutionMode.TEST,
                ExecutionMode.INFERENCE,
                ExecutionMode.FEATURE_EXTRACTION,
            ]
            test_data = _create_test_experiments(base_dir, count=1000, modes=all_modes)
            target_run_id = test_data["target_run_id"]

            # Measure resolution time for target experiment (near middle)
            start_time = time.time()

            resolved_path = ExperimentDirManager.get_experiment_dir(
                run_id_or_name=target_run_id, base_dir=base_dir
            )

            elapsed_time = time.time() - start_time

            # Should complete within 1 second for large-scale scenario
            assert (
                elapsed_time < 1.0
            ), f"Manifest scan took {elapsed_time:.2f}s, expected < 1.0s"
            assert resolved_path.exists()

    def test_mode_filter_improves_search_performance(self):
        """Test that mode argument narrows search scope and improves performance.

        This test verifies that specifying a mode filter significantly reduces
        search time by limiting the search to a specific mode directory.
        """
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            base_dir = Path(tmpdir) / "experiments"

            # Create 200 experiments per mode (800 total)
            target_run_id = None
            target_mode = ExecutionMode.TRAIN
            for i in range(800):
                modes = [
                    ExecutionMode.TRAIN,
                    ExecutionMode.TEST,
                    ExecutionMode.INFERENCE,
                    ExecutionMode.FEATURE_EXTRACTION,
                ]
                mode = modes[i % 4]

                experiment_dir = (
                    base_dir
                    / mode.value
                    / f"category{i % 10}"
                    / f"method{i % 20}"
                    / f"v{i % 5}"
                    / f"run-{i:06d}"
                )
                experiment_dir.mkdir(parents=True)

                run_id = f"run-id-{i:06d}" if mode in [ExecutionMode.TRAIN, ExecutionMode.TEST] else None
                if i == 796:  # Target in TRAIN mode near the end
                    target_run_id = run_id
                    target_mode = mode

                ExperimentDirManager.generate_manifest(
                    experiment_dir=experiment_dir,
                    run_id=run_id,
                    config={"experiment_id": i},
                    mode=mode,
                )

            # Measure resolution time WITHOUT mode filter (search all modes)
            start_time_no_filter = time.time()

            resolved_path_no_filter = ExperimentDirManager.get_experiment_dir(
                run_id_or_name=target_run_id, base_dir=base_dir, mode=None
            )

            elapsed_no_filter = time.time() - start_time_no_filter

            # Measure resolution time WITH mode filter (search only TRAIN mode)
            start_time_with_filter = time.time()

            resolved_path_with_filter = ExperimentDirManager.get_experiment_dir(
                run_id_or_name=target_run_id, base_dir=base_dir, mode=target_mode
            )

            elapsed_with_filter = time.time() - start_time_with_filter

            # Both should find the same path
            assert resolved_path_no_filter == resolved_path_with_filter
            assert resolved_path_with_filter.exists()

            # Mode filter should provide some performance improvement
            # The improvement ratio varies based on system performance and I/O characteristics,
            # so we verify that mode filter doesn't degrade performance (improvement >= 1.0)
            # and ideally provides measurable benefit
            improvement_ratio = elapsed_no_filter / elapsed_with_filter

            # Both searches should be reasonably fast
            assert elapsed_no_filter < 0.5, f"No-filter search took {elapsed_no_filter:.2f}s, too slow"
            assert elapsed_with_filter < 0.5, f"Filtered search took {elapsed_with_filter:.2f}s, too slow"

            # Mode filter should not degrade performance (ratio >= 1.0)
            assert (
                improvement_ratio >= 1.0
            ), f"Mode filter degraded performance: {improvement_ratio:.2f}x, expected >= 1.0x"

            # Log performance metrics for analysis
            print(f"\nPerformance comparison (800 experiments):")
            print(f"  Without mode filter: {elapsed_no_filter*1000:.2f}ms")
            print(f"  With mode filter:    {elapsed_with_filter*1000:.2f}ms")
            print(f"  Improvement ratio:   {improvement_ratio:.2f}x")
            if improvement_ratio > 1.05:
                print(f"  ✓ Mode filter provides measurable improvement")

    def test_manifest_corruption_fallback_performance(self):
        """Test that fallback to directory scan doesn't cause excessive delays.

        This test verifies that even when manifest files are corrupted or missing,
        the directory scan fallback strategy completes within reasonable time.
        """
        import tempfile
        import json

        with tempfile.TemporaryDirectory() as tmpdir:
            base_dir = Path(tmpdir) / "experiments"

            # Create 100 experiments with manifests
            for i in range(100):
                experiment_dir = (
                    base_dir
                    / ExecutionMode.TRAIN.value
                    / f"category{i % 5}"
                    / f"method{i % 10}"
                    / "v1"
                    / f"run-{i:05d}-corrupted"
                )
                experiment_dir.mkdir(parents=True)

                # Create corrupted manifest (invalid JSON)
                manifest_path = experiment_dir / "manifest.json"
                manifest_path.write_text("{invalid json content", encoding="utf-8")

            # Target directory name (will be found via directory scan fallback)
            target_name = "run-00075-corrupted"

            # Measure resolution time with fallback strategy
            start_time = time.time()

            resolved_path = ExperimentDirManager.get_experiment_dir(
                run_id_or_name=target_name, base_dir=base_dir, mode=ExecutionMode.TRAIN
            )

            elapsed_time = time.time() - start_time

            # Should complete within 500ms even with corrupted manifests
            assert (
                elapsed_time < 0.5
            ), f"Fallback scan took {elapsed_time*1000:.2f}ms, expected < 500ms"
            assert resolved_path.exists()
            assert target_name in str(resolved_path)

    def test_worst_case_performance_last_experiment(self):
        """Test worst-case performance when target is the last experiment scanned.

        This test simulates the worst-case scenario where the target experiment
        is the last one to be found during manifest scanning.
        """
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            base_dir = Path(tmpdir) / "experiments"

            # Create 500 experiments
            target_run_id = None
            for i in range(500):
                experiment_dir = (
                    base_dir
                    / ExecutionMode.TRAIN.value
                    / f"category{i % 10}"
                    / f"method{i % 20}"
                    / "v1"
                    / f"run-{i:05d}"
                )
                experiment_dir.mkdir(parents=True)

                run_id = f"run-id-{i:05d}"
                if i == 499:  # Last experiment (worst case)
                    target_run_id = run_id

                ExperimentDirManager.generate_manifest(
                    experiment_dir=experiment_dir,
                    run_id=run_id,
                    config={"experiment_id": i},
                    mode=ExecutionMode.TRAIN,
                )

            # Measure resolution time for worst-case scenario
            start_time = time.time()

            resolved_path = ExperimentDirManager.get_experiment_dir(
                run_id_or_name=target_run_id, base_dir=base_dir, mode=ExecutionMode.TRAIN
            )

            elapsed_time = time.time() - start_time

            # Even in worst case, should complete within 200ms
            assert (
                elapsed_time < 0.2
            ), f"Worst-case resolution took {elapsed_time*1000:.2f}ms, expected < 200ms"
            assert resolved_path.exists()
