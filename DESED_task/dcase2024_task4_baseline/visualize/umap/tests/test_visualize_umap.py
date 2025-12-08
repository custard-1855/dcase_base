"""Tests for UMAPVisualizer class and CLI argument parser."""

import argparse
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from visualize.umap.visualize_umap import UMAPVisualizer


class TestUMAPVisualizerDefaultConfig:
    """Test default configuration generation."""

    def test_get_default_config_structure(self) -> None:
        """Test that default config has all required keys."""
        config = UMAPVisualizer._get_default_config()

        assert "umap" in config
        assert "plot" in config
        assert "output" in config
        assert "logging" in config

    def test_get_default_config_umap_params(self) -> None:
        """Test that UMAP parameters have correct default values."""
        config = UMAPVisualizer._get_default_config()

        assert config["umap"]["n_neighbors"] == 15
        assert config["umap"]["min_dist"] == 0.1
        assert config["umap"]["metric"] == "euclidean"
        assert config["umap"]["random_state"] == 42

    def test_get_default_config_plot_params(self) -> None:
        """Test that plot parameters have correct default values."""
        config = UMAPVisualizer._get_default_config()

        assert config["plot"]["dpi"] == 300
        assert config["plot"]["figsize"] == (12, 8)
        assert config["plot"]["palette"] == "colorblind"

    def test_get_default_config_output_params(self) -> None:
        """Test that output parameters have correct default values."""
        config = UMAPVisualizer._get_default_config()

        assert config["output"]["dir"] == "output"

    def test_get_default_config_logging_params(self) -> None:
        """Test that logging parameters have correct default values."""
        config = UMAPVisualizer._get_default_config()

        assert config["logging"]["level"] == "INFO"


class TestParseArgs:
    """Test CLI argument parser."""

    def test_parse_args_class_separation_minimal(self) -> None:
        """Test parsing class_separation mode with minimal arguments."""
        test_args = [
            "class_separation",
            "--input",
            "test.npz",
        ]

        with patch("sys.argv", ["visualize_umap.py"] + test_args):
            args = UMAPVisualizer.parse_args()

        assert args.mode == "class_separation"
        assert args.input == "test.npz"
        assert args.model_type == "student"  # default

    def test_parse_args_class_separation_with_teacher(self) -> None:
        """Test parsing class_separation with teacher model."""
        test_args = [
            "class_separation",
            "--input",
            "test.npz",
            "--model_type",
            "teacher",
        ]

        with patch("sys.argv", ["visualize_umap.py"] + test_args):
            args = UMAPVisualizer.parse_args()

        assert args.mode == "class_separation"
        assert args.model_type == "teacher"

    def test_parse_args_domain_comparison(self) -> None:
        """Test parsing domain_comparison mode."""
        test_args = [
            "domain_comparison",
            "--inputs",
            "data1.npz",
            "data2.npz",
        ]

        with patch("sys.argv", ["visualize_umap.py"] + test_args):
            args = UMAPVisualizer.parse_args()

        assert args.mode == "domain_comparison"
        assert args.inputs == ["data1.npz", "data2.npz"]

    def test_parse_args_mixstyle_effect(self) -> None:
        """Test parsing mixstyle_effect mode."""
        test_args = [
            "mixstyle_effect",
            "--before",
            "before.npz",
            "--after",
            "after.npz",
        ]

        with patch("sys.argv", ["visualize_umap.py"] + test_args):
            args = UMAPVisualizer.parse_args()

        assert args.mode == "mixstyle_effect"
        assert args.before == "before.npz"
        assert args.after == "after.npz"

    def test_parse_args_with_custom_umap_params(self) -> None:
        """Test parsing with custom UMAP parameters."""
        test_args = [
            "class_separation",
            "--input",
            "test.npz",
            "--n_neighbors",
            "30",
            "--min_dist",
            "0.05",
            "--metric",
            "cosine",
        ]

        with patch("sys.argv", ["visualize_umap.py"] + test_args):
            args = UMAPVisualizer.parse_args()

        assert args.n_neighbors == 30
        assert args.min_dist == 0.05
        assert args.metric == "cosine"

    def test_parse_args_with_custom_plot_params(self) -> None:
        """Test parsing with custom plot parameters."""
        test_args = [
            "class_separation",
            "--input",
            "test.npz",
            "--dpi",
            "600",
            "--figsize",
            "16",
            "10",
            "--palette",
            "tab20",
        ]

        with patch("sys.argv", ["visualize_umap.py"] + test_args):
            args = UMAPVisualizer.parse_args()

        assert args.dpi == 600
        assert args.figsize == [16, 10]
        assert args.palette == "tab20"

    def test_parse_args_with_output_dir(self) -> None:
        """Test parsing with custom output directory."""
        test_args = [
            "class_separation",
            "--input",
            "test.npz",
            "--output_dir",
            "custom_output",
        ]

        with patch("sys.argv", ["visualize_umap.py"] + test_args):
            args = UMAPVisualizer.parse_args()

        assert args.output_dir == "custom_output"

    def test_parse_args_with_config_file(self) -> None:
        """Test parsing with config file path."""
        test_args = [
            "class_separation",
            "--input",
            "test.npz",
            "--config",
            "config.yaml",
        ]

        with patch("sys.argv", ["visualize_umap.py"] + test_args):
            args = UMAPVisualizer.parse_args()

        assert args.config == "config.yaml"


class TestLoadConfig:
    """Test YAML configuration loading."""

    def test_load_config_success(self, tmp_path: Path) -> None:
        """Test successful loading of YAML configuration file."""
        # Create a test YAML config file
        config_file = tmp_path / "test_config.yaml"
        config_content = """
umap:
  n_neighbors: 20
  min_dist: 0.2
  metric: cosine
plot:
  dpi: 600
  figsize: [16, 10]
  palette: tab20
output:
  dir: custom_output
logging:
  level: DEBUG
"""
        config_file.write_text(config_content)

        # Load config
        config = UMAPVisualizer.load_config(str(config_file))

        # Verify structure
        assert "umap" in config
        assert "plot" in config
        assert "output" in config
        assert "logging" in config

        # Verify values
        assert config["umap"]["n_neighbors"] == 20
        assert config["umap"]["min_dist"] == 0.2
        assert config["umap"]["metric"] == "cosine"
        assert config["plot"]["dpi"] == 600
        assert config["plot"]["figsize"] == [16, 10]
        assert config["plot"]["palette"] == "tab20"
        assert config["output"]["dir"] == "custom_output"
        assert config["logging"]["level"] == "DEBUG"

    def test_load_config_file_not_found(self) -> None:
        """Test that FileNotFoundError is raised for non-existent config file."""
        with pytest.raises(FileNotFoundError, match="Config file not found"):
            UMAPVisualizer.load_config("nonexistent_config.yaml")

    def test_load_config_partial_config(self, tmp_path: Path) -> None:
        """Test loading config with only some sections specified."""
        config_file = tmp_path / "partial_config.yaml"
        config_content = """
umap:
  n_neighbors: 25
plot:
  dpi: 450
"""
        config_file.write_text(config_content)

        config = UMAPVisualizer.load_config(str(config_file))

        # Verify only specified sections are present
        assert "umap" in config
        assert config["umap"]["n_neighbors"] == 25
        assert "plot" in config
        assert config["plot"]["dpi"] == 450

    def test_load_config_empty_file(self, tmp_path: Path) -> None:
        """Test loading an empty YAML file returns None or empty dict."""
        config_file = tmp_path / "empty_config.yaml"
        config_file.write_text("")

        config = UMAPVisualizer.load_config(str(config_file))

        # Empty YAML file should return None or empty dict
        assert config is None or config == {}


class TestMergeConfig:
    """Test configuration merging logic."""

    def test_merge_config_cli_overrides_yaml(self) -> None:
        """Test that CLI arguments override YAML config."""
        cli_args = argparse.Namespace(
            n_neighbors=30,
            min_dist=None,
            dpi=None,
        )
        yaml_config = {
            "umap": {"n_neighbors": 15, "min_dist": 0.2},
            "plot": {"dpi": 300},
        }
        defaults = UMAPVisualizer._get_default_config()

        merged = UMAPVisualizer.merge_config(cli_args, yaml_config, defaults)

        # CLI value should override
        assert merged["umap"]["n_neighbors"] == 30
        # YAML value should be used when CLI is None
        assert merged["umap"]["min_dist"] == 0.2
        # YAML value should override default
        assert merged["plot"]["dpi"] == 300

    def test_merge_config_yaml_overrides_defaults(self) -> None:
        """Test that YAML config overrides defaults."""
        cli_args = argparse.Namespace(
            n_neighbors=None,
            min_dist=None,
            dpi=None,
        )
        yaml_config = {
            "umap": {"n_neighbors": 20},
            "plot": {"palette": "tab10"},
        }
        defaults = UMAPVisualizer._get_default_config()

        merged = UMAPVisualizer.merge_config(cli_args, yaml_config, defaults)

        # YAML should override defaults
        assert merged["umap"]["n_neighbors"] == 20
        assert merged["plot"]["palette"] == "tab10"
        # Defaults should be used when not specified in YAML or CLI
        assert merged["umap"]["min_dist"] == 0.1

    def test_merge_config_empty_yaml(self) -> None:
        """Test config merging with empty YAML."""
        cli_args = argparse.Namespace(n_neighbors=None)
        yaml_config = {}
        defaults = UMAPVisualizer._get_default_config()

        merged = UMAPVisualizer.merge_config(cli_args, yaml_config, defaults)

        # Should use defaults when YAML is empty
        assert merged["umap"]["n_neighbors"] == 15
        assert merged["plot"]["dpi"] == 300


class TestUMAPVisualizerInit:
    """Test UMAPVisualizer initialization."""

    def test_init_with_default_config(self) -> None:
        """Test initialization with default configuration."""
        visualizer = UMAPVisualizer()

        assert visualizer.config is not None
        assert "umap" in visualizer.config
        assert visualizer.loader is not None
        assert visualizer.reducer is not None
        assert visualizer.generator is not None

    def test_init_with_custom_config(self) -> None:
        """Test initialization with custom configuration."""
        custom_config = {
            "umap": {
                "n_neighbors": 20,
                "min_dist": 0.2,
                "metric": "cosine",
                "random_state": 123,
            },
            "plot": {
                "dpi": 600,
                "figsize": (16, 10),
                "palette": "tab20",
            },
            "output": {"dir": "custom_output"},
            "logging": {"level": "DEBUG"},
        }

        visualizer = UMAPVisualizer(config=custom_config)

        # Check that components are initialized with custom config
        assert visualizer.reducer.n_neighbors == 20
        assert visualizer.reducer.min_dist == 0.2
        assert visualizer.reducer.metric == "cosine"
        assert visualizer.generator.dpi == 600
        assert visualizer.generator.figsize == (16, 10)


class TestRunClassSeparation:
    """Test run_class_separation execution flow."""

    def test_run_class_separation_student_model(self, tmp_path: Path) -> None:
        """Test class separation visualization with student model."""
        # Create a mock UMAPVisualizer with mocked components
        visualizer = UMAPVisualizer()

        # Mock the components
        visualizer.loader = MagicMock()
        visualizer.reducer = MagicMock()
        visualizer.generator = MagicMock()

        # Setup mock return values
        import numpy as np

        mock_features = np.random.randn(100, 384).astype(np.float32)
        mock_classes = np.random.randint(0, 21, size=100).astype(np.int32)
        mock_targets = np.zeros((100, 27), dtype=np.float32)
        mock_filenames = [f"file_{i}.wav" for i in range(100)]

        visualizer.loader.load_features.return_value = (
            mock_features,
            mock_classes,
            mock_targets,
            mock_filenames,
        )

        mock_embeddings = np.random.randn(100, 2).astype(np.float32)
        visualizer.reducer.fit_transform.return_value = mock_embeddings

        visualizer.generator.plot_class_separation.return_value = (
            "output.png",
            "output.pdf",
        )

        # Execute
        input_path = "test_features.npz"
        visualizer.run_class_separation(input_path, model_type="student")

        # Verify the correct sequence of calls
        visualizer.loader.load_features.assert_called_once_with(
            input_path,
            model_type="student",
        )
        visualizer.reducer.fit_transform.assert_called_once()
        visualizer.generator.plot_class_separation.assert_called_once()

        # Verify plot_class_separation was called with correct arguments
        call_args = visualizer.generator.plot_class_separation.call_args
        assert call_args is not None
        assert "embeddings" in call_args[1]
        assert "class_labels" in call_args[1]
        assert "class_names" in call_args[1]
        assert "title" in call_args[1]
        assert "filename_prefix" in call_args[1]
        assert "Student" in call_args[1]["title"]
        assert call_args[1]["filename_prefix"] == "class_separation_student"

    def test_run_class_separation_teacher_model(self, tmp_path: Path) -> None:
        """Test class separation visualization with teacher model."""
        visualizer = UMAPVisualizer()

        # Mock components
        visualizer.loader = MagicMock()
        visualizer.reducer = MagicMock()
        visualizer.generator = MagicMock()

        import numpy as np

        visualizer.loader.load_features.return_value = (
            np.random.randn(100, 384).astype(np.float32),
            np.random.randint(0, 21, size=100).astype(np.int32),
            np.zeros((100, 27), dtype=np.float32),
            [f"file_{i}.wav" for i in range(100)],
        )
        visualizer.reducer.fit_transform.return_value = np.random.randn(
            100,
            2,
        ).astype(np.float32)
        visualizer.generator.plot_class_separation.return_value = (
            "output.png",
            "output.pdf",
        )

        # Execute with teacher model
        visualizer.run_class_separation("test.npz", model_type="teacher")

        # Verify teacher model was passed correctly
        visualizer.loader.load_features.assert_called_once_with(
            "test.npz",
            model_type="teacher",
        )

        # Verify title contains "Teacher"
        call_args = visualizer.generator.plot_class_separation.call_args
        assert call_args is not None
        assert "Teacher" in call_args[1]["title"]
        assert call_args[1]["filename_prefix"] == "class_separation_teacher"

    def test_run_class_separation_uses_correct_class_names(
        self,
        tmp_path: Path,
    ) -> None:
        """Test that class separation uses unified class labels."""
        visualizer = UMAPVisualizer()

        # Store original loader's class_labels
        original_class_labels = visualizer.loader.class_labels

        visualizer.loader = MagicMock()
        visualizer.reducer = MagicMock()
        visualizer.generator = MagicMock()

        # Set class_labels property on mock loader
        visualizer.loader.class_labels = original_class_labels

        import numpy as np

        visualizer.loader.load_features.return_value = (
            np.random.randn(50, 384).astype(np.float32),
            np.random.randint(0, 21, size=50).astype(np.int32),
            np.zeros((50, 27), dtype=np.float32),
            [f"file_{i}.wav" for i in range(50)],
        )
        visualizer.reducer.fit_transform.return_value = np.random.randn(
            50,
            2,
        ).astype(np.float32)
        visualizer.generator.plot_class_separation.return_value = (
            "output.png",
            "output.pdf",
        )

        # Execute
        visualizer.run_class_separation("test.npz")

        # Verify class_names was passed to plot generator
        call_args = visualizer.generator.plot_class_separation.call_args
        assert call_args is not None
        class_names = call_args[1]["class_names"]
        assert isinstance(class_names, list)
        # Should have 21 classes (DESED 10 + MAESTRO 11)
        assert len(class_names) == 21


class TestRunDomainComparison:
    """Test run_domain_comparison execution flow."""

    def test_run_domain_comparison_multiple_datasets(self, tmp_path: Path) -> None:
        """Test domain comparison with multiple datasets."""
        visualizer = UMAPVisualizer()

        # Mock components
        visualizer.loader = MagicMock()
        visualizer.reducer = MagicMock()
        visualizer.generator = MagicMock()

        import numpy as np

        # Mock combined datasets (200 samples from 2 datasets)
        mock_features = np.random.randn(200, 384).astype(np.float32)
        mock_classes = np.random.randint(0, 21, size=200).astype(np.int32)
        mock_targets = np.zeros((200, 27), dtype=np.float32)
        mock_filenames = [f"desed_synth_{i}.wav" for i in range(100)] + [
            f"desed_real_{i}.wav" for i in range(100)
        ]

        visualizer.loader.load_multiple_datasets.return_value = (
            mock_features,
            mock_classes,
            mock_targets,
            mock_filenames,
        )

        mock_domain_labels = np.array([0] * 100 + [1] * 100, dtype=np.int32)
        visualizer.loader.extract_domain_labels.return_value = mock_domain_labels

        mock_embeddings = np.random.randn(200, 2).astype(np.float32)
        visualizer.reducer.fit_transform.return_value = mock_embeddings

        visualizer.generator.plot_domain_comparison.return_value = (
            "output.png",
            "output.pdf",
        )

        # Execute
        input_paths = ["dataset1.npz", "dataset2.npz"]
        visualizer.run_domain_comparison(input_paths, model_type="student")

        # Verify the correct sequence of calls
        visualizer.loader.load_multiple_datasets.assert_called_once_with(
            input_paths,
            model_type="student",
        )
        visualizer.loader.extract_domain_labels.assert_called_once_with(
            mock_filenames,
        )
        visualizer.reducer.fit_transform.assert_called_once()
        visualizer.generator.plot_domain_comparison.assert_called_once()

        # Verify plot_domain_comparison arguments
        call_args = visualizer.generator.plot_domain_comparison.call_args
        assert call_args is not None
        assert "embeddings" in call_args[1]
        assert "domain_labels" in call_args[1]
        assert "domain_names" in call_args[1]
        assert "title" in call_args[1]
        assert "filename_prefix" in call_args[1]

        # Verify domain names
        domain_names = call_args[1]["domain_names"]
        assert len(domain_names) == 4
        assert "DESED Synthetic" in domain_names
        assert "DESED Real" in domain_names
        assert "MAESTRO Training" in domain_names
        assert "MAESTRO Validation" in domain_names

    def test_run_domain_comparison_teacher_model(self, tmp_path: Path) -> None:
        """Test domain comparison with teacher model."""
        visualizer = UMAPVisualizer()

        visualizer.loader = MagicMock()
        visualizer.reducer = MagicMock()
        visualizer.generator = MagicMock()

        import numpy as np

        visualizer.loader.load_multiple_datasets.return_value = (
            np.random.randn(150, 384).astype(np.float32),
            np.random.randint(0, 21, size=150).astype(np.int32),
            np.zeros((150, 27), dtype=np.float32),
            [f"file_{i}.wav" for i in range(150)],
        )
        visualizer.loader.extract_domain_labels.return_value = np.random.randint(
            0,
            4,
            size=150,
        ).astype(np.int32)
        visualizer.reducer.fit_transform.return_value = np.random.randn(
            150,
            2,
        ).astype(np.float32)
        visualizer.generator.plot_domain_comparison.return_value = (
            "output.png",
            "output.pdf",
        )

        # Execute with teacher model
        input_paths = ["data1.npz", "data2.npz", "data3.npz"]
        visualizer.run_domain_comparison(input_paths, model_type="teacher")

        # Verify teacher model was passed
        visualizer.loader.load_multiple_datasets.assert_called_once_with(
            input_paths,
            model_type="teacher",
        )

        # Verify title contains "Teacher"
        call_args = visualizer.generator.plot_domain_comparison.call_args
        assert call_args is not None
        assert "Teacher" in call_args[1]["title"]
        assert call_args[1]["filename_prefix"] == "domain_comparison_teacher"

    def test_run_domain_comparison_single_dataset(self, tmp_path: Path) -> None:
        """Test domain comparison with a single dataset (edge case)."""
        visualizer = UMAPVisualizer()

        visualizer.loader = MagicMock()
        visualizer.reducer = MagicMock()
        visualizer.generator = MagicMock()

        import numpy as np

        visualizer.loader.load_multiple_datasets.return_value = (
            np.random.randn(50, 384).astype(np.float32),
            np.random.randint(0, 21, size=50).astype(np.int32),
            np.zeros((50, 27), dtype=np.float32),
            [f"desed_synth_{i}.wav" for i in range(50)],
        )
        visualizer.loader.extract_domain_labels.return_value = np.zeros(
            50,
            dtype=np.int32,
        )
        visualizer.reducer.fit_transform.return_value = np.random.randn(
            50,
            2,
        ).astype(np.float32)
        visualizer.generator.plot_domain_comparison.return_value = (
            "output.png",
            "output.pdf",
        )

        # Execute with single dataset
        input_paths = ["single_dataset.npz"]
        visualizer.run_domain_comparison(input_paths)

        # Should still work - load_multiple_datasets handles single file
        visualizer.loader.load_multiple_datasets.assert_called_once_with(
            input_paths,
            model_type="student",
        )


class TestRunMixStyleComparison:
    """Test run_mixstyle_comparison execution flow."""

    def test_run_mixstyle_comparison_basic(self, tmp_path: Path) -> None:
        """Test MixStyle comparison with student model."""
        visualizer = UMAPVisualizer()

        # Mock components
        visualizer.loader = MagicMock()
        visualizer.reducer = MagicMock()
        visualizer.generator = MagicMock()

        import numpy as np

        # Mock features before MixStyle (100 samples)
        mock_features_before = np.random.randn(100, 384).astype(np.float32)
        mock_classes_before = np.random.randint(0, 21, size=100).astype(np.int32)
        mock_targets_before = np.zeros((100, 27), dtype=np.float32)
        mock_filenames_before = [f"desed_synth_{i}.wav" for i in range(100)]

        # Mock features after MixStyle (100 samples)
        mock_features_after = np.random.randn(100, 384).astype(np.float32)
        mock_classes_after = np.random.randint(0, 21, size=100).astype(np.int32)
        mock_targets_after = np.zeros((100, 27), dtype=np.float32)
        mock_filenames_after = [f"desed_synth_{i}.wav" for i in range(100)]

        # Setup mock returns
        visualizer.loader.load_features.side_effect = [
            (
                mock_features_before,
                mock_classes_before,
                mock_targets_before,
                mock_filenames_before,
            ),
            (
                mock_features_after,
                mock_classes_after,
                mock_targets_after,
                mock_filenames_after,
            ),
        ]

        mock_domain_labels = np.zeros(100, dtype=np.int32)
        visualizer.loader.extract_domain_labels.return_value = mock_domain_labels

        # Mock combined UMAP embeddings (200 samples)
        mock_combined_embeddings = np.random.randn(200, 2).astype(np.float32)
        visualizer.reducer.fit_transform.return_value = mock_combined_embeddings

        visualizer.generator.plot_mixstyle_comparison.return_value = (
            "output.png",
            "output.pdf",
        )

        # Execute
        before_path = "before_mixstyle.npz"
        after_path = "after_mixstyle.npz"
        visualizer.run_mixstyle_comparison(before_path, after_path)

        # Verify load_features was called twice (before and after)
        assert visualizer.loader.load_features.call_count == 2
        visualizer.loader.load_features.assert_any_call(
            before_path,
            model_type="student",
        )
        visualizer.loader.load_features.assert_any_call(
            after_path,
            model_type="student",
        )

        # Verify domain labels were extracted twice
        assert visualizer.loader.extract_domain_labels.call_count == 2

        # Verify UMAP was called with combined features (200 samples)
        visualizer.reducer.fit_transform.assert_called_once()
        combined_features = visualizer.reducer.fit_transform.call_args[0][0]
        assert combined_features.shape == (200, 384)

        # Verify plot_mixstyle_comparison was called with correct embeddings
        visualizer.generator.plot_mixstyle_comparison.assert_called_once()
        call_args = visualizer.generator.plot_mixstyle_comparison.call_args[1]

        # Verify embeddings were split correctly
        assert call_args["embeddings_before"].shape == (100, 2)
        assert call_args["embeddings_after"].shape == (100, 2)
        assert call_args["domain_labels"].shape == (100,)
        assert len(call_args["domain_names"]) == 4

    def test_run_mixstyle_comparison_teacher_model(self, tmp_path: Path) -> None:
        """Test MixStyle comparison with teacher model."""
        visualizer = UMAPVisualizer()

        visualizer.loader = MagicMock()
        visualizer.reducer = MagicMock()
        visualizer.generator = MagicMock()

        import numpy as np

        # Setup mock returns
        visualizer.loader.load_features.return_value = (
            np.random.randn(50, 384).astype(np.float32),
            np.random.randint(0, 21, size=50).astype(np.int32),
            np.zeros((50, 27), dtype=np.float32),
            [f"file_{i}.wav" for i in range(50)],
        )
        visualizer.loader.extract_domain_labels.return_value = np.zeros(
            50,
            dtype=np.int32,
        )
        visualizer.reducer.fit_transform.return_value = np.random.randn(
            100,
            2,
        ).astype(np.float32)
        visualizer.generator.plot_mixstyle_comparison.return_value = (
            "output.png",
            "output.pdf",
        )

        # Execute with teacher model
        visualizer.run_mixstyle_comparison(
            "before.npz",
            "after.npz",
            model_type="teacher",
        )

        # Verify teacher model was used in both load_features calls
        assert visualizer.loader.load_features.call_count == 2
        for call in visualizer.loader.load_features.call_args_list:
            assert call[1]["model_type"] == "teacher"

        # Verify title contains "Teacher"
        call_args = visualizer.generator.plot_mixstyle_comparison.call_args[1]
        assert "Teacher" in call_args["title_before"]
        assert "Teacher" in call_args["title_after"]
        assert call_args["filename_prefix"] == "mixstyle_effect_teacher"

    def test_run_mixstyle_comparison_shared_embedding_space(
        self,
        tmp_path: Path,
    ) -> None:
        """Test that MixStyle comparison uses shared UMAP embedding space."""
        visualizer = UMAPVisualizer()

        visualizer.loader = MagicMock()
        visualizer.reducer = MagicMock()
        visualizer.generator = MagicMock()

        import numpy as np

        # Mock features with different sample counts
        n_before = 75
        n_after = 125

        visualizer.loader.load_features.side_effect = [
            (
                np.random.randn(n_before, 384).astype(np.float32),
                np.random.randint(0, 21, size=n_before).astype(np.int32),
                np.zeros((n_before, 27), dtype=np.float32),
                [f"before_{i}.wav" for i in range(n_before)],
            ),
            (
                np.random.randn(n_after, 384).astype(np.float32),
                np.random.randint(0, 21, size=n_after).astype(np.int32),
                np.zeros((n_after, 27), dtype=np.float32),
                [f"after_{i}.wav" for i in range(n_after)],
            ),
        ]

        visualizer.loader.extract_domain_labels.return_value = np.zeros(
            n_before,
            dtype=np.int32,
        )

        # Mock combined embeddings
        total_samples = n_before + n_after
        mock_combined_embeddings = np.random.randn(total_samples, 2).astype(
            np.float32
        )
        visualizer.reducer.fit_transform.return_value = mock_combined_embeddings

        visualizer.generator.plot_mixstyle_comparison.return_value = (
            "output.png",
            "output.pdf",
        )

        # Execute
        visualizer.run_mixstyle_comparison("before.npz", "after.npz")

        # Verify UMAP was called once with combined features
        visualizer.reducer.fit_transform.assert_called_once()

        # Verify the combined features have correct shape
        combined_features = visualizer.reducer.fit_transform.call_args[0][0]
        assert combined_features.shape[0] == total_samples
        assert combined_features.shape[1] == 384

        # Verify embeddings were split correctly based on sample counts
        call_args = visualizer.generator.plot_mixstyle_comparison.call_args[1]
        assert call_args["embeddings_before"].shape == (n_before, 2)
        assert call_args["embeddings_after"].shape == (n_after, 2)

        # Verify the split embeddings match the original combined embeddings
        embeddings_before = call_args["embeddings_before"]
        embeddings_after = call_args["embeddings_after"]

        # Check that concatenating them back gives the original
        np.testing.assert_array_equal(
            np.concatenate([embeddings_before, embeddings_after], axis=0),
            mock_combined_embeddings,
        )

    def test_run_mixstyle_comparison_domain_labels(self, tmp_path: Path) -> None:
        """Test that domain labels are correctly extracted and used."""
        visualizer = UMAPVisualizer()

        visualizer.loader = MagicMock()
        visualizer.reducer = MagicMock()
        visualizer.generator = MagicMock()

        import numpy as np

        # Create filenames from different domains
        mock_filenames = [
            "desed_synth_1.wav",
            "desed_real_2.wav",
            "maestro_train_3.wav",
            "maestro_val_4.wav",
        ]

        visualizer.loader.load_features.return_value = (
            np.random.randn(4, 384).astype(np.float32),
            np.random.randint(0, 21, size=4).astype(np.int32),
            np.zeros((4, 27), dtype=np.float32),
            mock_filenames,
        )

        # Mock domain labels (one from each domain)
        mock_domain_labels = np.array([0, 1, 2, 3], dtype=np.int32)
        visualizer.loader.extract_domain_labels.return_value = mock_domain_labels

        visualizer.reducer.fit_transform.return_value = np.random.randn(
            8,
            2,
        ).astype(np.float32)
        visualizer.generator.plot_mixstyle_comparison.return_value = (
            "output.png",
            "output.pdf",
        )

        # Execute
        visualizer.run_mixstyle_comparison("before.npz", "after.npz")

        # Verify extract_domain_labels was called twice
        assert visualizer.loader.extract_domain_labels.call_count == 2

        # Verify domain labels were passed to plot generator
        call_args = visualizer.generator.plot_mixstyle_comparison.call_args[1]
        domain_labels = call_args["domain_labels"]

        # Should use domain labels from "before" (first call)
        np.testing.assert_array_equal(domain_labels, mock_domain_labels)

        # Verify domain names are correct
        domain_names = call_args["domain_names"]
        assert domain_names == [
            "DESED Synthetic",
            "DESED Real",
            "MAESTRO Training",
            "MAESTRO Validation",
        ]
