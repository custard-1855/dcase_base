"""UMAP visualization CLI entry point and orchestrator.

This module provides the main entry point for UMAP-based feature visualization,
supporting three visualization modes:
1. Class separation: Visualize feature space colored by event classes
2. Domain comparison: Compare feature distributions across different domains
3. MixStyle effect: Compare features before/after MixStyle application
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Any, Optional

import numpy as np
import yaml
from local.classes_dict import (
    classes_labels_desed,
    classes_labels_maestro_real_eval,
)
from numpy.typing import NDArray

from visualize.umap.feature_loader import FeatureLoader
from visualize.umap.plot_generator import PlotGenerator
from visualize.umap.umap_reducer import UMAPReducer

LOGGER = logging.getLogger(__name__)


class UMAPVisualizer:
    """UMAP visualization system entry point.

    Orchestrates feature loading, UMAP dimensionality reduction, and plot generation
    for three visualization modes: class separation, domain comparison, and MixStyle effect.

    Attributes:
        config: Configuration dictionary with umap, plot, output, and logging settings.
        loader: FeatureLoader instance for reading .npz files.
        reducer: UMAPReducer instance for dimensionality reduction.
        generator: PlotGenerator instance for plot creation and file output.
        logger: Python logger for this class.

    """

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        """Initialize UMAPVisualizer with configuration.

        Args:
            config: Configuration dictionary. If None, uses default config.

        """
        self.config = config or self._get_default_config()

        # Create unified class labels (DESED 10 + MAESTRO 11)
        class_labels = self._create_unified_class_labels()

        # Initialize components with config
        self.loader = FeatureLoader(class_labels)

        umap_config = self.config["umap"]
        self.reducer = UMAPReducer(
            n_neighbors=umap_config["n_neighbors"],
            min_dist=umap_config["min_dist"],
            metric=umap_config["metric"],
            random_state=umap_config["random_state"],
        )

        plot_config = self.config["plot"]
        output_config = self.config["output"]
        self.generator = PlotGenerator(
            output_dir=output_config["dir"],
            dpi=plot_config["dpi"],
            figsize=tuple(plot_config["figsize"]),
            palette=plot_config["palette"],
        )

        self.logger = LOGGER

    def run_class_separation(
        self,
        input_path: str,
        model_type: str = "student",
    ) -> None:
        """Execute class separation visualization.

        Loads features from a single .npz file, reduces to 2D using UMAP,
        and generates a scatter plot colored by event class.

        Args:
            input_path: Path to .npz file containing features.
            model_type: "student" or "teacher" model type.

        """
        LOGGER.info("Running class separation visualization")
        LOGGER.info("Input: %s, Model: %s", input_path, model_type)

        # Load features
        features, primary_classes, _, _ = self.loader.load_features(
            input_path,
            model_type=model_type,
        )

        # Apply UMAP
        embeddings = self.reducer.fit_transform(features)

        # Get class names for legend
        class_names = list(self.loader.class_labels.keys())

        # Generate plot
        title = f"UMAP: Class Separation ({model_type.capitalize()} Model)"
        filename_prefix = f"class_separation_{model_type}"

        png_path, pdf_path = self.generator.plot_class_separation(
            embeddings=embeddings,
            class_labels=primary_classes,
            class_names=class_names,
            title=title,
            filename_prefix=filename_prefix,
        )

        LOGGER.info("Class separation visualization completed")
        LOGGER.info("Output files: %s, %s", png_path, pdf_path)

    def run_domain_comparison(
        self,
        input_paths: list[str],
        model_type: str = "student",
    ) -> None:
        """Execute domain comparison visualization.

        Loads features from multiple .npz files, reduces to 2D using UMAP,
        and generates a scatter plot colored by domain.

        Args:
            input_paths: List of paths to .npz files.
            model_type: "student" or "teacher" model type.

        """
        LOGGER.info("Running domain comparison visualization")
        LOGGER.info("Inputs: %s, Model: %s", input_paths, model_type)

        # Load and combine multiple datasets
        features, _, _, filenames = self.loader.load_multiple_datasets(
            input_paths,
            model_type=model_type,
        )

        # Extract domain labels from filenames
        domain_labels = self.loader.extract_domain_labels(filenames)

        # Apply UMAP
        embeddings = self.reducer.fit_transform(features)

        # Define domain names
        domain_names = [
            "DESED Synthetic",
            "DESED Real",
            "MAESTRO Training",
            "MAESTRO Validation",
        ]

        # Generate plot
        title = f"UMAP: Domain Comparison ({model_type.capitalize()} Model)"
        filename_prefix = f"domain_comparison_{model_type}"

        png_path, pdf_path = self.generator.plot_domain_comparison(
            embeddings=embeddings,
            domain_labels=domain_labels,
            domain_names=domain_names,
            title=title,
            filename_prefix=filename_prefix,
        )

        LOGGER.info("Domain comparison visualization completed")
        LOGGER.info("Output files: %s, %s", png_path, pdf_path)

    def run_mixstyle_comparison(
        self,
        before_path: str,
        after_path: str,
        model_type: str = "student",
    ) -> None:
        """Execute MixStyle effect comparison visualization.

        Loads features from two .npz files (before/after MixStyle), combines them,
        applies UMAP to create a shared embedding space, then generates side-by-side
        subplots to compare domain distributions.

        Args:
            before_path: Path to .npz file before MixStyle.
            after_path: Path to .npz file after MixStyle.
            model_type: "student" or "teacher" model type.

        """
        LOGGER.info("Running MixStyle effect comparison visualization")
        LOGGER.info(
            "Before: %s, After: %s, Model: %s",
            before_path,
            after_path,
            model_type,
        )

        # Load features from both checkpoints
        features_before, _, _, filenames_before = self.loader.load_features(
            before_path,
            model_type=model_type,
        )
        features_after, _, _, filenames_after = self.loader.load_features(
            after_path,
            model_type=model_type,
        )

        # Extract domain labels
        domain_labels_before = self.loader.extract_domain_labels(filenames_before)
        domain_labels_after = self.loader.extract_domain_labels(filenames_after)

        # Combine features for shared UMAP embedding space
        combined_features = np.concatenate([features_before, features_after], axis=0)
        LOGGER.info(
            "Combined %d samples (before) + %d samples (after) = %d total",
            len(features_before),
            len(features_after),
            len(combined_features),
        )

        # Apply UMAP to combined features
        combined_embeddings = self.reducer.fit_transform(combined_features)

        # Split embeddings back into before/after
        n_before = len(features_before)
        embeddings_before = combined_embeddings[:n_before]
        embeddings_after = combined_embeddings[n_before:]

        # Use domain labels from before (should match after)
        domain_labels = domain_labels_before

        # Define domain names
        domain_names = [
            "DESED Synthetic",
            "DESED Real",
            "MAESTRO Training",
            "MAESTRO Validation",
        ]

        # Generate subplot comparison
        title_before = f"Before MixStyle ({model_type.capitalize()})"
        title_after = f"After MixStyle ({model_type.capitalize()})"
        filename_prefix = f"mixstyle_effect_{model_type}"

        png_path, pdf_path = self.generator.plot_mixstyle_comparison(
            embeddings_before=embeddings_before,
            embeddings_after=embeddings_after,
            domain_labels=domain_labels,
            domain_names=domain_names,
            title_before=title_before,
            title_after=title_after,
            filename_prefix=filename_prefix,
        )

        LOGGER.info("MixStyle comparison visualization completed")
        LOGGER.info("Output files: %s, %s", png_path, pdf_path)

    @staticmethod
    def _create_unified_class_labels() -> dict[str, int]:
        """Create unified class labels (DESED 10 + MAESTRO 11).

        Returns:
            class_labels: Dictionary mapping class names to indices (0-20).

        """
        # Start with DESED classes (0-9)
        class_labels = dict(classes_labels_desed)

        # Add MAESTRO real eval classes (10-20)
        maestro_classes = sorted(classes_labels_maestro_real_eval)
        for i, class_name in enumerate(maestro_classes):
            class_labels[class_name] = 10 + i

        return class_labels

    @staticmethod
    def parse_args() -> argparse.Namespace:
        """Parse command-line arguments.

        Returns:
            args: Parsed arguments namespace.

        """
        parser = argparse.ArgumentParser(
            description="UMAP visualization for DCASE2024 SED features",
            formatter_class=argparse.RawDescriptionHelpFormatter,
        )

        # Create subparsers for different modes
        subparsers = parser.add_subparsers(dest="mode", required=True)

        # Common arguments for all modes
        def add_common_args(subparser: argparse.ArgumentParser) -> None:
            subparser.add_argument(
                "--model_type",
                type=str,
                default="student",
                choices=["student", "teacher"],
                help="Model type (default: student)",
            )
            subparser.add_argument(
                "--config",
                type=str,
                default=None,
                help="Path to YAML configuration file",
            )
            subparser.add_argument(
                "--output_dir",
                type=str,
                default=None,
                help="Output directory for plots",
            )
            subparser.add_argument(
                "--n_neighbors",
                type=int,
                default=None,
                help="UMAP n_neighbors parameter",
            )
            subparser.add_argument(
                "--min_dist",
                type=float,
                default=None,
                help="UMAP min_dist parameter",
            )
            subparser.add_argument(
                "--metric",
                type=str,
                default=None,
                help="UMAP distance metric",
            )
            subparser.add_argument(
                "--dpi",
                type=int,
                default=None,
                help="Output image DPI",
            )
            subparser.add_argument(
                "--figsize",
                type=int,
                nargs=2,
                default=None,
                help="Figure size (width height)",
            )
            subparser.add_argument(
                "--palette",
                type=str,
                default=None,
                help="Seaborn color palette name",
            )

        # Class separation mode
        parser_class = subparsers.add_parser(
            "class_separation",
            help="Visualize class separation in feature space",
        )
        parser_class.add_argument(
            "--input",
            type=str,
            required=True,
            help="Path to .npz file with features",
        )
        add_common_args(parser_class)

        # Domain comparison mode
        parser_domain = subparsers.add_parser(
            "domain_comparison",
            help="Compare feature distributions across domains",
        )
        parser_domain.add_argument(
            "--inputs",
            type=str,
            nargs="+",
            required=True,
            help="Paths to .npz files from different domains",
        )
        add_common_args(parser_domain)

        # MixStyle effect mode
        parser_mixstyle = subparsers.add_parser(
            "mixstyle_effect",
            help="Compare features before/after MixStyle",
        )
        parser_mixstyle.add_argument(
            "--before",
            type=str,
            required=True,
            help="Path to .npz file before MixStyle",
        )
        parser_mixstyle.add_argument(
            "--after",
            type=str,
            required=True,
            help="Path to .npz file after MixStyle",
        )
        add_common_args(parser_mixstyle)

        return parser.parse_args()

    @staticmethod
    def load_config(config_path: str) -> dict[str, Any]:
        """Load configuration from YAML file.

        Args:
            config_path: Path to YAML configuration file.

        Returns:
            config: Configuration dictionary.

        Raises:
            FileNotFoundError: If config file does not exist.

        """
        path = Path(config_path)
        if not path.exists():
            msg = f"Config file not found: {config_path}"
            raise FileNotFoundError(msg)

        with path.open("r") as f:
            config = yaml.safe_load(f)

        LOGGER.info("Loaded configuration from %s", config_path)
        return config

    @staticmethod
    def merge_config(
        cli_args: argparse.Namespace,
        yaml_config: dict[str, Any],
        defaults: dict[str, Any],
    ) -> dict[str, Any]:
        """Merge configuration from CLI, YAML, and defaults.

        Priority: CLI arguments > YAML config > defaults

        Args:
            cli_args: Command-line arguments namespace.
            yaml_config: Configuration from YAML file.
            defaults: Default configuration dictionary.

        Returns:
            merged: Merged configuration dictionary.

        """
        # Start with a deep copy of defaults
        merged = {
            "umap": dict(defaults["umap"]),
            "plot": dict(defaults["plot"]),
            "output": dict(defaults["output"]),
            "logging": dict(defaults["logging"]),
        }

        # Apply YAML config (overrides defaults)
        if "umap" in yaml_config:
            merged["umap"].update(yaml_config["umap"])
        if "plot" in yaml_config:
            merged["plot"].update(yaml_config["plot"])
        if "output" in yaml_config:
            merged["output"].update(yaml_config["output"])
        if "logging" in yaml_config:
            merged["logging"].update(yaml_config["logging"])

        # Apply CLI args (overrides YAML and defaults)
        if hasattr(cli_args, "n_neighbors") and cli_args.n_neighbors is not None:
            merged["umap"]["n_neighbors"] = cli_args.n_neighbors
        if hasattr(cli_args, "min_dist") and cli_args.min_dist is not None:
            merged["umap"]["min_dist"] = cli_args.min_dist
        if hasattr(cli_args, "metric") and cli_args.metric is not None:
            merged["umap"]["metric"] = cli_args.metric

        if hasattr(cli_args, "dpi") and cli_args.dpi is not None:
            merged["plot"]["dpi"] = cli_args.dpi
        if hasattr(cli_args, "figsize") and cli_args.figsize is not None:
            merged["plot"]["figsize"] = tuple(cli_args.figsize)
        if hasattr(cli_args, "palette") and cli_args.palette is not None:
            merged["plot"]["palette"] = cli_args.palette

        if hasattr(cli_args, "output_dir") and cli_args.output_dir is not None:
            merged["output"]["dir"] = cli_args.output_dir

        return merged

    @staticmethod
    def _get_default_config() -> dict[str, Any]:
        """Get default configuration.

        Returns:
            config: Default configuration dictionary.

        """
        return {
            "umap": {
                "n_neighbors": 15,
                "min_dist": 0.1,
                "metric": "euclidean",
                "random_state": 42,
            },
            "plot": {
                "dpi": 300,
                "figsize": (12, 8),
                "palette": "colorblind",
            },
            "output": {
                "dir": "output",
            },
            "logging": {
                "level": "INFO",
            },
        }


def main() -> None:
    """CLI entry point for UMAP visualization."""
    # Parse arguments
    args = UMAPVisualizer.parse_args()

    # Load config from YAML if specified
    yaml_config = {}
    if hasattr(args, "config") and args.config is not None:
        yaml_config = UMAPVisualizer.load_config(args.config)

    # Merge configurations
    defaults = UMAPVisualizer._get_default_config()
    config = UMAPVisualizer.merge_config(args, yaml_config, defaults)

    # Setup logging
    logging.basicConfig(
        level=getattr(logging, config["logging"]["level"]),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Initialize visualizer
    visualizer = UMAPVisualizer(config)

    # Execute mode-specific visualization
    try:
        if args.mode == "class_separation":
            visualizer.run_class_separation(args.input, args.model_type)
        elif args.mode == "domain_comparison":
            visualizer.run_domain_comparison(args.inputs, args.model_type)
        elif args.mode == "mixstyle_effect":
            visualizer.run_mixstyle_comparison(
                args.before,
                args.after,
                args.model_type,
            )
        else:
            msg = f"Unknown mode: {args.mode}"
            raise ValueError(msg)
    except Exception as e:
        LOGGER.error("Visualization failed: %s", e)
        sys.exit(1)


if __name__ == "__main__":
    main()
