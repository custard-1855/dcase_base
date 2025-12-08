"""Plot generation and output for UMAP visualization."""

import logging
from datetime import UTC, datetime
from pathlib import Path

import matplotlib.figure
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns  # type: ignore[import-untyped]
from matplotlib.axes import Axes
from numpy.typing import NDArray

LOGGER = logging.getLogger(__name__)

# Constants for UMAP visualization
N_UMAP_DIMENSIONS = 2  # UMAP reduces to 2D
N_CLASSES = 21  # DESED (10) + MAESTRO (11) classes
MAX_CLASS_INDEX = 20  # Classes are indexed 0-20


class PlotGenerator:
    """Generate publication-quality plots for UMAP visualization.

    Handles plot generation, styling, and file output for class separation,
    domain comparison, and MixStyle effect visualization.

    Attributes:
        output_dir: Output directory path for saving plots.
        dpi: Resolution for output images (300+ recommended for publication).
        figsize: Figure size as (width, height) in inches.
        palette: Seaborn color palette name (default: 'colorblind').
        font_sizes: Dictionary of font sizes for title, label, and legend.

    """

    def __init__(
        self,
        output_dir: str = "output",
        dpi: int = 300,
        figsize: tuple[int, int] = (12, 8),
        palette: str = "colorblind",
        *,
        font_size_title: int = 14,
        font_size_label: int = 12,
        font_size_legend: int = 10,
    ) -> None:
        """Initialize PlotGenerator.

        Args:
            output_dir: Output directory for saving plots.
            dpi: Resolution for output images (default: 300).
            figsize: Figure size as (width, height) in inches (default: (12, 8)).
            palette: Seaborn color palette name (default: 'colorblind').
            font_size_title: Font size for titles (default: 14).
            font_size_label: Font size for axis labels (default: 12).
            font_size_legend: Font size for legend (default: 10).

        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.dpi = dpi
        self.figsize = figsize
        self.palette = palette
        self.font_sizes = {
            "title": font_size_title,
            "label": font_size_label,
            "legend": font_size_legend,
        }

        LOGGER.debug(
            "PlotGenerator initialized: output_dir=%s, dpi=%d, figsize=%s, palette=%s",
            self.output_dir,
            self.dpi,
            self.figsize,
            self.palette,
        )

    def _get_color_palette(self, n_colors: int) -> list[tuple[float, float, float]]:
        """Get color palette with specified number of colors.

        Args:
            n_colors: Number of colors needed.

        Returns:
            colors: List of RGB color tuples.

        """
        # Get colors from seaborn palette
        colors = sns.color_palette(self.palette, n_colors=n_colors)

        # Convert to list of tuples (seaborn returns list-like objects)
        return [tuple(color) for color in colors]

    def _generate_filename(self, prefix: str, extension: str) -> str:
        """Generate filename with timestamp.

        Args:
            prefix: Filename prefix (e.g., 'class_separation').
            extension: File extension without dot (e.g., 'png', 'pdf').

        Returns:
            filename: Generated filename in format '{prefix}_{timestamp}.{extension}'.

        """
        timestamp = datetime.now(tz=UTC).strftime("%Y%m%d_%H%M%S")
        return f"{prefix}_{timestamp}.{extension}"

    def _plot_domain_scatter(
        self,
        ax: Axes,
        embeddings: NDArray[np.float32],
        domain_labels: NDArray[np.int32],
        domain_names: list[str],
        colors: list[tuple[float, float, float]],
        markers: list[str],
    ) -> None:
        """Plot domain-colored scatter plot on given axes.

        Helper method to reduce code duplication in domain-based visualizations.

        Args:
            ax: Matplotlib axes to plot on.
            embeddings: 2D UMAP embeddings with shape (N, 2).
            domain_labels: Domain indices with shape (N,).
            domain_names: List of domain names.
            colors: List of RGB color tuples for each domain.
            markers: List of marker shapes for each domain.

        """
        n_domains = len(domain_names)

        for domain_idx in range(n_domains):
            mask = domain_labels == domain_idx
            n_samples = np.sum(mask)

            if n_samples > 0:
                marker = markers[domain_idx % len(markers)]
                label = f"{domain_names[domain_idx]} (N={n_samples})"

                ax.scatter(
                    embeddings[mask, 0],
                    embeddings[mask, 1],
                    c=[colors[domain_idx]],
                    marker=marker,
                    label=label,
                    alpha=0.6,
                    s=40,
                    edgecolors="black",
                    linewidths=0.5,
                )

    def _apply_publication_style(
        self,
        fig: matplotlib.figure.Figure,
        ax: Axes,
    ) -> None:
        """Apply publication-quality styling to plot.

        Args:
            fig: Matplotlib figure object.
            ax: Matplotlib axes object.

        """
        # Set font sizes for title and axis labels
        ax.title.set_fontsize(self.font_sizes["title"])
        ax.xaxis.label.set_fontsize(self.font_sizes["label"])
        ax.yaxis.label.set_fontsize(self.font_sizes["label"])

        # Configure legend if it exists
        legend = ax.get_legend()
        if legend is not None:
            # Place legend outside plot area (right side)
            ax.legend(
                bbox_to_anchor=(1.05, 1),
                loc="upper left",
                fontsize=self.font_sizes["legend"],
                frameon=True,
                fancybox=False,
                shadow=False,
            )

        # Set tick label sizes
        ax.tick_params(axis="both", which="major", labelsize=10)

        # Enable grid for better readability
        ax.grid(visible=True, alpha=0.3, linestyle="--", linewidth=0.5)

        # Adjust layout to prevent label cutoff
        fig.tight_layout()

        LOGGER.debug("Applied publication style to plot")

    def plot_class_separation(
        self,
        embeddings: NDArray[np.float32],
        class_labels: NDArray[np.int32],
        class_names: list[str],
        title: str = "UMAP: Class Separation",
        filename_prefix: str = "class_separation",
    ) -> tuple[str, str]:
        """Generate class separation visualization plot.

        Creates a scatter plot showing UMAP embeddings colored by class labels,
        demonstrating class separability in the feature space.

        Args:
            embeddings: 2D UMAP embeddings with shape (N, 2).
            class_labels: Class indices with shape (N,), values in range [0, 20].
            class_names: List of 21 class names (DESED 10 + MAESTRO 11).
            title: Plot title (default: "UMAP: Class Separation").
            filename_prefix: Prefix for output filenames (default: "class_separation").

        Returns:
            tuple: (png_path, pdf_path) - Paths to saved PNG and PDF files.

        Raises:
            ValueError: If input shapes are invalid or class labels are out of range.

        """
        # Validate inputs
        if embeddings.shape[1] != N_UMAP_DIMENSIONS:
            msg = f"embeddings must be {N_UMAP_DIMENSIONS}D, got shape {embeddings.shape}"
            raise ValueError(msg)

        if len(embeddings) != len(class_labels):
            msg = (
                f"embeddings and class_labels length mismatch: "
                f"{len(embeddings)} != {len(class_labels)}"
            )
            raise ValueError(msg)

        if len(class_names) != N_CLASSES:
            msg = f"class_names must contain {N_CLASSES} names, got {len(class_names)}"
            raise ValueError(msg)

        if class_labels.min() < 0 or class_labels.max() > MAX_CLASS_INDEX:
            msg = (
                f"class labels must be in range [0, {MAX_CLASS_INDEX}], "
                f"got range [{class_labels.min()}, {class_labels.max()}]"
            )
            raise ValueError(msg)

        LOGGER.info(
            "Generating class separation plot: %d samples, 21 classes",
            len(embeddings),
        )

        # Create figure and axis
        fig, ax = plt.subplots(figsize=self.figsize)

        # Get colors for all classes
        colors = self._get_color_palette(N_CLASSES)

        # Plot each class separately for proper legend
        for class_idx in range(N_CLASSES):
            mask = class_labels == class_idx
            if np.any(mask):
                ax.scatter(
                    embeddings[mask, 0],
                    embeddings[mask, 1],
                    c=[colors[class_idx]],
                    label=class_names[class_idx],
                    alpha=0.6,
                    s=20,
                    edgecolors="none",
                )

        # Set labels and title
        ax.set_xlabel("UMAP Dimension 1")
        ax.set_ylabel("UMAP Dimension 2")
        ax.set_title(title)

        # Create legend
        ax.legend()

        # Apply publication style
        self._apply_publication_style(fig, ax)

        # Generate filenames and save
        png_filename = self._generate_filename(filename_prefix, "png")
        pdf_filename = self._generate_filename(filename_prefix, "pdf")

        png_path = str(self.output_dir / png_filename)
        pdf_path = str(self.output_dir / pdf_filename)

        fig.savefig(png_path, dpi=self.dpi, bbox_inches="tight")
        fig.savefig(pdf_path, format="pdf", bbox_inches="tight")

        plt.close(fig)

        LOGGER.info("Saved class separation plots: %s, %s", png_path, pdf_path)

        return png_path, pdf_path

    def plot_domain_comparison(
        self,
        embeddings: NDArray[np.float32],
        domain_labels: NDArray[np.int32],
        domain_names: list[str],
        title: str = "UMAP: Domain Comparison",
        filename_prefix: str = "domain_comparison",
    ) -> tuple[str, str]:
        """Generate domain comparison visualization plot.

        Creates a scatter plot showing UMAP embeddings colored by domain labels,
        with different marker shapes for each domain to enhance visual distinction.

        Args:
            embeddings: 2D UMAP embeddings with shape (N, 2).
            domain_labels: Domain indices with shape (N,).
            domain_names: List of domain names corresponding to indices.
            title: Plot title (default: "UMAP: Domain Comparison").
            filename_prefix: Prefix for output filenames (default: "domain_comparison").

        Returns:
            tuple: (png_path, pdf_path) - Paths to saved PNG and PDF files.

        Raises:
            ValueError: If input shapes are invalid or domain labels are out of range.

        """
        # Validate inputs
        if embeddings.shape[1] != N_UMAP_DIMENSIONS:
            msg = f"embeddings must be {N_UMAP_DIMENSIONS}D, got shape {embeddings.shape}"
            raise ValueError(msg)

        if len(embeddings) != len(domain_labels):
            msg = (
                f"embeddings and domain_labels length mismatch: "
                f"{len(embeddings)} != {len(domain_labels)}"
            )
            raise ValueError(msg)

        n_domains = len(domain_names)
        if domain_labels.min() < 0 or domain_labels.max() >= n_domains:
            msg = (
                f"domain labels must be in range [0, {n_domains - 1}], "
                f"got range [{domain_labels.min()}, {domain_labels.max()}]"
            )
            raise ValueError(msg)

        LOGGER.info(
            "Generating domain comparison plot: %d samples, %d domains",
            len(embeddings),
            n_domains,
        )

        # Create figure and axis
        fig, ax = plt.subplots(figsize=self.figsize)

        # Get colors for all domains
        colors = self._get_color_palette(n_domains)

        # Define marker shapes for different domains
        markers = ["o", "s", "^", "D", "v", "<", ">", "p", "*", "h"]

        # Plot each domain separately with different colors and markers
        for domain_idx in range(n_domains):
            mask = domain_labels == domain_idx
            n_samples = np.sum(mask)

            if n_samples > 0:
                marker = markers[domain_idx % len(markers)]
                label = f"{domain_names[domain_idx]} (N={n_samples})"

                ax.scatter(
                    embeddings[mask, 0],
                    embeddings[mask, 1],
                    c=[colors[domain_idx]],
                    marker=marker,
                    label=label,
                    alpha=0.6,
                    s=40,
                    edgecolors="black",
                    linewidths=0.5,
                )

        # Set labels and title
        ax.set_xlabel("UMAP Dimension 1")
        ax.set_ylabel("UMAP Dimension 2")
        ax.set_title(title)

        # Create legend
        ax.legend()

        # Apply publication style
        self._apply_publication_style(fig, ax)

        # Generate filenames and save
        png_filename = self._generate_filename(filename_prefix, "png")
        pdf_filename = self._generate_filename(filename_prefix, "pdf")

        png_path = str(self.output_dir / png_filename)
        pdf_path = str(self.output_dir / pdf_filename)

        fig.savefig(png_path, dpi=self.dpi, bbox_inches="tight")
        fig.savefig(pdf_path, format="pdf", bbox_inches="tight")

        plt.close(fig)

        LOGGER.info("Saved domain comparison plots: %s, %s", png_path, pdf_path)

        return png_path, pdf_path

    def plot_mixstyle_comparison(
        self,
        embeddings_before: NDArray[np.float32],
        embeddings_after: NDArray[np.float32],
        domain_labels: NDArray[np.int32],
        domain_names: list[str],
        title_before: str = "Before MixStyle",
        title_after: str = "After MixStyle",
        filename_prefix: str = "mixstyle_effect",
    ) -> tuple[str, str]:
        """Generate MixStyle effect comparison visualization with subplots.

        Creates two side-by-side subplots showing UMAP embeddings before and after
        MixStyle application, colored by domain labels. Both subplots share unified
        axis scales for direct comparison.

        Args:
            embeddings_before: 2D UMAP embeddings before MixStyle with shape (N, 2).
            embeddings_after: 2D UMAP embeddings after MixStyle with shape (N, 2).
            domain_labels: Domain indices with shape (N,).
            domain_names: List of domain names corresponding to indices.
            title_before: Title for left subplot (default: "Before MixStyle").
            title_after: Title for right subplot (default: "After MixStyle").
            filename_prefix: Prefix for output filenames (default: "mixstyle_effect").

        Returns:
            tuple: (png_path, pdf_path) - Paths to saved PNG and PDF files.

        Raises:
            ValueError: If input shapes are invalid or domain labels are out of range.

        """
        # Validate embeddings are 2D
        if embeddings_before.shape[1] != N_UMAP_DIMENSIONS:
            msg = f"embeddings must be {N_UMAP_DIMENSIONS}D, got shape {embeddings_before.shape}"
            raise ValueError(msg)

        if embeddings_after.shape[1] != N_UMAP_DIMENSIONS:
            msg = f"embeddings must be {N_UMAP_DIMENSIONS}D, got shape {embeddings_after.shape}"
            raise ValueError(msg)

        # Validate before and after have matching shapes
        if embeddings_before.shape != embeddings_after.shape:
            msg = (
                f"embeddings_before and embeddings_after must have same shape, "
                f"got {embeddings_before.shape} and {embeddings_after.shape}"
            )
            raise ValueError(msg)

        # Validate domain_labels length matches embeddings
        if len(embeddings_before) != len(domain_labels):
            msg = (
                f"embeddings and domain_labels length mismatch: "
                f"{len(embeddings_before)} != {len(domain_labels)}"
            )
            raise ValueError(msg)

        # Validate domain labels are in valid range
        n_domains = len(domain_names)
        if domain_labels.min() < 0 or domain_labels.max() >= n_domains:
            msg = (
                f"domain labels must be in range [0, {n_domains - 1}], "
                f"got range [{domain_labels.min()}, {domain_labels.max()}]"
            )
            raise ValueError(msg)

        LOGGER.info(
            "Generating MixStyle comparison plot: %d samples, %d domains",
            len(embeddings_before),
            n_domains,
        )

        # Create figure with two subplots side by side
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=self.figsize)

        # Get colors for all domains
        colors = self._get_color_palette(n_domains)

        # Define marker shapes for different domains
        markers = ["o", "s", "^", "D", "v", "<", ">", "p", "*", "h"]

        # Plot both subplots using helper method
        self._plot_domain_scatter(
            ax1,
            embeddings_before,
            domain_labels,
            domain_names,
            colors,
            markers,
        )
        self._plot_domain_scatter(
            ax2,
            embeddings_after,
            domain_labels,
            domain_names,
            colors,
            markers,
        )

        # Set labels and titles for both subplots
        ax1.set_xlabel("UMAP Dimension 1")
        ax1.set_ylabel("UMAP Dimension 2")
        ax1.set_title(title_before)

        ax2.set_xlabel("UMAP Dimension 1")
        ax2.set_ylabel("UMAP Dimension 2")
        ax2.set_title(title_after)

        # Create legends for both subplots
        ax1.legend()
        ax2.legend()

        # Unify axis scales across both subplots
        # Calculate combined min/max for both embeddings
        all_x = np.concatenate([embeddings_before[:, 0], embeddings_after[:, 0]])
        all_y = np.concatenate([embeddings_before[:, 1], embeddings_after[:, 1]])

        x_min, x_max = all_x.min(), all_x.max()
        y_min, y_max = all_y.min(), all_y.max()

        # Add small margin
        x_margin = (x_max - x_min) * 0.05
        y_margin = (y_max - y_min) * 0.05

        ax1.set_xlim(x_min - x_margin, x_max + x_margin)
        ax1.set_ylim(y_min - y_margin, y_max + y_margin)
        ax2.set_xlim(x_min - x_margin, x_max + x_margin)
        ax2.set_ylim(y_min - y_margin, y_max + y_margin)

        # Apply publication style to both subplots
        self._apply_publication_style(fig, ax1)
        self._apply_publication_style(fig, ax2)

        # Generate filenames and save
        png_filename = self._generate_filename(filename_prefix, "png")
        pdf_filename = self._generate_filename(filename_prefix, "pdf")

        png_path = str(self.output_dir / png_filename)
        pdf_path = str(self.output_dir / pdf_filename)

        fig.savefig(png_path, dpi=self.dpi, bbox_inches="tight")
        fig.savefig(pdf_path, format="pdf", bbox_inches="tight")

        plt.close(fig)

        LOGGER.info("Saved MixStyle comparison plots: %s, %s", png_path, pdf_path)

        return png_path, pdf_path
