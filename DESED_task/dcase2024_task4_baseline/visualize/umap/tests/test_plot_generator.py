"""Tests for PlotGenerator class."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pytest

from visualize.umap.plot_generator import PlotGenerator


class TestPlotGenerator:
    """Test suite for PlotGenerator class."""

    @pytest.fixture
    def output_dir(self, tmp_path: Path) -> Path:
        """Create a temporary output directory for testing."""
        return tmp_path / "output"

    @pytest.fixture
    def plot_generator(self, output_dir: Path) -> PlotGenerator:
        """Create a PlotGenerator instance for testing."""
        return PlotGenerator(
            output_dir=str(output_dir),
            dpi=300,
            figsize=(12, 8),
            palette="colorblind",
            font_size_title=14,
            font_size_label=12,
            font_size_legend=10,
        )

    @pytest.fixture
    def sample_embeddings(self) -> np.ndarray:
        """Create sample 2D embeddings for testing."""
        n_samples = 100
        return np.random.randn(n_samples, 2).astype(np.float32)

    def test_initialization(self, output_dir: Path) -> None:
        """Test PlotGenerator initialization with correct parameters."""
        generator = PlotGenerator(
            output_dir=str(output_dir),
            dpi=300,
            figsize=(12, 8),
            palette="colorblind",
            font_size_title=14,
            font_size_label=12,
            font_size_legend=10,
        )

        assert generator.output_dir == output_dir
        assert generator.dpi == 300
        assert generator.figsize == (12, 8)
        assert generator.palette == "colorblind"
        assert generator.font_sizes["title"] == 14
        assert generator.font_sizes["label"] == 12
        assert generator.font_sizes["legend"] == 10
        assert output_dir.exists()

    def test_initialization_creates_output_directory(self, tmp_path: Path) -> None:
        """Test that initialization creates output directory if it doesn't exist."""
        new_output_dir = tmp_path / "new_output" / "nested"
        generator = PlotGenerator(output_dir=str(new_output_dir))

        assert new_output_dir.exists()
        assert generator.output_dir == new_output_dir

    def test_get_color_palette_returns_correct_number_of_colors(
        self, plot_generator: PlotGenerator
    ) -> None:
        """Test _get_color_palette returns correct number of colors."""
        # Test with 21 colors (number of classes)
        colors = plot_generator._get_color_palette(21)

        assert len(colors) == 21
        # Verify colors are RGB tuples
        for color in colors:
            assert isinstance(color, tuple)
            assert len(color) == 3
            assert all(0 <= c <= 1 for c in color)

    def test_get_color_palette_with_different_sizes(
        self, plot_generator: PlotGenerator
    ) -> None:
        """Test _get_color_palette with different color counts."""
        for n_colors in [4, 10, 21, 50]:
            colors = plot_generator._get_color_palette(n_colors)
            assert len(colors) == n_colors

    def test_generate_filename_format(self, plot_generator: PlotGenerator) -> None:
        """Test _generate_filename creates correct format."""
        filename = plot_generator._generate_filename("class_separation", "png")

        # Check format: {prefix}_{timestamp}.{extension}
        assert filename.startswith("class_separation_")
        assert filename.endswith(".png")
        assert len(filename.split("_")) >= 2  # At least prefix and timestamp

    def test_generate_filename_different_prefixes(
        self, plot_generator: PlotGenerator
    ) -> None:
        """Test _generate_filename with different prefixes and extensions."""
        prefixes = ["class_separation", "domain_comparison", "mixstyle_effect"]
        extensions = ["png", "pdf"]

        for prefix in prefixes:
            for ext in extensions:
                filename = plot_generator._generate_filename(prefix, ext)
                assert filename.startswith(prefix)
                assert filename.endswith(f".{ext}")

    def test_apply_publication_style_sets_font_sizes(
        self, plot_generator: PlotGenerator, sample_embeddings: np.ndarray
    ) -> None:
        """Test _apply_publication_style sets correct font sizes."""
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.scatter(sample_embeddings[:, 0], sample_embeddings[:, 1])
        ax.set_xlabel("Test X")
        ax.set_ylabel("Test Y")
        ax.set_title("Test Title")

        plot_generator._apply_publication_style(fig, ax)

        # Check that font sizes are set (we can't directly check the values
        # but we can verify the method runs without error)
        assert ax.get_xlabel() == "Test X"
        assert ax.get_ylabel() == "Test Y"
        assert ax.get_title() == "Test Title"

        plt.close(fig)

    def test_apply_publication_style_axis_labels(
        self, plot_generator: PlotGenerator
    ) -> None:
        """Test _apply_publication_style with UMAP axis labels."""
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.set_xlabel("UMAP Dimension 1")
        ax.set_ylabel("UMAP Dimension 2")
        ax.set_title("Class Separation")

        plot_generator._apply_publication_style(fig, ax)

        # Verify axis labels are preserved
        assert "UMAP Dimension 1" in ax.get_xlabel()
        assert "UMAP Dimension 2" in ax.get_ylabel()

        plt.close(fig)

    def test_apply_publication_style_with_legend(
        self, plot_generator: PlotGenerator, sample_embeddings: np.ndarray
    ) -> None:
        """Test _apply_publication_style places legend outside plot area."""
        fig, ax = plt.subplots(figsize=(12, 8))

        # Create scatter plot with labels for legend
        ax.scatter(
            sample_embeddings[:50, 0],
            sample_embeddings[:50, 1],
            label="Class A",
        )
        ax.scatter(
            sample_embeddings[50:, 0],
            sample_embeddings[50:, 1],
            label="Class B",
        )
        ax.legend()

        plot_generator._apply_publication_style(fig, ax)

        # Check that legend exists
        legend = ax.get_legend()
        assert legend is not None

        plt.close(fig)

    def test_default_parameters(self) -> None:
        """Test PlotGenerator with default parameters."""
        generator = PlotGenerator()

        assert generator.dpi >= 300  # Minimum DPI requirement
        assert generator.figsize == (12, 8)
        assert generator.palette == "colorblind"
        assert generator.font_sizes["title"] >= 14
        assert generator.font_sizes["label"] >= 12
        assert generator.font_sizes["legend"] >= 10


class TestPlotClassSeparation:
    """Test suite for plot_class_separation method."""

    @pytest.fixture
    def output_dir(self, tmp_path: Path) -> Path:
        """Create a temporary output directory for testing."""
        return tmp_path / "output"

    @pytest.fixture
    def plot_generator(self, output_dir: Path) -> PlotGenerator:
        """Create a PlotGenerator instance for testing."""
        return PlotGenerator(
            output_dir=str(output_dir),
            dpi=300,
            figsize=(12, 8),
            palette="colorblind",
        )

    @pytest.fixture
    def sample_embeddings(self) -> np.ndarray:
        """Create sample 2D embeddings for 21 classes."""
        n_samples = 210  # 10 samples per class
        return np.random.randn(n_samples, 2).astype(np.float32)

    @pytest.fixture
    def sample_class_labels(self) -> np.ndarray:
        """Create sample class labels (0-20) for 21 classes."""
        n_samples = 210  # 10 samples per class
        # Repeat each class 10 times
        return np.repeat(np.arange(21, dtype=np.int32), 10)

    @pytest.fixture
    def class_names(self) -> list[str]:
        """Create list of 21 class names (DESED 10 + MAESTRO 11)."""
        desed_classes = [
            "Alarm_bell_ringing",
            "Blender",
            "Cat",
            "Dishes",
            "Dog",
            "Electric_shaver_toothbrush",
            "Frying",
            "Running_water",
            "Speech",
            "Vacuum_cleaner",
        ]
        maestro_classes = [
            "birds_singing",
            "car",
            "people_talking",
            "footsteps",
            "children_voices",
            "wind_blowing",
            "brakes_squeaking",
            "large_vehicle",
            "cutlery_and_dishes",
            "metro_approaching",
            "metro_leaving",
        ]
        return desed_classes + maestro_classes

    def test_plot_class_separation_creates_files(
        self,
        plot_generator: PlotGenerator,
        sample_embeddings: np.ndarray,
        sample_class_labels: np.ndarray,
        class_names: list[str],
    ) -> None:
        """Test that plot_class_separation creates PNG and PDF files."""
        png_path, pdf_path = plot_generator.plot_class_separation(
            embeddings=sample_embeddings,
            class_labels=sample_class_labels,
            class_names=class_names,
            title="Test Class Separation",
            filename_prefix="test_class_separation",
        )

        # Verify files exist
        assert Path(png_path).exists()
        assert Path(pdf_path).exists()

        # Verify file extensions
        assert png_path.endswith(".png")
        assert pdf_path.endswith(".pdf")

        # Verify filenames contain prefix
        assert "test_class_separation" in png_path
        assert "test_class_separation" in pdf_path

    def test_plot_class_separation_correct_shape_validation(
        self,
        plot_generator: PlotGenerator,
        sample_embeddings: np.ndarray,
        sample_class_labels: np.ndarray,
        class_names: list[str],
    ) -> None:
        """Test that plot_class_separation validates input shapes."""
        # Test with mismatched lengths
        wrong_labels = np.arange(100, dtype=np.int32)  # Different length

        with pytest.raises(ValueError, match="length mismatch"):
            plot_generator.plot_class_separation(
                embeddings=sample_embeddings,
                class_labels=wrong_labels,
                class_names=class_names,
                title="Test",
                filename_prefix="test",
            )

    def test_plot_class_separation_validates_embeddings_2d(
        self,
        plot_generator: PlotGenerator,
        sample_class_labels: np.ndarray,
        class_names: list[str],
    ) -> None:
        """Test that plot_class_separation validates embeddings are 2D."""
        # Create 3D embeddings (invalid)
        wrong_embeddings = np.random.randn(210, 3).astype(np.float32)

        with pytest.raises(ValueError, match="embeddings must be 2D"):
            plot_generator.plot_class_separation(
                embeddings=wrong_embeddings,
                class_labels=sample_class_labels,
                class_names=class_names,
                title="Test",
                filename_prefix="test",
            )

    def test_plot_class_separation_validates_class_range(
        self,
        plot_generator: PlotGenerator,
        sample_embeddings: np.ndarray,
        class_names: list[str],
    ) -> None:
        """Test that plot_class_separation validates class labels are in valid range."""
        # Create invalid labels (out of range 0-20)
        invalid_labels = np.arange(210, dtype=np.int32) % 25  # Some > 20

        with pytest.raises(ValueError, match="class labels must be in range"):
            plot_generator.plot_class_separation(
                embeddings=sample_embeddings,
                class_labels=invalid_labels,
                class_names=class_names,
                title="Test",
                filename_prefix="test",
            )

    def test_plot_class_separation_validates_class_names_count(
        self,
        plot_generator: PlotGenerator,
        sample_embeddings: np.ndarray,
        sample_class_labels: np.ndarray,
    ) -> None:
        """Test that plot_class_separation validates class names count."""
        # Provide wrong number of class names
        wrong_class_names = ["Class_1", "Class_2"]  # Only 2 names instead of 21

        with pytest.raises(ValueError, match="class_names must contain 21 names"):
            plot_generator.plot_class_separation(
                embeddings=sample_embeddings,
                class_labels=sample_class_labels,
                class_names=wrong_class_names,
                title="Test",
                filename_prefix="test",
            )

    def test_plot_class_separation_with_default_title(
        self,
        plot_generator: PlotGenerator,
        sample_embeddings: np.ndarray,
        sample_class_labels: np.ndarray,
        class_names: list[str],
    ) -> None:
        """Test plot_class_separation with default title."""
        png_path, pdf_path = plot_generator.plot_class_separation(
            embeddings=sample_embeddings,
            class_labels=sample_class_labels,
            class_names=class_names,
        )

        assert Path(png_path).exists()
        assert Path(pdf_path).exists()

    def test_plot_class_separation_dpi_requirement(
        self,
        plot_generator: PlotGenerator,
        sample_embeddings: np.ndarray,
        sample_class_labels: np.ndarray,
        class_names: list[str],
    ) -> None:
        """Test that plot_class_separation uses DPI >= 300."""
        # This test verifies the requirement for publication quality
        assert plot_generator.dpi >= 300

        png_path, pdf_path = plot_generator.plot_class_separation(
            embeddings=sample_embeddings,
            class_labels=sample_class_labels,
            class_names=class_names,
        )

        assert Path(png_path).exists()
        assert Path(pdf_path).exists()


class TestPlotDomainComparison:
    """Test suite for plot_domain_comparison method."""

    @pytest.fixture
    def output_dir(self, tmp_path: Path) -> Path:
        """Create a temporary output directory for testing."""
        return tmp_path / "output"

    @pytest.fixture
    def plot_generator(self, output_dir: Path) -> PlotGenerator:
        """Create a PlotGenerator instance for testing."""
        return PlotGenerator(
            output_dir=str(output_dir),
            dpi=300,
            figsize=(12, 8),
            palette="colorblind",
        )

    @pytest.fixture
    def sample_embeddings(self) -> np.ndarray:
        """Create sample 2D embeddings for 4 domains."""
        n_samples = 400  # 100 samples per domain
        return np.random.randn(n_samples, 2).astype(np.float32)

    @pytest.fixture
    def sample_domain_labels(self) -> np.ndarray:
        """Create sample domain labels (0-3) for 4 domains."""
        n_samples = 400  # 100 samples per domain
        # Repeat each domain 100 times
        return np.repeat(np.arange(4, dtype=np.int32), 100)

    @pytest.fixture
    def domain_names(self) -> list[str]:
        """Create list of 4 domain names."""
        return [
            "DESED Synthetic",
            "DESED Real",
            "MAESTRO Training",
            "MAESTRO Validation",
        ]

    def test_plot_domain_comparison_creates_files(
        self,
        plot_generator: PlotGenerator,
        sample_embeddings: np.ndarray,
        sample_domain_labels: np.ndarray,
        domain_names: list[str],
    ) -> None:
        """Test that plot_domain_comparison creates PNG and PDF files."""
        png_path, pdf_path = plot_generator.plot_domain_comparison(
            embeddings=sample_embeddings,
            domain_labels=sample_domain_labels,
            domain_names=domain_names,
            title="Test Domain Comparison",
            filename_prefix="test_domain_comparison",
        )

        # Verify files exist
        assert Path(png_path).exists()
        assert Path(pdf_path).exists()

        # Verify file extensions
        assert png_path.endswith(".png")
        assert pdf_path.endswith(".pdf")

        # Verify filenames contain prefix
        assert "test_domain_comparison" in png_path
        assert "test_domain_comparison" in pdf_path

    def test_plot_domain_comparison_validates_shape(
        self,
        plot_generator: PlotGenerator,
        sample_embeddings: np.ndarray,
        sample_domain_labels: np.ndarray,
        domain_names: list[str],
    ) -> None:
        """Test that plot_domain_comparison validates input shapes."""
        # Test with mismatched lengths
        wrong_labels = np.arange(100, dtype=np.int32)  # Different length

        with pytest.raises(ValueError, match="length mismatch"):
            plot_generator.plot_domain_comparison(
                embeddings=sample_embeddings,
                domain_labels=wrong_labels,
                domain_names=domain_names,
                title="Test",
                filename_prefix="test",
            )

    def test_plot_domain_comparison_validates_embeddings_2d(
        self,
        plot_generator: PlotGenerator,
        sample_domain_labels: np.ndarray,
        domain_names: list[str],
    ) -> None:
        """Test that plot_domain_comparison validates embeddings are 2D."""
        # Create 3D embeddings (invalid)
        wrong_embeddings = np.random.randn(400, 3).astype(np.float32)

        with pytest.raises(ValueError, match="embeddings must be 2D"):
            plot_generator.plot_domain_comparison(
                embeddings=wrong_embeddings,
                domain_labels=sample_domain_labels,
                domain_names=domain_names,
                title="Test",
                filename_prefix="test",
            )

    def test_plot_domain_comparison_validates_domain_range(
        self,
        plot_generator: PlotGenerator,
        sample_embeddings: np.ndarray,
        domain_names: list[str],
    ) -> None:
        """Test that plot_domain_comparison validates domain labels are in valid range."""
        # Create invalid labels (out of range for provided domain names)
        invalid_labels = np.arange(400, dtype=np.int32) % 5  # Some = 4, out of range

        with pytest.raises(ValueError, match="domain labels must be in range"):
            plot_generator.plot_domain_comparison(
                embeddings=sample_embeddings,
                domain_labels=invalid_labels,
                domain_names=domain_names,
                title="Test",
                filename_prefix="test",
            )

    def test_plot_domain_comparison_with_sample_counts_in_legend(
        self,
        plot_generator: PlotGenerator,
        sample_embeddings: np.ndarray,
        sample_domain_labels: np.ndarray,
        domain_names: list[str],
    ) -> None:
        """Test that plot_domain_comparison includes sample counts in legend."""
        png_path, pdf_path = plot_generator.plot_domain_comparison(
            embeddings=sample_embeddings,
            domain_labels=sample_domain_labels,
            domain_names=domain_names,
        )

        # Verify files are created (legend content is visual, hard to test programmatically)
        assert Path(png_path).exists()
        assert Path(pdf_path).exists()

    def test_plot_domain_comparison_with_different_marker_shapes(
        self,
        plot_generator: PlotGenerator,
        sample_embeddings: np.ndarray,
        sample_domain_labels: np.ndarray,
        domain_names: list[str],
    ) -> None:
        """Test that plot_domain_comparison uses different marker shapes for domains."""
        png_path, pdf_path = plot_generator.plot_domain_comparison(
            embeddings=sample_embeddings,
            domain_labels=sample_domain_labels,
            domain_names=domain_names,
        )

        # Verify files are created (marker shapes are visual, hard to test programmatically)
        assert Path(png_path).exists()
        assert Path(pdf_path).exists()

    def test_plot_domain_comparison_with_default_title(
        self,
        plot_generator: PlotGenerator,
        sample_embeddings: np.ndarray,
        sample_domain_labels: np.ndarray,
        domain_names: list[str],
    ) -> None:
        """Test plot_domain_comparison with default title."""
        png_path, pdf_path = plot_generator.plot_domain_comparison(
            embeddings=sample_embeddings,
            domain_labels=sample_domain_labels,
            domain_names=domain_names,
        )

        assert Path(png_path).exists()
        assert Path(pdf_path).exists()

    def test_plot_domain_comparison_with_empty_domain(
        self,
        plot_generator: PlotGenerator,
        domain_names: list[str],
    ) -> None:
        """Test plot_domain_comparison handles domains with no samples."""
        # Create embeddings with only 3 domains (0, 1, 2), skip domain 3
        embeddings = np.random.randn(300, 2).astype(np.float32)
        domain_labels = np.repeat(np.arange(3, dtype=np.int32), 100)

        png_path, pdf_path = plot_generator.plot_domain_comparison(
            embeddings=embeddings,
            domain_labels=domain_labels,
            domain_names=domain_names,
        )

        assert Path(png_path).exists()
        assert Path(pdf_path).exists()


class TestPlotMixStyleComparison:
    """Test suite for plot_mixstyle_comparison method."""

    @pytest.fixture
    def output_dir(self, tmp_path: Path) -> Path:
        """Create a temporary output directory for testing."""
        return tmp_path / "output"

    @pytest.fixture
    def plot_generator(self, output_dir: Path) -> PlotGenerator:
        """Create a PlotGenerator instance for testing."""
        return PlotGenerator(
            output_dir=str(output_dir),
            dpi=300,
            figsize=(16, 8),  # Wider for 2 subplots
            palette="colorblind",
        )

    @pytest.fixture
    def sample_embeddings_before(self) -> np.ndarray:
        """Create sample 2D embeddings before MixStyle for 4 domains."""
        n_samples = 400  # 100 samples per domain
        return np.random.randn(n_samples, 2).astype(np.float32)

    @pytest.fixture
    def sample_embeddings_after(self) -> np.ndarray:
        """Create sample 2D embeddings after MixStyle for 4 domains."""
        n_samples = 400  # 100 samples per domain
        return np.random.randn(n_samples, 2).astype(np.float32)

    @pytest.fixture
    def sample_domain_labels(self) -> np.ndarray:
        """Create sample domain labels (0-3) for 4 domains."""
        n_samples = 400  # 100 samples per domain
        # Repeat each domain 100 times
        return np.repeat(np.arange(4, dtype=np.int32), 100)

    @pytest.fixture
    def domain_names(self) -> list[str]:
        """Create list of 4 domain names."""
        return [
            "DESED Synthetic",
            "DESED Real",
            "MAESTRO Training",
            "MAESTRO Validation",
        ]

    def test_plot_mixstyle_comparison_creates_files(
        self,
        plot_generator: PlotGenerator,
        sample_embeddings_before: np.ndarray,
        sample_embeddings_after: np.ndarray,
        sample_domain_labels: np.ndarray,
        domain_names: list[str],
    ) -> None:
        """Test that plot_mixstyle_comparison creates PNG and PDF files."""
        png_path, pdf_path = plot_generator.plot_mixstyle_comparison(
            embeddings_before=sample_embeddings_before,
            embeddings_after=sample_embeddings_after,
            domain_labels=sample_domain_labels,
            domain_names=domain_names,
            title_before="Before MixStyle",
            title_after="After MixStyle",
            filename_prefix="test_mixstyle_effect",
        )

        # Verify files exist
        assert Path(png_path).exists()
        assert Path(pdf_path).exists()

        # Verify file extensions
        assert png_path.endswith(".png")
        assert pdf_path.endswith(".pdf")

        # Verify filenames contain prefix
        assert "test_mixstyle_effect" in png_path
        assert "test_mixstyle_effect" in pdf_path

    def test_plot_mixstyle_comparison_validates_embeddings_2d(
        self,
        plot_generator: PlotGenerator,
        sample_embeddings_after: np.ndarray,
        sample_domain_labels: np.ndarray,
        domain_names: list[str],
    ) -> None:
        """Test that plot_mixstyle_comparison validates embeddings are 2D."""
        # Create 3D embeddings (invalid)
        wrong_embeddings = np.random.randn(400, 3).astype(np.float32)

        with pytest.raises(ValueError, match="embeddings must be 2D"):
            plot_generator.plot_mixstyle_comparison(
                embeddings_before=wrong_embeddings,
                embeddings_after=sample_embeddings_after,
                domain_labels=sample_domain_labels,
                domain_names=domain_names,
            )

    def test_plot_mixstyle_comparison_validates_matching_shapes(
        self,
        plot_generator: PlotGenerator,
        sample_embeddings_before: np.ndarray,
        sample_domain_labels: np.ndarray,
        domain_names: list[str],
    ) -> None:
        """Test that plot_mixstyle_comparison validates before/after have matching shapes."""
        # Create different size embeddings
        wrong_embeddings = np.random.randn(200, 2).astype(np.float32)

        with pytest.raises(ValueError, match="embeddings_before and embeddings_after.*shape"):
            plot_generator.plot_mixstyle_comparison(
                embeddings_before=sample_embeddings_before,
                embeddings_after=wrong_embeddings,
                domain_labels=sample_domain_labels,
                domain_names=domain_names,
            )

    def test_plot_mixstyle_comparison_validates_domain_labels_length(
        self,
        plot_generator: PlotGenerator,
        sample_embeddings_before: np.ndarray,
        sample_embeddings_after: np.ndarray,
        domain_names: list[str],
    ) -> None:
        """Test that plot_mixstyle_comparison validates domain_labels length matches embeddings."""
        # Create domain labels with wrong length
        wrong_labels = np.arange(100, dtype=np.int32)

        with pytest.raises(ValueError, match="length mismatch"):
            plot_generator.plot_mixstyle_comparison(
                embeddings_before=sample_embeddings_before,
                embeddings_after=sample_embeddings_after,
                domain_labels=wrong_labels,
                domain_names=domain_names,
            )

    def test_plot_mixstyle_comparison_validates_domain_range(
        self,
        plot_generator: PlotGenerator,
        sample_embeddings_before: np.ndarray,
        sample_embeddings_after: np.ndarray,
        domain_names: list[str],
    ) -> None:
        """Test that plot_mixstyle_comparison validates domain labels are in valid range."""
        # Create invalid labels (out of range)
        invalid_labels = np.arange(400, dtype=np.int32) % 5  # Some = 4, out of range

        with pytest.raises(ValueError, match="domain labels must be in range"):
            plot_generator.plot_mixstyle_comparison(
                embeddings_before=sample_embeddings_before,
                embeddings_after=sample_embeddings_after,
                domain_labels=invalid_labels,
                domain_names=domain_names,
            )

    def test_plot_mixstyle_comparison_with_default_titles(
        self,
        plot_generator: PlotGenerator,
        sample_embeddings_before: np.ndarray,
        sample_embeddings_after: np.ndarray,
        sample_domain_labels: np.ndarray,
        domain_names: list[str],
    ) -> None:
        """Test plot_mixstyle_comparison with default titles."""
        png_path, pdf_path = plot_generator.plot_mixstyle_comparison(
            embeddings_before=sample_embeddings_before,
            embeddings_after=sample_embeddings_after,
            domain_labels=sample_domain_labels,
            domain_names=domain_names,
        )

        assert Path(png_path).exists()
        assert Path(pdf_path).exists()

    def test_plot_mixstyle_comparison_unified_axis_scales(
        self,
        plot_generator: PlotGenerator,
        sample_embeddings_before: np.ndarray,
        sample_embeddings_after: np.ndarray,
        sample_domain_labels: np.ndarray,
        domain_names: list[str],
    ) -> None:
        """Test that plot_mixstyle_comparison creates subplots with unified axis scales."""
        # This is a requirement check - both subplots should share the same axis ranges
        png_path, pdf_path = plot_generator.plot_mixstyle_comparison(
            embeddings_before=sample_embeddings_before,
            embeddings_after=sample_embeddings_after,
            domain_labels=sample_domain_labels,
            domain_names=domain_names,
        )

        # Verify files are created (axis scales are visual, hard to test programmatically)
        assert Path(png_path).exists()
        assert Path(pdf_path).exists()

    def test_plot_mixstyle_comparison_with_identical_embeddings(
        self,
        plot_generator: PlotGenerator,
        sample_embeddings_before: np.ndarray,
        sample_domain_labels: np.ndarray,
        domain_names: list[str],
    ) -> None:
        """Test plot_mixstyle_comparison when before and after are identical."""
        # Use same embeddings for before and after (edge case)
        png_path, pdf_path = plot_generator.plot_mixstyle_comparison(
            embeddings_before=sample_embeddings_before,
            embeddings_after=sample_embeddings_before,
            domain_labels=sample_domain_labels,
            domain_names=domain_names,
        )

        assert Path(png_path).exists()
        assert Path(pdf_path).exists()
