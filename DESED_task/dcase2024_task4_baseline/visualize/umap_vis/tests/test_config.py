"""設定ファイルとドキュメントの妥当性をテストするモジュール."""

import yaml
from pathlib import Path

import pytest


def get_project_root() -> Path:
    """プロジェクトルートディレクトリを取得."""
    # このテストファイルから相対的にプロジェクトルートを取得
    # test_config.py -> tests/ -> umap/ -> visualize/ -> dcase2024_task4_baseline/ -> DESED_task/ -> root/
    return Path(__file__).parent.parent.parent.parent.parent.parent


class TestConfigFile:
    """YAML設定ファイルの妥当性テスト."""

    def test_umap_visualization_yaml_exists(self) -> None:
        """config/umap_visualization.yamlファイルが存在することを確認."""
        config_path = get_project_root() / "config" / "umap_visualization.yaml"
        assert config_path.exists(), f"設定ファイルが存在しません: {config_path}"

    def test_umap_visualization_yaml_structure(self) -> None:
        """YAML設定ファイルが必須セクションを含むことを確認."""
        config_path = get_project_root() / "config" / "umap_visualization.yaml"
        with open(config_path) as f:
            config = yaml.safe_load(f)

        # 必須セクションの確認
        assert "umap" in config, "umapセクションが存在しません"
        assert "plot" in config, "plotセクションが存在しません"
        assert "output" in config, "outputセクションが存在しません"
        assert "logging" in config, "loggingセクションが存在しません"

    def test_umap_section_parameters(self) -> None:
        """umapセクションに必須パラメータが含まれることを確認."""
        config_path = get_project_root() / "config" / "umap_visualization.yaml"
        with open(config_path) as f:
            config = yaml.safe_load(f)

        umap_params = config["umap"]
        assert "n_neighbors" in umap_params
        assert "min_dist" in umap_params
        assert "metric" in umap_params
        assert "random_state" in umap_params

        # デフォルト値の確認（要件7.4）
        assert umap_params["n_neighbors"] == 15
        assert umap_params["min_dist"] == 0.1
        assert umap_params["metric"] == "euclidean"
        assert umap_params["random_state"] == 42

    def test_plot_section_parameters(self) -> None:
        """plotセクションに必須パラメータが含まれることを確認."""
        config_path = get_project_root() / "config" / "umap_visualization.yaml"
        with open(config_path) as f:
            config = yaml.safe_load(f)

        plot_params = config["plot"]
        assert "dpi" in plot_params
        assert "figsize" in plot_params
        assert "palette" in plot_params

        # 論文掲載品質の確認（要件6.2）
        assert plot_params["dpi"] >= 300, "DPIは300以上である必要があります"
        assert plot_params["palette"] == "colorblind", "色覚多様性対応パレットを使用する必要があります"

    def test_output_section_parameters(self) -> None:
        """outputセクションに必須パラメータが含まれることを確認."""
        config_path = get_project_root() / "config" / "umap_visualization.yaml"
        with open(config_path) as f:
            config = yaml.safe_load(f)

        output_params = config["output"]
        assert "dir" in output_params

    def test_logging_section_parameters(self) -> None:
        """loggingセクションに必須パラメータが含まれることを確認."""
        config_path = get_project_root() / "config" / "umap_visualization.yaml"
        with open(config_path) as f:
            config = yaml.safe_load(f)

        logging_params = config["logging"]
        assert "level" in logging_params
        assert logging_params["level"] in ["DEBUG", "INFO", "WARNING", "ERROR"]


class TestReadmeDocumentation:
    """README/使用例ドキュメントの妥当性テスト."""

    def test_visualize_readme_exists(self) -> None:
        """visualize/umap_vis/README.mdファイルが存在することを確認."""
        readme_path = get_project_root() / "DESED_task" / "dcase2024_task4_baseline" / "visualize" / "umap_vis" / "README.md"
        assert readme_path.exists(), f"READMEファイルが存在しません: {readme_path}"

    def test_readme_contains_usage_examples(self) -> None:
        """READMEに3つのモードの使用例が含まれることを確認."""
        readme_path = get_project_root() / "DESED_task" / "dcase2024_task4_baseline" / "visualize" / "umap_vis" / "README.md"
        with open(readme_path) as f:
            content = f.read()

        # 3つのモードのセクションが存在することを確認
        assert "class_separation" in content, "クラス分離性可視化の例が見つかりません"
        assert "domain_comparison" in content, "ドメイン比較の例が見つかりません"
        assert "mixstyle_effect" in content, "MixStyle効果比較の例が見つかりません"

    def test_readme_contains_class_list(self) -> None:
        """READMEに21クラスリストへの参照が含まれることを確認."""
        readme_path = get_project_root() / "DESED_task" / "dcase2024_task4_baseline" / "visualize" / "umap_vis" / "README.md"
        with open(readme_path) as f:
            content = f.read()

        # クラスリストまたはクラスに関する記述が存在することを確認
        assert "DESED" in content or "MAESTRO" in content, "DESEDまたはMAESTROクラスの記述が見つかりません"
        assert "21" in content or "10" in content or "11" in content, "クラス数の記述が見つかりません"

    def test_readme_contains_command_examples(self) -> None:
        """READMEにコマンドライン例が含まれることを確認."""
        readme_path = get_project_root() / "DESED_task" / "dcase2024_task4_baseline" / "visualize" / "umap_vis" / "README.md"
        with open(readme_path) as f:
            content = f.read()

        # コマンドライン例の存在確認
        assert "visualize_umap.py" in content, "スクリプト名が見つかりません"
        assert "class_separation" in content or "domain_comparison" in content or "mixstyle_effect" in content, "モード（サブコマンド）の例が見つかりません"
        assert "--input" in content or "--inputs" in content, "入力引数の例が見つかりません"

    def test_readme_contains_yaml_config_example(self) -> None:
        """READMEにYAML設定ファイル使用例が含まれることを確認."""
        readme_path = get_project_root() / "DESED_task" / "dcase2024_task4_baseline" / "visualize" / "umap_vis" / "README.md"
        with open(readme_path) as f:
            content = f.read()

        # YAML設定ファイルへの言及
        assert "--config" in content or "config" in content.lower(), "設定ファイルの使用例が見つかりません"
