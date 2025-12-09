"""Tests for DEVCONTAINER_GUIDE.md documentation.

このテストスイートは、DEVCONTAINER_GUIDE.mdが要件10.2, 10.3, 10.4, 10.5を
満たしていることを検証します。

検証項目:
- ファイル存在性
- 必須セクション（事前準備、トラブルシューティング、FAQ、設定ファイル解説）
- gh auth loginの説明
- docker volume pruneコマンド
- リソース管理方法
- wandbのオプショナル性
"""

from pathlib import Path


def test_devcontainer_guide_exists():
    """DEVCONTAINER_GUIDE.mdファイルが存在することを確認（要件10.2）"""
    guide_path = Path("DESED_task/dcase2024_task4_baseline/DEVCONTAINER_GUIDE.md")
    assert guide_path.exists(), "DEVCONTAINER_GUIDE.md must exist"


def test_devcontainer_guide_has_required_sections():
    """DEVCONTAINER_GUIDE.mdに必要なセクションが含まれていることを確認（要件10.2, 10.3）"""
    guide_path = Path("DESED_task/dcase2024_task4_baseline/DEVCONTAINER_GUIDE.md")
    content = guide_path.read_text()

    # 必須セクション（事前準備、トラブルシューティング、FAQ、設定ファイル解説）
    required_sections = [
        "# Devcontainer セットアップガイド",
        "## 事前準備",
        "### Docker Desktop のインストール",
        "### GitHub CLI (gh) 認証",
        "## セットアップ手順",
        "## トラブルシューティング",
        "### システムライブラリインストール失敗",
        "### Git submodule 初期化失敗",
        "### gh auth login 未実行エラー",
        "### ボリューム権限問題",
        "### メモリ不足警告",
        "### ディスク容量不足警告",
        "## FAQ",
        "### GPU 対応は可能ですか？",
        "### wandb を使用するには？",
        "### ディスク容量を管理するには？",
        "## 設定ファイル解説",
        "### devcontainer.json",
        "### Dockerfile",
    ]

    for section in required_sections:
        assert section in content, f"Section '{section}' must be present in DEVCONTAINER_GUIDE.md"


def test_devcontainer_guide_has_gh_auth_login_explanation():
    """gh auth loginの重要性が説明されていることを確認（要件10.2 - 事前準備の詳細）"""
    guide_path = Path("DESED_task/dcase2024_task4_baseline/DEVCONTAINER_GUIDE.md")
    content = guide_path.read_text()

    assert "gh auth login" in content, "Must explain 'gh auth login'"
    assert "GitHub CLI" in content, "Must mention GitHub CLI"


def test_devcontainer_guide_has_docker_volume_prune():
    """docker volume pruneコマンドが記載されていることを確認（要件10.3 - FAQ）"""
    guide_path = Path("DESED_task/dcase2024_task4_baseline/DEVCONTAINER_GUIDE.md")
    content = guide_path.read_text()

    assert "docker volume prune" in content, "Must include 'docker volume prune' command"


def test_devcontainer_guide_has_resource_management():
    """リソース不足時の対処法が記載されていることを確認（要件10.2 - リソース管理）"""
    guide_path = Path("DESED_task/dcase2024_task4_baseline/DEVCONTAINER_GUIDE.md")
    content = guide_path.read_text()

    # メモリ制限増加
    assert "メモリ制限" in content or "memory" in content.lower(), "Must explain memory limits"

    # バッチサイズ削減
    assert "バッチサイズ" in content or "batch" in content.lower(), "Must mention batch size"

    # ボリューム削除
    assert "ボリューム" in content or "volume" in content.lower(), "Must explain volume management"


def test_devcontainer_guide_has_wandb_optional():
    """wandbがオプショナルであることが記載されていることを確認（要件10.2 - wandb設定）"""
    guide_path = Path("DESED_task/dcase2024_task4_baseline/DEVCONTAINER_GUIDE.md")
    content = guide_path.read_text()

    assert "オプショナル" in content or "optional" in content.lower(), "Must indicate wandb is optional"
    assert "WANDB_API_KEY" in content, "Must mention WANDB_API_KEY"
