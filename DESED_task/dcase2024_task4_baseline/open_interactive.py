#!/usr/bin/env python3
"""
インタラクティブHTML可視化をブラウザで開くスクリプト

使用法:
    # UMAP可視化を開く
    python open_interactive.py visualization_outputs/umap/student/interactive_umap.html

    # または自動検索
    python open_interactive.py
"""

import argparse
import webbrowser
from pathlib import Path


def find_interactive_files(base_dir: Path = Path("visualization_outputs")) -> list:
    """インタラクティブHTMLファイルを検索"""
    html_files = []

    if base_dir.exists():
        # UMAP関連
        umap_dir = base_dir / "umap"
        if umap_dir.exists():
            for html in umap_dir.rglob("interactive_*.html"):
                html_files.append(html)

        # その他のHTML
        for html in base_dir.rglob("*.html"):
            if "interactive" in html.name or "plotly" in html.name:
                html_files.append(html)

    return sorted(set(html_files))


def open_in_browser(file_path: Path):
    """ブラウザでHTMLファイルを開く"""
    if not file_path.exists():
        print(f"エラー: ファイルが見つかりません: {file_path}")
        return False

    # 絶対パスに変換
    abs_path = file_path.resolve()

    print(f"ブラウザで開いています: {abs_path}")

    # ブラウザで開く
    webbrowser.open(f"file://{abs_path}")

    print("✓ ブラウザで開きました")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="インタラクティブHTML可視化をブラウザで開く"
    )
    parser.add_argument(
        "file",
        nargs="?",
        help="開くHTMLファイルのパス（省略時は自動検索）"
    )
    parser.add_argument(
        "--search-dir",
        default="visualization_outputs",
        help="検索するディレクトリ（デフォルト: visualization_outputs）"
    )

    args = parser.parse_args()

    if args.file:
        # 指定されたファイルを開く
        file_path = Path(args.file)
        open_in_browser(file_path)
    else:
        # 自動検索
        print("インタラクティブHTMLファイルを検索中...")
        html_files = find_interactive_files(Path(args.search_dir))

        if not html_files:
            print(f"エラー: {args.search_dir} にインタラクティブHTMLファイルが見つかりません")
            print("\n可視化を先に実行してください:")
            print("  bash run_visualizations.sh 1")
            return

        print(f"\n見つかったファイル: {len(html_files)}個")
        for i, html_file in enumerate(html_files, 1):
            rel_path = html_file.relative_to(Path.cwd())
            print(f"  [{i}] {rel_path}")

        if len(html_files) == 1:
            # 1つだけの場合は自動で開く
            print("\n自動的に開きます...")
            open_in_browser(html_files[0])
        else:
            # 複数ある場合は選択
            print("\n開くファイルを選択してください (1-{}, a=すべて, q=終了): ".format(len(html_files)), end="")
            choice = input().strip().lower()

            if choice == 'q':
                print("終了しました")
                return
            elif choice == 'a':
                print("\nすべてのファイルを開きます...")
                for html_file in html_files:
                    open_in_browser(html_file)
            else:
                try:
                    idx = int(choice) - 1
                    if 0 <= idx < len(html_files):
                        open_in_browser(html_files[idx])
                    else:
                        print("エラー: 無効な番号です")
                except ValueError:
                    print("エラー: 無効な入力です")


if __name__ == "__main__":
    main()
