# dcase_base

卒業論文の実験コード。DCASE 2024 Task 4 ベースライン（音響イベント検出, Mean Teacher + BEATs 埋め込み）に
CMT・MixStyle・cSEBBs を追加して比較している。セットアップから学習・評価までの手順は `README.md`、
詳細は `docs/`（`data.md` / `experiments.md` / `adding-a-method.md`）。

作業の進め方（調査 / 計画 / 実装 / デバッグ の定義と日本語応答）は次のファイルに従う:

@.github/copilot-instructions.md

## どこで何を実行するか

| 目的 | 実行場所 | コマンド |
|---|---|---|
| テスト | リポジトリ直下 | `uv run poe test`（= `pytest -m "not slow and not repro"`, 数十秒）, `uv run pytest`（CPU で数分） |
| 再現値の検査 | リポジトリ直下 | `uv run poe repro-check`（`exp/repro*` の結果を期待値と比較。既定の `pytest` には含まれない） |
| 学習・推論・データ準備 | `DESED_task/dcase2024_task4_baseline/` | `uv run train_pretrained.py ...` |
| 再現学習・ckpt の再評価 | リポジトリ直下 | `uv run poe repro` / `repro-short` / `repro-eval` / `sebbs`（一覧は `uv run poe`） |
| lint | リポジトリ直下 | `uv run ruff check <触ったファイル>`（設定は `pyproject.toml`） |

コードの本体はすべて `DESED_task/dcase2024_task4_baseline/` にある。そこを触るときは同ディレクトリの `CLAUDE.md` を読む。

## 変更を確認する

学習コード・設定・データ読み込みを変えたら `uv run pytest` を通す。実データも GPU も不要で、
合成データに対して「学習 → validation → dev-test 評価 → ckpt からの推論」を CPU で最後まで回す。

`local/sebbs_wrapper/tests/test_evaluator.py::test_evaluate_psds2` は既知の失敗
（`sed_scores_eval` 内部の TypeError）。これ以外が通れば緑とみなす。

ruff は全ルール有効（`select = ["ALL"]`）で、既存コードには数千件の指摘が残っている。
合格条件にはせず、自分が足した行に新しい指摘を増やさないことだけを目安にする。

## この環境で出来ないこと

実データでの学習は、データ（DESED は YouTube 由来を含む数十 GB）と GPU がある計算機でのみ可能。
手元に無い場合は smoke test までを確認範囲とし、実データでの結果は未確認と明記する。

## 依存関係

- 依存の追加・更新は `uv add` / `uv lock` で行い、`uv.lock` をコミットする。
- `sebbs` は git submodule を editable インストールしている（`pyproject.toml` の `[tool.uv.sources]`）。
  `uv sync` が Cython 拡張をビルドするので、`sox` / `libsox-dev` と submodule の初期化が前提。
- PyTorch Lightning は 1.9 系に固定。フック名は 1.9 の API（`validation_epoch_end` など）で書く。
