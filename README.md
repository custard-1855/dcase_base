# dcase_base

卒業論文の実験コード. DCASE 2024 Task 4 ベースライン (音響イベント検出, Mean Teacher + BEATs 埋め込み) に
CMT・MixStyle・cSEBBs を追加して比較している.

この README の 1 → 6 を上から順に実行すると, smoke test の実行と ① MT (ベースライン) の学習・評価まで進める.

| 手順 | 必要なもの | 所要の目安 |
|---|---|---|
| [1. セットアップ](#1-セットアップ) (smoke test まで) | Linux, インターネット接続. データも GPU も不要 | インストール + テスト 5 分程度 |
| [2. データと BEATs の準備](#2-データと-beats-の準備) | 空き容量 約 130 GB (3 で作る 16 kHz の写しを含む) | 回線による |
| [3. 埋め込み抽出](#3-埋め込み抽出) | GPU, 空き容量 約 60 GB | — (未計測) |
| [4. 学習](#4-学習) | GPU | 1 構成 約 1 時間 (RTX 4070) |
| [5. 評価](#5-評価) | 4 の出力 | 数分 |

詳細は `docs/` に分けてある.

| 文書 | 内容 |
|---|---|
| [docs/data.md](docs/data.md) | データの配置, 容量, 取得できないファイル, 埋め込みの中身 |
| [docs/experiments.md](docs/experiments.md) | 構成とフラグの一覧, 実行モード, 出力先, wandb, アブレーション, 再現値 |
| [docs/adding-a-method.md](docs/adding-a-method.md) | 手法を追加するときに触るファイル |
| [docs/reports/](docs/reports/) | 調査の記録 (再現学習の結果など) |
| [DESED_task/dcase2024_task4_baseline/CLAUDE.md](DESED_task/dcase2024_task4_baseline/CLAUDE.md) | コードの地図 (どのファイルが何をするか) |

`DESED_task/dcase2024_task4_baseline/README.md` は本家ベースラインのもので, このリポジトリに無いもの (devcontainer など) への言及が残っている. 手順はこの README に従う.

## コマンドを実行する場所

実行場所は 2 つあり, 間違えると相対パスが合わずに失敗する. 以下, コードブロックの先頭にどちらかを書く.

| 呼び方 | パス | ここで実行するもの |
|---|---|---|
| リポジトリ直下 | `dcase_base/` | `uv sync`, `uv run pytest`, `uv run poe ...` |
| レシピ | `dcase_base/DESED_task/dcase2024_task4_baseline/` | `uv run train_pretrained.py` など, スクリプトの直接実行 |

## 1. セットアップ

```bash
# sebbs が git submodule なので --recursive が必要
$ git clone --recursive https://github.com/custard-1855/dcase_base.git
# クローン済みの場合: git submodule update --init --recursive

$ curl -LsSf https://astral.sh/uv/0.7.12/install.sh | sh  # パッケージマネージャ uv
$ sudo apt install sox libsox-dev                         # libsox-dev は soxbindings のビルドに必要
$ cd dcase_base
$ uv sync                                                 # ライブラリのインストール (sebbs もここでビルドされる)
```

### 動作確認 (smoke test)

データや GPU が無くても, 合成データで「学習 → validation → dev-test 評価」と「ckpt からの推論」が通ることを確認できる.

```bash
# リポジトリ直下
$ uv run poe test   # モデルの forward など軽いテストのみ (数十秒)
$ uv run pytest     # 学習・推論を最後まで回すテストも含む (CPU で 5 分程度)
```

`local/sebbs_wrapper/tests/test_evaluator.py::test_evaluate_psds2` は既知の失敗 (`sed_scores_eval` 内部の TypeError).
これ以外が通れば環境は出来ている.

コードを変更する人は, コミット時・プッシュ時の検査も入れておく.

```bash
# リポジトリ直下
$ uv run prek install  # 設定は prek.toml. コミット時に構文・型検査, プッシュ時に軽いテストが走る
```

## 2. データと BEATs の準備

### データセット

`confs/pretrained.yaml` の `data.root` (既定: `/mnt/data/data/dcase/dataset`) の下にデータを置く.
下のコマンドは `--basedir` の下の `dcase/dataset/` に展開するので, 既定のままなら `data.root` の書き換えは不要.
別の場所に置いたら `data.root` の 1 行だけを書き換える.

```bash
# レシピ
$ uv run generate_dcase_task4_2024.py --basedir="/mnt/data/data"
```

- AudioSet 由来の音声は YouTube から取得するため, 取得できないファイルが出る. 欠損したファイルは学習時に読み飛ばされる.
- 容量と配置の詳細は [docs/data.md](docs/data.md).

### BEATs のチェックポイント

`BEATs_iter3_plus_AS2M.pt` (361 MB) をレシピの `pretrained_models/` に置く.
公式の配布元は OneDrive ([BEATs README](https://github.com/microsoft/unilm/blob/master/beats/README.md)).
wget で取得する場合は Hugging Face 上の第三者ミラーが使える (公式配布ではない点に注意).

```bash
# レシピ
$ mkdir -p pretrained_models
$ wget -O pretrained_models/BEATs_iter3_plus_AS2M.pt \
    https://huggingface.co/lpepino/beats_ckpts/resolve/main/BEATs_iter3_plus_AS2M.pt
$ sha256sum pretrained_models/BEATs_iter3_plus_AS2M.pt
# d43cbfad4d7b56381c061d7a24774f908d4d94c72961f6eb1d9090ff18cd8d34
```

## 3. 埋め込み抽出

BEATs の埋め込みを事前に計算して `embeddings/beats/*.hdf5` に保存する (合計 約 60 GB). 学習はこのファイルを読む.

```bash
# レシピ
$ uv run extract_embeddings.py --output_dir ./embeddings
# 別の場所の ckpt を使う場合: --beats_ckpt /path/to/BEATs_iter3_plus_AS2M.pt
```

最初に 16 kHz へのリサンプルが走り, `data.root` の下に `*_16k/` が作られる. 抽出はその後に始まる.

## 4. 学習

論文の表の構成には名前が付いていて (`mt`, `cmt`, `cmt_scale_mixstyle` など 7 つ), 名前を指定して学習する.
① MT (ベースライン) は `mt`.

```bash
# リポジトリ直下
$ uv run poe repro --list                                         # 構成名とフラグの一覧

# まず短い実行 (2 エポック・20 バッチ) で最後まで通ることを確認する
$ uv run poe repro --out exp/repro_check mt -- --fast_dev_run

# ① MT の学習 (150 エポック. RTX 4070 で約 1 時間)
$ uv run poe repro mt
```

- 出力はレシピの `exp/repro/mt/version_0/`. ログ (`train_*.log`), TensorBoard, ckpt, 最終指標 (`test_metrics.json`) が入る.
- 学習が終わると, 続けてベスト ckpt で dev-test の評価まで行う.
- `test_metrics.json` がある構成は飛ばされる. もう一度流すときは `--rotate` を付ける (前回分は `exp/repro.N/` に退避される).
  短い実行の出力先を `exp/repro_check` に分けているのはこのため.
- 構成名を省くと 7 構成を順に流す. CPU で流すときは `--gpus 0`.
- wandb は無効で流れる (ログイン不要). 有効にするときは `--wandb`.

フラグを自分で組み合わせる実行, アブレーション (`run_ablation.sh`), 10 エポックの短縮学習は [docs/experiments.md](docs/experiments.md).

## 5. 評価

4 の学習で評価まで済んでいるので, 結果を見るだけなら `test_metrics.json` を読む.
論文と比較する指標は teacher の次の 3 つ.

| 指標 | `test_metrics.json` のキー |
|---|---|
| PSDS1 | `test/teacher/psds1/psds_eval` |
| PSDS2 | `test/teacher/psds2/psds_eval` |
| mpAUC | `test/teacher/segment_mpauc/sed_scores_eval` |

```bash
# レシピ
$ grep -E '"test/teacher/(psds[12]/psds_eval|segment_mpauc/sed_scores_eval)"' exp/repro/mt/version_0/test_metrics.json
$ uv run tensorboard --logdir exp/repro  # 学習曲線
```

保存された ckpt をもう一度評価する場合:

```bash
# リポジトリ直下
$ uv run poe repro-eval mt  # ベスト ckpt を再評価 (約 3 分). 出力は exp/repro_eval/mt/
$ uv run poe sebbs mt       # ベスト ckpt に cSEBBs を掛けて評価 (約 7 分). 出力は exp/repro/mt_sebbs/
```

## 6. 期待される数値

① MT (`mt`) を `confs/pretrained.yaml` のまま (150 エポック, seed 42) 学習したときの値.

| | PSDS1 | PSDS2 | mpAUC |
|---|---|---|---|
| 論文 | 0.369 | 0.581 | 0.720 |
| 再現 1 回目 | 0.374 | 0.587 | 0.725 |
| 再現 2 回目 | 0.365 | 0.581 | 0.703 |

- 同じ seed でも実行ごとに値は揺れる (`training.deterministic` が `False`). 上の 2 回で PSDS は 0.01 弱, mpAUC は 0.02 程度の差がある.
  この程度の差なら手順は正しく進んでいると見てよい.
- 7 構成すべての値は [docs/experiments.md](docs/experiments.md#再現値), 論文値との比較の所見は [docs/reports/2026-11-12-repro-results.md](docs/reports/2026-11-12-repro-results.md).
- 欠損している音声の数が違うと値も変わりうる (YouTube から取得できるファイルは時期によって減る).

## 引用

本検討は電子情報通信学会にて発表しています.
```
竹本志恩, 川原亮一, “実環境を見据えた音による行動認識モデルの精度向上に関する一検討,” 信学技報, vol. 125, no. 385, NS2025-226, pp. 31-36, 2026 年 3 月.
```
