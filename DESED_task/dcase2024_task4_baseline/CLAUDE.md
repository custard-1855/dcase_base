# dcase2024_task4_baseline

DCASE 本家ベースラインを土台に、卒論用の手法を足したレシピ。スクリプトはこのディレクトリを
カレントにして実行する（`./confs`, `./embeddings`, `./pretrained_models`, `experiments/` を相対パスで参照するため）。

## 地図

| 場所 | 役割 | 由来 |
|---|---|---|
| `train_pretrained.py` | 学習・評価・推論の入口。`prepare_run`（引数と yaml の統合）→ `single_run` | 本家 + 改変 |
| `local/sed_trainer_pretrained.py` | LightningModule `SEDTask4`。損失、CMT、validation / test の指標計算、wandb | 本家 + 大幅改変 |
| `confs/pretrained.yaml` | 既定の設定。データのパスもここ | 本家 + 改変 |
| `desed_task/nnet/` | `CRNN` / `CNN` / `RNN`。`mixstyle.py` が MixStyle 実装 | 本家 + MixStyle 追加 |
| `desed_task/dataio/datasets.py` | Strong / Weak / Unlabeled の Dataset。埋め込みは hdf5 から読む | 本家 |
| `local/experiment_dir.py` | `experiments/{mode}/{category}/{method}/{variant}/run-*/` の出力先管理 | 卒論で追加 |
| `local/sebbs_wrapper/` | `sebbs` submodule の型付きラッパー。使い方は同ディレクトリの `README.md` | 卒論で追加 |
| `sebbs/` | cSEBBs 後処理（git submodule, MERL 製）。**編集しない** | 外部 |
| `run_*.sh` | 論文のアブレーション実験。引数の組み合わせの正本 | 卒論で追加 |
| `run_repro.sh` | 再現学習のキュー。論文の表の構成を順に流し、`exp/repro/{構成名}/version_N/test_metrics.json` に最終指標を残す（wandb は既定で無効） | 卒論で追加 |
| `run_repro_eval.sh` | 再現学習のベスト ckpt の再評価（`--sebbs` で cSEBBs）。構成のフラグは `run_repro.sh --list` から引く。結果は `exp/repro_eval/{構成名}/version_N/` | 卒論で追加 |
| `confs/pretrained_short.yaml` | 10 エポックの短縮学習用。`pretrained.yaml` を変えたら合わせて直す | 卒論で追加 |
| `extract_embeddings.py` | BEATs 埋め込みを `./embeddings/beats/*.hdf5` に事前計算 | 本家 + 改変 |
| `generate_dcase_task4_2024.py`, `dprep/` | データセットのダウンロードと MAESTRO の 10 秒分割 | 本家 |
| `tests/` | smoke test と合成データ生成（`conftest.py`）、再現値の検査（`test_repro.py`, 期待値は `repro_expected.json`） | 卒論で追加 |
| `local/pre_sed_trainer_pretrained.py`, `desed_task/dataio/pre_datasets.py` | 改変前の控え。どこからも import されていない | 旧版 |
| `optuna_pretrained.py`, `confs/optuna.yaml` | ハイパーパラメータ探索。smoke test の対象外 | 本家 |
| `README.md` | 本家ベースラインの README。devcontainer や `visualize/` など、このリポジトリに無いものへの言及が残る | 本家 |

本家由来のファイルは black 形式のまま。差分を小さく保つため、触る行だけを直す。

## 設定の優先順位

`prepare_run` は yaml を読んだ後に CLI 引数で上書きする。次のキーは引数に既定値があるため
**yaml に書いた値は常に上書きされ、効かない**。切り替えは CLI フラグで行う。

- `cmt.enabled` / `cmt.use_neg_sample` / `cmt.pos_neg_scale` / `cmt.phi_frame` / `cmt.warmup_epochs` ← `--cmt` ほか
- `sebbs.enabled` ← `--sebbs`
- `net.mixstyle_type` / `net.attn_type` ← `--mixstyle_type {disabled,only_mix,freqAttn}` / `--attn_type`

`wandb.use_wandb` は yaml の値が生きる（既定 `True`）。ログインしない実行では `WANDB_MODE=offline` を付ける。

`--gpus` は個数ではなく切り替え: `"1"` が GPU 1 枚、`"0"` が CPU。

## 実行モード

| 引数 | 動作 |
|---|---|
| なし | 学習 → ベスト ckpt で dev-test 評価 |
| `--fast_dev_run` | 2 epoch・20 batch に切り詰めた学習 + 評価 |
| `--test_from_checkpoint X` | 学習せず dev-test の指標を再計算 |
| `--eval_from_checkpoint X` | ラベルなしの `data.eval_folder` にスコアを書き出すだけ（指標なし） |

## データの約束事

- クラスは DESED 10 + MAESTRO 17 = 27（`local/classes_dict.py`）。`net.nclass` と `net.median_filter` の長さはこれに合わせる。
- 各セットは `*_folder_44k`（元音源）と `*_folder`（16 kHz）の組。16 kHz 側は初回実行時に自動生成される。
- 埋め込み hdf5 のキーは wav の stem。セットを足すときは `extract_embeddings.py` 末尾の辞書と
  `train_pretrained.py` の `get_embeddings_name(config, "<name>")` の両方に名前を足す。
- MAESTRO は `{scene}_{番号}-{開始}-{終了}.wav`。scene 名と番号で train / valid を分けている（`split_maestro`）。
- 欠損した音声は Dataset が `None` を返し、`SafeDataLoader` が読み飛ばす。件数は Dataset の構築時にセットごとに警告される。

## sebbs の import

`from sebbs.sebbs.csebbs import ...` の形で書く。外側の `sebbs` は submodule のディレクトリ、
内側が実パッケージ。このディレクトリが `sys.path` にあること（スクリプト実行時は自動、pytest は直下の `conftest.py`）と
editable インストールの両方に依存している。

## テストを足すとき

- 合成データは `tests/conftest.py` の `synthetic_config` が実データと同じ構成・形式で生成する。
  新しいデータセットやファイルを読むコードを足したら、ここにも同じ形式で足す。
- 再現対象の構成は `run_repro.sh --list` から読むので、`run_repro.sh` に足せば smoke test に入る。
  それ以外の CLI フラグの組み合わせは `tests/test_smoke.py` の `VARIANTS` に 1 行足す。
- 学習を回すテストには `@pytest.mark.slow` を付ける。
- リサンプルの並列処理（fork）は pytest 内でデッドロックするので、fixture で逐次実行に差し替えてある。
