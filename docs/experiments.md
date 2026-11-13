# 実験の流し方

README の「4. 学習」「5. 評価」「6. 期待される数値」の補足。
パスは断りが無ければレシピ（`DESED_task/dcase2024_task4_baseline/`）からの相対。

## 入口は 3 つ

| 入口 | 用途 | 出力先 | wandb |
|---|---|---|---|
| `uv run poe repro ...`（= `./run_repro.sh`） | 論文の表の 7 構成を、名前を指定して流す | `exp/repro/{構成名}/version_N/` | 既定で無効 |
| `./run_ablation.sh SET` | 論文のアブレーション実験を 1 セットまとめて流す | `experiments/train/{category}/{method}/{構成名}/` | 有効 |
| `uv run train_pretrained.py ...` | フラグを自分で組み合わせる | `--log_dir`（既定 `./exp/2024_baseline`）の `version_N/` | yaml の値（既定で有効） |

上の 2 つは `train_pretrained.py` にフラグを付けて呼ぶシェルスクリプトで、`--dry-run` を付けると実行するコマンドだけを表示する。
`uv run poe ...` はリポジトリ直下で、それ以外はレシピで実行する。

## 構成とフラグ

手法の切り替えはすべて `train_pretrained.py` の CLI フラグで行う。

| 手法 | フラグ | 意味 |
|---|---|---|
| Mean Teacher（ベースライン） | なし | |
| CMT | `--cmt` | 教師の擬似ラベルを後処理し、信頼度で重み付けした一貫性損失を使う |
| | `--use_neg_sample` | 信頼度の重みに負例（確率が `phi_neg` 以下）も使う |
| | `--pos_neg_scale` | 正例と負例の数の比で重みを正規化する |
| | `--phi_frame` / `--phi_clip` / `--phi_pos` / `--phi_neg` / `--warmup_epochs` | 閾値と、CMT を有効にするまでのエポック数 |
| MixStyle | `--mixstyle_type disabled` | なし（既定） |
| | `--mixstyle_type only_mix` | MixStyle のみ（周波数方向の注意なし） |
| | `--mixstyle_type freqAttn --attn_type default` | CNN による周波数注意つき（論文の MixStyle 行はこれ） |
| cSEBBs | `--sebbs` | validation のスコアで後処理を調整し、test で適用する |

`--attn_type` は拡張用に定義してあり、論文では `default` だけを使っている。

再現対象の 7 構成（`./run_repro.sh --list`）:

| 構成名 | フラグ |
|---|---|
| `mt`（① MT） | なし |
| `cmt_scale_mixstyle`（②） | `--cmt --use_neg_sample --pos_neg_scale --mixstyle_type freqAttn --attn_type default` |
| `cmt`（③） | `--cmt` |
| `mt_mixstyle` | `--mixstyle_type freqAttn --attn_type default` |
| `cmt_mixstyle` | `--cmt --mixstyle_type freqAttn --attn_type default` |
| `cmt_neg` | `--cmt --use_neg_sample` |
| `cmt_scale` | `--cmt --use_neg_sample --pos_neg_scale` |

### yaml に書いても効かないキー

`train_pretrained.py` の `prepare_run` は yaml を読んだ後に CLI 引数で上書きする。
次のキーは引数に既定値があるため、**yaml に書いた値は常に上書きされる**。

- `cmt.enabled` / `cmt.use_neg_sample` / `cmt.pos_neg_scale` / `cmt.phi_frame` / `cmt.warmup_epochs`
- `sebbs.enabled`
- `net.mixstyle_type` / `net.attn_type`

`--gpus` は個数ではなく切り替えで、`"1"` が GPU 1 枚、`"0"` が CPU。

## 再現学習（`run_repro.sh`）

```bash
# リポジトリ直下
$ uv run poe repro                 # 7 構成を順に（1 構成 約 1 時間）
$ uv run poe repro mt cmt          # 指定した構成だけ
$ uv run poe repro --rotate        # 前回分を exp/repro.N/ に退避して、もう一度流す
$ uv run poe repro-short           # 10 エポックの短縮学習（1 構成 約 7 分）。出力は exp/repro_short/
$ uv run poe repro mt --dry-run    # 実行するコマンドを表示するだけ
```

- `--` の後ろは `train_pretrained.py` にそのまま渡る（例: `-- --fast_dev_run`）。
- `test_metrics.json` がある構成は飛ばすので、途中で止まっても同じコマンドで再開できる。
- 失敗した構成があっても次へ進み、最後に終了コード 1 を返す。
- `exp/repro/queue.tsv` に構成ごとの commit・開始終了時刻・所要秒・終了コードが残る。
- 短縮学習は学習ロジックを変えたときの軽い確認用。設定は `confs/pretrained_short.yaml` で、`pretrained.yaml` を変えたら合わせて直す。

`exp/repro/{構成名}/version_N/` の中身:

| ファイル | 内容 |
|---|---|
| `test_metrics.json` | dev-test の最終指標（student / teacher） |
| `epoch=*-step=*.ckpt` | validation の指標が最良の ckpt。評価に使われるのはこれ |
| `last.ckpt` | 最終エポックの ckpt |
| `hparams.yaml` | 実行時の設定 |
| `events.out.tfevents.*` | TensorBoard のログ |
| `metrics_test/`, `class-wise-csv/` | 指標の内訳、クラス別の値 |
| `codecarbon/` | 消費電力の記録 |

## ckpt の再評価と cSEBBs（`run_repro_eval.sh`）

```bash
# リポジトリ直下
$ uv run poe repro-eval mt cmt_scale_mixstyle  # ベスト ckpt を再評価。出力は exp/repro_eval/{構成名}/
$ uv run poe sebbs mt                          # cSEBBs を掛けて評価。出力は exp/repro/{構成名}_sebbs/
```

- 1 本ずつ流す。結果を `lightning_logs/version_N` から出力先へ移す処理があり、同時に走らせると取り違える。
- `--test_from_checkpoint` は ckpt に保存された設定を使わず、渡した yaml と CLI フラグで評価する（issue #14）。
  スクリプトを使わずに直接呼ぶときは、学習時と同じフラグを付ける。
- cSEBBs を掛けても `psds_eval` 系の PSDS は変わらず、`sed_scores_eval` 系だけが変わる
  （[reports/2026-11-12-repro-results.md](reports/2026-11-12-repro-results.md)）。`test_step` が cSEBBs の出力に置き換えているのは
  `sed_scores_eval` 用のスコアだけで、`psds_eval` 用の検出結果はメディアンフィルタのまま、というのがコード上の読み（実行しての切り分けはしていない）。
  cSEBBs の効果は `test/teacher/psds{1,2}/sed_scores_eval` で見る。

## 直接実行（`train_pretrained.py`）

```bash
# レシピ
$ export WANDB_MODE=offline                 # wandb にログインしていない場合
$ uv run train_pretrained.py --fast_dev_run # 2 エポック・20 バッチの短い実行（CPU なら --gpus 0）
$ uv run train_pretrained.py --cmt --log_dir ./exp/my_cmt
```

| 引数 | 動作 |
|---|---|
| なし | 学習 → ベスト ckpt で dev-test 評価 |
| `--fast_dev_run` | 2 エポック・20 バッチに切り詰めた学習 + 評価 |
| `--resume_from_checkpoint X` | ckpt から学習を再開 |
| `--test_from_checkpoint X` | 学習せず dev-test の指標を再計算 |
| `--eval_from_checkpoint X` | ラベルなしの `data.eval_folder` にスコアを書き出すだけ（指標なし） |

### wandb

`confs/pretrained.yaml` の `wandb.use_wandb` は既定で `True` で、CLI からは無効にできない（`--use_wandb` は有効にする側だけ）。

- ログインせずに流す: `export WANDB_MODE=offline`（記録はローカルに残る）または `WANDB_MODE=disabled`。
- 完全に切る: `wandb.use_wandb: False` にした yaml を `--confs` で渡す。`run_repro.sh` はこの写しを自動で作っている
  （`exp/repro/pretrained.nowandb.yaml`）。

wandb が有効なとき、ckpt は `--log_dir` ではなく
`experiments/train/{category}/{method}/{variant}/` の下の run ディレクトリに保存される（`local/experiment_dir.py`）。
`{category}` などは `--category` / `--method` / `--variant` で指定する。

## アブレーション（`run_ablation.sh`）

```bash
# レシピ
$ ./run_ablation.sh cmt --list     # 構成名とフラグの一覧
$ ./run_ablation.sh cmt --dry-run  # 実行するコマンドを表示するだけ
$ ./run_ablation.sh cmt
```

| セット | 内容 | 構成数 |
|---|---|---|
| `mixstyle` | MixStyle の入れ方 | 2 |
| `cmt` | CMT の負例サンプリング | 4 |
| `a_times_b` | 学習手法 × 正則化・後処理 | 9 |

wandb を有効にして流すので、ログインするか `WANDB_MODE=offline` を付ける。
標準出力のログは `logs/{category}_{method}/` に残る。

## 再現値

`confs/pretrained.yaml` のまま（150 エポック、seed 42）、7 構成を 2 回ずつ学習した値。指標は teacher の
`test/teacher/psds{1,2}/psds_eval` と `test/teacher/segment_mpauc/sed_scores_eval`。

| 構成 | PSDS1（論文 / 1 回目 / 2 回目） | PSDS2 | mpAUC |
|---|---|---|---|
| `mt` | 0.369 / 0.374 / 0.365 | 0.581 / 0.587 / 0.581 | 0.720 / 0.725 / 0.703 |
| `cmt_scale_mixstyle` | 0.393 / 0.394 / 0.391 | 0.632 / 0.636 / 0.634 | 0.727 / 0.674 / 0.676 |
| `cmt` | 0.297 / 0.339 / 0.306 | 0.60 / 0.614 / 0.609 | 0.689 / 0.720 / 0.726 |
| `mt_mixstyle` | 0.382 / 0.379 / 0.372 | 0.607 / 0.607 / 0.601 | 0.704 / 0.688 / 0.710 |
| `cmt_mixstyle` | 0.387 / 0.388 / 0.387 | 0.637 / 0.643 / 0.643 | 0.722 / 0.691 / 0.692 |
| `cmt_neg` | 0.369 / 0.366 / 0.373 | 0.543 / 0.539 / 0.545 | 0.661 / 0.681 / 0.685 |
| `cmt_scale` | 0.368 / 0.369 / 0.359 | 0.621 / 0.623 / 0.618 | 0.699 / 0.679 / 0.677 |

- 1 回目: 2026-11-12 01:37〜、commit `ddca340`。[reports/2026-11-12-repro-results.md](reports/2026-11-12-repro-results.md) の値。
- 2 回目: 2026-11-12 14:04〜、commit `8aa335d`、`--rotate` で流し直したもの。レポートには入っていない。
- 同じ seed の 2 回の差は、PSDS1 で最大 0.032（`cmt`）、PSDS2 で最大 0.007、mpAUC で最大 0.022（`mt` と `mt_mixstyle`）。
  `training.deterministic` が `False` なので、同じ seed でも値は一致しない。
- seed を変えた実行はしていない。構成間の差がこの揺れより小さいところは、1 回の実行では順位を判断できない。

### 検査用のテスト

| コマンド（リポジトリ直下） | 内容 |
|---|---|
| `uv run poe repro-check` | `exp/repro*` の `test_metrics.json` を `tests/repro_expected.json` と比較。許容幅（`tolerance`）が `null` の間は skip |
| `uv run poe repro-ckpt-check` | `exp/repro` の ckpt を実データの一部に通し、スコアを `tests/repro_ckpt_expected.json` と比較（CPU で数分） |

どちらも既定の `uv run pytest` には含まれない。
