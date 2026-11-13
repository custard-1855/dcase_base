# 手法を追加するときに触るファイル

既存の 3 手法 (CMT・MixStyle・cSEBBs) が, それぞれ「損失」「モデル」「後処理」の追加例になっている.
足したい手法に近いものをなぞるのが早い. パスはレシピ (`DESED_task/dcase2024_task4_baseline/`) からの相対.
ファイルごとの役割は同ディレクトリの `CLAUDE.md` の「地図」にある.

## 全体の流れ

どの種類の手法でも, 設定は次の順に流れる.

```
CLI フラグ ──> prepare_run ──> config (dict) ──┬─> CRNN(**config["net"])       モデル
(train_pretrained.py)                         └─> SEDTask4.hparams            損失・後処理
```

1. `train_pretrained.py` の `prepare_run` が yaml を読み, CLI フラグの値を `configs[...]` に書き込む.
2. `single_run` が `configs["net"]` をそのまま `CRNN` に渡し, `configs` 全体を `SEDTask4` (`local/sed_trainer_pretrained.py`) に渡す.
3. `SEDTask4.__init__` が `self.hparams` から必要な値を取り出して属性にする.

## 共通で触るところ

| 順 | ファイル | やること |
|---|---|---|
| 1 | `train_pretrained.py` の `prepare_run` | `parser.add_argument` でフラグを足し, その値を `configs["<節>"]["<キー>"]` に書き込む |
| 2 | `confs/pretrained.yaml` | 同じキーを既定値つきで足す. `confs/pretrained_short.yaml` にも同じ変更を入れる |
| 3 | 手法の種類ごとのファイル (次節) | 本体の実装 |
| 4 | `tests/test_smoke.py` の `VARIANTS` | フラグの組み合わせを 1 行足す. 合成データでの学習〜評価のテストに入る |
| 5 | `run_repro.sh` または `run_ablation.sh` | 実データで流す構成を `add 構成名 "フラグ"` で足す |

- 1 で引数に既定値を付けると, yaml の値は常に CLI の値で上書きされる (既存の `--cmt` などはこの形).
  yaml の値を生かしたいときは, 引数の既定値を `None` にして「`None` でなければ上書き」と書く (`--phi_clip` がこの形).
- 5 で `run_repro.sh` に足した構成は, テスト (`REPRO_VARIANTS`) にも自動で入る.

## 種類ごとの追加箇所

### 損失・学習方法を足す (例: CMT)

| ファイル | CMT での実装 |
|---|---|
| `local/cmt.py` | 擬似ラベルの生成, 信頼度の重み, 一貫性損失の 3 関数. `SEDTask4` を第 1 引数 `self` として受け取る |
| `local/sed_trainer_pretrained.py` の `__init__` | `self.hparams["cmt"]` から `cmt_enabled`, `phi_*` などを取り出す |
| 同 `training_step` | 一貫性損失を計算する箇所で `cmt_active` のときだけ `local/cmt.py` の関数を呼び, そうでなければ Mean Teacher の損失を使う |

新しい損失は `local/` に別モジュールとして書き, `training_step` からは分岐と呼び出しだけにする.
`training_step` の中では, バッチが `[maestro, synth, strong, weak, unlabeled]` の順に並んでいる
(`training.batch_size` の順). どの範囲に損失を掛けるかはマスク (`mask_consistency` など) で決めている.

### モデルの層を足す (例: MixStyle)

| ファイル | MixStyle での実装 |
|---|---|
| `desed_task/nnet/mixstyle.py` | 層の本体 (`BasicMixStyleWrapper`, `FrequencyAttentionMixStyle`) |
| `desed_task/nnet/CNN.py` | `kwargs` から `mixstyle_type` を読み, `_create_mixstyle_layer` で層を作って `forward` に挟む. 学習時だけ適用 |
| `desed_task/nnet/CRNN.py` | 触っていない. 受け取った `kwargs` をそのまま `CNN` に渡している |

`configs["net"]` のキーは `CRNN(**config["net"])` でそのまま引数になるので, フラグの値を `configs["net"]` に入れれば `CNN` まで届く.
`tests/test_smoke.py` の `test_crnn_forward` が `MIXSTYLE_TYPES` の各値で forward を確認しているので, 選択肢を足したらここにも足す.

### 後処理を足す (例: cSEBBs)

| ファイル | cSEBBs での実装 |
|---|---|
| `local/sebbs_postprocess.py` | 後処理の適用 (`get_sebbs`) と, チューニング時の選択関数 |
| `local/sebbs_wrapper/` | `sebbs` submodule の型付きラッパー (`SEBBsTuner`, `SEBBsPredictor`) |
| `local/sed_trainer_pretrained.py` の `validation_step` | チューニング用に validation のスコアを貯める |
| 同 `on_test_start` | test の前に validation を 1 回流してスコアを集め直す (`--test_from_checkpoint` でも調整できるようにするため) |
| 同 `test_step` | 初回にチューニングし, `sebbs_enabled` のときは `sed_scores_eval` 用のスコアを `get_sebbs` の出力で置き換える |

置き換えているのは `sed_scores_eval` 用のスコアだけで, `psds_eval` 用の検出結果 (`decoded_*`) はメディアンフィルタのままになっている.
後処理を足すときは, どの指標の入力を差し替えるかを `test_step` で確認する.

`sebbs/` は外部の submodule なので編集しない. import は `from sebbs.sebbs.csebbs import ...` の形で書く.

### データセットを足す

`CLAUDE.md` の「データの約束事」を参照. 触るのは `confs/pretrained.yaml` の `data`, `extract_embeddings.py` 末尾の辞書,
`train_pretrained.py` (Dataset の構築と `get_embeddings_name`), `tests/conftest.py` の `synthetic_config` (合成データを同じ形式で足す).

## 確認の手順

```bash
# リポジトリ直下
$ uv run poe test                # 軽いテスト (数十秒)
$ uv run pytest                  # 合成データで学習〜評価まで (CPU で 5 分程度)
$ uv run ruff check <触ったファイル>
```

`uv run pytest` には, 合成データでの指標・損失・重みを保存値と比較するテストが入っている
(`tests/synthetic_expected.json`, `tests/synthetic_expected_trace.json`).

- フラグを付けないときの挙動を変えていなければ, 既存の構成の比較は通る. 落ちたら, 意図せず既存の計算を変えている.
- 新しく足した構成は保存値が無いので skip になる. 次のコマンドで保存値を作ってコミットする.
- 既存の計算を意図して変えた場合も, 同じコマンドで取り直す.

```bash
# リポジトリ直下
$ UPDATE_SYNTHETIC_EXPECTED=1 uv run pytest -k "synthetic or variant"
```

実データでは, まず短縮学習で確認してから本番を流す.

```bash
# リポジトリ直下 (run_repro.sh に構成を足した場合)
$ uv run poe repro-short <構成名>  # 10 エポック, 約 7 分
$ uv run poe repro <構成名>        # 150 エポック, 約 1 時間
```

ベースラインと比べるときは, 同じ seed でも実行ごとに値が揺れる点に注意する
(幅は [experiments.md の再現値](experiments.md#再現値)).

## 書き方の決まり

- PyTorch Lightning は 1.9 系に固定. フック名は 1.9 の API (`validation_epoch_end` など) で書く.
- 本家由来のファイル (`desed_task/`, `train_pretrained.py` など) は black 形式のまま. 差分を小さく保つため, 触る行だけを直す.
- ruff は全ルール有効で既存コードに指摘が多数残っている. 合格条件ではなく, 自分が足した行に指摘を増やさないことを目安にする.
- 依存の追加は `uv add` で行い, `uv.lock` もコミットする.
- 学習を回すテストには `@pytest.mark.slow` を付ける.
