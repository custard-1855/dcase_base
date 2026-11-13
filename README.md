卒業論文用に書いたコードを保存しています.

## 環境構築
### インストール
- 当リポジトリをクローン (`sebbs` が git submodule なので `--recursive` が必要)
```bash
$ git clone --recursive https://github.com/custard-1855/dcase_base.git
# クローン済みの場合: git submodule update --init --recursive
```
- パッケージマネージャ [uv](https://github.com/astral-sh/uv)でライブラリをインストール
```bash
$ curl -LsSf https://astral.sh/uv/0.7.12/install.sh | sh # uvをインストール
$ sudo apt install sox libsox-dev # libsox-dev は soxbindings のビルドに必要
$ cd dcase_base
$ uv sync # ライブラリインストール (submodule の sebbs もここでビルドされる)
$ . ./.venv/bin/activate # 仮想環境を起動
$ cd DESED_task/dcase2024_task4_baseline # 作業ディレクトリに移動
```

### 動作確認 (smoke test)
- データや GPU がなくても, 合成データで「学習 → validation → dev-test 評価」と「ckpt からの推論」が通ることを確認できる.
```bash
# dcase_base (リポジトリ直下) で実行
$ uv run poe test   # モデルの forward など軽いテストのみ (数十秒)
$ uv run pytest     # 学習・推論を最後まで回すテストも含む (CPU で 5 分程度)
$ uv run poe        # 登録済みのタスク一覧 (再現学習, ckpt の再評価など)
```

### コミット前・プッシュ前の検査
```bash
$ uv run prek install # 設定は prek.toml. コミット時に構文・型検査, プッシュ時に軽いテストが走る
```

### データ準備
- `confs/pretrained.yaml` は `/mnt/data/data/dcase/dataset` 以下にデータがある想定.
- 保存場所を変更する場合, `confs/pretrained.yaml`の`data.root`を書き換える(各パスは`root`からの相対).
- 参照: https://github.com/DCASE-REPO/DESED_task/tree/dcase24_baseline/recipes/dcase2024_task4_baseline
```bash
$ uv run generate_dcase_task4_2024.py --basedir="/mnt/data/data"
```
- 必要な空き容量の目安: 合成データの zip だけで約 20GB, MAESTRO は展開後 約 21GB.
- AudioSet 由来の音声 (weak / unlabeled / strong / validation) は YouTube から取得するため, 取得できないファイルが出る (`missing_files/` に一覧).

### BEATs のチェックポイント
- `BEATs_iter3_plus_AS2M.pt` (361MB) を `pretrained_models/` に置く.
- 公式の配布元は OneDrive ([BEATs README](https://github.com/microsoft/unilm/blob/master/beats/README.md)). 以前使われていた blob の直リンクは失効している.
- wget で取得する場合は Hugging Face 上の第三者ミラーが使える (公式配布ではない点に注意).
```bash
$ mkdir -p pretrained_models
$ wget -O pretrained_models/BEATs_iter3_plus_AS2M.pt \
    https://huggingface.co/lpepino/beats_ckpts/resolve/main/BEATs_iter3_plus_AS2M.pt
$ sha256sum pretrained_models/BEATs_iter3_plus_AS2M.pt
# d43cbfad4d7b56381c061d7a24774f908d4d94c72961f6eb1d9090ff18cd8d34
```

### 埋め込みの抽出
```bash
$ uv run extract_embeddings.py --output_dir ./embeddings
# 別の場所の ckpt を使う場合: --beats_ckpt /path/to/BEATs_iter3_plus_AS2M.pt
```


## 実験
- 各実験のため, 引数が細かく設定されている.
- 詳細は`run_ablation.sh`を参照(`./run_ablation.sh cmt --list` で構成と引数の一覧).
```bash
# dcase_base/DESED_task/dcase2024_task4_baseline
$ uv run train_pretrained.py --fast_dev_run # まず短い実行で確認 (CPU の場合は --gpus 0)
$ bash ./run_ablation.sh cmt
$ bash ./run_ablation.sh a_times_b
$ bash ./run_ablation.sh mixstyle
$ uv run train_pretrained.py
```

### args
- およそshファイルに記載の通り.
- 以下で補足する.

#### MixStyle
- mixstyle_type
  - disabled: MixStyleなし
  - only_mix: CNNなし
  - freqAttn: CNNあり
- attn_type
  - 今後の拡張のため定義
  - 今回はデフォルト値のみ使用

### wandb
```bash
$ export WANDB_MODE=offline
# $ WANDB_MODE=disabled
```

## 引用
本検討は電子情報通信学会にて発表しています.
```
竹本志恩, 川原亮一, “実環境を見据えた音による行動認識モデルの精度向上に関する一検討,” 信学技報, vol. 125, no. 385, NS2025-226, pp. 31-36, 2026 年 3 月.
```
