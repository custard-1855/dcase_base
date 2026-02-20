卒業論文用に書いたコードを保存しています.

## 環境構築
### インストール
- 当リポジトリをクローン
```bash
$ git clone https://github.com/custard-1855/dcase_base.git
```
- パッケージマネージャ [uv](https://github.com/astral-sh/uv)でライブラリをインストール
```bash
$ curl -LsSf https://astral.sh/uv/0.7.12/install.sh | sh # uvをインストール
$ sudo apt install sox
$ uv sync # ライブラリインストール
$ . ./.venv/bin/activate # 仮想環境を起動
$ cd dcase_base/DESED_task/dcase2024_task4_baseline # 作業ディレクトリに移動
```

### データ準備
- ここでは`/mnt`にデータを保存することを想定.
- 保存場所を変更する場合, `confs/pretrained.yaml`のデータへのパス変更が必要な点に注意.
- 参照: https://github.com/DCASE-REPO/DESED_task/tree/dcase24_baseline/recipes/dcase2024_task4_baseline
```bash
$ uv run generate_dcase_task4_2024.py --basedir="/mnt"
$ uv run extract_embeddings.py --output_dir ./embeddings
```

<!-- 
未検証
データやBEATsのckptダウンロードが足りない
 -->


## 実験
- 各実験のため, 引数が細かく設定されている.
- 詳細は各shファイルを参照.
```bash
# dcase_base/DESED_task/dcase2024_task4_baseline
$ . ./run_cmt_ablation.sh
$ . ./run_ablation_a_times_b.sh
$ . ./run_mixstyle.sh
$ uv run train_pretrained.py
```

### wandb
```bash
$ export WANDB_MODE=offline
# $ WANDB_MODE=disabled 
```
