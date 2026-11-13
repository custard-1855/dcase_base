# データと埋め込み

README の「2. データと BEATs の準備」「3. 埋め込み抽出」の補足.
パスは断りが無ければレシピ (`DESED_task/dcase2024_task4_baseline/`) からの相対.

## 置き場所の決まり方

データの置き場所は `confs/pretrained.yaml` の `data.root` の 1 か所で決まる.
`data` の他のパスは `root` からの相対で書いてあり, yaml を読んだ直後に絶対パスへ変換される
(`local/utils.py` の `resolve_data_paths`).

`generate_dcase_task4_2024.py --basedir=X` は `X/dcase/dataset/` に展開するので, `data.root` には `X/dcase/dataset` を書く.
既定値は `--basedir="/mnt/data/data"` と `data.root: "/mnt/data/data/dcase/dataset"` の組.

## 配置と容量

論文の実験に使った計算機での実測 (2026-11-14 時点). 全体で約 127 GB.
ファイル数は取得できた分で, 取得時期によって変わる (次節).

| セット | `data.root` からのパス | ファイル数 | 容量 |
|---|---|---|---|
| 合成 (train) | `dcase_synth/audio/train/synthetic21_train/soundscapes/` | 30006 | 17 GB |
| 合成 (validation) | `dcase_synth/audio/validation/synthetic21_validation/soundscapes/` | 7500 | 4.2 GB |
| strong (AudioSet) | `audio/train/strong_label_real/` | 1613 | 2.7 GB |
| weak | `audio/train/weak/` | 1317 | 2.2 GB |
| unlabeled | `audio/train/unlabel_in_domain/` | 12027 | 20 GB |
| dev-test (validation) | `audio/validation/validation/` | 958 | 1.6 GB |
| MAESTRO real (train) | `audio/maestro_real_train/` | 7503 | 13 GB |
| MAESTRO real (validation) | `audio/maestro_real_validation/` | 718 | 1.2 GB |

- 上は元音源 (44.1 kHz). 同じ階層に 16 kHz の写し `*_16k/` が作られ, 学習と埋め込み抽出はこちらを読む.
  容量の合計 127 GB には 16 kHz の写しとダウンロード時の中間ファイルを含む.
- ラベルは `metadata/` と `dcase_synth/metadata/` の tsv.
- ダウンロードの目安: 合成データの zip だけで約 20 GB, MAESTRO は展開後 約 21 GB.

### 16 kHz の写し

`extract_embeddings.py` と `train_pretrained.py` は, 開始時に `*_folder_44k` から `*_folder` (16 kHz) を作る
(`train_pretrained.py` の `resample_data_generate_durations`). 出力先の wav の数が元と同じなら作り直さない.
このときデータ側のディレクトリに書き込むので, データを読み取り専用で置く場合は先に 1 回作っておく.

## 取得できないファイル

AudioSet 由来の音声 (weak / unlabeled / strong / dev-test) は YouTube から取得するため, 公開停止された動画の分は取得できない.
取得できなかった一覧は, ダウンロードを実行したディレクトリの `missing_files/` に出る.

欠損への対処はコード側に入っている.

- tsv にあって音声が無いファイルは Dataset が `None` を返し, `SafeDataLoader` (`local/utils.py`) が読み飛ばす.
- 件数は Dataset の構築時にセットごとに警告として表示される.

欠損の数が違うと学習データも dev-test も変わるので, 指標は論文値や `docs/reports/` の値と完全には一致しない.

## 評価用セット (eval24)

`data.eval_folder` (`audio/eval24_16k`) は `generate_dcase_task4_2024.py` では取得されない.
使うのは `--eval_from_checkpoint` (ラベルなし音声にスコアを書き出すモード) だけで, 学習と dev-test 評価には不要.

`extract_embeddings.py` は eval24 も対象にしている. フォルダが無い場合の動作は未確認
(コード上は wav が 0 件のまま空の `eval.hdf5` を書く).

## BEATs のチェックポイント

- ファイル名は `BEATs_iter3_plus_AS2M.pt` (361 MB), 置き場所は `pretrained_models/`.
- 公式の配布元は OneDrive ([BEATs README](https://github.com/microsoft/unilm/blob/master/beats/README.md)).
  以前使われていた blob の直リンクは失効している.
- README に載せた Hugging Face のミラーは第三者によるもの. sha256 は
  `d43cbfad4d7b56381c061d7a24774f908d4d94c72961f6eb1d9090ff18cd8d34`.

## 埋め込み

`extract_embeddings.py --output_dir ./embeddings` が `embeddings/beats/{セット名}.hdf5` を書く. 合計 約 60 GB.

| ファイル | 元のセット | 容量 |
|---|---|---|
| `synth_train.hdf5` | 合成 (train) | 15 GB |
| `synth_val.hdf5` | 合成 (validation) | 3.6 GB |
| `strong_train.hdf5` | strong | 2.3 GB |
| `weak_train.hdf5`, `weak_val.hdf5` | weak (同じフォルダから 2 つ作り, 学習側で train / validation に分ける) | 各 1.9 GB |
| `unlabeled_train.hdf5` | unlabeled | 18 GB |
| `devtest.hdf5` | dev-test | 1.4 GB |
| `maestro_real_train.hdf5` | MAESTRO real (train) | 11 GB |
| `maestro_real_dev.hdf5` | MAESTRO real (validation) | 1.1 GB |
| `eval.hdf5` | eval24 | 3.2 GB |

各ファイルの中身は 3 つのデータセット.

| 名前 | 形 | 内容 |
|---|---|---|
| `filenames` | (N,) | wav の stem. Dataset はこれをキーに埋め込みを引く |
| `frame_embeddings` | (N, 768, 496) | フレームごとの埋め込み. 学習で使うのはこちら (`net.embedding_type: frame`) |
| `global_embeddings` | (N, 768) | クリップ全体の埋め込み |

注意点:

- 置き場所は `confs/pretrained.yaml` の `pretrained.extracted_embeddings_dir` (既定 `./embeddings`). レシピからの相対なので,
  `train_pretrained.py` はレシピをカレントにして実行する.
- 音声を追加・取り直したら埋め込みも作り直す. 埋め込みに無い音声を読むと `KeyError` になる.
- セットを増やすときは, `extract_embeddings.py` 末尾の辞書と `train_pretrained.py` の `get_embeddings_name(config, "<名前>")` の両方に名前を足す.
- `*.hdf5` / `*.pt` / `*.ckpt` / `*.wav` / `*.tsv` は `.gitignore` 済み. リポジトリには入らない.
