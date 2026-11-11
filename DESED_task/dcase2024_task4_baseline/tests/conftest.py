"""smoke test 用の合成データセット.

実データ (DESED / MAESTRO / BEATs 埋め込み) の代わりに, 同じディレクトリ構成・
同じファイル形式の極小データをノイズから生成する.
学習結果に意味はなく, パイプラインが最後まで通ることだけを確認するためのもの.
"""

import copy
import glob
import os
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import pytest
import soundfile as sf
import yaml

RECIPE_DIR = Path(__file__).resolve().parents[1]
BASE_CONFIG = RECIPE_DIR / "confs" / "pretrained.yaml"

ORIG_FS = 44100
CLIP_SEC = 10
N_CLIPS = 6  # DESED 系の各セットのクリップ数
MAESTRO_SEC = 12  # MAESTRO の「長尺」録音の長さ
EMB_DIM = 768  # BEATs の埋め込み次元
EMB_FRAMES = 496  # BEATs の 10 秒あたりのフレーム数


def _write_noise(path, seconds, rng):
    path.parent.mkdir(parents=True, exist_ok=True)
    audio = (0.1 * rng.standard_normal(int(seconds * ORIG_FS))).astype(np.float32)
    sf.write(path, audio, ORIG_FS)


def _strong_set(folder, tsv, prefix, classes, rng):
    """強ラベル付きセット (filename / onset / offset / event_label)."""
    rows = []
    for i in range(N_CLIPS):
        name = f"{prefix}_{i}.wav"
        _write_noise(folder / name, CLIP_SEC, rng)
        # 全クラスが正解ラベルに現れるよう, クラスを巡回させて 1 クリップ 4 イベント置く
        for j in range(4):
            label = classes[(4 * i + j) % len(classes)]
            onset = 0.5 + 2.2 * j
            rows.append((name, onset, onset + 1.5, label))
    tsv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=["filename", "onset", "offset", "event_label"]).to_csv(
        tsv, sep="\t", index=False
    )


def _weak_set(folder, tsv, classes, rng):
    """弱ラベル付きセット (filename / event_labels)."""
    rows = []
    for i in range(10):  # weak_split=0.9 で 9 train / 1 valid に分かれる
        name = f"weak_{i}.wav"
        _write_noise(folder / name, CLIP_SEC, rng)
        labels = [classes[i % len(classes)], classes[(i + 3) % len(classes)]]
        rows.append((name, ",".join(labels)))
    tsv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=["filename", "event_labels"]).to_csv(tsv, sep="\t", index=False)


def _maestro(root, audio_dir, meta_dir, rng):
    """MAESTRO Real を模した長尺録音を作り, 本物の前処理 (10 秒チャンク分割) にかける."""
    from dprep.download_prep_maestro import split_maestro_real
    from local.classes_dict import classes_labels_maestro_real

    classes = list(classes_labels_maestro_real.keys())
    download = root / "MAESTRO_original" / "maestro_dev"
    durations = []
    for split, column, per_scene in [("train", "train", 2), ("validation", "val", 1)]:
        stems = pd.read_csv(RECIPE_DIR / "dprep" / f"{split}_split.csv")[column]
        stems = sorted(Path(x).stem for x in stems)
        scenes = sorted({s.rsplit("_", 1)[0] for s in stems})
        for scene in scenes:
            for stem in [s for s in stems if s.rsplit("_", 1)[0] == scene][:per_scene]:
                _write_noise(download / "development_audio" / scene / f"{stem}.wav", MAESTRO_SEC, rng)
                ann = download / "development_annotation" / scene / f"{stem}.txt"
                ann.parent.mkdir(parents=True, exist_ok=True)
                with open(ann, "w") as f:
                    for k, label in enumerate(classes):
                        onset = float(k % (MAESTRO_SEC - 2))
                        f.write(f"{onset}\t{onset + 2.0}\t{label}\t0.9\n")
                durations.append((stem, float(MAESTRO_SEC)))  # 実データも拡張子なし
    pd.DataFrame(durations, columns=["filename", "duration"]).to_csv(
        download / "development_metadata.csv", sep="\t", index=False
    )
    split_maestro_real(str(download), str(audio_dir), str(meta_dir))


def _fake_embeddings(wav_folder, hdf5_path, rng):
    """extract_embeddings.py と同じレイアウトの hdf5 を乱数で作る."""
    stems = sorted(Path(x).stem for x in glob.glob(os.path.join(wav_folder, "*.wav")))
    assert stems, f"no wav files in {wav_folder}"
    hdf5_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(hdf5_path, "w") as f:
        f.create_dataset(
            "global_embeddings",
            data=rng.standard_normal((len(stems), EMB_DIM)).astype(np.float32),
        )
        f.create_dataset(
            "frame_embeddings",
            data=rng.standard_normal((len(stems), EMB_DIM, EMB_FRAMES)).astype(np.float32),
        )
        f.create_dataset("filenames", data=stems)


@pytest.fixture(scope="session")
def base_config():
    """リポジトリの confs/pretrained.yaml をそのまま読んだ設定."""
    with open(BASE_CONFIG) as f:
        return yaml.safe_load(f)


@pytest.fixture(scope="session")
def synthetic_config(tmp_path_factory, base_config):
    """合成データセットを生成し, それを指す設定 dict を返す (セッションで 1 回だけ生成)."""
    from local.classes_dict import classes_labels_desed
    from train_pretrained import resample_data_generate_durations

    rng = np.random.default_rng(0)
    root = tmp_path_factory.mktemp("dcase") / "dataset"
    audio, meta = root / "audio", root / "metadata"
    desed_classes = list(classes_labels_desed.keys())

    data = {
        "synth_folder_44k": audio / "train" / "synth",
        "synth_tsv": meta / "train" / "synth.tsv",
        "strong_folder_44k": audio / "train" / "strong_label_real",
        "strong_tsv": meta / "train" / "audioset_strong.tsv",
        "weak_folder_44k": audio / "train" / "weak",
        "weak_tsv": meta / "train" / "weak.tsv",
        "unlabeled_folder_44k": audio / "train" / "unlabel_in_domain",
        "synth_val_folder_44k": audio / "validation" / "synth_val",
        "synth_val_tsv": meta / "validation" / "synth_val.tsv",
        "synth_val_dur": meta / "validation" / "synth_val_durations.tsv",
        "test_folder_44k": audio / "validation" / "validation",
        "test_tsv": meta / "validation" / "validation.tsv",
        "test_dur": meta / "validation" / "validation_durations.tsv",
        "real_maestro_train_folder_44k": audio / "maestro_real_train",
        "real_maestro_train_tsv": meta / "maestro_real_train.tsv",
        "real_maestro_val_folder_44k": audio / "maestro_real_validation",
        "real_maestro_val_tsv": meta / "maestro_real_validation.tsv",
        "real_maestro_val_dur": meta / "maestro_real_durations.tsv",
        "eval_folder_44k": audio / "eval24",
    }
    for key in [k for k in data if k.endswith("_44k")]:
        data[key[: -len("_44k")]] = Path(str(data[key]) + "_16k")

    _strong_set(data["synth_folder_44k"], data["synth_tsv"], "synth", desed_classes, rng)
    _strong_set(data["strong_folder_44k"], data["strong_tsv"], "strong", desed_classes, rng)
    _strong_set(data["synth_val_folder_44k"], data["synth_val_tsv"], "val", desed_classes, rng)
    _strong_set(data["test_folder_44k"], data["test_tsv"], "test", desed_classes, rng)
    _weak_set(data["weak_folder_44k"], data["weak_tsv"], desed_classes, rng)
    for i in range(N_CLIPS):
        _write_noise(data["unlabeled_folder_44k"] / f"unlabel_{i}.wav", CLIP_SEC, rng)
    for i in range(N_CLIPS):  # ラベルなしの評価セット (--eval_from_checkpoint 用)
        _write_noise(data["eval_folder_44k"] / f"eval_{i}.wav", CLIP_SEC, rng)
    _maestro(root, audio, meta, rng)

    config = copy.deepcopy(base_config)
    config["data"].update({k: str(v) for k, v in data.items()})

    # 44.1kHz -> 16kHz のリサンプルと duration tsv 生成 (本番と同じコード).
    # ただし本番の fork ベースの並列実行は, torch を使った後の pytest プロセスでは
    # デッドロックするため, テストでは逐次実行に差し替える.
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(
            "local.resample_folder.process_map",
            lambda fn, args, **_: [fn(a) for a in args],
        )
        resample_data_generate_durations(config["data"])
        resample_data_generate_durations(config["data"], test_only=True, evaluation=True)

    emb_dir = root / "embeddings"
    for name, folder in {
        "synth_train": "synth_folder",
        "strong_train": "strong_folder",
        "weak_train": "weak_folder",
        "weak_val": "weak_folder",
        "unlabeled_train": "unlabeled_folder",
        "synth_val": "synth_val_folder",
        "devtest": "test_folder",
        "maestro_real_train": "real_maestro_train_folder",
        "maestro_real_dev": "real_maestro_val_folder",
        "eval": "eval_folder",
    }.items():
        _fake_embeddings(config["data"][folder], emb_dir / "beats" / f"{name}.hdf5", rng)
    config["pretrained"]["extracted_embeddings_dir"] = str(emb_dir)

    # CPU で数十秒に収めるための縮小設定 (ネットワーク構造は本番のまま)
    config["training"].update(
        {
            "batch_size": [2, 2, 2, 2, 2],
            "batch_size_val": 4,
            "num_workers": 0,
            "n_test_thresholds": 5,
            "enable_progress_bar": False,
        }
    )
    config["wandb"]["use_wandb"] = False
    return config
