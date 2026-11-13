"""再現学習の ckpt を実データの一部に通し, スコアを保存値と比較する (既定の pytest には含めない).

    uv run pytest -m repro_ckpt        # = uv run poe repro-ckpt-check (CPU で数分)

tests/test_repro.py (マーカー repro) は保存済みの test_metrics.json を読むだけだが,
こちらは ckpt をロードして dev-test の固定した一部のクリップで推論と評価を実行する.
整形やリファクタの前後で, 学習済みモデルの出力が変わっていないことを確認するためのもの.
ckpt, 実データ, 埋め込みのどれかが無い環境では skip する. 保存値は tests/repro_ckpt_expected.json.

意図して挙動を変えたとき, ckpt を学習し直したとき, torch などの版を上げたときは, 次で取り直す:

    UPDATE_REPRO_CKPT_EXPECTED=1 uv run pytest -m repro_ckpt

ckpt の場所は REPRO_CKPT_DIR で変えられる (既定は exp/repro. run_repro.sh の出力と同じ構成).
"""

import copy
import hashlib
import json
import os
from pathlib import Path

import pandas as pd
import pytest
import torch
import yaml
from test_smoke import REPRO_VARIANTS, _run, _test_metrics

RECIPE_DIR = Path(__file__).resolve().parents[1]
CKPT_DIR = Path(os.environ.get("REPRO_CKPT_DIR", RECIPE_DIR / "exp" / "repro"))
EXPECTED = Path(__file__).with_name("repro_ckpt_expected.json")

DESED_CLIPS_PER_CLASS = 10  # DESED: クラスごとに選ぶクリップ数
MAESTRO_RECORDINGS_PER_CLASS = 1  # MAESTRO: 評価対象のクラスごとに選ぶ録音数

# 保存値との許容幅 (絶対値). 同じ計算機の CPU での実測 (2026-11-13, 7 構成の全 535 項目):
#   同じスレッド数で別プロセス: 最大 2e-16 (psds2 の 4 項目のみ. 他はビット単位で一致)
#   16 スレッドと 1 スレッド:   最大 2e-8 (loss_strong), スコアの平均は 3e-9, 他の指標は一致
# スレッド数による揺れの約 50 倍を取っている. 後処理や特徴量を変えたときの差は 1e-4 以上.
TOLERANCE = 1e-6

pytestmark = pytest.mark.repro_ckpt


@pytest.fixture(autouse=True)
def _only_when_selected(request):
    """-m でマーカーを名指ししたときだけ実行する.

    pyproject.toml の addopts で既定では除外しているが, コマンドラインの -m はそれを上書きする.
    -m "not slow" のような指定で, 実データの評価が意図せず走らないようにする.
    """
    if "repro_ckpt" not in request.config.getoption("-m"):
        pytest.skip("opt-in (uv run pytest -m repro_ckpt)")


def select_desed_clips(tsv, durations):
    """DESED validation の一部: クラスごとに, そのクラスを含むクリップをファイル名順に先頭から選ぶ.

    全クラスの正解ラベルが入るようにするため (ファイル名順の先頭 N 件では少数クラスが欠ける).
    音声が無いクリップ (durations に無いもの. 配布元で欠けている) は対象にしない.
    """
    tsv = tsv[tsv["filename"].isin(durations["filename"])].dropna()
    clips = set()
    for _, rows in sorted(tsv.groupby("event_label")):
        clips.update(sorted(rows["filename"].unique())[:DESED_CLIPS_PER_CLASS])
    return sorted(clips)


def select_maestro_recordings(tsv):
    """MAESTRO validation の一部: 評価対象のクラスごとに, そのクラスを含む録音を名前順に選ぶ.

    評価は 10 秒クリップのスコアを録音単位に重ね合わせてから行うので, 録音の全クリップを使う.
    また, 正解ラベル (confidence > 0.5) が 1 つも無いクラスがあると評価が例外で止まるので,
    全クラスが入るように選ぶ. クリップ名は {scene}_{番号}-{開始}-{終了}.wav.
    """
    from local.classes_dict import classes_labels_maestro_real_eval

    tsv = tsv[tsv["confidence"] > 0.5]  # 正解として扱う閾値 (on_test_epoch_end と同じ)
    recordings = set()
    for label in sorted(classes_labels_maestro_real_eval):
        rows = tsv[tsv["event_label"] == label]
        of_label = sorted(rows["filename"].str.rsplit("-", n=2).str[0].unique())
        recordings.update(of_label[:MAESTRO_RECORDINGS_PER_CLASS])
    return sorted(recordings)


@pytest.fixture(scope="module")
def subset_config(tmp_path_factory):
    """dev-test を一部のクリップに絞った設定 (学習時の設定が元). データが無ければ skip.

    絞るのは正解ラベルと durations の tsv だけで, 音声と埋め込みは元の場所から読む
    (Dataset は tsv にあるクリップだけを読み, 埋め込みは hdf5 から stem で引く).
    """
    from local.utils import resolve_data_paths

    conf_file = CKPT_DIR / "pretrained.nowandb.yaml"
    if not conf_file.exists():
        pytest.skip(f"学習時の設定が無い: {conf_file}")
    with open(conf_file) as f:
        config = yaml.safe_load(f)
    data = resolve_data_paths(config["data"])

    # 相対パスはレシピのディレクトリから見たもの. テストは tmp にカレントを移して実行する
    emb_dir = RECIPE_DIR / config["pretrained"]["extracted_embeddings_dir"]
    config["pretrained"]["extracted_embeddings_dir"] = str(emb_dir)

    # --test_from_checkpoint でも validation 用の Dataset は作られるので, その分も要る
    needed = [data[k] for k in data if k.endswith(("_tsv", "_dur")) and "synth_maestro" not in k]
    needed += [data["test_folder"], data["real_maestro_val_folder"]]
    needed += [
        emb_dir / config["pretrained"]["model"] / f"{name}.hdf5"
        for name in ["devtest", "maestro_real_dev", "synth_val", "weak_val", "maestro_real_train"]
    ]
    missing = [str(p) for p in needed if not os.path.exists(p)]
    if missing:
        pytest.skip(f"実データまたは埋め込みが無い: {missing[0]} (ほか {len(missing) - 1} 件)")

    tmp_path = tmp_path_factory.mktemp("devtest_subset")
    desed_tsv = pd.read_csv(data["test_tsv"], sep="\t")
    desed_dur = pd.read_csv(data["test_dur"], sep="\t")
    desed_clips = select_desed_clips(desed_tsv, desed_dur)
    maestro_tsv = pd.read_csv(data["real_maestro_val_tsv"], sep="\t")
    recordings = select_maestro_recordings(maestro_tsv)
    maestro_tsv = maestro_tsv[maestro_tsv["filename"].str.rsplit("-", n=2).str[0].isin(recordings)]

    subset = {
        "test_tsv": desed_tsv[desed_tsv["filename"].isin(desed_clips)],
        "test_dur": desed_dur[desed_dur["filename"].isin(desed_clips)],
        "real_maestro_val_tsv": maestro_tsv,
    }
    for key, df in subset.items():
        data[key] = str(tmp_path / f"{key}.tsv")
        df.to_csv(data[key], sep="\t", index=False)
    # 16 kHz への変換は済んでいる前提. 変換元にも同じフォルダを指して, 変換 (元データの
    # ディレクトリへの書き込み) が起きないようにする
    data["test_folder_44k"] = data["test_folder"]

    config["training"].update({"num_workers": 0, "enable_progress_bar": False})
    return config, {
        "desed_clips": len(desed_clips),
        "maestro_clips": int(maestro_tsv["filename"].nunique()),
        "maestro_recordings": recordings,
    }


def _best_checkpoint(name):
    """ベスト ckpt (last.ckpt ではない). version が複数あれば最新 (run_repro_eval.sh と同じ)."""
    files = sorted(
        (CKPT_DIR / name).glob("version_*/epoch=*.ckpt"),
        key=lambda p: int(p.parent.name.split("_")[1]),
    )
    if not files:
        pytest.skip(f"{CKPT_DIR / name} に ckpt が無い")
    return files[-1]


def _score_summary(run_dir):
    """保存されたスコア TSV の要約 (クリップ数, 行数, クラスごとの平均スコア).

    指標は小さなサブセットでは 0 や同値になりやすいので, モデルの出力そのものも比べる.
    desed は後処理 (median filter) 後のフレーム単位のスコア, maestro は録音単位に
    重ね合わせた 1 秒セグメントのスコア.
    """
    summary = {}
    for model in ["student", "teacher"]:
        for dataset, sub in [("desed", "scenario1/scores"), ("maestro", "maestro/postprocessed")]:
            score_dirs = list(run_dir.rglob(f"metrics_test/{model}/{sub}"))
            assert len(score_dirs) == 1, f"{model}/{sub} のスコア出力先が見つからない"
            files = sorted(score_dirs[0].glob("*.tsv"))
            scores = pd.concat([pd.read_csv(f, sep="\t") for f in files])
            summary[f"{model}/{dataset}"] = {
                "clips": len(files),
                "rows": len(scores),
                "class_mean": scores.drop(columns=["onset", "offset"]).mean().to_dict(),
            }
    return summary


def _flatten(tree, prefix=""):
    """入れ子の dict を {"a/b/c": 値} に平らにする (pytest.approx の差分を読みやすくするため)."""
    flat = {}
    for key, value in tree.items():
        if isinstance(value, dict):
            flat.update(_flatten(value, f"{prefix}{key}/"))
        else:
            flat[f"{prefix}{key}"] = value
    return flat


@pytest.mark.parametrize("name", list(REPRO_VARIANTS))
def test_checkpoint_scores_unchanged(name, subset_config, tmp_path, monkeypatch):
    """ckpt を dev-test の一部に通した指標とスコアが, 保存した値から変わっていない."""
    config, subset = subset_config
    ckpt = _best_checkpoint(name)
    checkpoint = {"file": ckpt.name, "sha256": hashlib.sha256(ckpt.read_bytes()).hexdigest()[:16]}
    expected = json.loads(EXPECTED.read_text()) if EXPECTED.exists() else {}
    update = bool(os.environ.get("UPDATE_REPRO_CKPT_EXPECTED"))
    if not update:
        if name not in expected.get("configs", {}):
            pytest.skip("保存値が無い (UPDATE_REPRO_CKPT_EXPECTED=1 で作る)")
        if expected["configs"][name]["checkpoint"] != checkpoint:
            pytest.skip(
                f"保存値を記録した ckpt ({expected['configs'][name]['checkpoint']}) と違う: "
                f"{checkpoint} (UPDATE_REPRO_CKPT_EXPECTED=1 で取り直す)"
            )

    # ckpt に保存された設定は使われない (issue #14) ので, 構成のフラグを渡す
    args = [*REPRO_VARIANTS[name], "--test_from_checkpoint", str(ckpt)]
    # GPU で保存した ckpt は, そのままでは GPU の無い計算機で読めない (prepare_run の torch.load に
    # map_location が無い). テストは常に CPU で評価するので, CPU に読み替える
    load = torch.load
    monkeypatch.setattr(torch, "load", lambda *a, **kw: load(*a, **{"map_location": "cpu", **kw}))
    _run(copy.deepcopy(config), tmp_path, monkeypatch, args)
    actual = {
        "checkpoint": checkpoint,
        "metrics": _test_metrics(tmp_path),
        "scores": _score_summary(tmp_path),
    }

    if update:
        expected["subset"] = subset
        expected.setdefault("configs", {})[name] = actual
        EXPECTED.write_text(json.dumps(expected, indent=2, sort_keys=True) + "\n")
        return
    assert subset == expected["subset"], "サブセットが保存時と違う (データまたは選び方が変わった)"
    for key in ["metrics", "scores"]:
        assert _flatten(actual[key]) == pytest.approx(
            _flatten(expected["configs"][name][key]), abs=TOLERANCE, nan_ok=True
        ), key
