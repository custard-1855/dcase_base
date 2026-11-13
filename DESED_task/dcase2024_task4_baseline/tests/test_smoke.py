"""smoke test: 実データなしで「動くこと」だけを確認する.

    uv run pytest                 # 全部 (CPU で数分)
    uv run poe test               # 学習を回さない軽いテストのみ (数秒)

学習を通すテストは合成データ (tests/conftest.py) を使い, CPU 上で
学習 2 epoch -> validation -> ベスト ckpt のロード -> dev-test 評価 まで実行する.
推論のテストは, そこで保存された ckpt を --test_from_checkpoint (dev-test の再評価) と
--eval_from_checkpoint (ラベルなし音声へのスコア出力) に渡す.

同じ学習実行から, 最終指標 (synthetic_expected.json) と, ステップごとの損失・モデルの出力・
学習後の重み (synthetic_expected_trace.json) を取り出して保存値と比較する.
挙動を変えないはずの変更 (整形, リファクタ) で計算が変わったことを検知するためのもの.
"""

import copy
import json
import math
import os
import subprocess
from pathlib import Path
from typing import NamedTuple

import pytest
import torch
import yaml

RECIPE_DIR = Path(__file__).resolve().parents[1]
MIXSTYLE_TYPES = ["disabled", "only_mix", "freqAttn"]
SYNTHETIC_EXPECTED = Path(__file__).with_name("synthetic_expected.json")
SYNTHETIC_TRACE = Path(__file__).with_name("synthetic_expected_trace.json")


def test_config_matches_code(base_config):
    """confs/pretrained.yaml が読めて, クラス数などがコードと整合している."""
    from local.classes_dict import classes_labels_desed, classes_labels_maestro_real
    from local.experiment_dir import ExperimentConfig
    from train_pretrained import get_encoder

    encoder = get_encoder(base_config)
    n_classes = len(classes_labels_desed) + len(classes_labels_maestro_real)
    assert len(encoder.labels) == n_classes == base_config["net"]["nclass"]
    assert len(base_config["net"]["median_filter"]) == n_classes
    assert len(base_config["training"]["batch_size"]) == 5
    ExperimentConfig(**base_config["experiment"])


@pytest.mark.parametrize("mixstyle_type", MIXSTYLE_TYPES)
def test_crnn_forward(base_config, mixstyle_type):
    """CRNN (+BEATs 埋め込み) の forward が全 MixStyle 設定で通り, 出力形状が正しい."""
    from desed_task.nnet.CRNN import CRNN
    from train_pretrained import get_encoder

    net_conf = {**base_config["net"], "mixstyle_type": mixstyle_type}
    model = CRNN(**net_conf)
    model.train()  # MixStyle は学習時のみ有効

    batch, n_frames = 4, 626  # 10 秒 @16kHz, hop 256 (center=True) -> 626 frames
    mels = torch.randn(batch, base_config["feats"]["n_mels"], n_frames)
    embeddings = torch.randn(batch, base_config["net"]["embedding_size"], 496)
    strong, weak = model(mels, embeddings=embeddings)

    n_classes = base_config["net"]["nclass"]
    assert strong.shape == (batch, n_classes, n_frames // base_config["data"]["net_subsample"])
    assert weak.shape == (batch, n_classes)
    assert torch.isfinite(strong).all() and torch.isfinite(weak).all()
    assert 0.0 <= strong.min() and strong.max() <= 1.0
    assert get_encoder(base_config).n_frames == strong.shape[-1]


def test_beats_backbone_builds():
    """BEATs のモデル定義がチェックポイントなしで構築でき, forward が通る (重みは乱数)."""
    from local.beats.BEATs import BEATs, BEATsConfig

    # BEATs_iter3+ と同じ入出力形状で, 層数だけ減らした構成
    cfg = BEATsConfig({"input_patch_size": 16, "encoder_layers": 1})
    model = BEATs(cfg).eval()
    with torch.inference_mode():
        features, _ = model.extract_features(torch.randn(1, 16000 * 10))
    assert features.shape == (1, 496, 768)  # extract_embeddings.py が前提とする形状


def _run(config, tmp_path, monkeypatch, extra_args=(), modules=None):
    """train_pretrained.py を CLI と同じ経路 (prepare_run -> single_run) で実行する.

    modules にリストを渡すと, 実行中に構築された SEDTask4 をそこへ足す
    (実行後に dev-test のスコアを読むため. 本体の挙動は変えない).
    """
    import train_pretrained

    if modules is not None:
        build = train_pretrained.SEDTask4
        monkeypatch.setattr(
            train_pretrained,
            "SEDTask4",
            lambda *a, **kw: modules.append(build(*a, **kw)) or modules[-1],
        )
    monkeypatch.chdir(tmp_path)  # experiments/ などの相対パス出力を tmp に閉じ込める
    monkeypatch.setenv("WANDB_MODE", "offline")
    conf_file = tmp_path / "smoke.yaml"
    with open(conf_file, "w") as f:
        yaml.safe_dump(copy.deepcopy(config), f)

    argv = ["--confs", str(conf_file), "--gpus", "0", *extra_args]
    if not any(a in extra_args for a in ("--test_from_checkpoint", "--eval_from_checkpoint")):
        argv.append("--fast_dev_run")
    configs, args, test_state_dict, evaluation = train_pretrained.prepare_run(argv)
    return train_pretrained.single_run(
        configs,
        str(tmp_path / "exp"),
        args.gpus,
        args.resume_from_checkpoint,
        test_state_dict,
        args.fast_dev_run,
        evaluation,
    )


def _test_metrics(tmp_path):
    """run ディレクトリに保存された最終指標 (test_metrics.json) を読む."""
    files = list(tmp_path.rglob("test_metrics.json"))
    assert len(files) == 1, "test_metrics.json が 1 つだけ保存されているはず"
    with open(files[0]) as f:
        return json.load(f)


def _logged_series(tmp_path):
    """TensorBoard に記録された train/ と val/ のスカラー (タグ -> 記録順の全値).

    --fast_dev_run では log_every_n_steps=1 なので, train/ は全ステップの値
    (各損失項, CMT の統計量, 整合性損失の重み, 学習率), val/ はエポックごとの値になる.
    test/ は test_metrics.json と同じ値なので拾わない.
    """
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    series = {}
    for path in sorted(tmp_path.rglob("events.out.tfevents.*")):
        events = EventAccumulator(str(path), size_guidance={"scalars": 0})  # 0 = 間引かない
        events.Reload()
        for tag in events.Tags()["scalars"]:
            if tag.startswith(("train/", "val/")) and not tag.startswith("train/step/"):
                series.setdefault(tag, []).extend(e.value for e in events.Scalars(tag))
    assert "train/student/tot_loss" in series, "学習中の損失が TensorBoard に記録されていない"
    return series


def _position_weighted_mean(values):
    """並び (クリップ, フレーム, クラス, 層) が入れ替わると変わる要約値."""
    import numpy as np

    weights = (np.arange(values.size) % 997 + 1) / 997
    return float(np.mean(values * weights))


def _summarize_scores(scores):
    """クリップ名 -> スコア表 (onset, offset, クラスごとの列) を少数の値に要約する.

    生の配列は大きすぎて保存値にできないので, 形状, クラス列の統計量, 固定位置の 3 点,
    onset / offset も含めた位置重み付き平均を残す.
    """
    import numpy as np

    assert scores, "dev-test のスコアが空"
    table = np.concatenate([scores[clip].to_numpy(dtype=np.float64) for clip in sorted(scores)])
    values = table[:, 2:].ravel()
    return {
        "shape": [len(scores), *table[:, 2:].shape],
        "mean": float(values.mean()),
        "std": float(values.std()),
        "min": float(values.min()),
        "max": float(values.max()),
        "samples": [float(values[i * values.size // 3]) for i in range(3)],
        "weighted": _position_weighted_mean(table.ravel()),
    }


def _summarize_weights(state_dict, prefix):
    """学習後の重み (prefix で始まるキーをキー順に連結) の要約."""
    flat = torch.cat(
        [v.double().flatten() for k, v in sorted(state_dict.items()) if k.startswith(prefix)]
    )
    return {
        "size": flat.numel(),
        "abs_mean": float(flat.abs().mean()),
        "weighted": _position_weighted_mean(flat.numpy()),
    }


def _trace(tmp_path, module):
    """学習実行 1 回分の, 保存値と比較する途中経過 (損失, 出力, 重み).

    losses:  ステップごとの train/*, エポックごとの val/* (TensorBoard の記録).
    outputs: ベスト ckpt での dev-test 全クリップのスコア. student / teacher それぞれの
             後処理前 (unprocessed) と後処理後 (postprocessed: median filter か cSEBBs).
    weights: 学習終了時 (last.ckpt) の student / teacher の重み.
    """
    checkpoints = list(tmp_path.rglob("last.ckpt"))
    assert len(checkpoints) == 1, "last.ckpt が 1 つだけ保存されているはず"
    state_dict = torch.load(checkpoints[0], weights_only=False)["state_dict"]
    return {
        "losses": _logged_series(tmp_path),
        "outputs": {
            "student/unprocessed": _summarize_scores(
                module.test_buffer_sed_scores_eval_unprocessed_student
            ),
            "student/postprocessed": _summarize_scores(module.test_buffer_sed_scores_eval_student),
            "teacher/unprocessed": _summarize_scores(
                module.test_buffer_sed_scores_eval_unprocessed_teacher
            ),
            "teacher/postprocessed": _summarize_scores(module.test_buffer_sed_scores_eval_teacher),
        },
        "weights": {
            "sed_student": _summarize_weights(state_dict, "sed_student."),
            "sed_teacher": _summarize_weights(state_dict, "sed_teacher."),
        },
    }


def _flatten(tree, prefix=""):
    """入れ子の dict / list を {"a b[0]": 値} に平らにする (pytest.approx が入れ子を扱えないため)."""
    if isinstance(tree, dict):
        return {k2: v2 for k, v in tree.items() for k2, v2 in _flatten(v, f"{prefix} {k}").items()}
    if isinstance(tree, list):
        return {
            k2: v2 for i, v in enumerate(tree) for k2, v2 in _flatten(v, f"{prefix}[{i}]").items()
        }
    return {prefix.strip(): tree}


# 途中経過には 1e-10 程度の値 (train/cmt/pos_neg_ratio_*) や 1e-5 程度の値 (train/lr) があるので,
# 絶対誤差の許容幅は実質 0 にして相対誤差 1e-6 だけで比べる.
TRACE_ABS_TOL = 1e-15


def _dumps_compact(tree, depth=3, level=0):
    """上から depth 階層だけ改行し, その下 (系列や要約) は 1 行に収めた JSON (diff を読むため)."""
    if depth == 0 or not isinstance(tree, dict):
        return json.dumps(tree, sort_keys=True)
    pad = "  " * (level + 1)
    items = [
        f"{pad}{json.dumps(k)}: {_dumps_compact(tree[k], depth - 1, level + 1)}"
        for k in sorted(tree)
    ]
    return "{\n" + ",\n".join(items) + "\n" + "  " * level + "}"


def _dumps_indented(tree):
    """キーごとに改行した JSON (synthetic_expected.json の形式)."""
    return json.dumps(tree, indent=2, sort_keys=True)


def _assert_unchanged(path, name, actual, dumps=_dumps_indented, abs_tol=1e-9):
    """構成 name の値 actual が path の保存値と一致する.

    UPDATE_SYNTHETIC_EXPECTED=1 のときは, 比較せずに保存し直す.
    """
    expected = json.loads(path.read_text()) if path.exists() else {}

    if os.environ.get("UPDATE_SYNTHETIC_EXPECTED"):
        expected[name] = actual
        path.write_text(dumps(expected) + "\n")
        return
    if name not in expected:
        pytest.skip(f"{path.name} に {name} の保存値が無い (UPDATE_SYNTHETIC_EXPECTED=1 で作る)")
    # pytest.approx の失敗メッセージは dict を並び順で突き合わせるので, キー順に揃えてから比べる
    actual, saved = (dict(sorted(_flatten(tree).items())) for tree in (actual, expected[name]))
    assert actual == pytest.approx(saved, rel=1e-6, abs=abs_tol, nan_ok=True)


class Run(NamedTuple):
    """合成データで 1 回学習〜評価した結果."""

    name: str  # 構成名
    tmp_path: Path  # 出力先
    objective: float  # single_run の戻り値 (目的関数値)
    trace: dict  # 途中経過の要約 (_trace)


def _traced_run(name, extra_args, synthetic_config, tmp_path_factory):
    tmp_path = tmp_path_factory.mktemp(f"run_{name}")
    modules = []
    with pytest.MonkeyPatch.context() as mp:
        objective = _run(synthetic_config, tmp_path, mp, extra_args, modules)
    assert len(modules) == 1, "SEDTask4 は 1 回だけ構築されるはず"
    return Run(name, tmp_path, objective, _trace(tmp_path, modules[0]))


def _repro_variants():
    """再現対象の構成 (構成名 -> フラグ). 正本の run_repro.sh --list から読む."""
    out = subprocess.run(
        [str(RECIPE_DIR / "run_repro.sh"), "--list"], capture_output=True, text=True, check=True
    ).stdout
    return {line.split()[0]: line.split()[1:] for line in out.splitlines() if line.strip()}


REPRO_VARIANTS = _repro_variants()

# 再現対象の構成 (REPRO_VARIANTS) に無い引数の組み合わせ
VARIANTS = {
    "mixstyle_freqattn": ["--mixstyle_type", "freqAttn", "--attn_type", "default"],
    "mixstyle_only_mix": ["--mixstyle_type", "only_mix", "--attn_type", "default"],
    "sebbs": ["--sebbs"],
    "all_with_wandb_offline": [
        "--use_wandb",
        "--category", "smoke",
        "--method", "all",
        "--variant", "cmt_neg_scale_mixstyle_sebbs",
        "--cmt",
        "--use_neg_sample",
        "--pos_neg_scale",
        "--mixstyle_type", "freqAttn",
        "--attn_type", "default",
        "--sebbs",
    ],
}


@pytest.fixture(scope="module", params=list(VARIANTS))
def variant_run(request, synthetic_config, tmp_path_factory):
    """VARIANTS の構成を合成データで 1 回ずつ学習〜評価した結果."""
    return _traced_run(request.param, VARIANTS[request.param], synthetic_config, tmp_path_factory)


@pytest.mark.slow
def test_train_and_test_end_to_end(variant_run):
    """学習 -> validation -> ckpt 保存 -> dev-test 評価が例外なく完走する."""
    name, tmp_path, objective, _ = variant_run
    extra_args = VARIANTS[name]

    assert objective is not None and math.isfinite(objective)
    assert list(tmp_path.rglob("*.ckpt")), "checkpoint が保存されていない"
    metrics = _test_metrics(tmp_path)
    psds1 = metrics["test/teacher/psds1/sed_scores_eval"]
    assert objective == pytest.approx(psds1 + metrics["test/teacher/segment_mpauc/sed_scores_eval"])
    if "--use_wandb" in extra_args:
        run_dirs = list((tmp_path / "experiments" / "train" / "smoke" / "all").rglob("manifest.json"))
        assert run_dirs, "experiments/train/{category}/{method}/{variant}/ に manifest が無い"


@pytest.fixture(scope="module", params=list(REPRO_VARIANTS))
def repro_run(request, synthetic_config, tmp_path_factory):
    """再現対象の構成を合成データで 1 回ずつ学習〜評価した結果."""
    return _traced_run(
        request.param, REPRO_VARIANTS[request.param], synthetic_config, tmp_path_factory
    )


@pytest.mark.slow
def test_repro_variant_end_to_end(repro_run):
    """再現対象の全構成 (run_repro.sh) で, 学習 -> validation -> ckpt 保存 -> dev-test 評価が完走する."""
    _, tmp_path, objective, _ = repro_run

    assert objective is not None and math.isfinite(objective)
    assert list(tmp_path.rglob("*.ckpt")), "checkpoint が保存されていない"
    assert "test/teacher/psds1/sed_scores_eval" in _test_metrics(tmp_path)


@pytest.mark.slow
def test_synthetic_metrics_unchanged(repro_run):
    """合成データ・seed 固定・CPU での最終指標が, 保存した値から変わっていない.

    整形やリファクタで挙動が変わったことを検知する. 保存値は tests/synthetic_expected.json.
    意図して挙動を変えたとき, torch などの版を上げたときは, 次で取り直す
    (synthetic_expected_trace.json も同時に取り直される):

        UPDATE_SYNTHETIC_EXPECTED=1 uv run pytest -k "synthetic or variant"
    """
    _assert_unchanged(SYNTHETIC_EXPECTED, repro_run.name, _test_metrics(repro_run.tmp_path))


@pytest.mark.slow
def test_synthetic_trace_unchanged(repro_run):
    """合成データ・seed 固定・CPU での損失・出力・重みが, 保存した値から変わっていない.

    最終指標は合成データではほとんど 0 で鈍いので, 途中経過 (_trace) も比較する.
    保存値は tests/synthetic_expected_trace.json.
    取り直し方は test_synthetic_metrics_unchanged と同じ.
    """
    _assert_unchanged(
        SYNTHETIC_TRACE, repro_run.name, repro_run.trace, _dumps_compact, TRACE_ABS_TOL
    )


@pytest.mark.slow
def test_variant_numbers_unchanged(variant_run):
    """VARIANTS の構成でも, 最終指標と途中経過が保存した値から変わっていない."""
    _assert_unchanged(
        SYNTHETIC_TRACE, variant_run.name, variant_run.trace, _dumps_compact, TRACE_ABS_TOL
    )
    _assert_unchanged(SYNTHETIC_EXPECTED, variant_run.name, _test_metrics(variant_run.tmp_path))


@pytest.fixture(scope="module")
def trained_checkpoint(synthetic_config, tmp_path_factory):
    """推論テスト用に, 合成データで 1 回だけ学習して得た ckpt (cSEBBs 有効)."""
    tmp_path = tmp_path_factory.mktemp("train_for_inference")
    with pytest.MonkeyPatch.context() as mp:
        _run(synthetic_config, tmp_path, mp, ["--sebbs"])
    checkpoints = sorted(tmp_path.rglob("epoch=*.ckpt"))
    assert checkpoints, "checkpoint が保存されていない"
    return checkpoints[-1]


@pytest.mark.slow
@pytest.mark.parametrize("sebbs", [False, True], ids=["median_filter", "sebbs"])
def test_devtest_from_checkpoint(synthetic_config, trained_checkpoint, tmp_path, monkeypatch, sebbs):
    """--test_from_checkpoint: 学習なしで ckpt をロードし, dev-test の評価指標が出る."""
    extra_args = ["--test_from_checkpoint", str(trained_checkpoint), "--mode", "test"]
    objective = _run(synthetic_config, tmp_path, monkeypatch, extra_args + ["--sebbs"] * sebbs)

    assert objective is not None and math.isfinite(objective)
    assert not list(tmp_path.rglob("*.ckpt")), "テストのみの実行で ckpt が作られている"
    assert "test/teacher/psds1/sed_scores_eval" in _test_metrics(tmp_path)
    assert list(tmp_path.rglob("metrics_test/**/*.tsv")), "予測スコアが保存されていない"


@pytest.mark.slow
def test_inference_on_unlabeled_audio(synthetic_config, trained_checkpoint, tmp_path, monkeypatch):
    """--eval_from_checkpoint: ラベルなし音声に対して全クリップ分のスコアを書き出す."""
    import pandas as pd
    from local.classes_dict import classes_labels_desed, classes_labels_maestro_real

    _run(synthetic_config, tmp_path, monkeypatch, ["--eval_from_checkpoint", str(trained_checkpoint)])
    assert not list(tmp_path.rglob("test_metrics.json")), "指標の無い推論で指標が保存されている"

    eval_clips = {p.stem for p in Path(synthetic_config["data"]["eval_folder"]).glob("*.wav")}
    labels = list(classes_labels_desed) + list(classes_labels_maestro_real)
    for model in ["student", "teacher"]:
        for kind in ["unprocessed", "postprocessed"]:
            score_dirs = list(tmp_path.rglob(f"metrics_test/{model}_scores/{kind}"))
            assert len(score_dirs) == 1, f"{model}/{kind} のスコア出力先が見つからない"
            files = list(score_dirs[0].glob("*.tsv"))
            assert {f.stem for f in files} == eval_clips
            scores = pd.read_csv(files[0], sep="\t")
            assert list(scores.columns) == ["onset", "offset", *labels]
            assert len(scores) == 156  # 10 秒 / (hop 256 * subsample 4 / 16kHz)
            assert scores[labels].to_numpy().min() >= 0.0
            assert scores[labels].to_numpy().max() <= 1.0
