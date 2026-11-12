"""smoke test: 実データなしで「動くこと」だけを確認する.

    uv run pytest                 # 全部 (CPU で数分)
    uv run poe test               # 学習を回さない軽いテストのみ (数秒)

学習を通すテストは合成データ (tests/conftest.py) を使い, CPU 上で
学習 2 epoch -> validation -> ベスト ckpt のロード -> dev-test 評価 まで実行する.
推論のテストは, そこで保存された ckpt を --test_from_checkpoint (dev-test の再評価) と
--eval_from_checkpoint (ラベルなし音声へのスコア出力) に渡す.
"""

import copy
import json
import math
import os
import subprocess
from pathlib import Path

import pytest
import torch
import yaml

RECIPE_DIR = Path(__file__).resolve().parents[1]
MIXSTYLE_TYPES = ["disabled", "only_mix", "freqAttn"]
SYNTHETIC_EXPECTED = Path(__file__).with_name("synthetic_expected.json")


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


def _run(config, tmp_path, monkeypatch, extra_args=()):
    """train_pretrained.py を CLI と同じ経路 (prepare_run -> single_run) で実行する."""
    import train_pretrained

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


@pytest.mark.slow
@pytest.mark.parametrize("extra_args", VARIANTS.values(), ids=VARIANTS.keys())
def test_train_and_test_end_to_end(synthetic_config, tmp_path, monkeypatch, extra_args):
    """学習 -> validation -> ckpt 保存 -> dev-test 評価が例外なく完走する."""
    objective = _run(synthetic_config, tmp_path, monkeypatch, extra_args)

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
    """再現対象の構成を合成データで 1 回ずつ学習〜評価した結果 (構成名, 出力先, 目的関数値)."""
    tmp_path = tmp_path_factory.mktemp(f"repro_{request.param}")
    with pytest.MonkeyPatch.context() as mp:
        objective = _run(synthetic_config, tmp_path, mp, REPRO_VARIANTS[request.param])
    return request.param, tmp_path, objective


@pytest.mark.slow
def test_repro_variant_end_to_end(repro_run):
    """再現対象の全構成 (run_repro.sh) で, 学習 -> validation -> ckpt 保存 -> dev-test 評価が完走する."""
    _, tmp_path, objective = repro_run

    assert objective is not None and math.isfinite(objective)
    assert list(tmp_path.rglob("*.ckpt")), "checkpoint が保存されていない"
    assert "test/teacher/psds1/sed_scores_eval" in _test_metrics(tmp_path)


@pytest.mark.slow
def test_synthetic_metrics_unchanged(repro_run):
    """合成データ・seed 固定・CPU での最終指標が, 保存した値から変わっていない.

    整形やリファクタで挙動が変わったことを検知する. 保存値は tests/synthetic_expected.json.
    意図して挙動を変えたとき, torch などの版を上げたときは, 次で取り直す:

        UPDATE_SYNTHETIC_EXPECTED=1 uv run pytest -k test_synthetic_metrics_unchanged
    """
    name, tmp_path, _ = repro_run
    metrics = _test_metrics(tmp_path)
    expected = json.loads(SYNTHETIC_EXPECTED.read_text()) if SYNTHETIC_EXPECTED.exists() else {}

    if os.environ.get("UPDATE_SYNTHETIC_EXPECTED"):
        expected[name] = metrics
        SYNTHETIC_EXPECTED.write_text(json.dumps(expected, indent=2, sort_keys=True) + "\n")
        return
    if name not in expected:
        pytest.skip("保存値が無い (UPDATE_SYNTHETIC_EXPECTED=1 で作る)")
    assert metrics == pytest.approx(expected[name], rel=1e-6, abs=1e-9, nan_ok=True)


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
