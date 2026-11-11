"""smoke test: 実データなしで「動くこと」だけを確認する.

    uv run pytest                 # 全部 (CPU で数分)
    uv run pytest -m "not slow"   # 学習を回さない軽いテストのみ (数秒)

学習を通すテストは合成データ (tests/conftest.py) を使い, CPU 上で
学習 2 epoch -> validation -> ベスト ckpt のロード -> dev-test 評価 まで実行する.
"""

import copy
import math
from pathlib import Path

import pytest
import torch
import yaml

RECIPE_DIR = Path(__file__).resolve().parents[1]
MIXSTYLE_TYPES = ["disabled", "only_mix", "freqAttn"]


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

    argv = ["--confs", str(conf_file), "--gpus", "0", "--fast_dev_run", *extra_args]
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


# 各 run_*.sh が使う引数の組み合わせを代表するもの
VARIANTS = {
    "baseline": [],
    "cmt_neg_scale": ["--cmt", "--use_neg_sample", "--pos_neg_scale"],
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
    if "--use_wandb" in extra_args:
        run_dirs = list((tmp_path / "experiments" / "train" / "smoke" / "all").rglob("manifest.json"))
        assert run_dirs, "experiments/train/{category}/{method}/{variant}/ に manifest が無い"
