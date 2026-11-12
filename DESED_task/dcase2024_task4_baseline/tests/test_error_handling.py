"""黙って続行していた箇所が, 警告や例外として見えることを確認する.

対象: 欠損した音声の件数の報告, wandb 送信失敗時の警告, 損失が NaN のときの停止,
PSDS 計算失敗時の例外.
pytest の設定は UserWarning を無視するので, 警告は pytest.warns で明示的に捕まえる.
"""

import warnings

import numpy as np
import pandas as pd
import pytest
import soundfile as sf
import torch
from test_smoke import _run


def test_missing_audio_is_counted(base_config, tmp_path):
    """音声が無いサンプルは, Dataset の構築時に件数が警告され, 読むと None になる."""
    from desed_task.dataio.datasets import WeakSet
    from train_pretrained import get_encoder

    sf.write(tmp_path / "present.wav", np.zeros(16000, dtype=np.float32), 16000)
    entries = pd.DataFrame(
        {
            "filename": ["present.wav", "missing_0.wav", "missing_1.wav"],
            "event_labels": ["Speech", "Dog", "Cat"],
        }
    )

    with pytest.warns(UserWarning, match=r"音声ファイルが無いサンプルが 2 / 3 件"):
        dataset = WeakSet(str(tmp_path), entries, get_encoder(base_config))
    assert dataset[1] is None

    with warnings.catch_warnings():
        warnings.simplefilter("error")  # 欠損が無ければ警告しない
        WeakSet(str(tmp_path), entries[:1], get_encoder(base_config))


def test_wandb_log_failure_warns_once(monkeypatch):
    """送信 (wandb.log) が失敗しても例外は出ず, 初回だけ警告する."""
    import wandb
    from local.sed_trainer_pretrained import SEDTask4

    class Module:  # _maybe_wandb_log が触る属性だけを持つ代役
        hparams = {"wandb": {"use_wandb": True}}
        global_step = 0
        current_epoch = 0
        scheduler = None
        _wandb_log_failure_warned = False
        _warn_wandb_log_failure = SEDTask4._warn_wandb_log_failure

    def failing_log(*args, **kwargs):
        raise ConnectionError("offline")

    monkeypatch.setattr(wandb, "run", object())
    monkeypatch.setattr(wandb, "log", failing_log)
    module = Module()

    with pytest.warns(UserWarning, match="wandb への送信に失敗した.*offline"):
        SEDTask4._maybe_wandb_log(module, {"train/loss": torch.tensor(1.0)})
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # 2 回目以降は警告しない
        SEDTask4._maybe_wandb_log(module, {"train/loss": torch.tensor(1.0)})


@pytest.mark.slow
def test_training_stops_on_nan_loss(synthetic_config, tmp_path, monkeypatch):
    """損失が NaN になったら, 学習を続けずに例外で止まる."""

    def nan_loss(self, prediction, target):
        return prediction.sum() * float("nan")

    monkeypatch.setattr(torch.nn.BCELoss, "forward", nan_loss)
    with pytest.raises(RuntimeError, match="損失が NaN / inf になった"):
        _run(synthetic_config, tmp_path, monkeypatch)


def test_psds_failure_keeps_original_error():
    """計算に失敗した PSDS は, 元の例外を原因として付けて送出される."""
    import psds_eval
    from desed_task.evaluation.evaluation_measures import psds_results

    class FailingPSDS:
        def psds(self, **kwargs):
            raise psds_eval.psds.PSDSEvalError("no operating points")

    with pytest.raises(EnvironmentError, match="PSDS の計算に失敗した") as excinfo:
        psds_results(FailingPSDS())
    assert isinstance(excinfo.value.__cause__, psds_eval.psds.PSDSEvalError)
