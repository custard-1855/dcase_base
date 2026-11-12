"""再現学習の結果を期待値と比較する (既定の pytest には含めない).

    uv run pytest -m repro

読むのは学習・再評価が残した test_metrics.json だけで, 学習も評価も実行しない.
結果ファイルが無い構成は skip する. 期待値は tests/repro_expected.json.
"""

import json
from pathlib import Path

import pytest

RECIPE_DIR = Path(__file__).resolve().parents[1]
TRAIN_DIR = RECIPE_DIR / "exp" / "repro"  # run_repro.sh の出力
REEVAL_DIR = RECIPE_DIR / "exp" / "repro_eval"  # run_repro_eval.sh の出力

with open(Path(__file__).with_name("repro_expected.json")) as f:
    EXPECTED = json.load(f)
CONFIGS = list(EXPECTED["configs"])

pytestmark = pytest.mark.repro


def _latest_metrics(run_dir):
    """最新の version_N/test_metrics.json を読む (N が最大のもの). 無ければ skip."""
    files = sorted(
        run_dir.glob("version_*/test_metrics.json"),
        key=lambda p: int(p.parent.name.split("_")[1]),
    )
    if not files:
        pytest.skip(f"{run_dir.relative_to(RECIPE_DIR)} に test_metrics.json が無い")
    with open(files[-1]) as f:
        return json.load(f)


@pytest.mark.parametrize("name", CONFIGS)
def test_trained_metrics_within_tolerance(name):
    """学習結果が基準値の許容幅に収まる. 許容幅が未定 (null) の指標があれば skip."""
    metrics = _latest_metrics(TRAIN_DIR / name)
    expected = EXPECTED["configs"][name]
    if any(v["tolerance"] is None for v in expected.values()):
        pytest.skip("許容幅が未定 (複数 seed の計測後に repro_expected.json へ入れる)")
    for short, key in EXPECTED["metrics"].items():
        reference, tolerance = expected[short]["reference"], expected[short]["tolerance"]
        assert metrics[key] == pytest.approx(reference, abs=tolerance), short


@pytest.mark.parametrize("name", CONFIGS)
def test_reevaluation_matches_training(name):
    """再評価の結果が, 学習直後の評価と丸め誤差の範囲で一致する (学習の揺れは入らない)."""
    reevaluated = _latest_metrics(REEVAL_DIR / name)
    trained = _latest_metrics(TRAIN_DIR / name)
    keys = sorted(k for k in trained if k.startswith("test/"))
    assert keys, "比較する指標が無い"
    assert {k: reevaluated.get(k) for k in keys} == pytest.approx(
        {k: trained[k] for k in keys}, abs=1e-6, nan_ok=True
    )
