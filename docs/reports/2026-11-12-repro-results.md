# 再現学習の結果 (7 構成 + cSEBBs, 2026-11-12)

issue #6 の結果表. 論文 (ieice2025) の 7 構成を現在のコードで学習し直し, 論文値と比較した.

## 結論

- PSDS1 / PSDS2 は `cmt` を除く 6 構成で論文値から ±0.006 以内に収まった.
- `cmt` の PSDS1 は論文より 0.042 高い.
- mpAUC は 4 構成で ±0.02 を外れ, 構成間の順位も論文と変わった.
- 各構成 1 回の実行なので, 外れた差が seed による揺れか系統的な差かは判断できていない.
- ① mt の ckpt に cSEBBs を掛けた評価は完走した. ただし `psds_eval` 系の PSDS は後処理の前後で同じ値だった.

## 条件

| 項目 | 内容 |
|---|---|
| commit | `ddca340-dirty` (追跡ファイルの差分は `.gitignore` のみ) |
| 設定 | `confs/pretrained.yaml` (150 エポック, seed 42), wandb 無効 |
| 回数 | 各構成 1 回 |
| 実行 | `run_repro.sh` (`uv run poe repro`), 2026-11-12 01:37〜09:06, 終了コードはすべて 0 |
| 出力 | `exp/repro/{構成名}/version_0/test_metrics.json`, 実行記録は `exp/repro/queue.tsv` |
| 比較した指標 | teacher の `test/teacher/psds{1,2}/psds_eval` と `test/teacher/segment_mpauc/sed_scores_eval` |
| 判定 | 論文値から ±0.02 以内 |

構成ごとの追加フラグ (`./run_repro.sh --list`):

| 構成 | フラグ |
|---|---|
| ① mt | なし |
| ② cmt_scale_mixstyle | `--cmt --use_neg_sample --pos_neg_scale --mixstyle_type freqAttn --attn_type default` |
| ③ cmt | `--cmt` |
| mt_mixstyle | `--mixstyle_type freqAttn --attn_type default` |
| cmt_mixstyle | `--cmt --mixstyle_type freqAttn --attn_type default` |
| cmt_neg | `--cmt --use_neg_sample` |
| cmt_scale | `--cmt --use_neg_sample --pos_neg_scale` |

## 論文値との比較

各セルは「再現 / 論文 (差)」. **太字**は ±0.02 を外れたもの.

| 構成 | PSDS1 | PSDS2 | mpAUC | 所要 |
|---|---|---|---|---|
| ① mt | 0.374 / 0.369 (+0.005) | 0.587 / 0.581 (+0.006) | 0.725 / 0.720 (+0.005) | 3619 s |
| ② cmt_scale_mixstyle | 0.394 / 0.393 (+0.001) | 0.636 / 0.632 (+0.004) | **0.674 / 0.727 (−0.053)** | 4009 s |
| ③ cmt | **0.339 / 0.297 (+0.042)** | 0.614 / 0.60 (+0.014) | **0.720 / 0.689 (+0.031)** | 3809 s |
| mt_mixstyle | 0.379 / 0.382 (−0.003) | 0.607 / 0.607 (0.000) | 0.688 / 0.704 (−0.016) | 3831 s |
| cmt_mixstyle | 0.388 / 0.387 (+0.001) | 0.643 / 0.637 (+0.006) | **0.691 / 0.722 (−0.031)** | 4007 s |
| cmt_neg | 0.366 / 0.369 (−0.003) | 0.539 / 0.543 (−0.004) | 0.681 / 0.661 (+0.020) | 3807 s |
| cmt_scale | 0.369 / 0.368 (+0.001) | 0.623 / 0.621 (+0.002) | **0.679 / 0.699 (−0.020)** | 3835 s |

`cmt_neg` の mpAUC は +0.0197 で範囲内, `cmt_scale` の mpAUC は −0.0201 で僅かに範囲外.

### 所見

- **PSDS1 / PSDS2**: 最良の構成は論文と同じ (PSDS1 は `cmt_scale_mixstyle`, PSDS2 は `cmt_mixstyle`).
- **`cmt` の PSDS1**: CMT 単体での低下は mt 比 −0.035 で, 論文の −0.072 より小さい.
- **mpAUC**: 論文で最良の `cmt_scale_mixstyle` (0.727) が今回は最下位 (0.674), 最良は `mt` (0.725).

## cSEBBs (① mt の ckpt)

`uv run poe sebbs mt` で ① のベスト ckpt に cSEBBs を掛けて評価し, 完走した (約 7 分).
出力は `exp/repro/mt_sebbs/version_0/test_metrics.json`. 値は teacher.

| 指標 | mt | mt + cSEBBs | 差 |
|---|---|---|---|
| PSDS1 `sed_scores_eval` | 0.456 | 0.466 | +0.010 |
| PSDS2 `sed_scores_eval` | 0.717 | 0.725 | +0.008 |
| mpAUC `sed_scores_eval` | 0.725 | 0.727 | +0.002 |
| PSDS1 `psds_eval` | 0.374 | 0.374 | 0.000 |
| PSDS2 `psds_eval` | 0.587 | 0.587 | 0.000 |

`psds_eval` 系の値は後処理の前後で一致している. cSEBBs がこの系統の計算に反映されていない可能性があるが, 原因は調べていない.

## ckpt の再評価

`uv run poe repro-eval` で mt と cmt_scale_mixstyle のベスト ckpt を再評価した.
PSDS1 / PSDS2 / mpAUC は学習直後の値と小数 4 桁まで一致した (出力は `exp/repro_eval/{構成名}/`).

## 期待値ファイル

`DESED_task/dcase2024_task4_baseline/tests/repro_expected.json` に, 上の実測値を基準値 (`reference`), 論文値を `paper` として記録した.
許容幅 (`tolerance`) は `null` のままで, `uv run poe repro-check` のうち学習結果の検査は skip される.

## 未確認

- seed を変えた実行はしていない. mpAUC と `cmt` の PSDS1 の差が揺れの範囲かどうかは未判断.
- `training.deterministic` は `False`. 同じ seed で同じ値になるかは未確認.
- 論文の PSDS が `psds_eval` 系であることは値の一致からの推定で, 論文側のコードでは確認していない.
- cSEBBs で `psds_eval` 系の値が変わらない理由.
