"""cSEBBs 後処理: チューニング時の選択関数と, DESED / MAESTRO 両クラスへの適用."""

import numpy as np
import sed_scores_eval
from sebbs.sebbs.utils import sed_scores_from_sebbs

from .classes_dict import classes_labels_desed, classes_labels_maestro_real_eval


def select_best_auroc(
    csebbs: list,
    ground_truth: dict,
    audio_durations: dict,
    audio_ids=None,
    segment_length=1.0,
    **kwargs,
):
    """cSEBBsのハイパーパラメータチューニングでAUROCを最大化する選択関数.

    csebbs.tune()からのインターフェースに適合し、
    内部でsed_scores_eval.segment_based.aurocを使用する

    Args:
        csebbs: [(CSEBBsPredictor, sebbs_dict), ...]のリスト
        ground_truth: {audio_id: [(onset, offset, label), ...]}
        audio_durations: {audio_id: duration}
        audio_ids: 評価対象のaudio_idリスト（オプション）
        segment_length: セグメント長（秒）
        **kwargs: aurocに渡す追加引数

    Returns:
        (best_predictor, best_scores): 最良のpredictor、および各クラスのAUROC値

    """
    if audio_ids is not None:
        ground_truth = {aid: ground_truth[aid] for aid in audio_ids}
        audio_durations = {aid: audio_durations[aid] for aid in audio_ids}

    best_auroc = -1
    best_predictor = None
    best_auroc_values = None

    for predictor, sebbs_dict in csebbs:
        # SEBBsをsed_scores_eval形式に変換
        if audio_ids is not None:
            sebbs_dict = {aid: sebbs_dict[aid] for aid in audio_ids}

        scores = sed_scores_from_sebbs(
            sebbs_dict,
            sound_classes=predictor.sound_classes,
            audio_duration=audio_durations,
        )

        # AUROC計算
        auroc_values, _ = sed_scores_eval.segment_based.auroc(
            scores=scores,
            ground_truth=ground_truth,
            audio_durations=audio_durations,
            segment_length=segment_length,
            **kwargs,
        )

        # 平均AUROCで評価
        mean_auroc = np.mean(list(auroc_values.values()))

        if mean_auroc > best_auroc:
            best_auroc = mean_auroc
            best_predictor = predictor
            best_auroc_values = auroc_values

    return best_predictor, best_auroc_values


def get_sebbs(self, scores_all_classes, model_type="student"):
    """Apply cSEBBs post-processing to both DESED and MAESTRO classes.

    cSEBBs (change-point based Sound Event Bounding Boxes)は、
    フレームレベルの事後確率スコアから以下の処理によりイベント候補を生成:

    1. 変化点検出: ステップフィルタを用いてスコアの変化点を検出
    2. セグメント生成: 変化点を境界として候補セグメントを生成
    3. セグメントマージ: 類似スコアを持つ隣接セグメントを統合
    4. 信頼度計算: 各セグメントの平均スコアを信頼度として出力

    この関数では、DESEDとMAESTROの両クラスに対してcSEBBsを適用し、
    結果を統合して最終的なスコアDataFrameを生成する。

    Args:
        self: SEDTask4 instance (csebbs_predictor_desed, csebbs_predictor_maestroを保持)
        scores_all_classes: Dictionary of score DataFrames for all clips
                           各clipのDataFrameは全クラスのスコアを含む
                           形式: {clip_id: pd.DataFrame(onset, offset, class1, class2, ...)}
        model_type: 'student' or 'teacher'. モデルタイプを指定して適切なpredictorを選択

    Returns:
        sed_scores_postprocessed: Dictionary of post-processed score DataFrames
                                  cSEBBsにより生成されたイベント候補のスコア
                                  形式: {clip_id: pd.DataFrame(onset, offset, class1, class2, ...)}

    処理フロー:
        1. DESED/MAESTROクラスのスコアを個別に抽出
        2. 各データセットに最適化されたcSEBBs predictorで処理
        3. 生成されたイベント候補を統合
        4. sed_scores_eval形式のDataFrameに変換

    """
    # モデルタイプに応じたpredictorを選択
    if model_type == "teacher":
        csebbs_predictor_desed = self.csebbs_predictor_desed_teacher
        csebbs_predictor_maestro = self.csebbs_predictor_maestro_teacher
    else:
        csebbs_predictor_desed = self.csebbs_predictor_desed
        csebbs_predictor_maestro = self.csebbs_predictor_maestro

    # ステップ1: DESEDクラスに対するcSEBBs適用
    # DESEDは家庭内音（食器、掃除機、猫など）の10クラス
    desed_classes = list(classes_labels_desed.keys())
    keys_desed = ["onset", "offset"] + sorted(desed_classes)

    # 全クラスのスコアからDESEDクラスのみを抽出
    scores_desed_classes = {
        clip_id: scores_all_classes[clip_id][keys_desed] for clip_id in scores_all_classes.keys()
    }

    # cSEBBsでDESEDデータセットのスコアを後処理
    # 戻り値: {clip_id: [(onset, offset, class_name, confidence), ...]}
    csebbs_desed_events = csebbs_predictor_desed.predict(
        scores_desed_classes,
    )

    # ステップ2: MAESTROクラスに対するcSEBBs適用
    # MAESTROは都市音・屋内音（足音、会話、車など）の17クラス
    maestro_classes = sorted(classes_labels_maestro_real_eval)
    keys_maestro = ["onset", "offset"] + sorted(maestro_classes)

    # 全クラスのスコアからMAESTROクラスのみを抽出
    scores_maestro_classes = {
        clip_id: scores_all_classes[clip_id][keys_maestro] for clip_id in scores_all_classes.keys()
    }

    # cSEBBsでMAESTROデータセットのスコアを後処理
    # DESEDとは異なる音響特性に最適化されたパラメータを使用
    # 戻り値: {clip_id: [(onset, offset, class_name, confidence), ...]}
    csebbs_maestro_events = csebbs_predictor_maestro.predict(
        scores_maestro_classes,
    )

    # ステップ3: イベント候補の統合
    # DESEDとMAESTROの全クラスのリスト（重複なし、ソート済み）
    all_sound_classes = sorted(list(set(desed_classes + maestro_classes)))

    # 両データセットのclip IDを統合
    all_clip_ids = set(csebbs_desed_events.keys()) | set(csebbs_maestro_events.keys())

    # clip毎にDESEDとMAESTROのイベントを結合
    sebbs_all_events = {}
    for clip_id in all_clip_ids:
        # DESEDのイベントリストを取得（存在しない場合は空リスト）
        desed_events = csebbs_desed_events.get(clip_id, [])

        # MAESTROのイベントリストを取得（存在しない場合は空リスト）
        maestro_events = csebbs_maestro_events.get(clip_id, [])

        # イベントリストを結合してonset時間順にソート
        # 各イベント: (onset, offset, class_name, confidence)
        sebbs_all_events[clip_id] = sorted(
            desed_events + maestro_events,
            key=lambda x: x[0],  # onset時間でソート
        )

    # ステップ4: sed_scores_eval形式への変換
    # イベント候補リストからsed_scores_eval形式のDataFrameに変換
    # - 各イベントの時間範囲で該当クラスのスコアを設定
    # - イベントが存在しない時間・クラスは0.0で埋める
    # - 評価ツール(sed_scores_eval)で直接使用可能な形式
    sed_scores_postprocessed = sed_scores_from_sebbs(
        sebbs_all_events,
        sound_classes=all_sound_classes,
        fill_value=0.0,  # イベントが存在しない箇所は0.0
    )

    return sed_scores_postprocessed
