"""Evaluation utilities for SEBBs predictions.

This module provides type-safe wrappers for performance evaluation using
PSDS (Polyphonic Sound Detection Score), collar-based metrics, and
mpAUC/mAUC (mean partial/full AUROC) for MAESTRO dataset evaluation.
"""

from sed_scores_eval import collar_based, intersection_based, segment_based

from .types import (
    AudioDurationsInput,
    ClasswiseParam,
    GroundTruthInput,
    Scores,
    ScoresInput,
)


class SEBBsEvaluator:
    """Performance evaluator for SEBBs predictions.

    This class provides methods for evaluating SEBBs predictions using
    standard Sound Event Detection metrics:
    - PSDS (Polyphonic Sound Detection Score)
    - Collar-based F-score

    Example:
        >>> evaluator = SEBBsEvaluator()
        >>> psds, class_psds = evaluator.evaluate_psds(
        ...     scores=predictions, ground_truth=gt, audio_durations=durations
        ... )

    """

    @staticmethod
    def evaluate_psds(
        scores: ScoresInput,
        ground_truth: GroundTruthInput,
        audio_durations: AudioDurationsInput,
        *,
        dtc_threshold: float = 0.7,
        gtc_threshold: float = 0.7,
        cttc_threshold: float | None = None,
        alpha_ct: float = 0.0,
        alpha_st: float = 1.0,
        unit_of_time: str = "hour",
        max_efpr: float = 100.0,
    ) -> tuple[float, dict[str, float], dict, dict]:
        """Evaluate PSDS (Polyphonic Sound Detection Score).

        PSDS is a comprehensive metric for polyphonic sound event detection
        that considers detection tolerance, cross-triggers, and operates
        over a range of operating points.

        Args:
            scores: Path to or dict of SED posterior score DataFrames
            ground_truth: Path to or dict of ground truth event lists
            audio_durations: Path to or dict of audio durations
            dtc_threshold: Detection Tolerance Criterion threshold [0,1].
                Controls how much temporal overlap is required between
                detected and ground truth events.
            gtc_threshold: Ground Truth intersection Criterion threshold [0,1].
                Controls minimum coverage of ground truth events.
            cttc_threshold: Cross-Trigger Tolerance Criterion threshold [0,1].
                If None, cross-triggers not evaluated.
            alpha_ct: Cross-trigger penalization weight [0,1].
                Weight for penalizing cross-triggers (detecting wrong class).
            alpha_st: Self-trigger penalization weight [0,1].
                Weight for penalizing self-triggers (false positives).
            unit_of_time: Time unit for FPR computation ('hour', 'minute', 'second')
            max_efpr: Maximum effective FPR (false positives per time unit)
                to consider in PSD-ROC computation.

        Returns:
            Tuple of:
            - PSDS macro-average across classes
            - Dict mapping class names to class-wise PSDS values
            - Dict with detailed PSD-ROC data
            - Dict with effective intermediate statistics

        Reference:
            C. Bilen et al., "A Framework for the Robust Evaluation of Sound
            Event Detection", ICASSP 2020.

        Example:
            >>> psds, class_psds, psd_roc, stats = evaluator.evaluate_psds(
            ...     scores=predictions,
            ...     ground_truth=gt,
            ...     audio_durations=durations,
            ...     dtc_threshold=0.7,
            ...     gtc_threshold=0.7,
            ... )
            >>> print(f"PSDS: {psds:.3f}")
            >>> for cls, value in class_psds.items():
            ...     print(f"  {cls}: {value:.3f}")

        """
        return intersection_based.psds(
            scores=scores,
            ground_truth=ground_truth,
            audio_durations=audio_durations,
            dtc_threshold=dtc_threshold,
            gtc_threshold=gtc_threshold,
            cttc_threshold=cttc_threshold,
            alpha_ct=alpha_ct,
            alpha_st=alpha_st,
            unit_of_time=unit_of_time,
            max_efpr=max_efpr,
        )

    @staticmethod
    def evaluate_collar_based_fscore(
        scores: ScoresInput,
        ground_truth: GroundTruthInput,
        threshold: ClasswiseParam,
        *,
        onset_collar: float = 0.2,
        offset_collar: float = 0.2,
        offset_collar_rate: float = 0.2,
    ) -> tuple[dict[str, float], dict[str, float], dict[str, float]]:
        """Evaluate collar-based F-score.

        Collar-based evaluation allows tolerance windows around onset and
        offset times when matching detections to ground truth events.

        Args:
            scores: Path to or dict of SED posterior score DataFrames
            ground_truth: Path to or dict of ground truth event lists
            threshold: (Class-wise) detection threshold(s) to apply to scores
            onset_collar: Onset collar in seconds. Detections within this
                window before/after ground truth onset are considered correct.
            offset_collar: Minimum offset collar in seconds.
            offset_collar_rate: Offset collar as fraction of ground truth
                event duration. Actual collar is max(offset_collar,
                offset_collar_rate * event_duration).

        Returns:
            Tuple of dicts, each mapping class names to values:
            - F-scores (including 'macro_average')
            - Precision scores
            - Recall scores

        Example:
            >>> f_scores, precision, recall = evaluator.evaluate_collar_based_fscore(
            ...     scores=predictions,
            ...     ground_truth=gt,
            ...     threshold=0.5,
            ...     onset_collar=0.2,
            ...     offset_collar=0.2,
            ... )
            >>> print(f"Macro F1: {f_scores['macro_average']:.3f}")

        """
        return collar_based.fscore(
            scores=scores,
            ground_truth=ground_truth,
            threshold=threshold,
            onset_collar=onset_collar,
            offset_collar=offset_collar,
            offset_collar_rate=offset_collar_rate,
        )

    @staticmethod
    def find_best_collar_based_fscore(
        scores: ScoresInput,
        ground_truth: GroundTruthInput,
        *,
        onset_collar: float = 0.2,
        offset_collar: float = 0.2,
        offset_collar_rate: float = 0.2,
    ) -> tuple[
        dict[str, float],
        dict[str, float],
        dict[str, float],
        dict[str, float],
        dict[str, int],
    ]:
        """Find best collar-based F-score and optimal thresholds.

        Searches over score thresholds to find the operating point that
        maximizes collar-based F-score for each class.

        Args:
            scores: Path to or dict of SED posterior score DataFrames
            ground_truth: Path to or dict of ground truth event lists
            onset_collar: Onset collar in seconds
            offset_collar: Minimum offset collar in seconds
            offset_collar_rate: Offset collar as fraction of event duration

        Returns:
            Tuple of dicts, each mapping class names to values:
            - Best F-scores achieved (including 'macro_average')
            - Precision at best F-score operating points
            - Recall at best F-score operating points
            - Optimal detection thresholds
            - Number of TP, FP, FN counts at optimal thresholds

        Example:
            >>> f, p, r, thresholds, counts = evaluator.find_best_collar_based_fscore(
            ...     scores=predictions,
            ...     ground_truth=gt,
            ... )
            >>> print(f"Best macro F1: {f['macro_average']:.3f}")
            >>> print(f"Optimal thresholds: {thresholds}")

        """
        return collar_based.best_fscore(
            scores=scores,
            ground_truth=ground_truth,
            onset_collar=onset_collar,
            offset_collar=offset_collar,
            offset_collar_rate=offset_collar_rate,
        )

    @staticmethod
    def evaluate_psds1(
        scores: ScoresInput,
        ground_truth: GroundTruthInput,
        audio_durations: AudioDurationsInput,
    ) -> tuple[float, dict[str, float]]:
        """Evaluate PSDS1 with standard DCASE 2021-2024 Task 4 settings.

        Convenience method using PSDS1 configuration:
        - dtc_threshold = 0.7
        - gtc_threshold = 0.7
        - cttc_threshold = None
        - alpha_ct = 0.0
        - alpha_st = 1.0
        - max_efpr = 100.0

        Args:
            scores: Path to or dict of SED posterior score DataFrames
            ground_truth: Path to or dict of ground truth event lists
            audio_durations: Path to or dict of audio durations

        Returns:
            Tuple of (psds1_macro_average, class_wise_psds1)

        Example:
            >>> psds1, class_psds1 = evaluator.evaluate_psds1(
            ...     scores=predictions,
            ...     ground_truth=gt,
            ...     audio_durations=durations,
            ... )
            >>> print(f"PSDS1: {psds1:.3f}")

        """
        psds, single_class_psds, *_ = intersection_based.psds(
            scores=scores,
            ground_truth=ground_truth,
            audio_durations=audio_durations,
            dtc_threshold=0.7,
            gtc_threshold=0.7,
            cttc_threshold=None,
            alpha_ct=0.0,
            alpha_st=1.0,
            unit_of_time="hour",
            max_efpr=100.0,
        )
        return psds, single_class_psds

    @staticmethod
    def evaluate_psds2(
        scores: ScoresInput,
        ground_truth: GroundTruthInput,
        audio_durations: AudioDurationsInput,
    ) -> tuple[float, dict[str, float]]:
        """Evaluate PSDS2 with standard DCASE 2021-2024 Task 4 settings.

        Convenience method using PSDS2 configuration:
        - dtc_threshold = 0.1
        - gtc_threshold = 0.1
        - cttc_threshold = 0.3
        - alpha_ct = 0.5
        - alpha_st = 1.0
        - max_efpr = 100.0

        PSDS2 is more strict than PSDS1, penalizing cross-triggers and
        requiring tighter temporal matching.

        Args:
            scores: Path to or dict of SED posterior score DataFrames
            ground_truth: Path to or dict of ground truth event lists
            audio_durations: Path to or dict of audio durations

        Returns:
            Tuple of (psds2_macro_average, class_wise_psds2)

        Example:
            >>> psds2, class_psds2 = evaluator.evaluate_psds2(
            ...     scores=predictions,
            ...     ground_truth=gt,
            ...     audio_durations=durations,
            ... )
            >>> print(f"PSDS2: {psds2:.3f}")

        """
        psds, single_class_psds, *_ = intersection_based.psds(
            scores=scores,
            ground_truth=ground_truth,
            audio_durations=audio_durations,
            dtc_threshold=0.1,
            gtc_threshold=0.1,
            cttc_threshold=0.3,
            alpha_ct=0.5,
            alpha_st=1.0,
            unit_of_time="hour",
            max_efpr=100.0,
        )
        return psds, single_class_psds

    @staticmethod
    def evaluate_mpauc(
        scores: ScoresInput,
        ground_truth: GroundTruthInput,
        audio_durations: AudioDurationsInput,
        *,
        segment_length: float = 1.0,
        max_fpr: float = 0.1,
    ) -> tuple[float, dict[str, float]]:
        """Evaluate mpAUC (mean partial AUROC) for MAESTRO dataset evaluation.

        mpAUC is the mean of partial AUROC (Area Under ROC Curve) values,
        computed over a restricted FPR range [0, max_fpr]. This metric is
        useful for evaluating performance in the low false positive rate
        region, which is critical for MAESTRO urban and indoor sound events.

        Args:
            scores: Path to or dict of SED posterior score DataFrames
            ground_truth: Path to or dict of ground truth event lists
            audio_durations: Path to or dict of audio durations
            segment_length: Length of segments in seconds for evaluation (default 1.0)
            max_fpr: Maximum FPR for partial AUROC computation (default 0.1).
                Partial AUROC is computed in the range [0, max_fpr].

        Returns:
            Tuple of:
            - mpAUC macro-average across all sound classes
            - Dict mapping class names to class-wise mpAUC values

        Reference:
            McClain et al., "Leveraging Pre-trained Audio Embeddings for
            Multi-label Sound Event Classification", DCASE 2023.

        Example:
            >>> mpauc, class_mpauc = SEBBsEvaluator.evaluate_mpauc(
            ...     scores=maestro_scores,
            ...     ground_truth=maestro_gt,
            ...     audio_durations=maestro_durations,
            ...     segment_length=1.0,
            ...     max_fpr=0.1,
            ... )
            >>> print(f"mpAUC: {mpauc:.3f}")
            >>> for cls, value in class_mpauc.items():
            ...     print(f"  {cls}: {value:.3f}")

        """
        auroc_dict, _ = segment_based.auroc(
            scores=scores,
            ground_truth=ground_truth,
            audio_durations=audio_durations,
            segment_length=segment_length,
            max_fpr=max_fpr,
        )
        # segment_based.auroc returns dict with 'mean' and class-wise values
        mpauc_mean = auroc_dict['mean']
        # Remove 'mean' key to get class-wise dict
        class_mpauc = {k: v for k, v in auroc_dict.items() if k != 'mean'}
        return mpauc_mean, class_mpauc

    @staticmethod
    def evaluate_mauc(
        scores: ScoresInput,
        ground_truth: GroundTruthInput,
        audio_durations: AudioDurationsInput,
        *,
        segment_length: float = 1.0,
    ) -> tuple[float, dict[str, float]]:
        """Evaluate mAUC (mean AUROC) for sound event detection evaluation.

        mAUC is the mean of AUROC (Area Under ROC Curve) values across
        all sound classes. Unlike mpAUC, this computes the full AUROC
        over the entire FPR range [0, 1].

        Args:
            scores: Path to or dict of SED posterior score DataFrames
            ground_truth: Path to or dict of ground truth event lists
            audio_durations: Path to or dict of audio durations
            segment_length: Length of segments in seconds for evaluation (default 1.0)

        Returns:
            Tuple of:
            - mAUC macro-average across all sound classes
            - Dict mapping class names to class-wise mAUC (AUROC) values

        Example:
            >>> mauc, class_mauc = SEBBsEvaluator.evaluate_mauc(
            ...     scores=maestro_scores,
            ...     ground_truth=maestro_gt,
            ...     audio_durations=maestro_durations,
            ...     segment_length=1.0,
            ... )
            >>> print(f"mAUC: {mauc:.3f}")
            >>> for cls, value in class_mauc.items():
            ...     print(f"  {cls}: {value:.3f}")

        """
        auroc_dict, _ = segment_based.auroc(
            scores=scores,
            ground_truth=ground_truth,
            audio_durations=audio_durations,
            segment_length=segment_length,
        )
        # segment_based.auroc returns dict with 'mean' and class-wise values
        mauc_mean = auroc_dict['mean']
        # Remove 'mean' key to get class-wise dict
        class_mauc = {k: v for k, v in auroc_dict.items() if k != 'mean'}
        return mauc_mean, class_mauc


__all__ = ["SEBBsEvaluator"]
