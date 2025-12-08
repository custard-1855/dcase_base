"""Hyperparameter tuning utilities for SEBBs.

This module provides type-safe wrappers for hyperparameter tuning and
cross-validation functionality.
"""

from collections.abc import Callable
from pathlib import Path
from typing import List, Optional, Tuple, Union

from sebbs.sebbs import csebbs

from .predictor import SEBBsPredictor
from .types import (
    AudioDurationsInput,
    GroundTruthInput,
    Scores,
    ScoresInput,
    TuningConfig,
)


class SEBBsTuner:
    """Hyperparameter tuner for SEBBs prediction.

    This class provides methods for grid search over hyperparameters and
    cross-validation, wrapping the functionality from sebbs.csebbs module
    with improved type safety and documentation.

    Example:
        >>> tuner = SEBBsTuner()
        >>> predictor, metrics = tuner.tune_for_psds(
        ...     scores="/path/to/val/scores",
        ...     ground_truth="/path/to/val/gt.tsv",
        ...     audio_durations="/path/to/val/durations.tsv",
        ... )

    """

    @staticmethod
    def tune(
        scores: ScoresInput,
        ground_truth: GroundTruthInput,
        audio_durations: AudioDurationsInput,
        *,
        step_filter_lengths: tuple[float, ...] = (0.32, 0.48, 0.64),
        merge_thresholds_abs: tuple[float, ...] = (0.15, 0.2, 0.3),
        merge_thresholds_rel: tuple[float, ...] = (1.5, 2.0, 3.0),
        selection_fn: Callable,
        folds: list[set] | None = None,
        either_abs_or_rel_threshold: bool = True,
        **selection_kwargs,
    ) -> tuple | list:
        """Perform grid search over hyperparameters.

        Args:
            scores: Path to or dict of SED posterior score DataFrames
            ground_truth: Path to or dict of ground truth event lists
            audio_durations: Path to or dict of audio durations
            step_filter_lengths: Candidate values for step_filter_length
            merge_thresholds_abs: Candidate values for merge_threshold_abs
            merge_thresholds_rel: Candidate values for merge_threshold_rel
            selection_fn: Function to select best parameters. Should be one of:
                - csebbs.select_best_psds
                - csebbs.select_best_cbf
                - csebbs.select_best_psds_and_cbf
                - Custom function with same signature
            folds: Optional list of audio_id sets for fold-wise selection.
                If None, selects best parameters on whole dataset.
            either_abs_or_rel_threshold: If True, either merge_thresholds_rel
                OR merge_thresholds_abs is used (not both), reducing grid size.
            **selection_kwargs: Arguments forwarded to selection_fn

        Returns:
            If folds is None:
                Return value from selection_fn (typically SEBBsPredictor and metrics)
            If folds is provided:
                List of return values from selection_fn, one per fold

        Example:
            >>> from sebbs.sebbs.csebbs import select_best_psds
            >>> predictor, psds_values = SEBBsTuner.tune(
            ...     scores=val_scores,
            ...     ground_truth=val_gt,
            ...     audio_durations=val_durations,
            ...     selection_fn=select_best_psds,
            ...     dtc_threshold=0.7,
            ...     gtc_threshold=0.7,
            ... )

        """
        return csebbs.tune(
            scores=scores,
            ground_truth=ground_truth,
            audio_durations=audio_durations,
            step_filter_lengths=step_filter_lengths,
            merge_thresholds_abs=merge_thresholds_abs,
            merge_thresholds_rel=merge_thresholds_rel,
            selection_fn=selection_fn,
            folds=folds,
            either_abs_or_rel_threshold=either_abs_or_rel_threshold,
            **selection_kwargs,
        )

    @staticmethod
    def tune_for_psds(
        scores: ScoresInput,
        ground_truth: GroundTruthInput,
        audio_durations: AudioDurationsInput,
        *,
        step_filter_lengths: tuple[float, ...] = (0.32, 0.48, 0.64),
        merge_thresholds_abs: tuple[float, ...] = (0.15, 0.2, 0.3),
        merge_thresholds_rel: tuple[float, ...] = (1.5, 2.0, 3.0),
        dtc_threshold: float = 0.7,
        gtc_threshold: float = 0.7,
        cttc_threshold: float | None = None,
        alpha_ct: float = 0.0,
        unit_of_time: str = "hour",
        max_efpr: float = 100.0,
        classwise: bool = True,
    ) -> tuple[SEBBsPredictor, dict]:
        """Tune hyperparameters to optimize PSDS (Polyphonic Sound Detection Score).

        Convenience method for tuning with PSDS as optimization criterion.

        Args:
            scores: Path to or dict of SED posterior score DataFrames
            ground_truth: Path to or dict of ground truth event lists
            audio_durations: Path to or dict of audio durations
            step_filter_lengths: Candidate values for step_filter_length
            merge_thresholds_abs: Candidate values for merge_threshold_abs
            merge_thresholds_rel: Candidate values for merge_threshold_rel
            dtc_threshold: Detection tolerance criterion threshold (PSDS)
            gtc_threshold: Ground truth intersection criterion threshold (PSDS)
            cttc_threshold: Cross trigger tolerance criterion threshold (PSDS)
            alpha_ct: Cross trigger penalization weight (PSDS)
            unit_of_time: Time unit for FPR computation ('hour', 'minute', etc.)
            max_efpr: Maximum false positives per time unit to consider
            classwise: Whether to use different parameters per sound class

        Returns:
            Tuple of (best_predictor, class_wise_psds_values)

        Example:
            >>> predictor, psds_values = SEBBsTuner.tune_for_psds(
            ...     scores=val_scores,
            ...     ground_truth=val_gt,
            ...     audio_durations=val_durations,
            ...     dtc_threshold=0.7,
            ...     gtc_threshold=0.7,
            ... )
            >>> print(f"PSDS: {psds_values}")

        """
        predictor, metrics = csebbs.tune(
            scores=scores,
            ground_truth=ground_truth,
            audio_durations=audio_durations,
            step_filter_lengths=step_filter_lengths,
            merge_thresholds_abs=merge_thresholds_abs,
            merge_thresholds_rel=merge_thresholds_rel,
            selection_fn=csebbs.select_best_psds,
            dtc_threshold=dtc_threshold,
            gtc_threshold=gtc_threshold,
            cttc_threshold=cttc_threshold,
            alpha_ct=alpha_ct,
            unit_of_time=unit_of_time,
            max_efpr=max_efpr,
            classwise=classwise,
        )

        # Wrap base predictor
        wrapper = SEBBsPredictor.__new__(SEBBsPredictor)
        wrapper._predictor = predictor
        return wrapper, metrics

    @staticmethod
    def tune_for_collar_based_f1(
        scores: ScoresInput,
        ground_truth: GroundTruthInput,
        audio_durations: AudioDurationsInput,
        *,
        step_filter_lengths: tuple[float, ...] = (0.32, 0.48, 0.64),
        merge_thresholds_abs: tuple[float, ...] = (0.15, 0.2, 0.3),
        merge_thresholds_rel: tuple[float, ...] = (1.5, 2.0, 3.0),
        onset_collar: float = 0.2,
        offset_collar: float = 0.2,
        offset_collar_rate: float = 0.2,
        classwise: bool = True,
    ) -> tuple[SEBBsPredictor, dict]:
        """Tune hyperparameters to optimize collar-based F1-score.

        Convenience method for tuning with collar-based F1 as optimization criterion.
        The returned predictor will have detection_threshold set.

        Args:
            scores: Path to or dict of SED posterior score DataFrames
            ground_truth: Path to or dict of ground truth event lists
            audio_durations: Path to or dict of audio durations
            step_filter_lengths: Candidate values for step_filter_length
            merge_thresholds_abs: Candidate values for merge_threshold_abs
            merge_thresholds_rel: Candidate values for merge_threshold_rel
            onset_collar: Onset collar in seconds
            offset_collar: Minimum offset collar in seconds
            offset_collar_rate: Offset collar as rate of event length
            classwise: Whether to use different parameters per sound class

        Returns:
            Tuple of (best_predictor, class_wise_f1_values)
            Note: best_predictor will have detection_threshold set

        Example:
            >>> predictor, f1_values = SEBBsTuner.tune_for_collar_based_f1(
            ...     scores=val_scores,
            ...     ground_truth=val_gt,
            ...     audio_durations=val_durations,
            ... )
            >>> print(f"F1 scores: {f1_values}")

        """
        predictor, metrics = csebbs.tune(
            scores=scores,
            ground_truth=ground_truth,
            audio_durations=audio_durations,
            step_filter_lengths=step_filter_lengths,
            merge_thresholds_abs=merge_thresholds_abs,
            merge_thresholds_rel=merge_thresholds_rel,
            selection_fn=csebbs.select_best_cbf,
            onset_collar=onset_collar,
            offset_collar=offset_collar,
            offset_collar_rate=offset_collar_rate,
            classwise=classwise,
        )

        # Wrap base predictor
        wrapper = SEBBsPredictor.__new__(SEBBsPredictor)
        wrapper._predictor = predictor
        return wrapper, metrics

    @staticmethod
    def cross_validation(
        scores: ScoresInput,
        ground_truth: GroundTruthInput,
        audio_durations: AudioDurationsInput,
        folds: list[set],
        *,
        step_filter_lengths: tuple[float, ...] = (0.32, 0.48, 0.64),
        merge_thresholds_abs: tuple[float, ...] = (0.15, 0.2, 0.3),
        merge_thresholds_rel: tuple[float, ...] = (1.5, 2.0, 3.0),
        either_abs_or_rel_threshold: bool = True,
        selection_fn: Callable,
        return_sed_scores: bool = True,
        **selection_kwargs,
    ) -> tuple[list, Scores, Scores]:
        """Perform leave-one-out cross-validation.

        Tunes hyperparameters on all folds except one and generates predictions
        for the left-out fold. Repeats for all folds.

        Args:
            scores: Path to or dict of SED posterior score DataFrames
            ground_truth: Path to or dict of ground truth event lists
            audio_durations: Path to or dict of audio durations
            folds: List of audio_id sets, one per fold
            step_filter_lengths: Candidate values for step_filter_length
            merge_thresholds_abs: Candidate values for merge_threshold_abs
            merge_thresholds_rel: Candidate values for merge_threshold_rel
            either_abs_or_rel_threshold: If True, use either abs or rel threshold
            selection_fn: Function to select best parameters (e.g., select_best_psds)
            return_sed_scores: Whether to return sed_scores_eval format outputs
            **selection_kwargs: Arguments forwarded to selection_fn

        Returns:
            Tuple of:
            - List of tuned predictors (one per fold or dict of predictors)
            - Dict of SEBBs predictions for entire dataset
            - Dict of detections for entire dataset (if detection_threshold selected)

        Example:
            >>> from sebbs.sebbs.csebbs import select_best_psds_and_cbf
            >>> folds = [set1, set2, set3, set4, set5]
            >>> predictors, sebbs, dets = SEBBsTuner.cross_validation(
            ...     scores=scores,
            ...     ground_truth=gt,
            ...     audio_durations=durations,
            ...     folds=folds,
            ...     selection_fn=select_best_psds_and_cbf,
            ... )

        """
        return csebbs.cross_validation(
            scores=scores,
            ground_truth=ground_truth,
            audio_durations=audio_durations,
            folds=folds,
            step_filter_lengths=step_filter_lengths,
            merge_thresholds_abs=merge_thresholds_abs,
            merge_thresholds_rel=merge_thresholds_rel,
            either_abs_or_rel_threshold=either_abs_or_rel_threshold,
            selection_fn=selection_fn,
            return_sed_scores=return_sed_scores,
            **selection_kwargs,
        )


__all__ = ["SEBBsTuner"]
