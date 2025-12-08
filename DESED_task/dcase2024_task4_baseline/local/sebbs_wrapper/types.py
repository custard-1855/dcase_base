"""Type definitions for SEBBs wrapper layer.

This module provides strong typing for the SEBBs prediction and evaluation pipeline.
"""

from pathlib import Path
from typing import Dict, List, Protocol, Tuple, TypedDict, Union

import numpy as np
import pandas as pd
from numpy.typing import NDArray

# Sound Event Bounding Box types
SEBB = tuple[float, float, str, float]  # (onset, offset, class_label, confidence)
Detection = tuple[float, float, str]  # (onset, offset, class_label)
SEBBList = list[SEBB]
DetectionList = list[Detection]

# Score data types
ScoreDataFrame = pd.DataFrame
Scores = dict[str, ScoreDataFrame]
GroundTruth = dict[str, list[Detection]]
AudioDurations = dict[str, float]

# Path types
PathLike = Union[str, Path]
ScoresInput = Union[PathLike, Scores]
GroundTruthInput = Union[PathLike, GroundTruth]
AudioDurationsInput = Union[PathLike, AudioDurations]

# Parameter types
ClasswiseParam = Union[float, dict[str, float]]
OptionalClasswiseParam = Union[float, dict[str, float], None]


class PredictorConfig(TypedDict, total=False):
    """Configuration for CSEBBs predictor.

    Attributes:
        step_filter_length: Step filter length(s) for change point detection
        merge_threshold_abs: Absolute threshold(s) for segment merging
        merge_threshold_rel: Relative threshold(s) for segment merging
        detection_threshold: Detection/decision threshold(s)
        sound_classes: List of sound class labels

    """

    step_filter_length: ClasswiseParam
    merge_threshold_abs: ClasswiseParam
    merge_threshold_rel: ClasswiseParam
    detection_threshold: OptionalClasswiseParam
    sound_classes: list[str] | None


class TuningConfig(TypedDict, total=False):
    """Configuration for hyperparameter tuning.

    Attributes:
        step_filter_lengths: Candidate values for step_filter_length
        merge_thresholds_abs: Candidate values for merge_threshold_abs
        merge_thresholds_rel: Candidate values for merge_threshold_rel
        either_abs_or_rel_threshold: Use either abs or rel threshold (not both)

    """

    step_filter_lengths: tuple[float, ...]
    merge_thresholds_abs: tuple[float, ...]
    merge_thresholds_rel: tuple[float, ...]
    either_abs_or_rel_threshold: bool


class EvaluationConfig(TypedDict, total=False):
    """Configuration for PSDS/collar-based evaluation.

    Attributes:
        dtc_threshold: Detection tolerance criterion threshold (PSDS)
        gtc_threshold: Ground truth intersection criterion threshold (PSDS)
        cttc_threshold: Cross trigger tolerance criterion threshold (PSDS)
        alpha_ct: Cross trigger penalization weight (PSDS)
        alpha_st: Self-trigger penalization weight (PSDS)
        unit_of_time: Time unit for FPR computation (PSDS)
        max_efpr: Maximum false positives per time unit (PSDS)
        onset_collar: Onset collar in seconds (collar-based)
        offset_collar: Offset collar in seconds (collar-based)
        offset_collar_rate: Offset collar as rate of event length (collar-based)

    """

    # PSDS parameters
    dtc_threshold: float
    gtc_threshold: float
    cttc_threshold: float | None
    alpha_ct: float
    alpha_st: float
    unit_of_time: str
    max_efpr: float

    # Collar-based parameters
    onset_collar: float
    offset_collar: float
    offset_collar_rate: float


class PredictorProtocol(Protocol):
    """Protocol for predictor implementations."""

    def predict(
        self,
        scores: ScoresInput,
        audio_ids: list[str] | None = None,
        return_sed_scores: bool = False,
    ) -> dict[str, SEBBList] | Scores:
        """Predict SEBBs from scores."""
        ...

    def detect(
        self,
        scores: ScoresInput,
        detection_threshold: OptionalClasswiseParam = None,
        audio_ids: list[str] | None = None,
        return_sed_scores: bool = False,
    ) -> dict[str, DetectionList] | Scores:
        """Detect events from scores using threshold."""
        ...


class ChangeDetectionResult(TypedDict):
    """Result from change detection process.

    Attributes:
        seg_bounds: Segment boundaries
        mean_scores: Mean scores per segment
        min_scores: Minimum scores per segment
        max_scores: Maximum scores per segment

    """

    seg_bounds: NDArray[np.float64]
    mean_scores: NDArray[np.float64]
    min_scores: NDArray[np.float64]
    max_scores: NDArray[np.float64]


# Re-export common types
__all__ = [
    "SEBB",
    "AudioDurations",
    "AudioDurationsInput",
    "ChangeDetectionResult",
    "ClasswiseParam",
    "Detection",
    "DetectionList",
    "EvaluationConfig",
    "GroundTruth",
    "GroundTruthInput",
    "OptionalClasswiseParam",
    "PathLike",
    "PredictorConfig",
    "PredictorProtocol",
    "SEBBList",
    "ScoreDataFrame",
    "Scores",
    "ScoresInput",
    "TuningConfig",
]
