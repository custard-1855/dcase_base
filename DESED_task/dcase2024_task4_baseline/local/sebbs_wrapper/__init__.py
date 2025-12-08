"""SEBBs (Sound Event Bounding Boxes) wrapper layer.

This package provides a type-safe, well-documented wrapper around the SEBBs
submodule, improving code quality and maintainability without modifying the
original submodule code.

Main Components:
    - SEBBsPredictor: Predict sound event bounding boxes from scores
    - SEBBsTuner: Hyperparameter tuning utilities
    - SEBBsEvaluator: Performance evaluation metrics

Example Usage:
    Basic prediction:
        >>> from local.sebbs_wrapper import SEBBsPredictor
        >>> predictor = SEBBsPredictor(
        ...     step_filter_length=0.5, merge_threshold_abs=1.0, merge_threshold_rel=2.0
        ... )
        >>> sebbs = predictor.predict("/path/to/scores")

    Hyperparameter tuning:
        >>> from local.sebbs_wrapper import SEBBsTuner
        >>> tuner = SEBBsTuner()
        >>> predictor, metrics = tuner.tune_for_psds(
        ...     scores="/path/to/val/scores",
        ...     ground_truth="/path/to/val/gt.tsv",
        ...     audio_durations="/path/to/val/durations.tsv",
        ... )

    Evaluation:
        >>> from local.sebbs_wrapper import SEBBsEvaluator
        >>> evaluator = SEBBsEvaluator()
        >>> psds, class_psds = evaluator.evaluate_psds1(
        ...     scores=predictions, ground_truth=gt, audio_durations=durations
        ... )

Reference:
    J. Ebbers, F. Germain, G. Wichern and J. Le Roux,
    "Sound Event Bounding Boxes", Interspeech 2024.
    https://arxiv.org/abs/2406.04212
"""

from .evaluator import SEBBsEvaluator
from .predictor import SEBBsPredictor
from .tuner import SEBBsTuner
from .types import (
    SEBB,
    AudioDurations,
    Detection,
    DetectionList,
    EvaluationConfig,
    GroundTruth,
    PredictorConfig,
    Scores,
    SEBBList,
    TuningConfig,
)

__version__ = "0.1.0"

__all__ = [
    # Main classes
    "SEBBsPredictor",
    "SEBBsTuner",
    "SEBBsEvaluator",
    # Types
    "SEBB",
    "Detection",
    "SEBBList",
    "DetectionList",
    "Scores",
    "GroundTruth",
    "AudioDurations",
    "PredictorConfig",
    "TuningConfig",
    "EvaluationConfig",
]
