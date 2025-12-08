"""Predictor wrapper for SEBBs (Sound Event Bounding Boxes).

This module provides a type-safe, well-documented wrapper around the CSEBBsPredictor
from the sebbs submodule, improving code quality without modifying the submodule.
"""

from sebbs.sebbs.csebbs import CSEBBsPredictor as _CSEBBsPredictorBase

from .types import (
    ClasswiseParam,
    DetectionList,
    OptionalClasswiseParam,
    PredictorConfig,
    Scores,
    ScoresInput,
    SEBBList,
)


class SEBBsPredictor:
    """Wrapper for change-point based Sound Event Bounding Boxes prediction.

    This class provides a type-safe interface for predicting Sound Event Bounding
    Boxes (SEBBs) from frame-level posterior scores using change-point detection
    and segment merging algorithms.

    SEBBs are one-dimensional bounding boxes defined by:
    - Event onset time
    - Event offset time
    - Sound class label
    - Confidence score

    The final sound event detection can be derived by event-level thresholding
    of the SEBBs' confidences.

    Reference:
        J. Ebbers, F. Germain, G. Wichern and J. Le Roux,
        "Sound Event Bounding Boxes", Interspeech 2024.
        https://arxiv.org/abs/2406.04212

    Attributes:
        step_filter_length: Step filter length(s) for change point detection
        merge_threshold_abs: Absolute threshold(s) for segment merging
        merge_threshold_rel: Relative threshold(s) for segment merging
        detection_threshold: Detection/decision threshold(s)
        sound_classes: List of sound class labels

    """

    def __init__(
        self,
        step_filter_length: ClasswiseParam = 0.5,
        merge_threshold_abs: ClasswiseParam = 1.0,
        merge_threshold_rel: ClasswiseParam = 2.0,
        detection_threshold: OptionalClasswiseParam = None,
        sound_classes: list[str] | None = None,
    ) -> None:
        """Initialize SEBBs predictor.

        Args:
            step_filter_length: (Class-wise) step filter length(s) for change
                point detection. Can be a single float or dict mapping class
                names to floats.
            merge_threshold_abs: (Class-wise) absolute threshold(s) for segment
                merging. If the absolute difference between a min value in a gap
                segment and the max value in neighbouring event candidates is
                smaller than this threshold, segments may be merged.
            merge_threshold_rel: (Class-wise) relative threshold(s) for segment
                merging. If the relative difference (max_value/min_value) is
                smaller than this threshold, segments may be merged.
            detection_threshold: (Class-wise) detection/decision threshold(s).
                Used in detect() method to filter SEBBs by confidence.
            sound_classes: List of sound class labels. If None, inferred from
                input scores.

        """
        self._predictor = _CSEBBsPredictorBase(
            step_filter_length=step_filter_length,
            merge_threshold_abs=merge_threshold_abs,
            merge_threshold_rel=merge_threshold_rel,
            detection_threshold=detection_threshold,
            sound_classes=sound_classes,
        )

    @property
    def step_filter_length(self) -> ClasswiseParam:
        """Get step filter length parameter."""
        return self._predictor.step_filter_length

    @property
    def merge_threshold_abs(self) -> ClasswiseParam:
        """Get absolute merge threshold parameter."""
        return self._predictor.merge_threshold_abs

    @property
    def merge_threshold_rel(self) -> ClasswiseParam:
        """Get relative merge threshold parameter."""
        return self._predictor.merge_threshold_rel

    @property
    def detection_threshold(self) -> OptionalClasswiseParam:
        """Get detection threshold parameter."""
        return self._predictor.detection_threshold

    @property
    def sound_classes(self) -> list[str] | None:
        """Get sound class labels."""
        return self._predictor.sound_classes

    @classmethod
    def from_config(cls, config: PredictorConfig) -> "SEBBsPredictor":
        """Create predictor from configuration dictionary.

        Args:
            config: Configuration dictionary with predictor parameters

        Returns:
            Configured SEBBsPredictor instance

        Example:
            >>> config = {
            ...     "step_filter_length": 0.5,
            ...     "merge_threshold_abs": 1.0,
            ...     "merge_threshold_rel": 2.0,
            ... }
            >>> predictor = SEBBsPredictor.from_config(config)

        """
        return cls(**config)

    def predict(
        self,
        scores: ScoresInput,
        audio_ids: list[str] | None = None,
        return_sed_scores: bool = False,
    ) -> dict[str, SEBBList] | Scores:
        """Predict Sound Event Bounding Boxes from posterior scores.

        Args:
            scores: Either a path to directory containing SED posterior score
                DataFrames, or a dict of DataFrames keyed by audio_id. Format
                follows sed_scores_eval conventions.
            audio_ids: Optional list of audio IDs to restrict prediction to.
                If None, predict for all audio IDs in scores.
            return_sed_scores: If True, return outputs as SED scores DataFrames
                for direct use with sed_scores_eval evaluation. If False, return
                raw SEBBs lists.

        Returns:
            If return_sed_scores=False:
                Dict mapping audio_id to list of SEBBs. Each SEBB is a tuple:
                (onset, offset, class_label, confidence)

            If return_sed_scores=True:
                Dict mapping audio_id to pandas DataFrame in sed_scores_eval
                format with constant class scores over SEBB extents.

        Example:
            >>> predictor = SEBBsPredictor(step_filter_length=0.5)
            >>> sebbs = predictor.predict("/path/to/scores")
            >>> # sebbs['audio_001'] = [(1.5, 3.2, 'Speech', 0.87), ...]

        """
        return self._predictor.predict(
            scores=scores,
            audio_ids=audio_ids,
            return_sed_scores=return_sed_scores,
        )

    def detect(
        self,
        scores: ScoresInput,
        detection_threshold: OptionalClasswiseParam = None,
        audio_ids: list[str] | None = None,
        return_sed_scores: bool = False,
    ) -> dict[str, DetectionList] | Scores:
        """Predict SEBBs and derive detections via threshold.

        This method first predicts SEBBs using predict(), then applies
        detection_threshold to filter SEBBs by confidence, converting them
        to binary detections.

        Args:
            scores: Either a path to directory containing SED posterior score
                DataFrames, or a dict of DataFrames keyed by audio_id.
            detection_threshold: (Class-wise) detection/decision threshold(s).
                If None, uses self.detection_threshold. Must be provided either
                here or during initialization.
            audio_ids: Optional list of audio IDs to restrict detection to.
            return_sed_scores: If True, return binary SED scores DataFrames.
                If False, return detection lists.

        Returns:
            If return_sed_scores=False:
                Dict mapping audio_id to list of detections. Each detection is
                a tuple: (onset, offset, class_label)

            If return_sed_scores=True:
                Dict mapping audio_id to binary pandas DataFrame in
                sed_scores_eval format.

        Raises:
            AssertionError: If no detection_threshold provided

        Example:
            >>> predictor = SEBBsPredictor(detection_threshold=0.5)
            >>> detections = predictor.detect("/path/to/scores")
            >>> # detections['audio_001'] = [(1.5, 3.2, 'Speech'), ...]

        """
        return self._predictor.detect(
            scores=scores,
            detection_threshold=detection_threshold,
            audio_ids=audio_ids,
            return_sed_scores=return_sed_scores,
        )

    def detection_thresholding(
        self,
        sebbs: dict[str, SEBBList],
        detection_threshold: OptionalClasswiseParam = None,
        return_sed_scores: bool = False,
    ) -> dict[str, DetectionList] | Scores:
        """Apply detection threshold to existing SEBBs predictions.

        This method allows applying thresholds to pre-computed SEBBs without
        re-running the prediction pipeline.

        Args:
            sebbs: Dict mapping audio_id to list of SEBBs (onset, offset,
                class_label, confidence) as returned by predict().
            detection_threshold: (Class-wise) detection/decision threshold(s).
                If None, uses self.detection_threshold.
            return_sed_scores: If True, return binary SED scores DataFrames.
                If False, return detection lists.

        Returns:
            Detections in same format as detect() method.

        Raises:
            AssertionError: If no detection_threshold provided

        Example:
            >>> sebbs = predictor.predict(scores)
            >>> # Try different thresholds without re-prediction
            >>> dets_low = predictor.detection_thresholding(sebbs, 0.3)
            >>> dets_high = predictor.detection_thresholding(sebbs, 0.7)

        """
        return self._predictor.detection_thresholding(
            sebbs=sebbs,
            detection_threshold=detection_threshold,
            return_sed_scores=return_sed_scores,
        )

    def copy(self) -> "SEBBsPredictor":
        """Create a copy of this predictor instance.

        Returns:
            New SEBBsPredictor with same configuration

        Example:
            >>> predictor2 = predictor.copy()
            >>> # predictor2 has same parameters but independent state

        """
        base_copy = self._predictor.copy()
        wrapper = SEBBsPredictor.__new__(SEBBsPredictor)
        wrapper._predictor = base_copy
        return wrapper

    def __repr__(self) -> str:
        """String representation of predictor."""
        return (
            f"SEBBsPredictor("
            f"step_filter_length={self.step_filter_length}, "
            f"merge_threshold_abs={self.merge_threshold_abs}, "
            f"merge_threshold_rel={self.merge_threshold_rel}, "
            f"detection_threshold={self.detection_threshold}, "
            f"sound_classes={self.sound_classes})"
        )


__all__ = ["SEBBsPredictor"]
