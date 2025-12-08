"""Tests for SEBBsPredictor wrapper."""

import numpy as np
import pytest
from sed_scores_eval.base_modules.scores import create_score_dataframe

from local.sebbs_wrapper import SEBBsPredictor
from local.sebbs_wrapper.types import PredictorConfig


class TestSEBBsPredictor:
    """Test suite for SEBBsPredictor class."""

    def test_initialization_with_defaults(self):
        """Test predictor initialization with default parameters."""
        predictor = SEBBsPredictor()

        assert predictor.step_filter_length == 0.5
        assert predictor.merge_threshold_abs == 1.0
        assert predictor.merge_threshold_rel == 2.0
        assert predictor.detection_threshold is None
        assert predictor.sound_classes is None

    def test_initialization_with_custom_params(self):
        """Test predictor initialization with custom parameters."""
        predictor = SEBBsPredictor(
            step_filter_length=0.7,
            merge_threshold_abs=0.5,
            merge_threshold_rel=1.5,
            detection_threshold=0.3,
            sound_classes=['Speech', 'Music']
        )

        assert predictor.step_filter_length == 0.7
        assert predictor.merge_threshold_abs == 0.5
        assert predictor.merge_threshold_rel == 1.5
        assert predictor.detection_threshold == 0.3
        assert predictor.sound_classes == ['Speech', 'Music']

    def test_from_config(self):
        """Test predictor creation from configuration dict."""
        config: PredictorConfig = {
            'step_filter_length': 0.6,
            'merge_threshold_abs': 0.8,
            'merge_threshold_rel': 1.8,
            'sound_classes': ['Cat', 'Dog'],
        }

        predictor = SEBBsPredictor.from_config(config)

        assert predictor.step_filter_length == 0.6
        assert predictor.merge_threshold_abs == 0.8
        assert predictor.merge_threshold_rel == 1.8
        assert predictor.sound_classes == ['Cat', 'Dog']

    def test_predict_simple_case(self):
        """Test basic prediction functionality with simple synthetic data."""
        # Create simple synthetic scores (from SEBBs paper example)
        cos = (1 - np.cos(np.linspace(0, 2 * np.pi, 270))) / 2
        y = np.zeros(500)
        y[15:15 + len(cos)] += 0.98 * cos
        cos = (1 - np.cos(np.linspace(0, 2 * np.pi, 170))) / 2
        y[315:315 + len(cos)] += 0.2 * cos

        timestamps = np.linspace(0.0, 10.0, len(y) + 1)
        scores_df = create_score_dataframe(y[:, None], timestamps, event_classes=['a'])

        predictor = SEBBsPredictor(
            step_filter_length=1.0,
            merge_threshold_abs=1.0,
            merge_threshold_rel=2.0,
        )

        sebbs = predictor.predict({'audio_0': scores_df})

        # Should predict SEBBs
        assert 'audio_0' in sebbs
        assert len(sebbs['audio_0']) > 0

        # Each SEBB should be a tuple of (onset, offset, class, confidence)
        for sebb in sebbs['audio_0']:
            assert len(sebb) == 4
            onset, offset, class_label, confidence = sebb
            assert isinstance(onset, (int, float))
            assert isinstance(offset, (int, float))
            assert isinstance(class_label, str)
            assert isinstance(confidence, (int, float))
            assert onset < offset

    def test_detect_with_threshold(self):
        """Test detection thresholding functionality."""
        # Create simple synthetic scores
        y = np.array([0.0] * 100 + [0.8] * 100 + [0.0] * 100)
        timestamps = np.linspace(0.0, 3.0, len(y) + 1)
        scores_df = create_score_dataframe(y[:, None], timestamps, event_classes=['test'])

        predictor = SEBBsPredictor(
            step_filter_length=0.1,
            merge_threshold_abs=0.5,
            merge_threshold_rel=2.0,
        )

        detections = predictor.detect(
            {'audio_0': scores_df},
            detection_threshold=0.5,
        )

        # Should have detections
        assert 'audio_0' in detections

        # Each detection should be a tuple of (onset, offset, class)
        for detection in detections['audio_0']:
            assert len(detection) == 3
            onset, offset, class_label = detection
            assert isinstance(onset, (int, float))
            assert isinstance(offset, (int, float))
            assert isinstance(class_label, str)

    def test_copy(self):
        """Test predictor copying functionality."""
        predictor1 = SEBBsPredictor(
            step_filter_length=0.7,
            merge_threshold_abs=0.5,
        )

        predictor2 = predictor1.copy()

        # Should have same parameters
        assert predictor2.step_filter_length == 0.7
        assert predictor2.merge_threshold_abs == 0.5

        # But be different objects
        assert predictor1 is not predictor2
        assert predictor1._predictor is not predictor2._predictor

    def test_repr(self):
        """Test string representation of predictor."""
        predictor = SEBBsPredictor(
            step_filter_length=0.5,
            merge_threshold_abs=1.0,
            merge_threshold_rel=2.0,
        )

        repr_str = repr(predictor)

        assert 'SEBBsPredictor' in repr_str
        assert '0.5' in repr_str
        assert '1.0' in repr_str
        assert '2.0' in repr_str


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
