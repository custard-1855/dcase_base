"""Tests for SEBBsTuner wrapper."""

import numpy as np
import pytest
from sed_scores_eval.base_modules.scores import create_score_dataframe

from local.sebbs_wrapper import SEBBsPredictor, SEBBsTuner


class TestSEBBsTuner:
    """Test suite for SEBBsTuner class."""

    @pytest.fixture
    def synthetic_data(self):
        """Create synthetic scores, ground truth, and audio durations for testing."""
        # Create two audio samples with simple synthetic scores
        # Audio 0: Single event from 1.0s to 3.0s with high confidence
        y0 = np.concatenate([
            np.zeros(50),           # 0.0 - 1.0s: silence
            np.ones(100) * 0.9,     # 1.0 - 3.0s: event
            np.zeros(50),           # 3.0 - 4.0s: silence
        ])

        # Audio 1: Two events
        y1 = np.concatenate([
            np.zeros(25),           # 0.0 - 0.5s: silence
            np.ones(75) * 0.85,     # 0.5 - 2.0s: event 1
            np.zeros(25),           # 2.0 - 2.5s: silence
            np.ones(75) * 0.88,     # 2.5 - 4.0s: event 2
        ])

        timestamps0 = np.linspace(0.0, 4.0, len(y0) + 1)
        timestamps1 = np.linspace(0.0, 4.0, len(y1) + 1)

        scores = {
            'audio_0': create_score_dataframe(
                y0[:, None], timestamps0, event_classes=['test_class']
            ),
            'audio_1': create_score_dataframe(
                y1[:, None], timestamps1, event_classes=['test_class']
            ),
        }

        # Ground truth events
        ground_truth = {
            'audio_0': [
                (1.0, 3.0, 'test_class'),  # Event from 1.0s to 3.0s
            ],
            'audio_1': [
                (0.5, 2.0, 'test_class'),  # Event 1
                (2.5, 4.0, 'test_class'),  # Event 2
            ],
        }

        # Audio durations
        audio_durations = {
            'audio_0': 4.0,
            'audio_1': 4.0,
        }

        return scores, ground_truth, audio_durations

    def test_tune_for_psds_basic(self, synthetic_data):
        """Test basic PSDS tuning functionality."""
        scores, ground_truth, audio_durations = synthetic_data

        # Perform PSDS tuning with small parameter grid
        predictor, metrics = SEBBsTuner.tune_for_psds(
            scores=scores,
            ground_truth=ground_truth,
            audio_durations=audio_durations,
            step_filter_lengths=(0.2, 0.4),
            merge_thresholds_abs=(0.1, 0.2),
            merge_thresholds_rel=(1.5, 2.0),
            dtc_threshold=0.7,
            gtc_threshold=0.7,
        )

        # Should return a predictor and metrics
        assert isinstance(predictor, SEBBsPredictor)
        assert isinstance(metrics, dict)

        # Metrics should contain class-wise PSDS values
        assert 'test_class' in metrics
        assert isinstance(metrics['test_class'], (int, float))

    def test_tune_for_collar_based_f1(self, synthetic_data):
        """Test collar-based F1 tuning functionality."""
        scores, ground_truth, audio_durations = synthetic_data

        # Perform collar-based F1 tuning
        predictor, metrics = SEBBsTuner.tune_for_collar_based_f1(
            scores=scores,
            ground_truth=ground_truth,
            audio_durations=audio_durations,
            step_filter_lengths=(0.2, 0.4),
            merge_thresholds_abs=(0.1, 0.2),
            merge_thresholds_rel=(1.5, 2.0),
            onset_collar=0.2,
            offset_collar=0.2,
        )

        # Should return a predictor and metrics
        assert isinstance(predictor, SEBBsPredictor)
        assert isinstance(metrics, dict)

        # Predictor should have detection_threshold set
        assert predictor.detection_threshold is not None

        # Metrics should contain F1 scores
        assert 'test_class' in metrics
        assert isinstance(metrics['test_class'], (int, float))

    def test_tune_with_custom_selection_fn(self, synthetic_data):
        """Test generic tune() method with custom selection function."""
        from sebbs.sebbs.csebbs import select_best_psds

        scores, ground_truth, audio_durations = synthetic_data

        # Use generic tune() method with custom selection function
        result = SEBBsTuner.tune(
            scores=scores,
            ground_truth=ground_truth,
            audio_durations=audio_durations,
            step_filter_lengths=(0.2, 0.4),
            merge_thresholds_abs=(0.1, 0.2),
            merge_thresholds_rel=(1.5, 2.0),
            selection_fn=select_best_psds,
            dtc_threshold=0.7,
            gtc_threshold=0.7,
        )

        # Should return a tuple (predictor, metrics)
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_cross_validation_basic(self, synthetic_data):
        """Test basic cross-validation functionality."""
        from sebbs.sebbs.csebbs import select_best_psds

        scores, ground_truth, audio_durations = synthetic_data

        # Define two folds
        folds = [
            {'audio_0'},  # Fold 1
            {'audio_1'},  # Fold 2
        ]

        # Perform cross-validation
        result = SEBBsTuner.cross_validation(
            scores=scores,
            ground_truth=ground_truth,
            audio_durations=audio_durations,
            folds=folds,
            step_filter_lengths=(0.2, 0.4),
            merge_thresholds_abs=(0.1, 0.2),
            merge_thresholds_rel=(1.5, 2.0),
            selection_fn=select_best_psds,
            dtc_threshold=0.7,
            gtc_threshold=0.7,
        )

        # Should return a tuple of (predictors, sebbs, detections)
        assert isinstance(result, tuple)
        assert len(result) == 3


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
