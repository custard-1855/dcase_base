# SEBBs Wrapper

Type-safe wrapper layer for Sound Event Bounding Boxes (SEBBs) functionality.

## Overview

This package provides a clean, well-documented interface to the SEBBs submodule without modifying the original code. It adds:

- **Strong typing** with explicit type annotations
- **Improved documentation** with detailed docstrings
- **Better ergonomics** through convenience methods
- **Isolation** from submodule changes

## Architecture

```
sebbs_wrapper/
├── __init__.py         # Public API exports
├── types.py            # Type definitions
├── predictor.py        # SEBBsPredictor wrapper
├── tuner.py            # SEBBsTuner utilities
└── evaluator.py        # SEBBsEvaluator metrics
```

### Design Pattern: Wrapper (Adapter)

This implementation uses the **Wrapper pattern** to:
1. Delegate to the underlying `sebbs` submodule
2. Provide type-safe interfaces
3. Add convenience methods
4. Maintain backward compatibility

## Usage

### Basic Prediction

```python
from local.sebbs_wrapper import SEBBsPredictor

# Create predictor with default parameters
predictor = SEBBsPredictor(
    step_filter_length=0.5,
    merge_threshold_abs=1.0,
    merge_threshold_rel=2.0
)

# Predict SEBBs from scores
sebbs = predictor.predict('/path/to/scores')

# sebbs['audio_001'] = [
#     (onset, offset, class_label, confidence),
#     (1.5, 3.2, 'Speech', 0.87),
#     ...
# ]
```

### Hyperparameter Tuning

```python
from local.sebbs_wrapper import SEBBsTuner

# Tune for PSDS (Polyphonic Sound Detection Score)
predictor, metrics = SEBBsTuner.tune_for_psds(
    scores='/path/to/validation/scores',
    ground_truth='/path/to/validation/gt.tsv',
    audio_durations='/path/to/validation/durations.tsv'
)

print(f"Best PSDS values: {metrics}")
```

### Evaluation

```python
from local.sebbs_wrapper import SEBBsEvaluator

evaluator = SEBBsEvaluator()

# Evaluate PSDS1 (DCASE 2024 Task 4 standard)
psds, class_psds = evaluator.evaluate_psds1(
    scores=predictions,
    ground_truth=gt,
    audio_durations=durations
)

print(f"PSDS1: {psds:.3f}")
for cls, value in class_psds.items():
    print(f"  {cls}: {value:.3f}")
```

### Detection with Threshold

```python
# Predict SEBBs and apply detection threshold
detections = predictor.detect(
    scores='/path/to/scores',
    detection_threshold=0.5,
    return_sed_scores=False
)

# detections['audio_001'] = [
#     (onset, offset, class_label),
#     (1.5, 3.2, 'Speech'),
#     ...
# ]
```

## Type Safety

All public interfaces use explicit type annotations:

```python
from local.sebbs_wrapper.types import (
    SEBB,           # Tuple[float, float, str, float]
    Detection,      # Tuple[float, float, str]
    SEBBList,       # List[SEBB]
    Scores,         # Dict[str, pd.DataFrame]
    PredictorConfig,
)

# Type-safe configuration
config: PredictorConfig = {
    'step_filter_length': 0.5,
    'merge_threshold_abs': 1.0,
    'merge_threshold_rel': 2.0,
}
predictor = SEBBsPredictor.from_config(config)
```

## Advanced Usage

### Cross-Validation

```python
from sebbs.sebbs.csebbs import select_best_psds_and_cbf

# Define folds
folds = [fold1_ids, fold2_ids, fold3_ids, fold4_ids, fold5_ids]

# Run cross-validation
predictors, sebbs, detections = SEBBsTuner.cross_validation(
    scores=all_scores,
    ground_truth=all_gt,
    audio_durations=all_durations,
    folds=folds,
    selection_fn=select_best_psds_and_cbf,
)
```

### Custom Selection Function

```python
def my_custom_selection(csebbs_list, ground_truth, audio_durations, **kwargs):
    """Custom parameter selection logic."""
    # Implement your selection criteria
    best_predictor = ...
    best_metrics = ...
    return best_predictor, best_metrics

# Use custom selection
predictor, metrics = SEBBsTuner.tune(
    scores=val_scores,
    ground_truth=val_gt,
    audio_durations=val_durations,
    selection_fn=my_custom_selection,
    # Pass custom kwargs
    my_custom_param=42,
)
```

## Benefits

### 1. Submodule Independence

The wrapper isolates your code from changes in the `sebbs` submodule:

```python
# If sebbs updates, only the wrapper needs adjustment
# Your application code remains unchanged
```

### 2. Type Safety

IDE autocomplete and type checking work properly:

```python
# Type checker knows the return types
sebbs: Dict[str, SEBBList] = predictor.predict(scores)
psds: float, class_psds: Dict[str, float] = evaluator.evaluate_psds1(...)
```

### 3. Better Documentation

Rich docstrings with examples:

```python
help(SEBBsPredictor.predict)  # Detailed documentation
```

### 4. Convenience Methods

Shortcuts for common operations:

```python
# Before (using submodule directly)
from sebbs.sebbs import csebbs
predictor, _ = csebbs.tune(
    scores=scores, gt=gt, durations=durations,
    selection_fn=csebbs.select_best_psds,
    dtc_threshold=0.7, gtc_threshold=0.7,
    cttc_threshold=None, alpha_ct=0.0,
    alpha_st=1.0, unit_of_time='hour', max_efpr=100.0
)

# After (using wrapper)
predictor, _ = SEBBsTuner.tune_for_psds(scores, gt, durations)
```

## Migration from Direct Submodule Usage

### Before

```python
from sebbs.sebbs import csebbs
from sebbs.sebbs.utils import sed_scores_from_sebbs

predictor = csebbs.CSEBBsPredictor(...)
predictor, _ = csebbs.tune(...)
```

### After

```python
from local.sebbs_wrapper import SEBBsPredictor, SEBBsTuner
from sebbs.sebbs.utils import sed_scores_from_sebbs  # Keep utilities

predictor = SEBBsPredictor(...)
predictor, _ = SEBBsTuner.tune_for_psds(...)
```

## Reference

Based on:
> J. Ebbers, F. Germain, G. Wichern and J. Le Roux,
> "Sound Event Bounding Boxes", Interspeech 2024.
> https://arxiv.org/abs/2406.04212

## Future Enhancements

Potential improvements:
- [ ] Async prediction support
- [ ] Batch processing utilities
- [ ] Caching layer for tuning results
- [ ] Progress callbacks for long operations
- [ ] Configuration file support (YAML/JSON)
