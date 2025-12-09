# Product Overview

DCASE 2024 Task 4 Sound Event Detection (SED) system - a deep learning pipeline for detecting and localizing acoustic events in audio recordings.

## Core Capabilities

- **Sound Event Detection**: Frame-level and clip-level event detection with temporal localization
- **Semi-supervised Learning**: Teacher-student architecture leveraging labeled and unlabeled data
- **Multi-scale Analysis**: CRNN-based model with optional BEATs embedding integration
- **Post-processing Pipeline**: cSEBBs (change-point based Sound Event Bounding Boxes) for robust event segmentation

## Target Use Cases

- **Audio Scene Analysis**: Detecting and categorizing acoustic events in 10-second audio clips
- **Research & Benchmarking**: DCASE competition baseline and experimentation platform
- **Model Analysis**: Visualization tools (UMAP, reliability analysis) for model interpretation
- **Hyperparameter Optimization**: Optuna-based tuning for post-processing and model configuration

## Value Proposition

Production-grade implementation of state-of-the-art sound event detection combining:
- Teacher-student semi-supervised learning with EMA updates
- Confident Mean Teacher (CMT) strategy for pseudo-labeling
- Advanced data augmentation (Mixup, SpecAugment, MixStyle)
- Comprehensive evaluation metrics (PSDS, F1-score, intersection-based)

---
_Focus on patterns and purpose, not exhaustive feature lists_
