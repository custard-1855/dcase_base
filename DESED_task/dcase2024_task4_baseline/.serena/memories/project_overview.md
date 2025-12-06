# DCASE2024 Task 4 Baseline - Project Overview

## Project Purpose
This project is the baseline system for **DCASE2024 Task 4: Sound Event Detection in Domestic Environments with Heterogeneous Training Dataset and Potentially Missing Labels**.

The main goal is to detect sound events in domestic environments using:
- **DESED** dataset (synthetic strong, strong from AudioSet, weak, unlabeled)
- **MAESTRO** dataset (soft-labeled strong annotations from crowdsourcing)

## Key Challenges
- Handling heterogeneous training datasets with different annotation procedures
- Managing potentially missing labels across datasets
- Combining datasets with partially shared sound event classes
- Long-form audio processing (MAESTRO has audio clips several minutes long)

## Technical Stack

### Core Framework
- **Python**: >=3.11
- **PyTorch Lightning**: 1.9.*
- **PyTorch/TorchAudio**: For deep learning and audio processing

### Pre-trained Models
- **BEATs**: State-of-the-art pre-trained model for AudioSet classification
- Frame-level embeddings used in late-fusion with CRNN baseline

### Key Libraries
- `pytorch-lightning==1.9.*`: Training framework
- `torchmetrics>=0.10.0`: Metrics computation
- `dcase_util>=0.2.16`: DCASE utilities
- `desed>=1.3.6`: DESED dataset utilities
- `sed_scores_eval>=0.0.3`: Sound event detection evaluation
- `resampy==0.4.3`: Audio resampling
- `intervaltree>=2.4.0`: Interval operations

### Experiment Tracking & Optimization
- `wandb>=0.22.3`: Experiment tracking
- `optuna>=4.6.0`: Hyperparameter optimization
- `tensorboard>=2.20.0`: Training visualization
- `codecarbon>=3.0.4`: Energy consumption tracking

### Development Tools
- `pre-commit>=2.3.0`: Git hooks
- `black==19.10b0`: Code formatter (legacy version)
- `ruff`: Modern linter and formatter (configured in root pyproject.toml)

## Model Architecture
- **CRNN** (Convolutional Recurrent Neural Network)
- **Mean-Teacher** framework for semi-supervised learning
- Late fusion of BEATs embeddings with CNN features
- Attention pooling layer
- Handling of missing labels through masking

## Datasets
1. **DESED**: 10-second clips with synthetic strong, strong (AudioSet), weak, and unlabeled data
2. **MAESTRO**: Long-form audio (several minutes) with soft-labeled strong annotations

## System Type
- macOS (Darwin 25.1.0)
- Development environment managed with `uv` package manager
