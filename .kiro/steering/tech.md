# Technology Stack

## Architecture

Deep learning pipeline for sound event detection with teacher-student semi-supervised learning architecture. CRNN-based (CNN + RNN) model with optional pre-trained BEATs embeddings.

## Core Technologies

- **Language**: Python 3.11+
- **Framework**: PyTorch Lightning 1.9.x
- **Runtime**: Python 3.11+ with uv package manager
- **Audio Processing**: torchaudio, resampy, sox/soxbindings

## Key Libraries

- **Deep Learning**: pytorch-lightning (1.9.*), torchaudio (2.8.0+), torchmetrics (0.10.0+)
- **Experiment Tracking**: wandb (0.22.3+), tensorboard (2.20.0+), codecarbon (3.0.4+)
- **Audio/DCASE Tools**: dcase-util (0.2.16+), desed (1.3.6+), psds-eval (0.5.3+), sed-scores-eval
- **Optimization**: optuna (4.6.0+) with optuna-dashboard
- **Visualization**: plotly (6.5.0+), seaborn (0.13.2+), umap-learn (0.5.9+)
- **Storage**: h5py (3.14.0+) for HDF5 embeddings

## Development Standards

### Code Quality
- **Linter/Formatter**: Ruff (configured in pyproject.toml)
- **Line Length**: 100 characters
- **Lint Rules**: ALL enabled with selective ignores (D1/docstrings, TD/todos, PD011)
- **Unfixable**: F401 (unused imports), F841 (unused variables) for explicit removal
- **Pre-commit**: Automated hooks for code quality

### Type Safety
- **Type Checker**: mypy (dev dependency)
- **Type Stubs**: pandas-stubs, types-setuptools for third-party libraries
- **Type Annotations**: Enforced in refactored modules (e.g., sed_trainer_pretrained, sebbs_wrapper)

### Testing
- **Unit Tests**: pytest-based test suites for refactored modules
  - `sebbs_wrapper/tests/`: Wrapper layer tests
  - `local/test_sed_trainer_pretrained.py`: Type annotation validation, helper method tests
- **Manual Validation**: Visualization tools (UMAP, reliability diagrams)

### Logging
Custom logger at `src.library.logger.LOGGER` (Ruff-configured)

## Development Environment

### Required Tools
- Python 3.11+ (pinned in `.python-version`)
- uv package manager (dependency management)
- git with submodules (PSDS_Eval, sebbs)

### Container-based Development
- **Devcontainer**: `.devcontainer/` configuration for reproducible environment
- **Security**: Verification documents for secrets/resource management
- **Setup**: `post_create_command.sh` for automated environment initialization

### Common Commands
```bash
# Setup: uv sync
# Train: python DESED_task/dcase2024_task4_baseline/train_pretrained.py
# Extract embeddings: python DESED_task/dcase2024_task4_baseline/extract_embeddings.py
# Optimize: python DESED_task/dcase2024_task4_baseline/optuna_pretrained.py
# Visualize: python DESED_task/dcase2024_task4_baseline/visualize/visualize_umap.py
```

## Key Technical Decisions

- **PyTorch Lightning 1.9.x**: Frozen version for reproducibility (not latest 2.x)
- **16 kHz Audio**: Resampling from 44 kHz for efficiency vs. quality trade-off
- **HDF5 Embeddings**: Pre-computed BEATs features stored for training efficiency
- **Teacher-Student EMA**: 0.999 decay rate for stable pseudo-labels
- **Ruff over Black/Flake8**: Modern all-in-one linter/formatter (note: black 19.10b0 in deps is legacy)
- **Wandb Primary**: Main experiment tracker over TensorBoard

---
_Document standards and patterns, not every dependency_
