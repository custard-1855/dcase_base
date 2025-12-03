# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a DCASE 2024 Task 4 baseline system for **Sound Event Detection (SED) in Domestic Environments** with heterogeneous training datasets and potentially missing labels. The system handles two datasets:
- **DESED**: Domestic environment sounds (10 classes)
- **MAESTRO**: Crowdsourced annotations with soft labels (17 classes)

The baseline uses a **CRNN** (CNN + Bidirectional GRU) architecture with **BEATs pretrained embeddings** in a late-fusion approach, trained using **Mean Teacher** semi-supervised learning.

## Common Commands

### Environment Setup
```bash
# Create conda environment (run line by line if issues occur)
bash conda_create_environment.sh
```

### Data Preparation
```bash
# Download full dataset (change basedir as needed)
python generate_dcase_task4_2024.py --basedir="../../data"

# Download specific parts only
python generate_dcase_task4_2024.py --only_strong
python generate_dcase_task4_2024.py --only_real
python generate_dcase_task4_2024.py --only_synth
python generate_dcase_task4_2024.py --only_maestro

# Extract BEATs embeddings (required before training)
python extract_embeddings.py --output_dir ./embeddings

# Extract embeddings for evaluation set
python extract_embeddings.py --eval_set
```

### Training
```bash
# Basic training with default config
python train_pretrained.py

# Or using uv (as seen in run_exp.sh)
uv run train_pretrained.py

# Train with custom config
python train_pretrained.py --conf_file confs/my_config.yaml

# Train on specific GPU (GPU indexes start from 1!)
python train_pretrained.py --gpus 1

# WARNING: --gpus 0 will use CPU, not GPU 0

# Change log directory
python train_pretrained.py --log_dir ./exp/my_experiment

# Fast development run (for debugging)
python train_pretrained.py --fast_dev_run

# Resume from checkpoint
python train_pretrained.py --resume_from_checkpoint /path/to/file.ckpt
```

### Evaluation
```bash
# Test from checkpoint
python train_pretrained.py --test_from_checkpoint /path/to/model.ckpt

# Evaluate on eval set
python train_pretrained.py --eval_from_checkpoint /path/to/model.ckpt
```

### Hyperparameter Tuning
```bash
# Run Optuna tuning (n_jobs should match CUDA_VISIBLE_DEVICES)
python optuna_pretrained.py --log_dir MY_TUNING_EXP --n_jobs X

# GPUs must be in exclusive compute mode: nvidia-smi -c 2

# Tune median filter after finding best config
python optuna_pretrained.py --log_dir MY_TUNING_EXP_MEDIAN --n_jobs X \
  --test_from_checkpoint best_checkpoint.ckpt --confs path/to/best.yaml
```

### Experiment Scripts
```bash
# Run experiment with CMT
bash run_exp_cmt.sh

# Run standard experiment
bash run_exp.sh

# Run Optuna with GMM
bash run_optuna_gmm.sh
```

### Monitoring
```bash
# View TensorBoard logs
tensorboard --logdir=./exp/2024_baseline

# Energy consumption logs are in:
# ./exp/2024_baseline/version_X/codecarbon/emissions_baseline_training.csv
# ./exp/2024_baseline/version_X/codecarbon/emissions_baseline_test.csv
```

## Architecture & Code Structure

### High-Level Architecture

The system implements a **Mean Teacher** semi-supervised learning framework with the following key components:

1. **Feature Extraction**: Log-mel spectrograms (128 mel bins) + BEATs pretrained embeddings (768-dim)
2. **Model**: CRNN with attention pooling
3. **Training Strategy**:
   - Student-teacher architecture with EMA (ρ=0.999)
   - Supervised loss on labeled data (strong, weak, synthetic)
   - Self-supervised loss (MSE) on unlabeled data
   - Ramp-up schedule for consistency loss weight

### Key Design Patterns for Missing Labels

Since DESED and MAESTRO have partially overlapping classes with different annotation procedures, the baseline implements:

1. **Class Mapping** (`local/classes_dict.py`):
   - Maps certain MAESTRO classes to DESED classes (e.g., "people talking" → "Speech")
   - See `maestro_desed_alias` dict and `process_tsvs()` function in `train_pretrained.py`

2. **Logit Masking** (`desed_task/nnet/CRNN.py`):
   - Output logits are masked for classes without annotations in current dataset
   - Applied in loss computation AND attention pooling layer

3. **Within-Dataset Mixup**:
   - Mixup augmentation only mixes samples from the same dataset
   - Prevents mixing incompatible annotations

4. **Long-Form Audio Handling** (`local/sed_trainer_pretrained.py`):
   - MAESTRO clips are long-form (minutes)
   - Uses overlap-add at logit level over sliding windows

### Module Organization

```
desed_task/
├── dataio/          # Data loading and sampling
│   ├── datasets.py       # StronglyAnnotatedSet, WeakSet, UnlabeledSet
│   ├── pre_datasets.py   # Additional dataset classes
│   └── sampler.py        # ConcatDatasetBatchSampler for multi-dataset batching
├── nnet/            # Neural network modules
│   ├── CNN.py            # Convolutional blocks
│   ├── CRNN.py           # Main model: CNN + RNN + attention + embedding fusion
│   ├── RNN.py            # Bidirectional GRU
│   ├── mixstyle.py       # MixStyle augmentation
│   └── pre_CNN.py        # Pre-processing CNN layers
├── utils/           # Utilities
│   ├── encoder.py        # ManyHotEncoder, CatManyHotEncoder for multi-dataset labels
│   └── schedulers.py     # ExponentialWarmup scheduler
└── evaluation/      # Evaluation metrics and utilities

local/
├── sed_trainer_pretrained.py  # Main trainer (SEDTask4) - PyTorch Lightning module
├── classes_dict.py             # Class definitions and mappings
├── resample_folder.py          # Audio resampling utilities
├── utils.py                    # MAC calculation, TSV processing
└── beats/                      # BEATs embedding extraction code
```

### Configuration System

All experiments are controlled via YAML config files in `confs/`:

- `pretrained.yaml`: Main configuration file
- Sections:
  - `pretrained`: BEATs model settings (frozen, embeddings dir)
  - `cmt`: Confidence-based Masked Training params
  - `sat`: Self-Adaptive Training params
  - `sebbs`: SEBBs feature flag
  - `training`: Batch sizes, epochs, learning rate schedules, mixup settings
  - `data`: All dataset paths (44kHz and 16kHz versions)
  - `feats`: Log-mel spectrogram parameters
  - `net`: Model architecture (CNN layers, RNN cells, attention, median filters)
  - `opt`: Optimizer settings

**Important**: Default data paths assume `../../data` structure. Update paths in config YAML if data is elsewhere.

### Custom Features (Beyond Original Baseline)

This repository includes experimental features:

1. **CMT (Confidence-based Masked Training)**:
   - Enable with `--cmt` flag
   - Configurable `phi_clip` and `phi_frame` thresholds
   - Warmup epochs supported

2. **SAT (Self-Adaptive Training)**:
   - Enable with `--sat` flag
   - Strong augmentation types: cutmix, frame_shift_time_mask
   - GMM-based adaptive thresholding

3. **SEBBs**:
   - Enable with `--sebbs` flag
   - Additional module in `sebbs/` directory

## Data Pipeline Details

### Automatic Preprocessing

On first run, `train_pretrained.py` automatically:
1. Resamples all audio from 44kHz → 16kHz (requires write permissions on data folder)
2. Generates duration TSV files for validation sets
3. Caches resampled data to avoid repeated processing

### Dataset Types

The system handles 5 different data sources with different batch sizes:
```yaml
batch_size: [12, 6, 6, 12, 24]  # [maestro, synth, strong, weak, unlabel]
```

- **maestro**: MAESTRO real recordings (long-form, soft labels)
- **synth**: DESED synthetic data (strong labels)
- **strong**: AudioSet strong labels (10s clips)
- **weak**: DESED weak labels (clip-level only)
- **unlabel**: Unlabeled in-domain data (for Mean Teacher)

### Multi-Encoder System

Uses `CatManyHotEncoder` (in `desed_task/utils/encoder.py`) to handle:
- DESED encoder: 10 classes
- MAESTRO encoder: 17 classes
- Combined 27-class output space with selective masking

## Energy Consumption Tracking

**Mandatory for DCASE submissions**. The system uses CodeCarbon:

```python
from codecarbon import OfflineEmissionsTracker
tracker = OfflineEmissionsTracker(
    gpu_ids=[torch.cuda.current_device()],
    country_iso_code="CAN"
)
```

Implementation is in `local/sed_trainer_pretrained.py`. Output CSV files contain GPU/CPU/RAM energy breakdowns.

## Evaluation Metrics

- **PSDS** (Polyphonic Sound Detection Score) - computed with `sed_scores_eval` library
- Submissions require **timestamped scores**, not just detected events
- Multiple scenarios evaluated (PSDS-scenario1, mean pAUC)
- Median filtering applied post-hoc (per-class filter lengths in config)

## Important Gotchas

1. **GPU Indexing**: `--gpus` parameter starts from 1, not 0. Using `--gpus 0` runs on CPU.

2. **Data Paths**: Config assumes `../../data` relative to repo. First run creates `*_16k` folders.

3. **Embeddings Required**: Must run `extract_embeddings.py` before training with `use_embeddings: True`.

4. **Multi-Dataset Batching**: The `ConcatDatasetBatchSampler` ensures each batch contains samples from all datasets.

5. **Class Masking**: When modifying model outputs, remember masking is applied in both loss computation AND attention pooling.

6. **Weights & Biases**: Set `use_wandb: True` in config and ensure wandb is configured.

7. **Deterministic Training**: Set `deterministic: True` in config for reproducibility, but note this may reduce performance.

8. **Pre-commit Hooks**: Repository uses pre-commit (see `.pre-commit-config.yaml` if it exists).

## Working with This Codebase

### When Adding New Features

- Configuration changes go in `confs/pretrained.yaml`
- Model architecture changes go in `desed_task/nnet/CRNN.py`
- Training logic changes go in `local/sed_trainer_pretrained.py`
- New augmentations can be added in `desed_task/data_augm.py`

### When Running Experiments

- Use shell scripts (`run_exp.sh`) for reproducible experiment tracking
- Log directory structure: `./exp/<experiment_name>/version_X/`
- Always specify `--wandb_dir` for proper organization
- Check `docs/reports/` for experiment documentation (in Japanese)

### When Debugging

- Use `--fast_dev_run` for quick sanity checks
- Enable progress bar: `enable_progress_bar: True` in config
- Check TensorBoard for training curves
- Validation runs every N epochs (configurable via `validation_interval`)

## Japanese Language Support

This repository uses Japanese for:
- Comments in some configuration files
- Documentation in `docs/` directory
- Copilot instructions (`.github/copilot-instructions.md`)
- Some variable naming and logging

When working with Japanese users, respond in Japanese and follow the workflow defined in copilot instructions.
