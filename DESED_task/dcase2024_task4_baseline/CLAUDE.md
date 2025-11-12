# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

DCASE2024 Task 4 Baseline for Sound Event Detection in domestic environments with heterogeneous training datasets and potentially missing labels. Uses Mean-Teacher semi-supervised learning with pre-trained BEATs embeddings, combining DESED and MAESTRO datasets.

## Development Workflow (Japanese)

**IMPORTANT**: This project follows Japanese workflow conventions defined in `.github/copilot-instructions.md`:

- Always respond in **日本語** (Japanese)
- **調査 (Investigation)**: Document findings in `docs/reports/`
- **計画 (Planning)**: Write minimal requirements to `docs/tasks.md` (delete previous content first, don't write code)
- **実装 (Implementation)**: Implement ONLY what is specified in `docs/tasks.md` (no extra implementation, no debugging here)
- **デバッグ (Debug)**: Show debugging steps only

## Architecture Overview
### High-Level Flow
```
Audio (10s, 16kHz)
  → Mel-Spectrogram (128 bins)
  → CNN (7 layers) ────┐
                       ├→ Concatenate → Bidirectional GRU → Attention Pooling
  BEATs Embeddings ────┘                                      ↓
                                                    Strong (frame) + Weak (clip) predictions
```

### Core Architecture: CRNN with Mean-Teacher

**Student/Teacher Model** (`desed_task/nnet/CRNN.py`):
- **CNN**: 7-layer ConvNet with filters [16, 32, 64, 128, 128, 128, 128]
- **Embedding Integration**: BEATs embeddings (768-dim) aggregated via adaptive pooling (pool1d) and concatenated with CNN output
- **RNN**: Single-layer Bidirectional GRU (192 cells)
- **Attention**: Class-wise attention pooling for weak predictions

**Mean-Teacher** (`local/sed_trainer_pretrained.py`):
- Teacher is EMA of student (decay=0.999)
- Self-supervised loss on unlabeled data using teacher predictions
- Loss weighting ramps up to const_max=2 over 50 warmup epochs

### Multi-Dataset Handling (Key Innovation)

The baseline handles **two heterogeneous datasets** with partially overlapping classes:

1. **DESED**: 10 domestic sounds (Alarm_bell_ringing, Blender, Cat, Dishes, Dog, Electric_shaver_toothbrush, Frying, Running_water, Speech, Vacuum_cleaner)
2. **MAESTRO**: 17 environmental sounds (cutlery and dishes, furniture dragging, people talking, children voices, coffee machine, footsteps, vehicles, metro, etc.)

**Total: 27 classes** (10 DESED + 17 MAESTRO)

**Class Mapping** (`local/classes_dict.py`):
- Some MAESTRO classes map to DESED: "people talking" → "Speech", "cutlery and dishes" → "Dishes", "dog_bark" → "Dog"
- Mapping applied via `process_tsvs()` in `local/utils.py`

**Missing Label Strategy**:
- Each dataset may lack annotations for the other's classes
- `classes_mask` parameter masks invalid classes in:
  - Loss computation (only compute loss for valid classes)
  - Attention pooling (cannot attend to missing classes)
- See `CRNN._get_logits_one_head()` at `desed_task/nnet/CRNN.py:157-183`

**Dataset-Specific Rules**:
- Mixup performed ONLY within same dataset (no cross-dataset mixing)
- MAESTRO long-form audio uses overlap-add at logit level over sliding windows
- See `train_pretrained.py:99-156` for MAESTRO splitting logic

### Training Data Organization

**5 concurrent datasets** with specific batch sizes:
```python
batch_sizes = [12, 6, 6, 12, 24]  # [maestro_real, synth, strong, weak, unlabel]
```

1. **MAESTRO real**: Soft-labeled strong annotations from crowdsourcing
2. **DESED synthetic**: Synthetic soundscapes with strong labels
3. **DESED strong**: Audioset strong-labeled real audio
4. **DESED weak**: Clip-level labels only
5. **DESED unlabeled**: For semi-supervised learning

Uses `ConcatDatasetBatchSampler` (`desed_task/dataio/sampler.py`) to batch from multiple datasets simultaneously. See `train_pretrained.py:486-493`.

### Encoders and Data Loading

**Two-Dataset Encoder** (`train_pretrained.py:62-82`):
- `CatManyHotEncoder` wraps two `ManyHotEncoder` instances (one for DESED, one for MAESTRO)
- Handles multi-hot encoding for 27 classes
- Frame-level encoding matches CNN output temporal resolution

**Dataset Classes** (`desed_task/dataio/datasets.py`):
- `StronglyAnnotatedSet`: Frame-level labels
- `WeakSet`: Clip-level labels only
- `UnlabeledSet`: No labels (for semi-supervised)

**Safe Data Loading** (`local/sed_trainer_pretrained.py:72-100`):
- Custom `SafeDataLoader` and `SafeCollate` handle missing audio files gracefully
- Returns None for missing files, skipped by `_NoneSafeIterator`

## Configuration System

**Main Config**: `confs/pretrained.yaml`

Key sections:
- `pretrained`: BEATs settings (`e2e: False` means use pre-extracted embeddings)
- `training`: Batch sizes, learning rates, warmup, mixup settings
- `data`: Dataset paths (expects `../../data` by default)
- `net`: CRNN architecture (filters, RNN cells, dropout, attention)
- `feats`: Mel-spectrogram parameters (n_mels=128, n_fft=2048, hop=256)

### Experimental Features

Command-line arguments for experimental features (`train_pretrained.py:698-752`):

**MixStyle** (domain generalization):
```bash
--attn_type default        # Attention mechanism type
--attn_deepen 2            # Attention depth
--mixstyle_type resMix     # MixStyle variant (or "disabled")
```

**CMT** (Confidence-based Mean Teacher):
```bash
--cmt                      # Enable CMT
--warmup_epochs 50         # CMT warmup
```

**cSEBBs** (change-point detection post-processing):
```bash
--sebbs                    # Enable SEBBS post-processing
```

**WandB Integration**:
```bash
--wandb_dir /path/to/logs  # WandB logging directory
```

## Energy Consumption Tracking

**MANDATORY** for DCASE challenge submissions. Uses CodeCarbon to track GPU/CPU/RAM energy.

Implementation in `local/sed_trainer_pretrained.py`:
```python
from codecarbon import OfflineEmissionsTracker
tracker = OfflineEmissionsTracker(
    gpu_ids=[torch.cuda.current_device()],
    country_iso_code="CAN"
)
```

Results saved to:
- `./exp/2024_baseline/version_X/codecarbon/emissions_baseline_training.csv`
- `./exp/2024_baseline/version_X/codecarbon/emissions_baseline_test.csv`

Must submit full CSV files for GPU, CPU, and RAM usage.

## Evaluation Metrics

- **PSDS** (Polyphonic Sound Detection Score): Scenarios 1 and 2 using `sed_scores_eval` library (NOT `psds_eval`)
- **mAUC**: Segment-based mean Area Under Curve
- **Energy Consumption**: kWh for training and inference
- **MACs**: Multiply-accumulate operations for 10s audio (use `calculate_macs()` in `local/utils.py`)

**Key**: Submit timestamped scores (not detected events) for accurate PSDS computation.

## Important Implementation Details

1. **GPU Indexing**: GPU indexes start from 1. `--gpus 0` = CPU, `--gpus 1` = GPU 0.

2. **First Run Setup**: Creates resampled 16kHz data from 44.1kHz sources. Needs write permissions on data directory.

3. **Embeddings Required**: With `e2e: False` (default), must run `extract_embeddings.py` before training.

4. **Missing Files**: Code includes filtering at `train_pretrained.py:207-212` via `filter_nonexistent_files()`.

5. **Validation Split**: MAESTRO split is deterministic by scene (5 scenes: cafe_restaurant, city_center, grocery_store, metro_station, residential_area) with 90/10 train/val split per scene. See `split_maestro()` at `train_pretrained.py:99-156`.

6. **Median Filtering**: Per-class median filter lengths tuned via Optuna. Applied in post-processing. See `net.median_filter` in config.

7. **Attention Masking**: Attention pooling masks both padding (temporal) and invalid classes. See `CRNN._get_logits_one_head()` at lines 166-171.

8. **Domain Identification Prohibited**: System must not leverage domain information (MAESTRO vs DESED) during inference.

## File Organization

```
train_pretrained.py              # Main training script
confs/pretrained.yaml            # Main configuration

local/                           # Task-specific implementations
  sed_trainer_pretrained.py      # PyTorch Lightning module (SEDTask4)
  classes_dict.py                # Class definitions and MAESTRO→DESED mapping
  utils.py                       # TSV processing, MACs calculation
  beats/                         # BEATs model implementation

desed_task/                      # Core SED framework
  nnet/
    CRNN.py                      # Main CRNN with embedding integration
    CNN.py                       # CNN backbone
    RNN.py                       # Bidirectional GRU
  dataio/
    datasets.py                  # Dataset classes (Strong/Weak/Unlabeled)
    sampler.py                   # ConcatDatasetBatchSampler
  utils/
    encoder.py                   # ManyHotEncoder, CatManyHotEncoder
    scaler.py                    # TorchScaler for feature normalization
    postprocess.py               # ClassWiseMedianFilter
  evaluation/
    evaluation_measures.py       # PSDS and metrics computation

sebbs/                           # cSEBBs post-processing module
```

## Key Entry Points

- **Training Loop**: `local/sed_trainer_pretrained.py` → `SEDTask4.training_step()`
- **Validation**: `SEDTask4.validation_step()` → computes PSDS on both DESED and MAESTRO
- **Model Forward**: `desed_task/nnet/CRNN.py` → `CRNN.forward(x, embeddings, classes_mask)`
- **Data Loading**: `train_pretrained.py:414-493` → creates 5 datasets and batch sampler
- **Loss Computation**: Check `sed_trainer_pretrained.py` for masked loss handling
