# Project Structure

## Organization Philosophy

Research-oriented structure with clear separation between:
- **Core library code** (`desed_task/`): Reusable model/data/evaluation components
- **Local experiments** (`local/`): Project-specific trainers and utilities
- **Utility scripts**: Top-level scripts for data generation, embedding extraction, optimization
- **Analysis tools** (`visualize/`): Post-hoc model interpretation

Follows DCASE baseline conventions with extensions for CMT, BEATs, and visualization.

## Directory Patterns

### Core Library - `DESED_task/dcase2024_task4_baseline/desed_task/`
**Purpose**: Reusable SED components (models, data loaders, evaluation)
**Structure**:
- `nnet/`: Neural network modules (CRNN.py, CNN.py, RNN.py)
- `dataio/`: Audio preprocessing and dataset classes
- `evaluation/`: Metrics computation (PSDS, F1)
- `data_augm.py`: Augmentation strategies (Mixup, SpecAugment)
- `utils/`: Shared utilities (encoders, scalers, schedulers)

### Experiment Code - `DESED_task/dcase2024_task4_baseline/local/`
**Purpose**: Project-specific training logic and configuration
**Structure**:
- `sed_trainer_pretrained.py`: Main LightningModule (teacher-student training)
- `beats/`: BEATs embedding model integration
- `sebbs_wrapper/`: Type-safe wrapper layer for SEBBs submodule (predictor, tuner, evaluator)
- `classes_dict.py`: Event class definitions
- `utils.py`: Local helper functions

### Configuration - `DESED_task/dcase2024_task4_baseline/confs/`
**Purpose**: YAML configs for training/inference/data
**Pattern**: Hydra-style hierarchical configs (not shown in detail, assumed standard)

### Post-processing - `DESED_task/dcase2024_task4_baseline/sebbs/`
**Purpose**: cSEBBs algorithm (git submodule)
**Structure**:
- `sebbs/csebbs.py`: Main change-point detection
- `sebbs/median_filter.py`: Class-wise filtering
- `tests/`: Unit tests for post-processing

**Wrapper Pattern**: `local/sebbs_wrapper/` provides type-safe adapter layer without modifying submodule:
- Explicit type annotations (types.py)
- Convenience methods (tuner.py, predictor.py)
- Isolation from submodule changes

### Visualization - `DESED_task/dcase2024_task4_baseline/visualize/`
**Purpose**: Model analysis and interpretation tools
**Structure**:
- `visualize_umap.py`: UMAP embedding visualization
- `visualize_reliability.py`: Reliability diagrams
- `generate_analysis_report.py`: Comprehensive analysis output
- `get_features/`: Feature extraction for visualization
- `visualization_utils.py`: Shared plotting utilities

### Data & Outputs - `DESED_task/dcase2024_task4_baseline/`
**Storage** (not version-controlled):
- `data/`: Raw audio and metadata
- `embeddings/beats/`: Pre-computed HDF5 embeddings
- `exp/`: Experiment outputs (checkpoints, logs)
- `wandb/`: Wandb run artifacts

### Utility Scripts - `DESED_task/dcase2024_task4_baseline/`
**Purpose**: Top-level orchestration
**Examples**:
- `train_pretrained.py`: Training entry point
- `extract_embeddings.py`: BEATs feature extraction
- `optuna_pretrained.py`: Hyperparameter tuning
- `generate_dcase_task4_2024.py`: Dataset generation
- `run_exp.sh`, `run_exp_cmt.sh`: Shell wrappers

## Naming Conventions

- **Python Files**: snake_case (e.g., `sed_trainer_pretrained.py`, `data_augm.py`)
- **Classes**: PascalCase (e.g., `CRNN`, `SEDTask4`, `BEATsModel`)
- **Functions**: snake_case (e.g., `train_model`, `extract_features`)
- **Configs**: lowercase YAML (e.g., `default.yaml`, `strong_real.yaml`)

## Import Organization

```python
# Standard library
import os
from pathlib import Path

# Third-party (grouped by category)
import torch
import pytorch_lightning as pl
from torchaudio import transforms

# DCASE utilities
from desed_task.nnet.CRNN import CRNN
from desed_task.utils.encoder import ManyHotEncoder

# Local project
from local.classes_dict import classes_labels
from local.utils import batched_decode_preds
```

**No Path Aliases**: Standard relative/absolute imports (no `@/` style)

## Code Organization Principles

- **Lightning Modules**: Training logic in `local/sed_trainer_pretrained.py`, model architecture in `desed_task/nnet/`
- **Configuration-driven**: Hydra/OmegaConf for experiment management (assumed pattern)
- **Submodules**: External tools (PSDS_Eval, sebbs) as git submodules for reproducibility
- **Separation of Concerns**: Data (desed_task/dataio) → Model (desed_task/nnet) → Training (local/) → Evaluation (desed_task/evaluation)
- **Pre-computed Features**: BEATs embeddings stored in HDF5 to avoid re-computation during training
- **Wrapper Pattern**: Type-safe adapters for submodules (e.g., `sebbs_wrapper/`) to maintain backward compatibility while adding type safety

---
_Document patterns, not file trees. New files following patterns shouldn't require updates_
