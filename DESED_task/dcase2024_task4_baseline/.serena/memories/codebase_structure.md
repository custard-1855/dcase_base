# Codebase Structure

## Project Root: `dcase2024_task4_baseline/`

```
dcase2024_task4_baseline/
├── README.md                          # Main documentation
├── requirements.txt                   # Python dependencies
├── 
├── # Main Scripts
├── train_pretrained.py                # Main training script
├── extract_embeddings.py              # Extract BEATs embeddings
├── optuna_pretrained.py               # Hyperparameter tuning with Optuna
├── generate_dcase_task4_2024.py       # Download datasets
├── 
├── # Execution Scripts
├── run_exp.sh                         # Batch experiment runner
├── run_exp_cmt.sh                     # CMT experiments
├── run_optuna_gmm.sh                  # Optuna tuning script
├── run_test_from_new_runs.sh          # Test multiple runs
├── 
├── # Utility Scripts
├── count_dataset.py                   # Dataset statistics
├── check_maestro_split.py             # Verify MAESTRO splits
├── 
├── # Configuration
├── confs/
│   ├── pretrained.yaml                # Main baseline configuration
│   ├── before_pretrained.yaml         # Alternative config
│   ├── optuna.yaml                    # Optuna search space
│   └── optuna_gmm.yaml                # Optuna GMM config
├── 
├── # Core Package: desed_task/
├── desed_task/
│   ├── dataio/                        # Data loading and processing
│   │   ├── datasets.py                # Dataset classes (StronglyAnnotatedSet, WeakSet, UnlabeledSet)
│   │   └── samplers.py                # Custom batch samplers (ConcatDatasetBatchSampler)
│   ├── nnet/                          # Neural network models
│   │   ├── CRNN.py                    # Main CRNN model
│   │   ├── CNN.py                     # CNN component
│   │   ├── RNN.py                     # RNN component (BidirectionalGRU)
│   │   └── attention.py               # Attention mechanisms
│   ├── utils/                         # Utilities
│   │   ├── encoder.py                 # Label encoders (ManyHotEncoder, CatManyHotEncoder)
│   │   ├── schedulers.py              # Learning rate schedulers
│   │   └── ...
│   ├── evaluation/                    # Evaluation metrics and tools
│   └── data_augm.py                   # Data augmentation
├── 
├── # Local Modules
├── local/
│   ├── sed_trainer_pretrained.py      # SEDTask4 Lightning module
│   ├── classes_dict.py                # Class mappings (DESED ↔ MAESTRO)
│   ├── resample_folder.py             # Audio resampling utilities
│   ├── utils.py                       # Helper functions (MAC calculation, TSV processing)
│   └── beats/                         # BEATs model code
├── 
├── # Data Preparation
├── dprep/
│   ├── download_prep_maestro.py       # MAESTRO download script
│   ├── train_split.csv                # Train split definition
│   └── validation_split.csv           # Validation split definition
├── 
├── # Visualization (Custom)
├── visualize/                         # Visualization scripts (project-specific)
├── 
├── # Data Analysis
├── data_analysis/                     # Dataset analysis tools
├── 
├── # SEBBs Module
├── sebbs/                             # Self-Ensemble Boundary-aware Bootstrap
│   ├── tests/                         # Unit tests
│   │   ├── test_change_detection.py
│   │   ├── test_csebbs.py
│   │   └── test_median_filter.py
│   └── ...                            # SEBBs implementation
├── 
├── # Output Directories (generated during runtime)
├── embeddings/                        # Pre-computed BEATs embeddings
├── data/                              # Downloaded/resampled datasets
├── exp/                               # Experiment logs and checkpoints
├── wandb/                             # Weights & Biases logs
└── local/                             # Temporary/local files
```

## Key Modules

### `desed_task/` - Core Framework
Main package containing:
- **dataio**: Dataset classes and custom samplers for handling multiple datasets
- **nnet**: Neural network architectures (CRNN, CNN, RNN, Attention)
- **utils**: Encoders, schedulers, and other utilities
- **evaluation**: Metrics computation and evaluation tools

### `local/` - Project-Specific Code
Contains baseline-specific implementations:
- `sed_trainer_pretrained.py`: PyTorch Lightning module (`SEDTask4`)
- `classes_dict.py`: Event class mappings between DESED and MAESTRO
- BEATs model integration

### `confs/` - Configuration Files
YAML configuration files for:
- Model architecture
- Training hyperparameters
- Data paths
- Experiment settings

## Data Flow

1. **Download**: `generate_dcase_task4_2024.py` → downloads raw datasets to `../../data/`
2. **Resample**: `train_pretrained.py` → resamples 44kHz to 16kHz on first run
3. **Extract Embeddings**: `extract_embeddings.py` → creates embeddings in `./embeddings/`
4. **Train**: `train_pretrained.py` → trains model, saves to `./exp/`
5. **Evaluate**: Checkpoints tested on dev-test or eval sets

## Important File Paths

### Configuration Defaults
- Data directory: `../../data` (relative to baseline directory)
- Embeddings: `./embeddings`
- Experiment logs: `./exp/2024_baseline`
- Checkpoints: `./exp/2024_baseline/version_X/checkpoints/`

### Pre-commit Config
- Project level: `/Users/takehonshion/work/iniad/dcase_base/DESED_task/.pre-commit-config.yaml`

### Ruff Config
- Root level: `/Users/takehonshion/work/iniad/dcase_base/pyproject.toml`

## Entry Points

Main entry points for the project:
1. `train_pretrained.py` - Training and evaluation
2. `extract_embeddings.py` - Embedding extraction
3. `optuna_pretrained.py` - Hyperparameter optimization
4. `generate_dcase_task4_2024.py` - Dataset preparation
