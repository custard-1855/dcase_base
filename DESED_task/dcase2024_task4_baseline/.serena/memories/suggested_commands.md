# Suggested Commands

## Environment Setup

### Create Conda Environment
```bash
# Run line by line from conda_create_environment.sh (recommended)
bash conda_create_environment.sh
```

## Data Management

### Download Datasets
```bash
# Download all datasets (training and development)
python generate_dcase_task4_2024.py --basedir="../../data"

# Download only specific parts
python generate_dcase_task4_2024.py --only_strong  # Strong labels only
python generate_dcase_task4_2024.py --only_real    # Weak, unlabeled, validation
python generate_dcase_task4_2024.py --only_synth   # Synthetic data only
python generate_dcase_task4_2024.py --only_maestro # MAESTRO dataset only
```

## Model Training & Inference

### Extract Embeddings (Required before training)
```bash
# Extract BEATs embeddings for training data
python extract_embeddings.py --output_dir ./embeddings

# Extract embeddings for evaluation set
python extract_embeddings.py --eval_set
```

### Train Model
```bash
# Basic training with default settings
uv run train_pretrained.py

# Training with custom GPU
uv run train_pretrained.py --gpu 1

# Training with custom log directory
uv run train_pretrained.py --log_dir="./exp/my_experiment"

# Fast development run (for debugging)
uv run train_pretrained.py --fast_dev_run

# Resume from checkpoint
uv run train_pretrained.py --resume_from_checkpoint /path/to/file.ckpt
```

### Test Model
```bash
# Test from checkpoint on dev-test set
python train_pretrained.py --test_from_checkpoint /path/to/checkpoint.ckpt

# Evaluate on evaluation set
python train_pretrained.py --eval_from_checkpoint /path/to/checkpoint.ckpt
```

### Run Experiments (Batch Training)
```bash
# Run CMT experiments
bash run_exp_cmt.sh

# Run standard experiments
bash run_exp.sh

# Run Optuna hyperparameter tuning
bash run_optuna_gmm.sh
```

## Hyperparameter Tuning

### Optuna-based Tuning
```bash
# Start hyperparameter tuning
python optuna_pretrained.py --log_dir MY_TUNING_EXP --n_jobs X

# Tune median filter lengths (after finding best config)
python optuna_pretrained.py \
  --log_dir MY_TUNING_EXP_4_MEDIAN \
  --n_jobs X \
  --test_from_checkpoint best_checkpoint.ckpt \
  --confs path/to/best.yaml
```

## Code Quality & Formatting

### Run Pre-commit Hooks
```bash
# Install pre-commit hooks
pre-commit install

# Run all hooks on all files
pre-commit run --all-files

# Run specific hook
pre-commit run black --all-files
pre-commit run trailing-whitespace --all-files
```

### Format Code (Ruff - Modern)
```bash
# Format code with ruff (from project root)
cd /Users/takehonshion/work/iniad/dcase_base
ruff format .

# Check and fix linting issues
ruff check --fix .

# Check only (no fixes)
ruff check .
```

### Format Code (Black - Legacy)
```bash
# Format specific file
black path/to/file.py

# Format all Python files in directory
black .
```

## Monitoring & Visualization

### TensorBoard
```bash
# View training logs
tensorboard --logdir="path/to/exp_folder"

# Example with default directory
tensorboard --logdir="./exp/2024_baseline"
```

### View Energy Consumption Reports
```bash
# Training energy consumption
cat ./exp/2024_baseline/version_X/codecarbon/emissions_baseline_training.csv

# Test/inference energy consumption
cat ./exp/2024_baseline/version_X/codecarbon/emissions_baseline_test.csv
```

## System Commands (macOS/Darwin)

### Common Darwin Commands
```bash
# List files (with hidden files)
ls -la

# Search for files
find . -name "*.py"

# Search in files (use ripgrep if available, faster than grep)
rg "pattern" --type py

# Directory navigation
cd path/to/directory
pwd

# Git operations
git status
git log --oneline -10
git diff

# Process monitoring
top
ps aux | grep python
```

## Configuration

### Modify Configuration
Edit YAML files in `confs/` directory:
- `confs/pretrained.yaml`: Main configuration for pretrained model baseline
- `confs/optuna.yaml`: Optuna hyperparameter search config
- `confs/optuna_gmm.yaml`: Optuna GMM-specific config

### Important Notes
- GPU indexing starts from 1 in training scripts (not 0!)
  - `--gpu 0` uses CPU
  - `--gpu 1` uses first GPU
- Data path defaults to `../../data` - update in YAML if different
- First run creates resampled data (44kHz → 16kHz) requiring write permissions
