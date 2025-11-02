# DCASE/SED Codebase - Comprehensive Analysis

## Project Overview
This is a Sound Event Detection (SED) system for DCASE 2024 Task 4 (Weakly-supervised SED with Domain Adaptation). It implements a sophisticated semi-supervised learning approach with Mean Teacher (MT) framework and optional Confident Mean Teacher (CMT) extensions.

**Repository Path:** `/Users/takehonshion/work/iniad/dcase_base/DESED_task/dcase2024_task4_baseline/`

---

## 1. SED Models Implemented

### 1.1 Main Model Architecture: CRNN (Convolutional Recurrent Neural Network)

**File:** `/Users/takehonshion/work/iniad/dcase_base/DESED_task/dcase2024_task4_baseline/desed_task/nnet/CRNN.py` (371 lines)

**Architecture Components:**
- **CNN Backbone:** Feature extraction layers
  - 7 convolutional blocks with configurable parameters
  - Supports multiple activation functions (ReLU, LeakyReLU, GLU)
  - Batch normalization or Layer normalization
  - Average pooling layers
  - FrequencyAttentionMixStyle layers integrated at multiple points

- **RNN Module:** Temporal context learning
  - Bidirectional GRU (BGRU) for sequence modeling
  - Configurable hidden size (default: 192 cells)
  - Bidirectional outputs concatenated (384 dimensions)

- **Output Heads:** Dual prediction tasks
  - Frame-level predictions (strong labels)
  - Clip-level predictions (weak labels)
  - Sigmoid activation for multi-label classification
  - Optional attention mechanism for temporal weighting

**Key Features:**
- **Embedding Integration:** Supports pre-trained embeddings (BEATs) injection
  - Frame-level embedding processing with secondary GRU
  - Multiple aggregation types: frame, global, interpolate, pool1d
  - Shrinking and concatenation transformations
  
- **Data Augmentation:**
  - SpecAugment (time and frequency masking)
  - Dropstep recurrent masking
  - Mixup support for synthetic data augmentation
  
- **Teacher-Student Architecture:**
  - Supports two identical models (student for training, teacher for consistency)
  - EMA (Exponential Moving Average) updates for teacher

**Configuration (pretrained.yaml):**
```yaml
net:
  n_in_channel: 1
  nclass: 27  # DESED + MAESTRO classes
  dropout: 0.2
  rnn_layers: 1
  n_RNN_cell: 192
  activation: glu
  rnn_type: BGRU
  kernel_size: [3, 3, 3, 3, 3, 3, 3]
  nb_filters: [16, 32, 64, 128, 128, 128, 128]
  pooling: [[2,2], [2,2], [1,2], [1,2], [1,2], [1,2], [1,2]]
  use_embeddings: True
  embedding_size: 768
  embedding_type: frame
  aggregation_type: pool1d
  median_filter: [3, 9, 9, 5, 5, 5, 9, 7, 11, 9, 7, 3, 9, 13, 7, 1, 13, 3, 13, 7, 5, 5, 1, 13, 17, 13, 15]
```

### 1.2 Pretrained Models: BEATs (Audio Pre-Training with Acoustic Tokenizers)

**Files:** 
- `/Users/takehonshion/work/iniad/dcase_base/DESED_task/dcase2024_task4_baseline/local/beats/BEATs.py` (808 lines)
- `/Users/takehonshion/work/iniad/dcase_base/DESED_task/dcase2024_task4_baseline/local/beats/backbone.py` (808 lines)

**Purpose:** Extract semantic audio embeddings from raw audio
- Pre-trained on AudioSet for general audio understanding
- Provides both global and frame-level embeddings (768-dim)
- Transformer-based architecture for contextual representation
- Uses Kaldi-compatible filterbank features

**Embedding Extraction Mode:**
- Can be frozen (default: `freezed: True`)
- Can be end-to-end trainable
- Pre-extracted embeddings stored in HDF5 files

### 1.3 Supporting CNN Architecture

**File:** `/Users/takehonshion/work/iniad/dcase_base/DESED_task/dcase2024_task4_baseline/desed_task/nnet/CNN.py`

**Features:**
- Modular CNN blocks with sequential addition
- MixStyle integration for domain adaptation
- FrequencyAttentionMixStyle layers at multiple positions
- Configurable number of filters and kernel sizes

### 1.4 RNN Modules

**File:** `/Users/takehonshion/work/iniad/dcase_base/DESED_task/dcase2024_task4_baseline/desed_task/nnet/RNN.py` (54 lines)

**Components:**
- `BidirectionalGRU`: Main recurrent unit for temporal modeling
- `BidirectionalLSTM`: Alternative recurrent architecture (not actively used)

---

## 2. Inference/Prediction Code

### 2.1 Main Training Module with Inference

**File:** `/Users/takehonshion/work/iniad/dcase_base/DESED_task/dcase2024_task4_baseline/local/sed_trainer_pretrained.py` (2208 lines)

**Class:** `SEDTask4` (PyTorch Lightning Module)

**Key Inference Methods:**

#### `detect()` Method (Line 556)
```python
def detect(self, mel_feats, model, embeddings=None, **kwargs):
    if embeddings is None:
        return model(self.scaler(self.take_log(mel_feats)), **kwargs)
    else:
        return model(self.scaler(self.take_log(mel_feats)), embeddings=embeddings, **kwargs)
```
- Takes mel-spectrogram features and optional embeddings
- Applies scaler normalization and log transformation
- Returns both strong (frame-level) and weak (clip-level) predictions

#### `validation_step()` Method (Line 769)
- Evaluates on validation data
- Produces both student and teacher predictions
- Computes F1 scores and PSDS metrics
- Handles weak and strong label evaluation separately

#### `test_step()` Method (Line 1171)
- Final inference on test/evaluation data
- cSEBBs (change-point based Sound Event Bounding Boxes) tuning
- Post-processing with median filters
- Separate pipelines for DESED and MAESTRO classes

### 2.2 Embedding Extraction Pipeline

**File:** `/Users/takehonshion/work/iniad/dcase_base/DESED_task/dcase2024_task4_baseline/extract_embeddings.py` (260 lines)

**Purpose:** Pre-extract and cache BEATs embeddings

**Key Components:**
- `WavDataset`: Loads audio files and pads to 10 seconds
- `extract()` function: Batch processing with HDF5 storage
- Supports GPU acceleration
- Stores both global (768-dim) and frame-level embeddings (768-dim x 496 frames)

**Usage:**
```bash
python extract_embeddings.py --output_dir ./embeddings --config confs/pretrained.yaml
```

### 2.3 Post-processing

**File:** `/Users/takehonshion/work/iniad/dcase_base/DESED_task/dcase2024_task4_baseline/desed_task/utils/postprocess.py`

**Features:**
- `ClassWiseMedianFilter`: Applies per-class median filtering
- Reduces false positives from frame-level predictions
- Class-wise window sizes configurable

### 2.4 cSEBBs Post-processing

**File:** `/Users/takehonshion/work/iniad/dcase_base/DESED_task/dcase2024_task4_baseline/sebbs/sebbs/csebbs.py` (741 lines)

**Purpose:** Convert frame-level scores to event bounding boxes

**Key Methods:**
- `predict()`: Generate SEBBs from posterior scores
- `tune()`: Grid search hyperparameters for PSDS maximization
- Change-point detection for precise event boundaries
- Adaptive segment merging

---

## 3. Audio Data Processing

### 3.1 Audio Reading and Preprocessing

**File:** `/Users/takehonshion/work/iniad/dcase_base/DESED_task/dcase2024_task4_baseline/desed_task/dataio/pre_datasets.py` and `datasets.py` (485, 478 lines)

**Key Functions:**

#### `read_audio()`
```python
def read_audio(file, multisrc, random_channel, pad_to, test=False):
    mixture, fs = torchaudio.load(file)
    if not multisrc:
        mixture = to_mono(mixture, random_channel)
    if pad_to is not None:
        mixture, onset_s, offset_s, padded_indx = pad_audio(mixture, pad_to, fs, test=test)
    return mixture, onset_s, offset_s, padded_indx
```

#### `pad_audio()`
- Pads audio to fixed length (10 seconds = 160,000 samples at 16kHz)
- Random cropping during training
- Deterministic cropping (from start) during testing
- Tracks onset/offset times and padding indices

#### `to_mono()`
- Converts multi-channel audio to mono
- Option for random channel selection

#### `process_labels()`
- Adjusts label timings based on audio cropping
- Clips labels to audio duration boundaries

### 3.2 Feature Extraction

**Feature Configuration (feats):**
```yaml
feats:
  n_mels: 128           # Mel-frequency bins
  n_filters: 2048       # FFT size
  hop_length: 256       # Frame hop in samples
  n_window: 2048        # Window size
  sample_rate: 16000    # Target sample rate
  f_min: 0              # Min frequency
  f_max: 8000           # Max frequency (Nyquist: 8000 Hz for 16kHz)
```

**Implementation:** `SEDTask4._init_mel_spec()`
- Uses `torchaudio.transforms.MelSpectrogram`
- Logarithmic amplitude transformation
- Instance-level or dataset-level scaler normalization

**Log Transformation:**
```python
amp_to_db = AmplitudeToDB(stype="amplitude")
amp_to_db.amin = 1e-5  # As in librosa
return amp_to_db(mels).clamp(min=-50, max=80)
```

### 3.3 Audio Data Augmentation

**File:** `/Users/takehonshion/work/iniad/dcase_base/DESED_task/dcase2024_task4_baseline/desed_task/data_augm.py`

**Augmentation Methods:**

1. **Mixup**
   - Interpolates between random pairs of samples
   - Beta distribution (alpha=0.2, beta=0.2)
   - "soft" mode: weighted interpolation
   - "hard" mode: logical OR combination
   - Applied within dataset boundaries (no mixing datasets)

2. **Frame Shift**
   - Random Gaussian shift with std=90 samples
   - Corresponding label shift in time domain
   - Preserves event alignment

3. **Add Noise**
   - White noise with configurable SNR
   - SNR range: 6-30 dB (tunable)
   - Per-batch or per-sample application

4. **SpecAugment (in CRNN model)**
   - TimeMasking: masks temporal regions
   - FrequencyMasking: masks frequency regions
   - Configurable probability and length

### 3.4 Dataset Classes

**StronglyAnnotatedSet:**
- Audio + frame-level labels (onset/offset + event)
- Supports embeddings loading from HDF5
- Multi-source handling
- Confidence-weighted labels support

**WeakSet:**
- Audio + clip-level labels only
- Converts to binary presence/absence

**UnlabeledSet:**
- Audio only (for semi-supervised learning)
- Used in Mean Teacher training

**PreDatasets** (alternative implementation):
- Similar functionality with different interface

### 3.5 Data Resampling Pipeline

**File:** `/Users/takehonshion/work/iniad/dcase_base/DESED_task/dcase2024_task4_baseline/local/resample_folder.py`

- Converts audio to 16 kHz (target sample rate)
- Supports batch processing
- Caches resampled audio to avoid redundant computation

---

## 4. Dependencies and Libraries

### 4.1 Audio Processing Libraries

**Primary:**
- `torchaudio >= 2.8.0`: Core audio operations
  - Audio loading: `torchaudio.load()`
  - Mel-spectrogram: `torchaudio.transforms.MelSpectrogram`
  - Amplitude-to-dB: `torchaudio.transforms.AmplitudeToDB`
  - Kaldi-compliance features: `torchaudio.compliance.kaldi`

**Secondary:**
- `librosa`: Resampling operations
  - `librosa.resample()` for sample rate conversion
- `scipy`: Signal processing
  - Median filtering: `scipy.ndimage.median_filter`

### 4.2 Deep Learning Framework

- `torch >= 2.0`: Core tensor operations
- `pytorch-lightning == 1.9.*`: Training loop abstraction
  - DataLoaders, early stopping, checkpointing
  - Distributed training support
- `torchmetrics >= 0.10.0`: Evaluation metrics

### 4.3 Data Processing

- `h5py >= 3.14.0`: HDF5 embedding storage
- `pandas >= 1.0`: TSV metadata handling
- `numpy >= 1.19.0`: Numerical operations

### 4.4 Evaluation Metrics

- `sed-scores-eval >= 0.0.3`: SED-specific metrics
  - PSDS (Polyphonic Sound Detection Score)
  - Intersection-based F1
  - Frame-wise metrics
- `psds-eval >= 0.5.3`: PSDS computation
- `sed_scores_eval`: Comprehensive SED evaluation

### 4.5 Utilities

- `pytorch-lightning`: Training orchestration
- `wandb >= 0.22.3`: Experiment tracking and logging
- `tensorboard >= 2.20.0`: Visualization
- `codecarbon >= 3.0.4`: CO2 emissions tracking
- `dcase-util >= 0.2.16`: DCASE challenge utilities
- `desed >= 1.3.6`: DESED dataset utilities
- `thop >= 0.1.1`: FLOPs/MACs computation
- `tqdm`: Progress bars
- `pyyaml`: Configuration file parsing

---

## 5. Real-time/Streaming Inference

### 5.1 Current State

**No dedicated real-time inference module exists.** The codebase is optimized for:
- Full-audio (10-second clip) processing
- Batch inference
- Frame-by-frame predictions on pre-extracted features

### 5.2 Inference Bottlenecks

1. **Fixed-length input:** Audio padding to 10 seconds
2. **Pre-computed embeddings:** External BEATs model required
3. **Batch processing:** DataLoader dependency
4. **Post-processing:** cSEBBs tuning requires validation data

### 5.3 Potential Streaming Implementation

To implement streaming/real-time inference, would need:

1. **Sliding window processing:**
   - Buffer audio in 10-second windows
   - Use frame overlap for smoothing
   - Gradual shift of prediction windows

2. **Incremental embedding extraction:**
   - Process audio chunks through BEATs model
   - Cache embeddings for current window

3. **Online median filtering:**
   - Update filter state incrementally

4. **Event boundary estimation:**
   - Simplified cSEBBs without full grid search
   - Use pre-tuned parameters

### 5.4 Current Inference Patterns

**Batch Inference (test_step):**
```
Audio Files → Audio Loading → Mel-spectrogram → Scaler Normalization 
→ Model Forward (student/teacher) → Median Filter → cSEBBs → Event Predictions
```

**Timeline:** ~100-200ms for 10-second audio on GPU

---

## 6. Project Structure

```
DESED_task/dcase2024_task4_baseline/
├── confs/
│   ├── pretrained.yaml              # Main configuration
│   ├── before_pretrained.yaml        # Alternative config
│   └── optuna.yaml                   # Hyperparameter tuning config
│
├── desed_task/
│   ├── dataio/
│   │   ├── datasets.py               # StronglyAnnotatedSet, WeakSet, UnlabeledSet
│   │   ├── pre_datasets.py           # Alternative dataset classes
│   │   ├── sampler.py                # Batch sampler for mixed datasets
│   │   └── __init__.py               # ConcatDatasetBatchSampler
│   │
│   ├── nnet/
│   │   ├── CRNN.py                   # Main CRNN model (371 lines)
│   │   ├── CNN.py                    # CNN backbone with MixStyle (varies)
│   │   ├── RNN.py                    # Bidirectional GRU/LSTM modules
│   │   ├── pre_CNN.py                # Pre-training variant
│   │   ├── mixstyle.py               # MixStyle and FrequencyAttentionMixStyle
│   │   └── __init__.py               # Model registration
│   │
│   ├── evaluation/
│   │   └── evaluation_measures.py     # PSDS, intersection-based F1
│   │
│   ├── utils/
│   │   ├── encoder.py                # ManyHotEncoder, CatManyHotEncoder (275 lines)
│   │   ├── scaler.py                 # TorchScaler for normalization
│   │   ├── schedulers.py             # ExponentialWarmup, etc.
│   │   ├── postprocess.py            # ClassWiseMedianFilter
│   │   ├── download.py               # Download utilities
│   │   └── torch_utils.py            # Utility functions
│   │
│   └── data_augm.py                  # mixup, frame_shift, add_noise
│
├── local/
│   ├── sed_trainer_pretrained.py     # Main training loop (2208 lines)
│   ├── beats/
│   │   ├── BEATs.py                  # BEATs model implementation
│   │   ├── backbone.py               # TransformerEncoder
│   │   ├── modules.py                # Transformer modules
│   │   ├── quantizer.py              # Vector quantizer
│   │   ├── Tokenizers.py             # Audio tokenization
│   │   └── __init__.py
│   │
│   ├── utils.py                      # Utilities (MACs calculation, etc.)
│   ├── classes_dict.py               # Class label mappings
│   ├── resample_folder.py            # Batch resampling
│   └── beats_resample_folder.py      # BEATs-specific resampling
│
├── sebbs/
│   ├── sebbs/
│   │   ├── csebbs.py                 # CSEBBsPredictor (741 lines)
│   │   ├── utils.py                  # Score utilities
│   │   ├── median_filter.py          # Median filtering
│   │   └── change_detection.py       # Change point detection
│   │
│   ├── tests/                        # Unit tests
│   ├── scripts/
│   │   └── dcase2024.py              # Integration script
│   └── setup.py
│
├── dprep/
│   ├── download_prep_maestro.py      # MAESTRO dataset preparation
│   └── ...
│
├── train_pretrained.py               # Main training entry point (776 lines)
├── optuna_pretrained.py              # Hyperparameter optimization
├── extract_embeddings.py             # Embedding extraction (260 lines)
├── generate_dcase_task4_2024.py      # Dataset generation
├── chk.py                            # Checkpoint utilities
│
└── run_*.sh                          # Experiment scripts
    ├── run_exp.sh                    # Main experiment
    ├── run_attention_experiments.sh
    ├── run_attn2mix_exp.sh
    └── ...

PSDS_Eval/                            # Evaluation utilities
└── meta/
    └── metrics_test/                 # Test data for metrics
```

---

## 7. Training Pipeline

### 7.1 Main Entry Point

**File:** `train_pretrained.py` (776 lines)

**Workflow:**
1. Load configuration from YAML
2. Resample training/validation data to 16kHz
3. Generate dataset duration files
4. Create encoders for DESED and MAESTRO classes
5. Initialize datasets (strong, weak, unlabeled, validation, test)
6. Create model, optimizer, schedulers
7. Launch PyTorch Lightning trainer
8. Perform inference on test set

### 7.2 Key Training Features

**Mean Teacher Semi-Supervised Learning:**
- Student model trained on labeled + unlabeled data
- Teacher model updated via EMA (factor: 0.999)
- Consistency loss between student and teacher on unlabeled data
- Supervised loss on labeled data

**Optional Confident Mean Teacher (CMT):**
- Applies confidence-weighted thresholding
- Clip-level threshold: phi_clip (default: 0.5)
- Frame-level threshold: phi_frame (default: 0.7)
- Confidence-weighted consistency loss

**Batch Composition (mixed datasets):**
- MAESTRO synthetic (12 samples)
- Synthetic DESED (6 samples)
- Strong DESED (6 samples)
- Weak DESED (12 samples)
- Unlabeled (24 samples)

**Training Hyperparameters:**
```yaml
training:
  batch_size: [12, 6, 6, 12, 24]
  n_epochs: 300
  early_stop_patience: 200
  ema_factor: 0.999
  const_max: 2                 # Unlabeled loss weight
  n_epochs_warmup: 50
  epoch_decay: 100
  gradient_clip: 5.0
  mixup: soft
  mixup_prob: 0.5
  val_thresholds: [0.5]
  n_test_thresholds: 50
  validation_interval: 10
```

### 7.3 Optimization

**Optimizer:** Adam
- Learning rate: 0.001 (scheduler: ExponentialWarmup)
- Gradient clipping: 5.0

**Scheduler:**
- Exponential warmup for first 50 epochs
- Decay factor: epoch_decay (100)

### 7.4 Evaluation

**Metrics:**
- F1 score (frame-wise for strong labels, segment-wise for weak)
- PSDS (Polyphonic Sound Detection Score)
- Intersection-based metrics

**Validation Frequency:** Every 10 epochs

**Early Stopping:** 200 epochs patience on PSDS metric

---

## 8. Configuration System

### 8.1 YAML Configuration Structure

**File:** `confs/pretrained.yaml` (149 lines)

**Main Sections:**
- `pretrained`: Model initialization (BEATs)
- `cmt`: Confident Mean Teacher settings
- `sebbs`: cSEBBs post-processing enable
- `training`: Training hyperparameters
- `scaler`: Feature normalization
- `data`: Dataset paths
- `opt`: Optimizer settings
- `feats`: Feature extraction parameters
- `net`: Model architecture parameters

### 8.2 Command-line Overrides

Training supports CLI argument overrides:
```bash
python train_pretrained.py \
    --mixstyle_type moreMix \
    --wandb_dir moreMix \
    --cmt \
    --sebbs
```

---

## 9. Experiment Tracking

### 9.1 Weights & Biases (W&B) Integration

**Configuration:**
```yaml
net:
  use_wandb: True
  wandb_dir: "MixStyle"          # Project name
```

**Logged Metrics:**
- Training/validation losses
- F1 scores (weak/strong)
- PSDS scores
- Learning rate
- Training step

### 9.2 TensorBoard Logging

- Automatic scalar logging
- Checkpoints saved to `./lightning_logs/`

---

## 10. Summary

### Strengths
1. **Comprehensive SED system** for multi-class, multi-label event detection
2. **Semi-supervised learning** with Mean Teacher framework
3. **Multi-dataset support** with domain adaptation (DESED + MAESTRO)
4. **Flexible model architecture** supporting embeddings injection
5. **Extensive post-processing** with cSEBBs for precise event boundaries
6. **Production-ready** with wandb integration and experiment tracking

### Limitations
1. **No real-time inference** implementation
2. **Fixed-length input** (10 seconds) requirement
3. **Dependency on pre-extracted embeddings** for efficient training
4. **Evaluation-dependent** post-processing (cSEBBs tuning requires validation set)

### Key Files for Modification
- **Model architecture:** `desed_task/nnet/CRNN.py`
- **Training logic:** `local/sed_trainer_pretrained.py`
- **Inference:** `local/sed_trainer_pretrained.py` (test_step)
- **Data loading:** `desed_task/dataio/datasets.py`
- **Configuration:** `confs/pretrained.yaml`

