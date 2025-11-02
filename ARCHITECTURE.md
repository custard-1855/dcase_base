# DCASE SED System Architecture

## System Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    DCASE 2024 Task 4 SED Pipeline                       │
└─────────────────────────────────────────────────────────────────────────┘

INPUT STAGE
├─ Raw Audio Files
│  ├─ Training: Strong/Weak/Unlabeled labels
│  ├─ Sample rate: 16 kHz (44 kHz → 16 kHz resampling)
│  └─ Duration: 10 seconds (padded/cropped)
│
├─ Audio Preprocessing (desed_task/dataio/)
│  ├─ Mono conversion
│  ├─ Fixed-length padding (10 sec = 160,000 samples)
│  ├─ Label alignment (onset/offset adjustment)
│  └─ Return: (audio, onset, offset, padding_index)
│
└─ Feature Extraction (local/sed_trainer_pretrained.py)
   ├─ Mel-spectrogram (128 bins, 2048 FFT, 256 hop)
   ├─ Amplitude-to-dB conversion (clamp: -50 to 80 dB)
   ├─ Instance-level normalization (TorchScaler)
   └─ Output: (batch, 128 mel bins, ~625 frames)


EMBEDDING STAGE (Optional Pre-processing)
├─ BEATs Model (local/beats/BEATs.py)
│  ├─ Input: Raw audio (44 kHz or 16 kHz)
│  ├─ Architecture: Transformer encoder (12 layers, 768 dim)
│  ├─ Output: 
│  │  ├─ Global embedding (768-dim) - clip-level representation
│  │  └─ Frame embeddings (768-dim × 496 frames) - temporal details
│  └─ Storage: HDF5 files (embeddings/beats/*.hdf5)
│
└─ Embedding Extraction Pipeline (extract_embeddings.py)
   ├─ Batch processing
   ├─ GPU acceleration
   └─ Pre-computed storage for training efficiency


MAIN MODEL STAGE
├─ CRNN Architecture (desed_task/nnet/CRNN.py)
│
├─┬─ CNN Backbone (desed_task/nnet/CNN.py)
│ │
│ ├─ ConvBlock 1: in=1   → filters=16  → kernel=3×3, pool=2×2
│ │  └─ MixStyle (pre)
│ │
│ ├─ ConvBlock 2: in=16  → filters=32  → kernel=3×3, pool=2×2
│ │  └─ MixStyle (post)
│ │
│ ├─ ConvBlock 3: in=32  → filters=64  → kernel=3×3, pool=1×2
│ │  └─ MixStyle (post)
│ │
│ ├─ ConvBlock 4: in=64  → filters=128 → kernel=3×3, pool=1×2
│ │  └─ MixStyle (post)
│ │
│ ├─ ConvBlock 5: in=128 → filters=128 → kernel=3×3, pool=1×2
│ │  └─ MixStyle (post)
│ │
│ ├─ ConvBlock 6: in=128 → filters=128 → kernel=3×3, pool=1×2
│ │  └─ MixStyle (post)
│ │
│ └─ ConvBlock 7: in=128 → filters=128 → kernel=3×3, pool=1×2
│    └─ Output shape: (batch, 128 channels, ~39 frames)
│
├─┬─ RNN Module (desed_task/nnet/RNN.py)
│ │
│ └─ BidirectionalGRU
│    ├─ Input: (batch, 39 frames, 128 features from CNN)
│    ├─ Hidden units: 192 (each direction)
│    └─ Output: (batch, 39 frames, 384) [192 × 2 bidirectional]
│
├─┬─ Embedding Integration (Optional)
│ │
│ ├─ Frame embedding preprocessing
│ │  ├─ If frame: GRU(512 hidden) on 768-dim embeddings
│ │  ├─ Output: 1024-dim (512×2 bidirectional)
│ │  └─ Shrink: FC(1024→128) + LayerNorm
│ │
│ ├─ Concatenation
│ │  ├─ RNN output (384) + Shrunk embedding (128) → 512
│ │  └─ FC(512→128) for final fusion
│ │
│ └─ Output: Enhanced features (batch, frames, 128)
│
└─┬─ Output Heads
  │
  ├─ Frame-level (Strong) Predictions
  │  ├─ Input: RNN output (batch, 39 frames, 384)
  │  ├─ Optional attention: FC(384→27) softmax
  │  ├─ Main output: FC(384→27)
  │  └─ Sigmoid activation → (batch, 39 frames, 27 classes)
  │
  └─ Clip-level (Weak) Predictions
     ├─ Global pooling over frames
     ├─ FC(384→27)
     └─ Sigmoid activation → (batch, 27 classes)


TEACHER-STUDENT TRAINING STAGE (local/sed_trainer_pretrained.py)
├─ Dual Model Architecture
│  ├─ Student Model (trained with gradients)
│  └─ Teacher Model (updated via EMA, no gradients)
│     └─ EMA update: teacher = 0.999 × teacher + 0.001 × student
│
├─ Semi-Supervised Learning
│  ├─ Labeled data (strong + weak):
│  │  └─ Supervised loss (BCE) on both heads
│  │
│  └─ Unlabeled data:
│     ├─ Student predictions
│     ├─ Teacher predictions (no grad)
│     └─ Consistency loss: KL divergence or BCE
│
├─ Optional: Confident Mean Teacher (CMT)
│  ├─ Step 1: Thresholding
│  │  ├─ Clip-level: φ_clip (default 0.5)
│  │  └─ Frame-level: φ_frame (default 0.7)
│  │
│  ├─ Step 2: Class-wise median filtering
│  │  └─ Per-class filter window sizes
│  │
│  └─ Step 3: Confidence-weighted consistency loss
│     └─ Loss weight = confidence score × standard weight
│
└─ Data Augmentation
   ├─ Mixup (soft/hard mode, within-dataset only)
   ├─ SpecAugment (time/freq masking in model)
   ├─ Frame shift (with label adjustment)
   └─ Gaussian noise (SNR 6-30 dB)


INFERENCE STAGE
├─ Feature Extraction
│  ├─ Mel-spectrogram (same as training)
│  └─ Normalization (same scaler)
│
├─ Model Inference
│  ├─ Student model predictions (test_step)
│  ├─ Teacher model predictions (optional comparison)
│  └─ Returns: strong predictions (batch, frames, classes)
│             weak predictions (batch, classes)
│
└─ Post-Processing
   ├─ 1. Median Filtering (ClassWiseMedianFilter)
   │    ├─ Per-class filtering
   │    ├─ Window sizes: [3, 9, 9, 5, 5, 5, 9, 7, 11, 9, 7, 3, 9, 13, 7, 1, 13, 3, 13, 7, 5, 5, 1, 13, 17, 13, 15]
   │    └─ Output: Smoothed frame-level scores
   │
   ├─ 2. cSEBBs (Change-point based Sound Event Bounding Boxes)
   │    ├─ Change-point detection (onset/offset)
   │    ├─ Segment merging based on thresholds
   │    │  ├─ Absolute threshold: merge_threshold_abs
   │    │  └─ Relative threshold: merge_threshold_rel
   │    └─ Output: Event bounding boxes (onset, offset, class, confidence)
   │
   └─ 3. Event Prediction
      ├─ Format: TSV with columns [filename, onset, offset, event_label, confidence]
      └─ Evaluation: PSDS, F1-score, intersection-based metrics


EVALUATION STAGE
├─ Metrics Computation
│  ├─ Frame-level F1 (strong labels)
│  ├─ Segment-wise F1 (weak labels)
│  ├─ PSDS (Polyphonic Sound Detection Score)
│  │  └─ Considers temporal tolerance and class overlap
│  ├─ Intersection-based metrics
│  └─ Per-class and macro-averaged scores
│
├─ Validation Loop (every 10 epochs)
│  ├─ Student model evaluation
│  ├─ Teacher model evaluation
│  └─ Early stopping on PSDS (patience: 200 epochs)
│
└─ Test Inference
   ├─ cSEBBs tuning on validation set
   │  └─ Grid search for hyperparameters (PSDS maximization)
   └─ Final prediction on test set


DATA FLOW SUMMARY
┌──────────────┐
│  Audio Files │
└──────┬───────┘
       ↓
┌──────────────────────┐
│ Audio Processing &   │
│ Feature Extraction   │
└──────┬───────────────┘
       ↓
┌──────────────────────┐
│ BEATs Embeddings     │ (optional, pre-computed)
│ (768-dim per frame)  │
└──────┬───────────────┘
       ↓
┌──────────────────────────────────────┐
│ CRNN Model                           │
│ (CNN+RNN+embedding fusion)           │
│ Output: frame & clip predictions     │
└──────┬───────────────────────────────┘
       ↓
┌──────────────────────┐
│ Post-processing      │
│ (Median Filter +     │
│  cSEBBs)             │
└──────┬───────────────┘
       ↓
┌──────────────────────┐
│ Event Predictions    │
│ (onset, offset,      │
│  event, confidence)  │
└──────────────────────┘


KEY DIMENSIONS
┌─────────────────────────────────────────────────┐
│ Audio:            (batch, 1, 160000) samples    │
│ Mel-spectrogram:  (batch, 128, 625) frames      │
│ CNN output:       (batch, 128, 39) frames       │
│ RNN output:       (batch, 39, 384) dims         │
│ Frame pred:       (batch, 39, 27) classes       │
│ Clip pred:        (batch, 27) classes           │
│ BEATs embed:      (batch, 768, 496) dims        │
│ Classes:          27 (DESED + MAESTRO)          │
│ Frames:           ~625 @ 16ms/frame             │
└─────────────────────────────────────────────────┘
